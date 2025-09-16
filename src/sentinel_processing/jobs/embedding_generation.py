#!/usr/bin/env python3
"""
Clay Embedding Generation Job - Spark with Pandas UDF
"""
import sys
import time
import datetime
import torch
import pandas as pd
import numpy as np
from io import BytesIO
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.functions import predict_batch_udf
from sentinel_processing.config import CONFIG
from sentinel_processing.lib.sedona_utils import create_sedona_session
from sentinel_processing.lib.clay_utils import ClayModelSingleton

def make_clay_predict_fn():
    """Create compiled Clay model predict function"""
    
    # Prefer MPS on Apple Silicon, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Loading compiled Clay model on {device}...")
    model = ClayModelSingleton.get_model().to(device)
    print("Compiled Clay model loaded successfully")
    
    def predict(clay_tensors: np.ndarray) -> np.ndarray:
        """Predict embeddings for batch of clay tensors"""
        batch_start = time.time()
        
        try:
            # Load all tensors in batch
            batch_data = [np.load(BytesIO(blob)) for blob in clay_tensors]
            
            # Stack per-chip tensors
            pixels_arrays = [data['pixels'] for data in batch_data]
            time_arrays = [data['time'] for data in batch_data]
            latlon_arrays = [data['latlon'] for data in batch_data]
            
            # Use shared tensors from first chip
            waves = torch.from_numpy(batch_data[0]['waves']).to(device)
            gsd = torch.from_numpy(batch_data[0]['gsd']).to(device)
            
            # Create datacube exactly like the working example
            datacube = {
                'pixels': torch.from_numpy(np.stack(pixels_arrays)).to(device),
                'time': torch.from_numpy(np.stack(time_arrays)).to(device),
                'latlon': torch.from_numpy(np.stack(latlon_arrays)).to(device),
                'waves': waves,
                'gsd': gsd
            }
            
            with torch.no_grad():
                embeddings = model(datacube)
            
            batch_time = time.time() - batch_start
            time_per_chip = batch_time / len(clay_tensors)
            print(f"{datetime.datetime.now().strftime('%H:%M:%S')} - Batch of {len(clay_tensors)} chips in {batch_time:.2f}s ({time_per_chip:.3f}s/chip)")
            return embeddings.cpu().numpy()
            
        except Exception as e:
            print(f"Batch embedding generation failed: {e}")
            return np.zeros((len(clay_tensors), 1024), dtype=np.float32)  # Compiled model outputs 1024 dims
    
    return predict

def main():
    start_time = time.time()
    
    # Create Spark session with Sedona
    spark = create_sedona_session("ClayEmbeddingGeneration")
    
    # Configuration
    input_table = CONFIG.jobs.embedding_generation.input_table
    output_table = CONFIG.jobs.embedding_generation.output_table
    batch_size = CONFIG.jobs.embedding_generation.get('batch_size', 32)
    
    try:

        
        # Read chips from Iceberg table (limit for testing)
        chips_df = spark.read \
            .format("iceberg") \
            .load(f"sentinel.{input_table}")
        
        print(f"Processing {chips_df.count()} chips (test batch)")
        
        # Create compiled Clay UDF with automatic batching
        clay_udf = predict_batch_udf(
            make_clay_predict_fn,
            return_type=ArrayType(FloatType()),
            batch_size=batch_size  # Compiled model can handle larger batches
        )
        
        embeddings_df = chips_df \
            .withColumn("global_embedding", clay_udf("clay_tensor")) \
            .withColumn("model_version", lit("clay-v1.5")) \
            .withColumn("created_at", current_timestamp()) \
            .select("id", "scene_id", "datetime", "geohash", "geometry", "global_embedding", "model_version", "created_at")
        
        # Create Iceberg table if it doesn't exist
        schema_ddl = embeddings_df._jdf.schema().toDDL()
        create_table = f"""
            CREATE TABLE IF NOT EXISTS sentinel.{output_table} (
                {schema_ddl}
            ) USING ICEBERG
            PARTITIONED BY (truncate(geohash, 4), month(datetime))
            TBLPROPERTIES (
                'write.target-file-size-bytes'='134217728',
                'write.parquet.compression-codec'='zstd'
            )
        """
        print(create_table)
        spark.sql(create_table)
        
        # Write to Iceberg table
        embeddings_df.write \
            .format("iceberg") \
            .mode("append") \
            .save(f"sentinel.{output_table}")
        
        processed_count = embeddings_df.count()
        
        print(f"Successfully processed chips in {(time.time() - start_time) / 60:.1f}m")
        
        # Show sample results
        print("Sample embeddings:")
        embeddings_df.select("id", "scene_id", "model_version").show(5, truncate=False)
        
    finally:
        spark.stop()



if __name__ == "__main__":
    main()