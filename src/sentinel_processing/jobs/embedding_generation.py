#!/usr/bin/env python3
"""
Clay Embedding Generation Job - Basic approach with compiled model
"""
import time
import torch
import numpy as np
from io import BytesIO

from pyspark.sql.functions import current_timestamp, lit, col
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.functions import predict_batch_udf
from sentinel_processing.config import CONFIG, get_ssm_parameter
from sentinel_processing.lib.sedona_utils import create_sedona_session
from sentinel_processing.lib.clay_utils import ClayModelSingleton

def make_clay_predict_fn(expected_batches):
    """Create compiled Clay model predict function"""
    
    model = ClayModelSingleton.get_model()
    device = next(model.parameters()).device
    
    def predict(clay_tensors: np.ndarray) -> np.ndarray:
        """Predict embeddings for batch of clay tensors"""
        batch_start = time.time()
        
        try:
            # Load tensors from binary blobs
            batch_data = [np.load(BytesIO(blob)) for blob in clay_tensors]
            
            # Create datacube
            datacube = {
                'pixels': torch.from_numpy(np.stack([d['pixels'] for d in batch_data])).to(device),
                'time': torch.from_numpy(np.stack([d['time'] for d in batch_data])).to(device),
                'latlon': torch.from_numpy(np.stack([d['latlon'] for d in batch_data])).to(device),
                'waves': torch.from_numpy(batch_data[0]['waves']).to(device),
                'gsd': torch.from_numpy(batch_data[0]['gsd']).to(device),
            }
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = model(datacube)
            
            result = embeddings.cpu().numpy()
            
            batch_time = time.time() - batch_start
            throughput = len(clay_tensors) / batch_time
            progress_pct = min(100, (batch_count / expected_batches) * 100) if expected_batches > 0 else 0
            print(f"Batch {batch_count}/{expected_batches} ({progress_pct:.0f}%): {len(clay_tensors)} chips in {batch_time:.2f}s ({throughput:.1f} chips/s)")
            
            return result
            
        except Exception as e:
            print(f"Batch {batch_count}/{expected_batches} failed: {e}")
            return np.zeros((len(clay_tensors), 1024), dtype=np.float32)
    
    return predict

def main():
    start_time = time.time()
    
    spark = create_sedona_session("ClayEmbeddingGeneration")
    
    input_table = CONFIG.jobs.embedding_generation.input_table
    output_table = CONFIG.jobs.embedding_generation.output_table
    batch_size = CONFIG.jobs.embedding_generation.get('batch_size', 32)
    
    try:
        database_name = CONFIG.sentinel.database_name
        chips_df = spark.read.format("iceberg").load(f"{database_name}.{input_table}")
        total_chips = chips_df.count()
        expected_batches = (total_chips + batch_size - 1) // batch_size  # Ceiling division
        print(f"Processing {total_chips:,} chips with batch size {batch_size}")
        print(f"Expected {expected_batches} batches")
        
        # Create Clay UDF with progress tracking
        clay_udf = predict_batch_udf(
            lambda: make_clay_predict_fn(expected_batches),
            return_type=ArrayType(FloatType()),
            batch_size=batch_size
        )
        
        embeddings_df = chips_df \
            .withColumn("embedding", clay_udf("clay_tensor")) \
            .withColumn("model_version", lit("clay-v1.5")) \
            .withColumn("created_at", current_timestamp()) \
            .select("id", "geohash", "datetime", "embedding", "model_version", "created_at") \
            .sortWithinPartitions("geohash", "id", "datetime")
        
        # Create minimal embeddings table
        schema_ddl = embeddings_df._jdf.schema().toDDL()
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {database_name}.{output_table} (
                {schema_ddl}
            ) USING ICEBERG
            PARTITIONED BY (truncate(geohash, 4), month(datetime))
            TBLPROPERTIES (
                'write.sort-order'='geohash,id,datetime'
            )
        """)
        
        # Write embeddings
        embeddings_df.write.format("iceberg").mode("append").save(f"{database_name}.{output_table}")
        
        total_time = time.time() - start_time
        avg_throughput = total_chips / total_time if total_time > 0 else 0
        print(f"Completed {total_chips:,} chips in {total_time / 60:.1f}m (avg {avg_throughput:.1f} chips/s)")
        
        print(f"Embedding generation completed for {total_chips:,} chips")
        
    finally:
        spark.stop()



if __name__ == "__main__":
    main()