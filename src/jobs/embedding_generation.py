#!/usr/bin/env python3
"""
Embedding Generation Job
Generate Clay GeoFM embeddings from processed chips
"""
import os
import boto3
import torch
import numpy as np
import pandas as pd
from src.config.config import CONFIG
from src.lib.sedona_utils import create_sedona_session
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

def get_ssm_parameter(name):
    """Get parameter from SSM Parameter Store"""
    ssm = boto3.client('ssm')
    return ssm.get_parameter(Name=name)['Parameter']['Value']

def load_clay_model():
    """Load Clay model from S3"""
    model_path = "/tmp/clay_model.pth"
    if not os.path.exists(model_path):
        bucket_name = get_ssm_parameter('/airflow/variables/bucket_name')
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, f"models/{CONFIG.jobs.embedding_generation.model_path}", model_path)
    return torch.load(model_path, map_location='cpu')

@pandas_udf(returnType=ArrayType(FloatType()))
def generate_embeddings(chip_rasters):
    """Generate Clay embeddings for chip rasters"""
    model = load_clay_model()
    model.eval()
    
    embeddings = []
    for raster_bytes in chip_rasters:
        try:
            # Convert raster bytes to tensor (simplified)
            # In practice, you'd need proper raster -> tensor conversion
            tensor = torch.randn(1, 4, 256, 256)  # Placeholder
            
            with torch.no_grad():
                embedding = model(tensor).squeeze().numpy()
            
            embeddings.append(embedding.tolist())
        except Exception as e:
            print(f"Error generating embedding: {e}")
            embeddings.append([0.0] * 768)  # Default embedding size
    
    return embeddings

def main():
    # Use config values (with automatic CLI overrides)
    batch_size = CONFIG.jobs.embedding_generation.batch_size
    
    spark = create_sedona_session("SentinelEmbeddingGeneration")
    
    try:
        # Read processed chips without embeddings
        processed_chips = spark.table("sentinel.chips_processed").filter(col("embedding").isNull())
        
        # Generate embeddings
        embedded_chips = processed_chips.withColumn(
            "embedding",
            generate_embeddings(col("chip_raster"))
        ).withColumn("updated_at", current_timestamp())
        
        # Update processed chips table with embeddings using MERGE
        embedded_chips.createOrReplaceTempView("embedded_updates")
        
        spark.sql("""
            MERGE INTO sentinel.chips_processed t
            USING embedded_updates s
            ON t.chip_id = s.chip_id AND t.datetime = s.datetime
            WHEN MATCHED THEN UPDATE SET 
                embedding = s.embedding,
                updated_at = s.updated_at
        """)
        
        print(f"Generated embeddings for {processed_chips.count()} chips")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()