#!/usr/bin/env python3
"""
Change Detection Job - Compute cosine similarity between consecutive embeddings
"""
import sys
import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit, when
from sentinel_processing.config import CONFIG, get_ssm_parameter
from sentinel_processing.lib.sedona_utils import create_sedona_session

def compute_change_scores(chip_group: pd.DataFrame) -> pd.DataFrame:
    """Compute cosine similarity to previous observation for each chip location"""
    
    # Sort by datetime
    chip_group = chip_group.sort_values('datetime').reset_index(drop=True)
    
    def cosine_similarity(a, b):
        """Compute cosine similarity between two vectors"""
        if a is None or b is None:
            return None
        a = np.array(a)
        b = np.array(b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = np.dot(a, b) / (norm_a * norm_b)
        return float(np.clip(similarity, -1.0, 1.0))  # Clamp to valid range
    
    # Compute similarities
    similarities = []
    for i in range(len(chip_group)):
        if i == 0:
            similarities.append(0.0)  # First observation is baseline
        else:
            curr_embedding = chip_group.iloc[i]['embedding']
            prev_embedding = chip_group.iloc[i-1]['embedding']
            sim = cosine_similarity(curr_embedding, prev_embedding)
            similarities.append(sim)
    
    # Create result with new columns
    return chip_group.assign(
        change_score=similarities
    )

def main():
    start_time = time.time()
    
    # Create Spark session with Sedona
    spark = create_sedona_session("ChangeDetection")
    
    # Configuration
    input_table = CONFIG.jobs.embedding_generation.output_table
    output_table = CONFIG.jobs.change_detection.output_table
    
    try:
        # Join chips with embeddings for change detection
        database_name = CONFIG.sentinel.database_name
        chips_df = spark.read.format("iceberg").load(f"{database_name}.{CONFIG.jobs.chip_extraction.output_table}")
        embeddings_df = spark.read.format("iceberg").load(f"{database_name}.{input_table}")
        
        joined_df = chips_df.join(embeddings_df, ["id", "geohash", "datetime"]).select(
            "id", "geohash", "datetime", "embedding"
        )
        
        total_embeddings = embeddings_df.count()
        print(f"Processing {total_embeddings:,} embeddings for change detection")
        
        # Group by geohash (spatial location) and compute change scores
        from pyspark.sql.types import StructType, StructField, FloatType
        
        output_schema = StructType(joined_df.schema.fields + [
            StructField("change_score", FloatType(), True)
        ])
        
        change_df = joined_df.groupBy("geohash").applyInPandas(
            compute_change_scores,
            schema=output_schema
        ).select("id", "geohash", "datetime", "change_score", current_timestamp().alias("created_at")) \
         .sortWithinPartitions("geohash", "id", "datetime")
        
        # Create minimal changes table
        schema_ddl = change_df._jdf.schema().toDDL()
        create_table = f"""
            CREATE TABLE IF NOT EXISTS {database_name}.{output_table} (
                {schema_ddl}
            ) USING ICEBERG
            PARTITIONED BY (truncate(geohash, 4), month(datetime))
            TBLPROPERTIES (
                'write.sort-order'='geohash,id,datetime'
            )
        """
        print(create_table)
        spark.sql(create_table)
        
        # Write to Iceberg table
        change_df.write \
            .format("iceberg") \
            .mode("overwrite") \
            .save(f"{database_name}.{output_table}")
        
        print(f"Change detection completed in {(time.time() - start_time) / 60:.1f}m")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()