#!/usr/bin/env python3
"""
Temporal Merge Job
Fill incomplete chips with previous data for consistent embeddings
"""
from src.config.config import CONFIG
from src.lib.sedona_utils import create_sedona_session
from pyspark.sql.functions import *
from pyspark.sql.window import Window

def main():
    # Use config values (with automatic CLI overrides)
    lookback_days = CONFIG.jobs.temporal_merge.lookback_days
    
    spark = create_sedona_session("SentinelTemporalMerge")
    
    try:
        # Create processed chips table if not exists
        spark.sql("""
            CREATE TABLE IF NOT EXISTS sentinel.chips_processed (
                chip_id STRING,
                datetime TIMESTAMP,
                chip_raster BINARY,
                is_complete BOOLEAN,
                cloud_coverage FLOAT,
                geohash STRING,
                embedding ARRAY<FLOAT>,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            ) USING ICEBERG
            PARTITIONED BY (substring(geohash, 1, 4))
            TBLPROPERTIES (
                'write.target-file-size-bytes'='134217728',
                'write.delete.target-file-size-bytes'='67108864',
                'write.merge.target-file-size-bytes'='134217728',
                'write.parquet.compression-codec'='zstd',
                'write.metadata.delete-after-commit.enabled'='true',
                'write.metadata.previous-versions-max'='10',
                'write.distribution-mode'='hash'
            )
        """)
        
        spark.sql("""
            ALTER TABLE sentinel.chips_processed 
            WRITE ORDERED BY geohash, chip_id, datetime
        """)
        
        # Read raw chips
        raw_chips = spark.table("sentinel.chips_raw")
        
        # Define window for temporal gap filling
        window_spec = Window.partitionBy("chip_id").orderBy("datetime")
        
        # Fill gaps with previous complete chips
        filled_chips = raw_chips.withColumn(
            "filled_raster",
            when(col("is_complete"), col("chip_raster"))
            .otherwise(
                last(when(col("is_complete"), col("chip_raster")), ignorenulls=True)
                .over(window_spec.rowsBetween(Window.unboundedPreceding, -1))
            )
        ).filter(col("filled_raster").isNotNull()) \
         .withColumn("chip_raster", col("filled_raster")) \
         .withColumn("embedding", lit(None).cast("array<float>")) \
         .withColumn("updated_at", current_timestamp()) \
         .drop("filled_raster")
        
        # Write to processed chips table
        filled_chips.write.mode("overwrite").insertInto("sentinel.chips_processed")
        
        print(f"Temporal merge completed: {filled_chips.count()} chips")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()