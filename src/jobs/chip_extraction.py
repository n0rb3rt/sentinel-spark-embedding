#!/usr/bin/env python3
"""
Sentinel-2 Chip Extraction Job
Extracts 256px chips from Sentinel-2 scenes with geohash partitioning
"""
from sedona.stac.client import Client
from pyspark.sql.functions import *
from ..config.config import CONFIG
from ..lib.sedona_utils import get_global_chips, create_sedona_session
from ..lib.raster_utils import process_scene_region, scene_chips_schema
import boto3

def get_ssm_parameter(name):
    """Get parameter from SSM Parameter Store"""
    ssm = boto3.client('ssm')
    return ssm.get_parameter(Name=name)['Parameter']['Value']

def main():
    # Get runtime configuration
    bucket_name = get_ssm_parameter('/airflow/variables/bucket_name')
    aoi_bounds = CONFIG.jobs.chip_extraction.aoi_bounds
    output_path = f's3://{bucket_name}/data/sentinel/'
    start_date = CONFIG.jobs.chip_extraction.start_date
    end_date = CONFIG.jobs.chip_extraction.end_date
    
    # Create Spark session
    spark = create_sedona_session("SentinelChipExtraction")
    
    try:
        # Query Sentinel-2 scenes
        print(f"Querying scenes for AOI: {aoi_bounds}, dates: {start_date}/{end_date}")

        client = Client.open("https://earth-search.aws.element84.com/v1")
        
        scenes_df = client.search(
            collection_id="sentinel-2-l2a",
            bbox=aoi_bounds,
            datetime=[start_date, end_date],
            return_dataframe=True
        ).select("id", "datetime", "grid:code", "geometry", "assets")
        
        # Convert HTTPS URLs to S3 paths
        scenes_s3 = scenes_df.select(
            "id", "datetime", "geometry",
            *[
                regexp_replace(
                    col(f"assets.{band}.href"), 
                    "https://sentinel-cogs.s3.us-west-2.amazonaws.com", 
                    "s3://sentinel-cogs"
                ).alias(f"{band}_s3") 
              for band in ['blue', 'green', 'red', 'nir', 'scl']
            ]
        ).repartition("id", "datetime").cache()

        print(f"Found {scenes_df.count()} scenes")
        
        # Create URL lookup for broadcast
        scene_urls = scenes_s3.select("id", "datetime", "blue_s3", "green_s3", 
                                     "red_s3", "nir_s3", "scl_s3").collect()
        
        url_lookup = {(row['id'], str(row['datetime'])): {
            'blue_s3': row['blue_s3'],
            'green_s3': row['green_s3'], 
            'red_s3': row['red_s3'],
            'nir_s3': row['nir_s3'],
            'scl_s3': row['scl_s3']
        } for row in scene_urls}

        broadcast_urls = spark.sparkContext.broadcast(url_lookup)
        
        # Generate chips and create scene-chip pairs
        get_global_chips(spark, aoi_bounds).createOrReplaceTempView("chips")
        scenes_s3.createOrReplaceTempView("scenes")
        
        scene_chip_pairs = spark.sql("""
            SELECT s.id, s.datetime, c.chip_id, c.geohash, c.region_id,
                   ST_Contains(s.geometry, c.chip_geometry) as is_complete,
                   c.minx, c.miny, c.maxx, c.maxy, c.x, c.y
            FROM scenes s
            CROSS JOIN chips c
            WHERE ST_Intersects(s.geometry, c.chip_geometry)
        """)
        
        # Process chips with 6x6 tiling optimization
        print("Processing scene-chip pairs with 6x6 tiling...")

        # Add these debug lines to see what's happening:
        print(f"Unique scenes: {scene_chip_pairs.select('id').distinct().count()}")
        print(f"Unique regions: {scene_chip_pairs.select('region_id').distinct().count()}")
        print(f"Scene-region combinations: {scene_chip_pairs.select('id', 'region_id').distinct().count()}")

        
        # Add region batching to reduce S3 connection overhead
        scene_chip_pairs_batched = scene_chip_pairs.withColumn(
            "region_batch", expr("hash(region_id) % 4")
        ).cache()
        
        # Group by scene AND region batch for balanced S3 connection reuse
        all_chips = scene_chip_pairs_batched.groupBy("id", "datetime", "region_batch").applyInPandas(
            lambda df: process_scene_region(df, broadcast_urls),
            scene_chips_schema
        )
        
        all_chips_with_metadata = all_chips.withColumn("created_at", current_timestamp())

        # Create raw chips table with optimized partitioning
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS sentinel.chips_raw (
                chip_id STRING,
                datetime TIMESTAMP,
                chip_raster BINARY,
                is_complete BOOLEAN,
                cloud_coverage FLOAT,
                geohash STRING,
                created_at TIMESTAMP
            ) USING ICEBERG
            LOCATION '{output_path}'
            PARTITIONED BY (
                truncate(4, geohash),
                year(datetime)
            )
            TBLPROPERTIES (
                'write.target-file-size-bytes'='134217728',
                'write.parquet.compression-codec'='zstd',
                'write.metadata.delete-after-commit.enabled'='true',
                'write.metadata.previous-versions-max'='5',
                'write.distribution-mode'='hash',
                'write.sort-order'='geohash,chip_id,datetime'
            )
        """)
        
        # Pre-sort data by Iceberg sort order before write
        all_chips_sorted = all_chips_with_metadata.orderBy("geohash", "chip_id", "datetime")
        all_chips_sorted.write.mode("append").insertInto("sentinel.chips_raw")
        
        chip_count = all_chips_with_metadata.count()
        print(f"Successfully processed {chip_count} chips")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()