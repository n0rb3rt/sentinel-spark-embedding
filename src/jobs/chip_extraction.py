#!/usr/bin/env python3
"""
Sentinel-2 Chip Extraction Job
Extracts 256px chips from Sentinel-2 scenes with geohash partitioning
"""
from sedona.stac.client import Client
from pyspark.sql.functions import *
from ..config.config import CONFIG
from ..lib.sedona_utils import get_global_chips, create_sedona_session
from ..lib.raster_utils import process_scene_chips, scene_chips_schema
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
    
    # Sentinel-2 bands to process
    bands = ['blue', 'green', 'red', 'nir', 'scl']
    
    # Create Spark session
    spark = create_sedona_session("SentinelChipExtraction")
    
    try:
        # 1. Search for Sentinel-2 scenes
        print(f"Querying scenes for AOI: {aoi_bounds}, dates: {start_date}/{end_date}")
        client = Client.open("https://earth-search.aws.element84.com/v1")
        
        scenes_raw = client.search(
            collection_id="sentinel-2-l2a",
            bbox=aoi_bounds,
            datetime=[start_date, end_date],
            return_dataframe=True
        ).select("id", "datetime", "geometry", "assets")
        
        print(f"Found {scenes_raw.count()} scenes")
        
        # 2. Prepare raw scene data
        scenes_s3 = scenes_raw.select(
            "id",
            "datetime", 
            "geometry",
            *[regexp_replace(col(f"assets.{band}.href"), 
                           "https://sentinel-cogs.s3.us-west-2.amazonaws.com", 
                           "s3://sentinel-cogs").alias(f"{band}_s3") 
              for band in bands]
        ).repartition("id", "datetime").cache()
        
        # Broadcast scene URLs for UDF
        url_lookup = {(row['id'], str(row['datetime'])): {
            f'{band}_s3': row[f'{band}_s3'] for band in bands
        } for row in scenes_s3.select("id", "datetime", *[f"{band}_s3" for band in bands]).collect()}
        
        broadcast_urls = spark.sparkContext.broadcast(url_lookup)
        
        # 3. Join scenes with chip grid
        chips_df = get_global_chips(spark, aoi_bounds)
        
        scene_chip_pairs = scenes_s3.crossJoin(chips_df) \
            .filter(expr("ST_Intersects(geometry, chip_geometry)")) \
            .withColumn("is_complete", expr("ST_Contains(geometry, chip_geometry)")) \
            .drop("geometry", "chip_geometry")
        
        # 4. Process chips with UDF
        print("Processing scene-chip pairs...")
        all_chips = scene_chip_pairs.groupBy("id", "datetime", "region_id").applyInPandas(
            lambda df: process_scene_chips(df, broadcast_urls),
            scene_chips_schema
        )
        
        # 5. Optimize for Iceberg write
        processed_chips = all_chips \
            .repartition(expr("substring(geohash, 1, 4)")) \
            .sortWithinPartitions("geohash", "chip_id", "datetime") \
            .withColumn("created_at", current_timestamp())
        
        # 6. Create Iceberg table
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS sentinel.chips_raw (
                chip_id STRING, 
                datetime TIMESTAMP, 
                chip_raster BINARY,
                is_complete BOOLEAN, 
                cloud_coverage FLOAT, 
                geohash STRING,
                created_at TIMESTAMP
            ) USING ICEBERG LOCATION '{output_path}'
            PARTITIONED BY (
                truncate(4, geohash), 
                year(datetime)
            )
            TBLPROPERTIES (
                'write.target-file-size-bytes'='134217728',
                'write.parquet.compression-codec'='zstd',
                'write.sort-order'='geohash,chip_id,datetime'
            )
        """)
        
        # 7. Write to Iceberg table
        processed_chips.write.mode("append").insertInto("sentinel.chips_raw")
        
        print(f"Successfully processed chips")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()