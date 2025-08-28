#!/usr/bin/env python3
"""
Sentinel-2 Chip Extraction Job
Extracts 256px chips from Sentinel-2 scenes with geohash partitioning
"""
from sedona.stac.client import Client
from pyspark.sql.functions import *
from ..config.config import CONFIG, CLAY_METADATA
from ..lib.sedona_utils import get_global_chips, create_sedona_session
from ..lib.raster_utils import SceneChipProcessor
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
    bands = CONFIG.sentinel.bands
    band_meta = CLAY_METADATA['sentinel-2-l2a'].bands

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
        
        # 2. Prepare raw scene data with explicit naming
        scenes_s3 = scenes_raw.select(
            col("id").alias("scene_id"),
            "datetime", 
            col("geometry").alias("scene_geom"),
            *[regexp_replace(col(f"assets.{band}.href"), 
                           "https://sentinel-cogs.s3.us-west-2.amazonaws.com", 
                           "s3://sentinel-cogs").alias(f"{band}_s3") 
              for band in bands]
        ).repartition("scene_id", "datetime").cache()
        
        # Broadcast scene URLs for UDF
        url_lookup = {(row['scene_id'], str(row['datetime'])): {
            f'{band}_s3': row[f'{band}_s3'] for band in bands
        } for row in scenes_s3.select("scene_id", "datetime", *[f"{band}_s3" for band in bands]).collect()}
        
        broadcast_urls = spark.sparkContext.broadcast(url_lookup)
        
        # 3. Join scenes with chip grid
        chips_df = get_global_chips(spark, aoi_bounds)
        
        scene_chip_pairs = scenes_s3.crossJoin(chips_df) \
            .filter(expr("ST_Intersects(scene_geom, chip_geometry)")) \
            .withColumn("is_complete", expr("ST_Contains(scene_geom, chip_geometry)")) \
            .drop("scene_geom", "chip_geometry")
        
        print(f"Found {scene_chip_pairs.count()} chips")
        
        # 4. Process chips with Clay normalization in UDF
        print("Processing scene-chip pairs...")
        normalized_chips = scene_chip_pairs.groupBy("scene_id", "datetime", "region_id").applyInPandas(
            lambda df: SceneChipProcessor.process(df, broadcast_urls),
            SceneChipProcessor.schema
        ).sortWithinPartitions("geohash", "id", "datetime") \
         .withColumn("created_at", current_timestamp())
        
        # Create table with partition transforms using SQL
        schema_ddl = normalized_chips._jdf.schema().toDDL()
        spark.sql(f"""
            CREATE OR REPLACE TABLE sentinel.normalized_chips (
                {schema_ddl}
            ) USING iceberg
            PARTITIONED BY (truncate(geohash, 4), month(datetime))
            LOCATION '{output_path}'
            TBLPROPERTIES (
                'write.target-file-size-bytes'='134217728',
                'write.parquet.compression-codec'='zstd',
                'write.sort-order'='geohash,id,datetime'
            )
        """)
        
        # Write to Iceberg table
        normalized_chips.writeTo("sentinel.normalized_chips").append()
        
        print(f"Successfully processed chips")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()