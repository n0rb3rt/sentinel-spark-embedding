"""Sedona utility functions and UDFs"""
import math
from pyspark.sql.functions import expr
from pyspark import SparkConf
from pyspark.sql import SparkSession
from sedona.spark import SedonaContext
from ..config.config import CONFIG

def create_sedona_session(app_name="SentinelProcessing"):
    """Create Sedona-enabled Spark session with unified config"""
    spark_conf = SparkConf().setAll(CONFIG.spark.items())
    spark_session = SparkSession.builder \
        .appName(app_name) \
        .config(conf=spark_conf) \
        .getOrCreate()
    return SedonaContext.create(spark_session)

def get_global_chips(spark, aoi_bounds, cell_size=CONFIG.jobs.chip_extraction.chip_size_degrees):
    """Generate global 256px grid chips using Sedona SQL"""
    return spark.sql(f"""
        WITH chip_coords AS (
            SELECT 
                concat('chip_', x, '_', y) as chip_id, 
                x, y,
                -180 + (x * {cell_size}) as minx, 
                -90 + (y * {cell_size}) as miny,
                -180 + ((x + 1) * {cell_size}) as maxx, 
                -90 + ((y + 1) * {cell_size}) as maxy
            FROM (
                SELECT explode(sequence(
                    {math.floor((aoi_bounds[0] + 180) / cell_size)}, 
                    {math.ceil((aoi_bounds[2] + 180) / cell_size) - 1}
                )) as x
            ) CROSS JOIN (
                SELECT explode(sequence(
                    {math.floor((aoi_bounds[1] + 90) / cell_size)}, 
                    {math.ceil((aoi_bounds[3] + 90) / cell_size) - 1}
                )) as y
            )
        ),
        chip_geometries AS (
            SELECT *,
                ST_MakeEnvelope(minx, miny, maxx, maxy, 4326) as chip_geometry
            FROM chip_coords
        )
        SELECT *,
            ST_GeoHash(ST_Centroid(chip_geometry), 6) as geohash,
            floor(x / 6) as region_x,
            floor(y / 6) as region_y,
            concat('region_', floor(x / 6), '_', floor(y / 6)) as region_id
        FROM chip_geometries
    """)

