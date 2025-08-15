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

def get_global_chips(spark, aoi_bounds):
    """Generate global grid chips using Sedona SQL with pre-calculated tile bounds"""
    from .raster_utils import get_chip_size_degrees
    cell_size = get_chip_size_degrees()
    region_size = int(CONFIG.jobs.chip_extraction.region_size_chips)
    chip_pixels = int(CONFIG.jobs.chip_extraction.chip_size_pixels)
    
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
            concat('region_', floor(x / {region_size}), '_', floor(y / {region_size})) as region_id,
            -- Pre-calculate region bounds for UDF optimization
            CAST(-180 + (floor(x / {region_size}) * {region_size} * {cell_size}) AS DOUBLE) as region_minx,
            CAST(-90 + (floor(y / {region_size}) * {region_size} * {cell_size}) AS DOUBLE) as region_miny,
            CAST(-180 + ((floor(x / {region_size}) + 1) * {region_size} * {cell_size}) AS DOUBLE) as region_maxx,
            CAST(-90 + ((floor(y / {region_size}) + 1) * {region_size} * {cell_size}) AS DOUBLE) as region_maxy,
            -- Pre-calculate chip position within region
            CAST((x % {region_size}) * {chip_pixels} AS INT) as chip_start_x,
            CAST((y % {region_size}) * {chip_pixels} AS INT) as chip_start_y
        FROM chip_geometries
    """)

