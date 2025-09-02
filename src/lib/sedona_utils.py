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
    region_size = CONFIG.jobs.chip_extraction.region_size_chips
    chip_pixels = CONFIG.jobs.chip_extraction.chip_size_pixels
    
    return spark.sql(f"""
        WITH variables AS (
            SELECT 
                CAST({cell_size} AS DOUBLE) as cell_size,
                CAST({region_size} AS DOUBLE) as region_size, 
                CAST({chip_pixels} AS DOUBLE) as chip_pixels,
                CAST({region_size} AS DOUBLE) * CAST({cell_size} AS DOUBLE) as region_cell_size
        ),
        chip_coords AS (
            SELECT 
                -- Chip boundaries
                concat('chip_', x, '_', y) as chip_id, 
                x, y,
                -180.0 + (x * v.cell_size) as minx, 
                -90.0 + (y * v.cell_size) as miny,
                -180.0 + ((x + 1) * v.cell_size) as maxx, 
                -90.0 + ((y + 1) * v.cell_size) as maxy,
                -- Region info for UDF batch reads
                floor(x / v.region_size) as region_x,
                floor(y / v.region_size) as region_y,
                concat('region_', floor(x / v.region_size), '_', floor(y / v.region_size)) as region_id,
                -180.0 + (floor(x / v.region_size) * v.region_cell_size) as region_minx,
                -90.0 + (floor(y / v.region_size) * v.region_cell_size) as region_miny,
                -180.0 + ((floor(x / v.region_size) + 1) * v.region_cell_size) as region_maxx,
                -90.0 + ((floor(y / v.region_size) + 1) * v.region_cell_size) as region_maxy,
                (x % v.region_size) * v.chip_pixels as chip_start_x,
                (y % v.region_size) * v.chip_pixels as chip_start_y
            FROM variables v
            CROSS JOIN (
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
        ),
        chip_centers AS (
            SELECT *,
                ST_Centroid(chip_geometry) as chip_center
            FROM chip_geometries
        )
        SELECT *,
            ST_GeoHash(chip_center, 6) as geohash,
            ST_Y(chip_center) as chip_center_lat,
            ST_X(chip_center) as chip_center_lon
        FROM chip_centers
    """)

