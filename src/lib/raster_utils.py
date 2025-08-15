"""Raster processing utilities"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from io import BytesIO
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
from ..config.config import CONFIG

def get_chip_size_degrees():
    """Calculate chip size in degrees from pixel size and ground resolution"""
    pixels = CONFIG.jobs.chip_extraction.chip_size_pixels
    meters_per_pixel = CONFIG.jobs.chip_extraction.ground_resolution_meters
    meters_per_degree = 111320  # approximate at equator
    
    chip_size_meters = pixels * meters_per_pixel
    return chip_size_meters / meters_per_degree

def create_multiband_geotiff(bands_dict, transform, crs='EPSG:4326'):
    """Create multiband GeoTIFF from band dictionary"""
    buffer = BytesIO()
    chip_pixels = CONFIG.jobs.chip_extraction.chip_size_pixels
    
    with rasterio.open(buffer, 'w', driver='GTiff', compress='lz4',
                      height=chip_pixels, width=chip_pixels, count=len(bands_dict),
                      dtype=list(bands_dict.values())[0].dtype,
                      crs=crs, transform=transform) as dst:
        
        for i, (name, band) in enumerate(bands_dict.items(), 1):
            dst.write(band, i)
            dst.set_band_description(i, name)
    
    return buffer.getvalue()

def extract_reprojected_chip(src, chip_bounds_wgs84, size=256):
    """Extract and reproject chip from raster source"""
    target_transform = from_bounds(*chip_bounds_wgs84, size, size)
    target_array = np.empty((size, size), dtype=src.dtypes[0])
    
    reproject(
        source=rasterio.band(src, 1),
        destination=target_array,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=target_transform,
        dst_crs='EPSG:4326',
        resampling=Resampling.bilinear
    )
    
    return target_array

scene_chips_schema = StructType([
    StructField("chip_id", StringType()),
    StructField("datetime", TimestampType()),
    StructField("chip_raster", BinaryType()),
    StructField("is_complete", BooleanType()),
    StructField("cloud_coverage", FloatType()),
    StructField("geohash", StringType())
])

def process_scene_chips(df, broadcast_urls) -> pd.DataFrame:
    """Process chips for a scene region using configurable tiling optimization"""
    if df.empty:
        return pd.DataFrame()
        
    scene_id = df.iloc[0]['id']
    datetime_str = str(df.iloc[0]['datetime'])
    region_id = df.iloc[0]['region_id']
    urls = broadcast_urls.value.get((scene_id, datetime_str))
    
    if not urls:
        return pd.DataFrame()
    
    results = []
    region_size = CONFIG.jobs.chip_extraction.region_size_chips
    chip_pixels = CONFIG.jobs.chip_extraction.chip_size_pixels
    tile_pixels = region_size * chip_pixels
    
    with rasterio.Env(**CONFIG.rasterio):
        # One connection per region
        with rasterio.open(urls['blue_s3']) as blue_src, \
             rasterio.open(urls['green_s3']) as green_src, \
             rasterio.open(urls['red_s3']) as red_src, \
             rasterio.open(urls['nir_s3']) as nir_src, \
             rasterio.open(urls['scl_s3']) as scl_src:
            
            print(f"Processing {len(df)} chips in region {region_id} for scene {scene_id}")
            
            try:
                # Use pre-calculated region bounds from upstream SQL
                region_minx = df.iloc[0]['region_minx']
                region_miny = df.iloc[0]['region_miny']
                region_maxx = df.iloc[0]['region_maxx']
                region_maxy = df.iloc[0]['region_maxy']
                
                # Read entire region from all bands
                region_bands = {
                    'blue': extract_reprojected_chip(blue_src, (region_minx, region_miny, region_maxx, region_maxy), tile_pixels),
                    'green': extract_reprojected_chip(green_src, (region_minx, region_miny, region_maxx, region_maxy), tile_pixels),
                    'red': extract_reprojected_chip(red_src, (region_minx, region_miny, region_maxx, region_maxy), tile_pixels),
                    'nir': extract_reprojected_chip(nir_src, (region_minx, region_miny, region_maxx, region_maxy), tile_pixels),
                    'scl': extract_reprojected_chip(scl_src, (region_minx, region_miny, region_maxx, region_maxy), tile_pixels)
                }
                
                # Extract individual chips from the region
                for _, chip in df.iterrows():
                    # Use pre-calculated chip positions from upstream SQL
                    chip_start_x = int(chip['chip_start_x'])
                    chip_start_y = int(chip['chip_start_y'])
                    
                    chip_bands = {
                        band: data[chip_start_y:chip_start_y+chip_pixels, chip_start_x:chip_start_x+chip_pixels]
                        for band, data in region_bands.items()
                    }
                    
                    # Create individual chip GeoTIFF
                    chip_bounds = (float(chip['minx']), float(chip['miny']), 
                                 float(chip['maxx']), float(chip['maxy']))
                    chip_transform = from_bounds(*chip_bounds, chip_pixels, chip_pixels)
                    chip_raster = create_multiband_geotiff(chip_bands, chip_transform)
                    
                    results.append({
                        'chip_id': chip['chip_id'],
                        'datetime': chip['datetime'],
                        'chip_raster': chip_raster,
                        'is_complete': chip['is_complete'],
                        'cloud_coverage': float(chip_bands['scl'][chip_bands['scl'] > 0].mean()) if chip_bands['scl'][chip_bands['scl'] > 0].size > 0 else 0.0,
                        'geohash': chip['geohash']
                    })
                    
            except Exception as e:
                print(f"Error processing region {region_id}: {e}")
                return pd.DataFrame()
    
    return pd.DataFrame(results)