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

def create_multiband_geotiff(bands_dict, transform, crs='EPSG:4326'):
    """Create multiband GeoTIFF from band dictionary"""
    buffer = BytesIO()
    
    with rasterio.open(buffer, 'w', driver='GTiff', compress='lz4',
                      height=256, width=256, count=len(bands_dict),
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
    """Process chips for a scene group"""
    scene_id = df.iloc[0]['id']
    datetime_str = str(df.iloc[0]['datetime'])
    urls = broadcast_urls.value.get((scene_id, datetime_str))
    
    if not urls:
        return pd.DataFrame()
    
    results = []
    
    with rasterio.Env(**CONFIG.rasterio):
        with rasterio.open(urls['blue_s3']) as blue_src, \
             rasterio.open(urls['green_s3']) as green_src, \
             rasterio.open(urls['red_s3']) as red_src, \
             rasterio.open(urls['nir_s3']) as nir_src, \
             rasterio.open(urls['scl_s3']) as scl_src:
            
            print(f"Processing {len(df)} chips for scene {scene_id} at {datetime_str}")

            for _, row in df.iterrows():
                try:
                    chip_bounds = (float(row['minx']), float(row['miny']), 
                                 float(row['maxx']), float(row['maxy']))
                    target_transform = from_bounds(*chip_bounds, 256, 256)
                    
                    bands = {
                        'blue': extract_reprojected_chip(blue_src, chip_bounds),
                        'green': extract_reprojected_chip(green_src, chip_bounds),
                        'red': extract_reprojected_chip(red_src, chip_bounds),
                        'nir': extract_reprojected_chip(nir_src, chip_bounds),
                        'scl': extract_reprojected_chip(scl_src, chip_bounds)
                    }
                    
                    chip_raster = create_multiband_geotiff(bands, target_transform)
                    
                    results.append({
                        'chip_id': row['chip_id'],
                        'datetime': row['datetime'],
                        'chip_raster': chip_raster,
                        'is_complete': row['is_complete'],
                        'cloud_coverage': float(bands['scl'][bands['scl'] > 0].mean()),
                        'geohash': row['geohash']
                    })
                        
                except Exception as e:
                    print(f"Error processing {row['chip_id']}: {e}")
                    continue
    
    return pd.DataFrame(results)

def process_scene_region(df, broadcast_urls) -> pd.DataFrame:
    """Process chips for a scene region using 6x6 tiling optimization"""
    if df.empty:
        return pd.DataFrame()
        
    scene_id = df.iloc[0]['id']
    datetime_str = str(df.iloc[0]['datetime'])
    region_id = df.iloc[0]['region_id']
    urls = broadcast_urls.value.get((scene_id, datetime_str))
    
    if not urls:
        return pd.DataFrame()
    
    results = []
    
    with rasterio.Env(**CONFIG.rasterio):
        # One connection per region
        with rasterio.open(urls['blue_s3']) as blue_src, \
             rasterio.open(urls['green_s3']) as green_src, \
             rasterio.open(urls['red_s3']) as red_src, \
             rasterio.open(urls['nir_s3']) as nir_src, \
             rasterio.open(urls['scl_s3']) as scl_src:
            
            print(f"Processing {len(df)} chips in region {region_id} for scene {scene_id}")
            
            # Group chips by 6x6 tiles within this region
            df['tile_x'] = df['x'] // 6
            df['tile_y'] = df['y'] // 6
            
            for (tile_x, tile_y), tile_chips in df.groupby(['tile_x', 'tile_y']):
                try:
                    # Calculate 6x6 tile bounds (1536x1536 pixels)
                    tile_minx = -180 + (tile_x * 6 * 0.023)
                    tile_miny = -90 + (tile_y * 6 * 0.023)
                    tile_maxx = tile_minx + (6 * 0.023)
                    tile_maxy = tile_miny + (6 * 0.023)
                    
                    # Read 6x6 tile from all bands (1536x1536 pixels)
                    tile_bands = {
                        'blue': extract_reprojected_chip(blue_src, (tile_minx, tile_miny, tile_maxx, tile_maxy), 1536),
                        'green': extract_reprojected_chip(green_src, (tile_minx, tile_miny, tile_maxx, tile_maxy), 1536),
                        'red': extract_reprojected_chip(red_src, (tile_minx, tile_miny, tile_maxx, tile_maxy), 1536),
                        'nir': extract_reprojected_chip(nir_src, (tile_minx, tile_miny, tile_maxx, tile_maxy), 1536),
                        'scl': extract_reprojected_chip(scl_src, (tile_minx, tile_miny, tile_maxx, tile_maxy), 1536)
                    }
                    
                    # Extract individual chips from the 6x6 tile
                    for _, chip in tile_chips.iterrows():
                        # Calculate chip position within tile (0-5 range)
                        chip_x_in_tile = int(chip['x'] % 6)
                        chip_y_in_tile = int(chip['y'] % 6)
                        
                        # Extract 256x256 chip from 1536x1536 tile
                        chip_start_x = chip_x_in_tile * 256
                        chip_start_y = chip_y_in_tile * 256
                        
                        chip_bands = {
                            band: data[chip_start_y:chip_start_y+256, chip_start_x:chip_start_x+256]
                            for band, data in tile_bands.items()
                        }
                        
                        # Create individual chip GeoTIFF
                        chip_bounds = (float(chip['minx']), float(chip['miny']), 
                                     float(chip['maxx']), float(chip['maxy']))
                        chip_transform = from_bounds(*chip_bounds, 256, 256)
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
                    print(f"Error processing tile {tile_x},{tile_y}: {e}")
                    continue
    
    return pd.DataFrame(results)