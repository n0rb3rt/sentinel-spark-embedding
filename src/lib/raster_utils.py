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

def extract_reprojected_chip(src, chip_bounds_wgs84):
    """Extract and reproject chip from raster source"""
    target_transform = from_bounds(*chip_bounds_wgs84, 256, 256)
    target_array = np.empty((256, 256), dtype=src.dtypes[0])
    
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