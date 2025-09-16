"""Raster processing utilities"""
import math
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from io import BytesIO
import pandas as pd
from pyspark.sql.types import *
from sentinel_processing.config import CONFIG
from sentinel_processing.lib.clay_utils import load_clay_metadata
from contextlib import ExitStack
import torch

BANDS = CONFIG.sentinel.bands
CHIP_PIXELS = CONFIG.jobs.chip_extraction.chip_size_pixels
REGION_SIZE = CONFIG.jobs.chip_extraction.region_size_chips
GROUND_RESOLUTION_METERS = CONFIG.jobs.chip_extraction.ground_resolution_meters

def get_chip_size_degrees():
    """Calculate chip size in degrees from pixel size and ground resolution"""
    meters_per_degree = 111320  # approximate at equator
    chip_size_meters = CHIP_PIXELS * GROUND_RESOLUTION_METERS
    return chip_size_meters / meters_per_degree

class SceneChipProcessor:
    """Processes scene chips with tensor-optimized numpy processing"""
    
    schema = StructType([
        StructField("id", StringType()),
        StructField("datetime", TimestampType()),
        StructField("scene_id", StringType()),
        StructField("geohash", StringType()),
        StructField("scl_mean", FloatType()),  # SCL band mean for cloud filtering
        StructField("clay_tensor", BinaryType()),  # np.savez_compressed blob
        StructField("geotiff", BinaryType()),  # GeoTIFF blob for visualization
        StructField("geometry", BinaryType())  # WKB polygon geometry
    ])
    
    @staticmethod
    def _extract_reprojected_region(src, bounds_wgs84, size):
        """Extract and reproject region from raster source using rasterio"""
        target_transform = from_bounds(*bounds_wgs84, size, size)
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
    
    @staticmethod
    def _create_geotiff_blob(raw_chip_data, chip_bounds):
        """Create GeoTIFF blob from raw uint16 chip data"""
        transform = from_bounds(*chip_bounds, CHIP_PIXELS, CHIP_PIXELS)
        
        buffer = BytesIO()
        with rasterio.open(
            buffer, 'w', driver='GTiff',
            height=CHIP_PIXELS, width=CHIP_PIXELS, count=raw_chip_data.shape[0],
            dtype=np.uint16, crs='EPSG:4326', transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(raw_chip_data)
        
        return buffer.getvalue()
    
    @staticmethod
    def _create_clay_tensor_blob(normalized_chip, latlon_array, time_array, waves_array, gsd_array):
        """Create numpy compressed blob"""
        buffer = BytesIO()
        np.savez_compressed(buffer,
            pixels=normalized_chip,
            time=time_array,
            latlon=latlon_array,
            waves=waves_array,
            gsd=gsd_array
        )
        return buffer.getvalue()
    
    @staticmethod
    def process(df, broadcast_urls) -> pd.DataFrame:
        """Process chips with numpy - tensor-optimized for Clay embeddings"""
        if df.empty:
            return pd.DataFrame()
            
        scene_id = df.iloc[0]['scene_id']
        scene_dt = df.iloc[0]['datetime']
        region_id = df.iloc[0]['region_id']
        region_minx = df.iloc[0]['region_minx']
        region_miny = df.iloc[0]['region_miny']
        region_maxx = df.iloc[0]['region_maxx']
        region_maxy = df.iloc[0]['region_maxy']
        urls = broadcast_urls.value.get((scene_id, str(scene_dt)))
        region_bounds = (region_minx, region_miny, region_maxx, region_maxy)
        
        if not urls:
            return pd.DataFrame()
        
        results = []

        clay_metadata = load_clay_metadata()
        s2_bands = clay_metadata['sentinel-2-l2a'].bands
        s2_waves = s2_bands.wavelength
        gsd = clay_metadata['sentinel-2-l2a'].gsd
        clay_means = np.array([s2_bands.mean.blue, s2_bands.mean.green, s2_bands.mean.red, s2_bands.mean.nir], dtype=np.float32)
        clay_stds = np.array([s2_bands.std.blue, s2_bands.std.green, s2_bands.std.red, s2_bands.std.nir], dtype=np.float32)
        week = scene_dt.isocalendar().week * 2 * np.pi / 52
        hour = scene_dt.hour * 2 * np.pi / 24
    
        # Create numpy arrays directly
        time_array = np.array([math.sin(week), math.cos(week), math.sin(hour), math.cos(hour)], dtype=np.float32)
        waves_array = np.array([s2_waves.blue, s2_waves.green, s2_waves.red, s2_waves.nir], dtype=np.float32)
        gsd_array = np.array([gsd], dtype=np.float32)
        
        try: 
            print(f"Processing {len(df)} chips in region {region_id} for scene {scene_id} at {scene_dt}")
            
            # Use rasterio for precise region extraction
            tile_pixels = REGION_SIZE * CHIP_PIXELS
            region_array = np.empty((len(BANDS), tile_pixels, tile_pixels), dtype=np.uint16)
            
            with rasterio.Env(**CONFIG.rasterio):
                with ExitStack() as stack:
                    # Open bands for reading
                    sources = [stack.enter_context(rasterio.open(urls[f'{band}_s3'])) for band in BANDS]
                    
                    # Extract region using rasterio range reads
                    for i, src in enumerate(sources):
                        region_array[i] = SceneChipProcessor._extract_reprojected_region(src, region_bounds, tile_pixels)
            
            # Normalize only spectral bands (first 4) for Clay processing
            normalized_region = (region_array[:4].astype(np.float32) - clay_means[:, None, None]) / clay_stds[:, None, None]
            
            # Extract chips (only lat/lon varies)
            for _, chip in df.iterrows():
                x, y = int(chip['chip_start_x']), int(chip['chip_start_y'])
                chip_bounds = (chip['minx'], chip['miny'], chip['maxx'], chip['maxy'])
                chip_lat, chip_lon = chip['chip_center_lat'], chip['chip_center_lon']
                
                lat_rad = chip_lat * np.pi / 180
                lon_rad = chip_lon * np.pi / 180
                
                latlon_array = np.array([math.sin(lat_rad), math.cos(lat_rad), math.sin(lon_rad), math.cos(lon_rad)], dtype=np.float32)
                
                # Extract chip data
                raw_chip = region_array[:, y:y+CHIP_PIXELS, x:x+CHIP_PIXELS]
                normalized_chip = normalized_region[:, y:y+CHIP_PIXELS, x:x+CHIP_PIXELS]
                
                results.append({
                    'id': chip['chip_id'],
                    'datetime': chip['datetime'],
                    'scene_id': chip['scene_id'],
                    'geohash': chip['geohash'],
                    'scl_mean': float(np.mean(raw_chip[4])),
                    'clay_tensor': SceneChipProcessor._create_clay_tensor_blob(normalized_chip, latlon_array, time_array, waves_array, gsd_array),
                    'geotiff': SceneChipProcessor._create_geotiff_blob(raw_chip, chip_bounds),
                    'geometry': chip['chip_wkb']  
                })
                
        except Exception as e:
            print(f"Error processing region {region_id}: {e}")
            print(f"Region bounds were: {region_bounds}")
            print(f"Scene ID: {scene_id}")
            return pd.DataFrame()

        return pd.DataFrame(results)