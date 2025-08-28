"""Raster processing utilities"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from io import BytesIO
import pandas as pd
from pyspark.sql.types import *
from ..config.config import CONFIG, CLAY_METADATA
from contextlib import ExitStack

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
    """Processes scene chips with encapsulated schema and logic"""
    
    schema = StructType([
        StructField("id", StringType()),
        StructField("datetime", TimestampType()),
        StructField("scene_id", StringType()),
        StructField("geohash", StringType()),
        StructField("normalized_raster", BinaryType()),
        StructField("lat", DoubleType()),
        StructField("lon", DoubleType()),
        StructField("week_sin", DoubleType()),
        StructField("week_cos", DoubleType()),
        StructField("hour_sin", DoubleType()),
        StructField("hour_cos", DoubleType()),
        StructField("lat_sin", DoubleType()),
        StructField("lat_cos", DoubleType()),
        StructField("lon_sin", DoubleType()),
        StructField("lon_cos", DoubleType())
    ])
    
    @staticmethod
    def _create_multiband_geotiff(band_views, transform, crs='EPSG:4326', dtype=None):
        """Create multiband GeoTIFF from band views without copying data"""
        buffer = BytesIO()
        
        if dtype is None:
            dtype = band_views[0].dtype
        
        with rasterio.open(buffer, 'w', driver='GTiff', compress='lz4',
                          height=CHIP_PIXELS, width=CHIP_PIXELS, count=len(band_views),
                          dtype=dtype,
                          crs=crs, transform=transform) as dst:
            
            for i, band_view in enumerate(band_views, 1):
                dst.write(band_view, i)
        
        return buffer.getvalue()
    
    @staticmethod
    def _extract_reprojected_chip(src, chip_bounds_wgs84, size=256):
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
    
    @staticmethod
    def process(df, broadcast_urls) -> pd.DataFrame:
        """Process chips with Clay normalization and temporal encoding"""
        if df.empty:
            return pd.DataFrame()
            
        scene_id = df.iloc[0]['scene_id']
        datetime_str = str(df.iloc[0]['datetime'])
        region_id = df.iloc[0]['region_id']
        urls = broadcast_urls.value.get((scene_id, datetime_str))
        
        if not urls:
            return pd.DataFrame()
        
        results = []
        tile_pixels = REGION_SIZE * CHIP_PIXELS
        
        # Clay normalization constants for first 4 bands
        s2_bands = CLAY_METADATA['sentinel-2-l2a'].bands
        band_means = [s2_bands.mean.blue, s2_bands.mean.green, s2_bands.mean.red, s2_bands.mean.nir]
        band_stds = [s2_bands.std.blue, s2_bands.std.green, s2_bands.std.red, s2_bands.std.nir]
        
        with rasterio.Env(**CONFIG.rasterio):
            with ExitStack() as stack:
                # Only process first 4 bands for Clay
                sources = [stack.enter_context(rasterio.open(urls[f'{band}_s3'])) for band in BANDS[:4]]
                
                print(f"Processing {len(df)} chips in region {region_id} for scene {scene_id} at {datetime_str}")
                
                try:
                    region_bounds = (df.iloc[0]['region_minx'], df.iloc[0]['region_miny'], 
                                   df.iloc[0]['region_maxx'], df.iloc[0]['region_maxy'])
                    
                    # Read all bands into 3D array
                    region_array = np.empty((4, tile_pixels, tile_pixels), dtype=np.uint16)
                    for i, src in enumerate(sources):
                        region_array[i] = SceneChipProcessor._extract_reprojected_chip(src, region_bounds, tile_pixels)
                    
                    # Normalize entire region with Clay constants
                    normalized_region = np.empty((4, tile_pixels, tile_pixels), dtype=np.float32)
                    for i in range(4):
                        normalized_region[i] = (region_array[i] / 10000.0 - band_means[i]) / band_stds[i]
                    
                    # Extract individual chips from normalized region
                    for _, chip in df.iterrows():
                        x, y = int(chip['chip_start_x']), int(chip['chip_start_y'])
                        
                        # Extract normalized chip as views
                        chip_array = normalized_region[:, y:y+CHIP_PIXELS, x:x+CHIP_PIXELS]
                        band_views = [chip_array[i] for i in range(4)]
                        
                        # Create normalized GeoTIFF
                        chip_bounds = (chip['minx'], chip['miny'], chip['maxx'], chip['maxy'])
                        chip_transform = from_bounds(*chip_bounds, CHIP_PIXELS, CHIP_PIXELS)
                        normalized_raster = SceneChipProcessor._create_multiband_geotiff(
                            band_views, chip_transform, dtype=np.float32
                        )
                        
                        # Calculate temporal encodings
                        dt = pd.to_datetime(chip['datetime'])
                        lat, lon = (chip['miny'] + chip['maxy']) / 2, (chip['minx'] + chip['maxx']) / 2
                        
                        results.append({
                            'id': chip['chip_id'],
                            'datetime': chip['datetime'],
                            'scene_id': chip['scene_id'],
                            'geohash': chip['geohash'],
                            'normalized_raster': normalized_raster,
                            'lat': lat,
                            'lon': lon,
                            'week_sin': np.sin(dt.isocalendar().week * 2 * np.pi / 52),
                            'week_cos': np.cos(dt.isocalendar().week * 2 * np.pi / 52),
                            'hour_sin': np.sin(dt.hour * 2 * np.pi / 24),
                            'hour_cos': np.cos(dt.hour * 2 * np.pi / 24),
                            'lat_sin': np.sin(lat * np.pi / 180),
                            'lat_cos': np.cos(lat * np.pi / 180),
                            'lon_sin': np.sin(lon * np.pi / 180),
                            'lon_cos': np.cos(lon * np.pi / 180)
                        })
                        
                except Exception as e:
                    print(f"Error processing region {region_id}: {e}")
                    return pd.DataFrame()
        
        return pd.DataFrame(results)