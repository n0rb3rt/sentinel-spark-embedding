"""Raster processing utilities"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from io import BytesIO
import pandas as pd
from pyspark.sql.types import *
from ..config.config import CONFIG
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
        StructField("chip_id", StringType()),
        StructField("datetime", TimestampType()),
        StructField("chip_raster", BinaryType()),
        StructField("is_complete", BooleanType()),
        StructField("cloud_coverage", FloatType()),
        StructField("geohash", StringType())
    ])
    
    @staticmethod
    def _create_multiband_geotiff(band_views, transform, crs='EPSG:4326'):
        """Create multiband GeoTIFF from band views without copying data"""
        buffer = BytesIO()
        
        with rasterio.open(buffer, 'w', driver='GTiff', compress='lz4',
                          height=CHIP_PIXELS, width=CHIP_PIXELS, count=len(band_views),
                          dtype=band_views[0].dtype,
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
        tile_pixels = REGION_SIZE * CHIP_PIXELS
        
        with rasterio.Env(**CONFIG.rasterio):
            with ExitStack() as stack:
                sources = [stack.enter_context(rasterio.open(urls[f'{band}_s3'])) for band in BANDS]
                
                print(f"Processing {len(df)} chips in region {region_id} for scene {scene_id} at {datetime_str}")
                
                try:
                    # Get region bounds and read all bands
                    region_bounds = (df.iloc[0]['region_minx'], df.iloc[0]['region_miny'], 
                                   df.iloc[0]['region_maxx'], df.iloc[0]['region_maxy'])
                    
                    region_data = [SceneChipProcessor._extract_reprojected_chip(src, region_bounds, tile_pixels) for src in sources]
                    
                    # Extract individual chips using views (no copies)
                    for _, chip in df.iterrows():
                        x, y = int(chip['chip_start_x']), int(chip['chip_start_y'])
                        
                        # Create views for all bands
                        band_views = [data[y:y+CHIP_PIXELS, x:x+CHIP_PIXELS] for data in region_data]
                        
                        # Create GeoTIFF directly from views
                        chip_bounds = (chip['minx'], chip['miny'], chip['maxx'], chip['maxy'])
                        chip_transform = from_bounds(*chip_bounds, CHIP_PIXELS, CHIP_PIXELS)
                        chip_raster = SceneChipProcessor._create_multiband_geotiff(band_views, chip_transform)
                        
                        # Calculate cloud coverage from SCL view
                        scl_index = BANDS.index('scl')
                        scl_view = band_views[scl_index]
                        cloud_coverage = float(scl_view[scl_view > 0].mean()) if scl_view[scl_view > 0].size > 0 else 0.0
                        
                        results.append({
                            'chip_id': chip['chip_id'],
                            'datetime': chip['datetime'],
                            'chip_raster': chip_raster,
                            'is_complete': chip['is_complete'],
                            'cloud_coverage': cloud_coverage,
                            'geohash': chip['geohash']
                        })
                        
                except Exception as e:
                    print(f"Error processing region {region_id}: {e}")
                    return pd.DataFrame()
        
        return pd.DataFrame(results)