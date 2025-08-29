"""Raster processing utilities"""
import math
import numpy as np
import xarray as xr
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
    """Processes scene chips with tensor-optimized xarray processing"""
    
    schema = StructType([
        StructField("id", StringType()),
        StructField("datetime", TimestampType()),
        StructField("scene_id", StringType()),
        StructField("geohash", StringType()),
        StructField("clay_tensor", BinaryType()),  # NetCDF blob with all tensor data
        StructField("geotiff", BinaryType()),  # GeoTIFF blob for visualization
        StructField("geometry", BinaryType())  # WKB polygon geometry
    ])
    
    @staticmethod
    def _extract_reprojected_region(src, bounds_wgs84, size):
        """Extract and reproject chip from raster source using rasterio"""
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
        from rasterio.transform import from_bounds
        
        minx, miny, maxx, maxy = chip_bounds
        transform = from_bounds(minx, miny, maxx, maxy, CHIP_PIXELS, CHIP_PIXELS)
        
        buffer = BytesIO()
        with rasterio.open(
            buffer, 'w', driver='GTiff',
            height=CHIP_PIXELS, width=CHIP_PIXELS, count=4,
            dtype=np.uint16, crs='EPSG:4326', transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(raw_chip_data)
        
        return buffer.getvalue()
    
    @staticmethod
    def _create_clay_tensor_blob(chip_data, datetime):
        """Create embedding-ready NetCDF blob with all Clay model inputs"""
        dt = pd.to_datetime(datetime)

        lat_center = float(chip_data.y.mean())
        lon_center = float(chip_data.x.mean())
        
        # Create xarray Dataset with all Clay inputs
        clay_dataset = xr.Dataset({
            # Main tensor data (4, 256, 256) - Clay expects (band, height, width)
            'pixels': chip_data.transpose('band', 'y', 'x').astype(np.float32),
            
            # Temporal features
            'temporal': xr.DataArray([
                dt.isocalendar().week,
                dt.hour,
                np.sin(dt.isocalendar().week * 2 * np.pi / 52),
                np.cos(dt.isocalendar().week * 2 * np.pi / 52),
                np.sin(dt.hour * 2 * np.pi / 24),
                np.cos(dt.hour * 2 * np.pi / 24)
            ], dims=['temporal_features']),
            
            # Spatial features
            'spatial': xr.DataArray([
                lat_center,
                lon_center,
                np.sin(lat_center * np.pi / 180),
                np.cos(lat_center * np.pi / 180),
                np.sin(lon_center * np.pi / 180),
                np.cos(lon_center * np.pi / 180)
            ], dims=['spatial_features'])
        })
        
        # Serialize to compressed NetCDF bytes
        buffer = BytesIO()
        clay_dataset.to_netcdf(
            buffer, 
            engine='h5netcdf', 
            encoding={'pixels': {'zlib': True, 'complevel': 6}}
        )
        return buffer.getvalue()
    
    @staticmethod
    def process(df, broadcast_urls) -> pd.DataFrame:
        """Process chips with xarray - tensor-optimized for Clay embeddings"""
        if df.empty:
            return pd.DataFrame()
            
        scene_id = df.iloc[0]['scene_id']
        datetime_str = str(df.iloc[0]['datetime'])
        region_id = df.iloc[0]['region_id']
        region_minx = df.iloc[0]['region_minx']
        region_miny = df.iloc[0]['region_miny']
        region_maxx = df.iloc[0]['region_maxx']
        region_maxy = df.iloc[0]['region_maxy']
        urls = broadcast_urls.value.get((scene_id, datetime_str))
        region_bounds = (region_minx, region_miny, region_maxx, region_maxy)
        
        if not urls:
            return pd.DataFrame()
        
        results = []
        
        # Clay normalization constants
        s2_bands = CLAY_METADATA['sentinel-2-l2a'].bands
        clay_stats = xr.DataArray(
            [[s2_bands.mean.blue, s2_bands.mean.green, s2_bands.mean.red, s2_bands.mean.nir],
             [s2_bands.std.blue, s2_bands.std.green, s2_bands.std.red, s2_bands.std.nir]],
            dims=['stat', 'band'],
            coords={'stat': ['mean', 'std'], 'band': ['blue', 'green', 'red', 'nir']}
        )
        
        try: 
            print(f"Processing {len(df)} chips in region {region_id} for scene {scene_id} at {datetime_str}")
            
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
            
            # Convert to xarray with region coordinates
            region_data = xr.DataArray(
                region_array,
                dims=['band', 'y', 'x'],
                coords={
                    'band': ['blue', 'green', 'red', 'nir'],
                    'y': np.linspace(region_maxy, region_miny, tile_pixels),
                    'x': np.linspace(region_minx, region_maxx, tile_pixels)
                }
            )
            
            # Normalize in-place
            region_data = (region_data / 10000.0 - clay_stats.sel(stat='mean')) / clay_stats.sel(stat='std')
            
            # Extract individual chips
            for _, chip in df.iterrows():
                x, y = int(chip['chip_start_x']), int(chip['chip_start_y'])
                
                # Extract raw data for GeoTIFF (before normalization)
                raw_chip = region_array[:, y:y+CHIP_PIXELS, x:x+CHIP_PIXELS]
                chip_bounds = (chip['minx'], chip['miny'], chip['maxx'], chip['maxy'])
                
                # Extract normalized data for Clay tensor (coordinates already set)
                chip_data = region_data.isel(x=slice(x, x + CHIP_PIXELS), y=slice(y, y + CHIP_PIXELS))
                
                results.append({
                    'id': chip['chip_id'],
                    'datetime': chip['datetime'],
                    'scene_id': chip['scene_id'],
                    'geohash': chip['geohash'],
                    'clay_tensor': SceneChipProcessor._create_clay_tensor_blob(chip_data, chip['datetime']),
                    'geotiff': SceneChipProcessor._create_geotiff_blob(raw_chip, chip_bounds),
                    'geometry': chip['chip_wkb']  
                })
                
        except Exception as e:
            print(f"Error processing region {region_id}: {e}")
            print(f"Region bounds were: {region_bounds}")
            print(f"Scene ID: {scene_id}")
            return pd.DataFrame()

        return pd.DataFrame(results)