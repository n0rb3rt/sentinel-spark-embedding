# Sentinel-2 Chip Processing Pipeline with Clay GeoFM

This project creates an AWS infrastructure for processing Sentinel-2 satellite imagery into 256px chips with temporal analysis and Clay GeoFM embeddings using EMR and PySpark.

## Architecture

- **EMR Cluster**: Spark cluster with Apache Sedona for geospatial processing
- **S3 Bucket**: Storage for Iceberg data and processed chips
- **Glue Catalog**: Metadata store for Iceberg tables with geohash partitioning
- **Clay GeoFM**: Foundation model for generating embeddings from satellite imagery
- **PySpark Jobs**: Extract chips from Sentinel-2 scenes and generate embeddings

## Development Setup

1. Create virtual environment:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

2. Install for development (includes PySpark for local testing):
   ```bash
   # Auto-detect architecture and install dev dependencies
   ./build.sh --dev
   
   # Or specify architecture explicitly
   ./build.sh --dev --arch cuda    # For CUDA development
   ./build.sh --dev --arch mps     # For Apple Silicon
   ./build.sh --dev --arch cpu     # For CPU-only
   ```

## Production Build & Deployment

1. Build distribution assets for deployment:
   ```bash
   # Auto-detect architecture and create distribution
   ./build.sh
   
   # Or specify target architecture
   ./build.sh --arch cuda    # For EMR/GPU deployment
   ./build.sh --arch cpu     # For CPU-only deployment
   ```

   The build process:
   - Downloads Clay model checkpoint and metadata
   - Builds Python wheel and packaged venv
   - Exports Clay model for target architecture
   - Creates dist/ directory with all deployment assets

2. Deploy infrastructure:
   ```bash
   cdk deploy
   ```

   The deployment process:
   - Uploads venv, wheel, and model to S3
   - Creates EMR cluster with proper dependency management

## Running the Pipeline

### 1. Extract chips from Sentinel-2 imagery:
```bash
uv run python src/sentinel_processing/jobs/chip_extraction.py \
  jobs.chip_extraction.aoi_bounds='[-122.5,-37.8,-122.3,-37.7]' \
  jobs.chip_extraction.start_date=2024-01-01 \
  jobs.chip_extraction.end_date=2024-01-31
```

**Output Table Schema (`chips`):**
| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique identifier for each 256px chip |
| `datetime` | timestamp | Acquisition date and time of the imagery |
| `scene_id` | string | Source Sentinel-2 scene identifier |
| `geohash` | string | Geohash for spatial partitioning |
| `scl_mean` | float | SCL band mean for cloud filtering |
| `clay_tensor` | binary | Compressed numpy arrays for Clay model input |
| `geotiff` | binary | GeoTIFF blob for visualization |
| `geometry` | binary | WKB polygon geometry |3

### 2. Generate Clay GeoFM embeddings:
```bash
uv run python src/sentinel_processing/jobs/embedding_generation.py \
  jobs.embedding_generation.input_table=chips \
  jobs.embedding_generation.output_table=embeddings
```

**Output Table Schema (`embeddings`):**
| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Links back to source chip |
| `scene_id` | string | Source Sentinel-2 scene identifier |
| `datetime` | timestamp | Acquisition date and time of the imagery |
| `geohash` | string | Geohash for spatial partitioning |
| `geometry` | binary | WKB polygon geometry |
| `global_embedding` | array<float> | 1024-dimensional Clay GeoFM embedding vector |
| `model_version` | string | Clay model version used for embedding |
| `created_at` | timestamp | Processing timestamp |

## Clay Foundation Model Results

The pipeline leverages Clay GeoFM (Geospatial Foundation Model) for generating meaningful embeddings from Sentinel-2 imagery:

### Input vs Embedding Visualization
![Input vs Embedding](docs/images/input_vs_embedding.png)

This visualization demonstrates how Clay GeoFM transforms raw Sentinel-2 spectral data into semantically meaningful embeddings. The left panel shows the original satellite imagery, while the right panel displays the learned embedding space where similar land cover types cluster together.

### Top Feature Analysis
![Top Features](docs/images/top_features.png)

Analysis of the most important spectral features learned by the Clay model. This helps understand which wavelengths and band combinations are most discriminative for different land cover classifications, providing insights into the model's decision-making process for geospatial analysis.

## Project Layout

```
sentinel-spark-embedding/
├── .cache/                         # Build cache directory
│   ├── model/                      # Clay model checkpoint and exports
│   ├── pip/                        # pip cache for faster builds
│   └── venvs/                      # Cached packaged environments
├── dist/                           # Distribution assets (production builds)
│   ├── venv/venv.tar.gz           # Packaged Python environment
│   ├── lib/*.whl                   # Python wheel
│   ├── model/                      # Exported Clay models
│   └── src/                        # Source code
├── src/
│   ├── infra/                      # CDK infrastructure modules
│   │   ├── __init__.py
│   │   ├── sentinel_stack.py       # Main CDK stack
│   │   └── emr_gpu_stack.py        # EMR GPU cluster stack
│   └── sentinel_processing/        # Python package for data processing
│       ├── jobs/
│       │   ├── __init__.py
│       │   ├── chip_extraction.py  # Sentinel-2 chip extraction job
│       │   └── embedding_generation.py # Clay GeoFM embedding job
│       ├── lib/
│       │   ├── __init__.py
│       │   ├── sedona_utils.py     # Geospatial processing utilities
│       │   ├── raster_utils.py     # Raster processing and chip extraction
│       │   └── clay_utils.py       # Clay model integration
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       └── config.yaml             # Default configuration
├── scripts/
│   └── export_clay_model.py        # Clay model export utility
├── docker/
│   └── Dockerfile.build            # Docker build configuration
├── docs/
│   └── images/                     # Documentation images
├── notebooks/
│   └── pyspark_sedona_rasterio.ipynb # Development notebook
├── sample-geospatial-foundation-models-on-aws/ # Reference implementation
├── app.py                          # CDK app entry point
├── build.sh                        # Build script with dev/production modes
├── pyproject.toml                  # Python package configuration
├── cdk.json                        # CDK configuration
└── README.md
```

**Key Components:**
- `.cache/`: Build cache for models, dependencies, and packaged environments
- `dist/`: Production distribution assets created by `./build.sh`
- `infra/`: CDK constructs for AWS resources (EMR, S3, Glue)
- `jobs/`: Main processing jobs for chip extraction and embedding generation
- `lib/`: Utility modules for geospatial processing, raster handling, and Clay model integration
- `build.sh`: Unified build script supporting dev mode (`--dev`) and production builds
- `config.py/config.yaml`: Configuration management with CLI override support

## Build Modes

**Development Mode** (`./build.sh --dev`):
- Installs editable package with dev dependencies (includes PySpark)
- Skips distribution asset creation for faster builds
- Ideal for local development and testing

**Production Mode** (`./build.sh`):
- Creates packaged venv, wheel, and source distribution
- Exports Clay model for target architecture
- Generates complete `dist/` directory for deployment

## Cleanup

```bash
cdk destroy
```
