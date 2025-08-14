# Sentinel-2 Chip Processing Pipeline

Production pipeline for processing Sentinel-2 satellite imagery into 256px chips with temporal analysis and Clay GeoFM embeddings.

## Architecture

- **PySpark + Apache Sedona**: Geospatial processing and chip extraction
- **Apache Iceberg**: Geohash-partitioned storage in AWS Glue catalog
- **Clay GeoFM**: Foundation model for generating embeddings
- **SageMaker + Airflow**: Serverless orchestration via Unified Studio

## Project Structure

```
├── src/
│   ├── config/          # Spark and processing configuration
│   ├── lib/             # Utility functions and UDFs
│   └── jobs/            # Main processing jobs
├── dags/                # Airflow DAGs for orchestration
├── docker/              # Custom Spark container
├── infrastructure/cdk/  # AWS CDK infrastructure
└── scripts/             # Build and deployment scripts
```

## Quick Start

1. **Setup CDK Environment**:
   ```bash
   cd infrastructure/cdk
   pip install -r requirements.txt
   cd ../..
   ```

2. **Deploy Everything**:
   ```bash
   ./deploy.sh
   ```

3. **Upload Clay Model**:
   ```bash
   BUCKET=$(aws ssm get-parameter --name '/airflow/variables/bucket_name' --query 'Parameter.Value' --output text)
   aws s3 cp clay_model.pth s3://$BUCKET/models/clay/v1.0/model.pth
   ```

4. **Deploy DAG to Airflow**:
   - Download from S3: `s3://bucket/airflow/dags/sentinel_processing_dag.py`
   - Upload to your Airflow environment
   - Enable the DAG in Airflow UI

Configuration is automatically stored in SSM Parameter Store and read by Airflow at runtime.

## Running Jobs

**Via Airflow UI**: Enable and trigger the `sentinel_chip_processing` DAG

**Individual Jobs** (for testing):
```bash
# Install dev dependencies first
pip install -r requirements-dev.txt

# Run jobs
python scripts/run_job.py chip_extraction --start-date 2024-01-15 --end-date 2024-01-16
python scripts/run_job.py temporal_merge
python scripts/run_job.py embedding_generation
```

## Development Workflow

**Code Changes**: Just run `./deploy.sh` - it rebuilds container and redeploys code

**Infrastructure Changes**: Modify CDK stack and run `./deploy.sh`

**Dependency Changes**: Update `docker/requirements.txt` and run `./deploy.sh`

## Processing Pipeline

1. **Chip Extraction**: Query STAC → intersect with global grid → extract 256px chips
2. **Temporal Merge**: Fill incomplete chips with previous data for consistency
3. **Embedding Generation**: Generate Clay GeoFM embeddings for change detection

## Key Features

- **Geohash Partitioning**: Level 8 for data, Level 4 for partitioning
- **Temporal Gap Filling**: Ensures consistent chip coverage over time
- **Serverless Execution**: SageMaker PySparkProcessor with auto-scaling
- **Iceberg Storage**: ACID transactions and schema evolution