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

1. **Setup dev environment**:
   ```bash
   git clone git@ssh.gitlab.aws.dev:jnmeehan/sentinel-spark-embedding.git
   cd sentinel-spark-embedding/
   brew install python@3.12
   python3.12 -m venv .venv
   source .venv/bin/activate
   export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)
   ```

2. **Deploy CDK infrastructure**:
   ```bash
   CDK_EXEC_ROLE=$(aws cloudformation list-stack-resources \
     --stack-name CDKToolkit \
     --query 'StackResourceSummaries[?ResourceType==`AWS::IAM::Role` && contains(LogicalResourceId, `CloudFormationExecutionRole`)].PhysicalResourceId' \
     --output text)
   
   aws lakeformation put-data-lake-settings \
     --data-lake-settings "{\"DataLakeAdmins\":[{\"DataLakePrincipalIdentifier\":\"$CDK_EXEC_ROLE\"}]}"
   
   cdk deploy
   ```

3. **Run a job**:
   ```bash
   python -m src.jobs.chip_extraction \
     jobs.chip_extraction.aoi_bounds='[-122.5,-37.8,-122.3,-37.7]' \
     jobs.chip_extraction.start_date=2024-01-01 \
     jobs.chip_extraction.end_date=2024-01-31
   ```

## Processing Pipeline

1. **Chip Extraction**: Query STAC → intersect with global grid → extract 256px chips
2. **Temporal Merge**: Fill incomplete chips with previous data for consistency
3. **Embedding Generation**: Generate Clay GeoFM embeddings for change detection
