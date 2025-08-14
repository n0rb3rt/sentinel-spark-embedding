#!/usr/bin/env python3
"""
Run individual Spark processing jobs using SageMaker PySparkProcessor
"""
import argparse
import boto3
from datetime import datetime, timedelta
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.processing import ProcessingInput
from ..config.config import CONFIG

def get_ssm_parameter(name):
    """Get parameter from SSM Parameter Store"""
    ssm = boto3.client('ssm')
    return ssm.get_parameter(Name=name)['Parameter']['Value']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job_type', choices=['chip_extraction', 'temporal_merge', 'embedding_generation'])
    parser.add_argument('--start-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--end-date', default=(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))
    parser.add_argument('--wait', action='store_true', help='Wait for job completion')
    parser.add_argument('--logs', action='store_true', help='Show logs in real-time')
    args = parser.parse_args()

    # Get configuration from SSM
    role_arn = get_ssm_parameter('/airflow/variables/sagemaker_role_arn')
    bucket_name = get_ssm_parameter('/airflow/variables/bucket_name')
    container_uri = get_ssm_parameter('/airflow/variables/spark_container_uri')
    
    # Get job configuration from config
    sagemaker_config = CONFIG.sagemaker[args.job_type]
    job_config = CONFIG.jobs[args.job_type]
    
    # Create SageMaker Spark configuration
    spark_config = [{"Classification": "spark-defaults", "Properties": dict(CONFIG.spark)}]
    spark_env = {
        'AWS_SPARK_CONFIG_MODE': '2',
        'PYTHONPATH': '/opt/ml/processing/input/src'
    }
    
    # Create PySparkProcessor
    processor = PySparkProcessor(
        framework_version='3.5',
        role=role_arn,
        instance_type=sagemaker_config.instance_type,
        instance_count=sagemaker_config.instance_count,
        volume_size_in_gb=sagemaker_config.volume_size_in_gb,
        base_job_name=f'sentinel-{args.job_type.replace("_", "-")}',
        image_uri=container_uri,
        env=spark_env
    )
    
    # Prepare arguments - pass config overrides via CLI
    if args.job_type == 'chip_extraction':
        arguments = [
            f'jobs.chip_extraction.aoi_bounds={list(job_config.aoi_bounds)}',
            f'date_range={args.start_date},{args.end_date}',
            f'output_path=s3://{bucket_name}/data/chips/'
        ]
    elif args.job_type == 'temporal_merge':
        arguments = [f'jobs.temporal_merge.lookback_days={job_config.lookback_days}']
    else:  # embedding_generation
        arguments = []
    
    # Run job using source code
    processor.run(
        submit_app=f's3://{bucket_name}/code/jobs/{args.job_type}.py',
        arguments=arguments,
        inputs=[ProcessingInput(
            source=f's3://{bucket_name}/code/',
            destination='/opt/ml/processing/input/src'
        )],
        configuration=spark_config,
        wait=args.wait,
        logs=args.logs
    )

    if args.wait:
        print(f"✅ {args.job_type} job completed successfully")
    else:
        print(f"✅ {args.job_type} job submitted successfully")
        print("Monitor at: https://console.aws.amazon.com/sagemaker/home#/processing-jobs")
        print(f"Job name: {processor.latest_job_name}")
        print(f"CloudWatch logs: /aws/sagemaker/ProcessingJobs")

if __name__ == '__main__':
    main()