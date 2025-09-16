#!/usr/bin/env python3
"""
EMR utilities for submitting Spark jobs
"""
import boto3
from sentinel_processing.config import get_ssm_parameter, CONFIG

def get_session():
    """Get boto3 session"""
    return boto3.Session()

def submit_to_emr(job_key: str, script_name: str, **kwargs) -> None:
    """Submit a job to EMR cluster using config-based settings.
    
    Args:
        job_key: Job key in config (e.g., 'chip_extraction', 'embedding_generation')
        script_name: Job script filename (e.g., 'chip_extraction.py')
    """
    # Get job configuration
    job_config = getattr(CONFIG.jobs, job_key)
    spark_config = job_config.spark
    
    emr_cluster_id = get_ssm_parameter('/sentinel-spark/emr-cluster-id')
    s3_bucket = get_ssm_parameter('/sentinel-spark/s3-bucket')
    emr = get_session().client('emr')
    
    # Use pre-staged artifacts from bootstrap (no archives needed)
    args = [
        'spark-submit',
        '--deploy-mode', 'cluster',
        '--py-files', f's3://{s3_bucket}/dist/lib/sentinel_processing-{CONFIG.version}-py3-none-any.whl',
        '--conf', 'spark.yarn.appMasterEnv.PYSPARK_PYTHON=/mnt1/opt/venv/bin/python',
        '--conf', 'spark.yarn.executorEnv.PYSPARK_PYTHON=/mnt1/opt/venv/bin/python',
        '--conf', 'spark.yarn.am.extraJavaOptions=-Djava.io.tmpdir=/mnt/tmp',
        '--conf', 'spark.yarn.executor.extraJavaOptions=-Djava.io.tmpdir=/mnt/tmp',

    ]
    
    # Add packages from job config
    if spark_config.packages:
        args.extend(['--packages', ','.join(spark_config.packages)])
    
    # Add the script path
    args.append(f's3://{s3_bucket}/dist/src/sentinel_processing/jobs/{script_name}')
    
    response = emr.add_job_flow_steps(
        JobFlowId=emr_cluster_id,
        Steps=[{
            'Name': job_key,
            'ActionOnFailure': 'CONTINUE',
            'HadoopJarStep': {
                'Jar': 'command-runner.jar',
                'Args': args
            }
        }]
    )
    
    print(f"{job_key} job submitted to EMR cluster {emr_cluster_id}")
    print(f"Step ID: {response['StepIds'][0]}")
    print(f"Packages: {spark_config.packages}")
    print(f"Include model: {spark_config.include_model}")
    
    return response['StepIds'][0]

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m sentinel_processing.lib.emr_utils <job_key>")
        print("Example: python -m sentinel_processing.lib.emr_utils chip_extraction")
        sys.exit(1)
    
    job_key = sys.argv[1]
    script_name = f"{job_key}.py"
    submit_to_emr(job_key, script_name)
