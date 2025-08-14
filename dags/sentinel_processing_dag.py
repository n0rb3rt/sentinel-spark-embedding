"""
Sentinel-2 Processing DAG
Orchestrates chip extraction, temporal merge, and embedding generation
"""
from airflow import DAG
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerProcessingOperator
from airflow.providers.amazon.aws.hooks.ssm import SsmHook
from datetime import datetime, timedelta
from omegaconf import OmegaConf
from pathlib import Path

# Get configuration from SSM at DAG parse time
def get_ssm_parameter(parameter_name):
    ssm_hook = SsmHook()
    return ssm_hook.get_parameter(parameter_name)['Parameter']['Value']

# Load unified configuration
config_path = Path(__file__).parent.parent / "config.yaml"
CONFIG = OmegaConf.load(config_path)

# Load SSM parameters
SAGEMAKER_ROLE_ARN = get_ssm_parameter('/airflow/variables/sagemaker_role_arn')
BUCKET_NAME = get_ssm_parameter('/airflow/variables/bucket_name')
SPARK_CONTAINER_URI = get_ssm_parameter('/airflow/variables/spark_container_uri')

default_args = {
    'owner': 'sentinel-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'sentinel_chip_processing',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    description='Process Sentinel-2 imagery into chips and embeddings'
)

def create_spark_job(task_id, script, args, instance_type='m5.xlarge', instance_count=2, volume_gb=50):
    """Create SageMaker Spark processing job with common config"""
    return SageMakerProcessingOperator(
        task_id=task_id,
        config={
            'ProcessingJobName': f'sentinel-{task_id.replace("_", "-")}-{{{{ ds }}}}',
            'ProcessingResources': {
                'ClusterConfig': {
                    'InstanceType': instance_type,
                    'InstanceCount': instance_count,
                    'VolumeSizeInGB': volume_gb
                }
            },
            'ProcessingInputs': [
                {
                    'InputName': 'code',
                    'S3Input': {
                        'S3Uri': f's3://{BUCKET_NAME}/code/',
                        'LocalPath': '/opt/ml/processing/input/code',
                        'S3DataType': 'S3Prefix',
                        'S3InputMode': 'File'
                    }
                }
            ],
            'AppSpecification': {
                'ImageUri': SPARK_CONTAINER_URI,
                'ContainerEntrypoint': ['spark-submit', f'/opt/ml/processing/input/code/src/jobs/{script}'] + args
            },
            'Environment': {
                'AWS_SPARK_CONFIG_MODE': '2'
            },
            'RoleArn': SAGEMAKER_ROLE_ARN
        },
        dag=dag
    )

extract_chips = create_spark_job(
    'extract_chips',
    'chip_extraction.py',
    [f'jobs.chip_extraction.aoi_bounds={list(CONFIG.jobs.chip_extraction.aoi_bounds)}',
     'date_range={{ ds }},{{ next_ds }}',
     f'output_path=s3://{BUCKET_NAME}/data/chips/'],
    instance_type=CONFIG.sagemaker.chip_extraction.instance_type,
    instance_count=CONFIG.sagemaker.chip_extraction.instance_count,
    volume_gb=CONFIG.sagemaker.chip_extraction.volume_size_in_gb
)

temporal_merge = create_spark_job(
    'temporal_merge',
    'temporal_merge.py',
    [f'jobs.temporal_merge.lookback_days={CONFIG.jobs.temporal_merge.lookback_days}'],
    instance_type=CONFIG.sagemaker.temporal_merge.instance_type,
    instance_count=CONFIG.sagemaker.temporal_merge.instance_count,
    volume_gb=CONFIG.sagemaker.temporal_merge.volume_size_in_gb
)

generate_embeddings = create_spark_job(
    'generate_embeddings',
    'embedding_generation.py',
    [f'jobs.embedding_generation.batch_size={CONFIG.jobs.embedding_generation.batch_size}'],
    instance_type=CONFIG.sagemaker.embedding_generation.instance_type,
    instance_count=CONFIG.sagemaker.embedding_generation.instance_count,
    volume_gb=CONFIG.sagemaker.embedding_generation.volume_size_in_gb
)

# Define dependencies
extract_chips >> temporal_merge >> generate_embeddings