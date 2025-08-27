"""
CDK Stack for Sentinel-2 Processing Infrastructure
"""
from aws_cdk import (
    Stack,
    CfnOutput,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_iam as iam,
    aws_glue as glue,
    aws_ecr_assets as ecr_assets,
    aws_ssm as ssm,
    RemovalPolicy
)
from aws_cdk.aws_lakeformation import CfnPermissions as LF
from constructs import Construct

class SentinelProcessingStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # S3 Bucket (single bucket with folders)
        self.bucket = s3.Bucket(
            self, "SentinelBucket",
            bucket_name=f"sentinel-{self.account}-{self.region}",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )
        
        # SageMaker Execution Role
        self.sagemaker_role = iam.Role(
            self, "SageMakerProcessingRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
                iam.ArnPrincipal(f"arn:aws:iam::{self.account}:role/Admin")
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
            ]
        )
        
        # Docker Image Asset (builds and pushes automatically)
        self.docker_asset = ecr_assets.DockerImageAsset(
            self, "SentinelSparkImage",
            directory="docker",
            file="Dockerfile",
            platform=ecr_assets.Platform.LINUX_AMD64,
            build_args={
                "BUILDKIT_INLINE_CACHE": "1"
            }
        )
        
        # Glue Database
        self.glue_database = glue.CfnDatabase(
            self, "SentinelGlueDatabase",
            catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name="sentinel",
                description="Sentinel-2 chip processing database"
            )
        )
        
        # Lake Formation permissions
        database_resource = LF.ResourceProperty(database_resource=LF.DatabaseResourceProperty(name="sentinel"))
        table_resource = LF.ResourceProperty(table_resource=LF.TableResourceProperty(database_name="sentinel", table_wildcard={}))
        
        admin_principal = LF.DataLakePrincipalProperty(data_lake_principal_identifier=f"arn:aws:iam::{self.account}:role/Admin")
        sagemaker_principal = LF.DataLakePrincipalProperty(data_lake_principal_identifier=self.sagemaker_role.role_arn)
        
        admin_db_perms = LF(self, "AdminDatabasePermissions",
            data_lake_principal=admin_principal, resource=database_resource, permissions=["ALL"])
        admin_table_perms = LF(self, "AdminTablePermissions", 
            data_lake_principal=admin_principal, resource=table_resource, permissions=["ALL"])
        
        sagemaker_db_perms = LF(self, "SageMakerDatabasePermissions",
            data_lake_principal=sagemaker_principal, resource=database_resource, permissions=["ALL"])
        sagemaker_table_perms = LF(self, "SageMakerTablePermissions",
            data_lake_principal=sagemaker_principal, resource=table_resource, permissions=["ALL"])
        
        # Ensure permissions are created after database
        admin_db_perms.add_dependency(self.glue_database)
        admin_table_perms.add_dependency(admin_db_perms)
        sagemaker_db_perms.add_dependency(self.glue_database)
        sagemaker_table_perms.add_dependency(sagemaker_db_perms)
        
        # Add S3 permissions
        self.bucket.grant_read_write(self.sagemaker_role)
        
        # Add Glue permissions
        self.sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "glue:GetDatabase",
                    "glue:GetTable",
                    "glue:CreateTable",
                    "glue:UpdateTable",
                    "glue:DeleteTable",
                    "glue:GetPartitions"
                ],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:catalog",
                    f"arn:aws:glue:{self.region}:{self.account}:database/sentinel",
                    f"arn:aws:glue:{self.region}:{self.account}:table/sentinel/*"
                ]
            )
        )
        
        # Add SSM permissions for reading configuration
        self.sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ssm:GetParameter",
                    "ssm:GetParameters"
                ],
                resources=[
                    f"arn:aws:ssm:{self.region}:{self.account}:parameter/airflow/variables/*"
                ]
            )
        )
        
        # Add permissions to access Sentinel-2 COGs (public dataset)
        self.sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:ListBucket"
                ],
                resources=[
                    "arn:aws:s3:::sentinel-cogs",
                    "arn:aws:s3:::sentinel-cogs/*"
                ]
            )
        )
        
        # CDK Outputs for Airflow configuration
        CfnOutput(self, "SageMakerRoleArn", 
                 value=self.sagemaker_role.role_arn,
                 description="SageMaker execution role ARN")
        
        CfnOutput(self, "BucketName",
                 value=self.bucket.bucket_name,
                 description="S3 bucket name")
        
        CfnOutput(self, "ContainerImageUri",
                 value=self.docker_asset.image_uri,
                 description="Docker image URI")
        
        # Store configuration in SSM Parameter Store for Airflow
        ssm.StringParameter(self, "AirflowSageMakerRole",
                           parameter_name="/airflow/variables/sagemaker_role_arn",
                           string_value=self.sagemaker_role.role_arn)
        
        ssm.StringParameter(self, "AirflowBucket",
                           parameter_name="/airflow/variables/bucket_name",
                           string_value=self.bucket.bucket_name)
        
        ssm.StringParameter(self, "AirflowContainerUri",
                           parameter_name="/airflow/variables/spark_container_uri",
                           string_value=self.docker_asset.image_uri)
        
        # Deploy source code
        s3deploy.BucketDeployment(self, "DeployCode",
            sources=[s3deploy.Source.asset("src")],
            destination_bucket=self.bucket,
            destination_key_prefix="code"
        )
        
        # Deploy DAG to S3 (for MWAA or manual download)
        s3deploy.BucketDeployment(self, "DeployDAG",
            sources=[s3deploy.Source.asset("dags")],
            destination_bucket=self.bucket,
            destination_key_prefix="airflow/dags"
        )
        