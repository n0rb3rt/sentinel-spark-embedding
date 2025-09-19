#!/usr/bin/env python3

from aws_cdk import (
    aws_s3 as s3,
    aws_glue as glue,
    aws_ssm as ssm,
    RemovalPolicy
)
from constructs import Construct
from sentinel_processing.config import CONFIG

class StorageConstruct(Construct):
    def __init__(self, scope: Construct, construct_id: str, account: str, region: str):
        super().__init__(scope, construct_id)
        
        # S3 Bucket
        self.bucket = s3.Bucket(
            self, "Bucket",
            bucket_name=f"sentinel-{account}-{region}",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )
        
        # Glue Database
        self.database = glue.CfnDatabase(
            self, "Database",
            catalog_id=account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name=CONFIG.sentinel.database_name,
                description="Sentinel-2 chip processing database v2"
            )
        )
        
