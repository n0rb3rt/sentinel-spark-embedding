#!/usr/bin/env python3

from aws_cdk import Stack, CfnOutput
from aws_cdk import aws_ssm as ssm
from constructs import Construct

from .vpc import VpcConstruct
from .storage import StorageConstruct
from .opensearch import OpenSearchConstruct
from .permissions import PermissionsConstruct
from .emr import EMRConstruct

class SentinelStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, 
                 vpc_id: str = None, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # VPC (existing or new)
        vpc = VpcConstruct(self, "VPC", vpc_id=vpc_id)
        
        # Storage (S3 + Glue)
        storage = StorageConstruct(self, "Storage", self.account, self.region)
        
        # EMR Cluster (creates its own roles)
        emr = EMRConstruct(self, "EMR", vpc, storage)
        
        # OpenSearch (needs EMR role for access policy)
        opensearch = OpenSearchConstruct(self, "OpenSearch", vpc, emr)
        
        # Permissions (assigns policies to existing roles)
        permissions = PermissionsConstruct(self, "Permissions", storage, opensearch, emr)
        
        # SSM Parameters
        ssm.StringParameter(self, "S3BucketParameter", parameter_name="/sentinel/s3-bucket", string_value=storage.bucket.bucket_name)
        ssm.StringParameter(self, "OpenSearchEndpointParameter", parameter_name="/sentinel/opensearch/endpoint", string_value=f"https://{opensearch.domain.domain_endpoint}")
        ssm.StringParameter(self, "EMRClusterIdParameter", parameter_name="/sentinel/emr-cluster-id", string_value=emr.cluster.ref)
        
        # Stack-level outputs
        CfnOutput(self, "EMRInstanceRoleArn", value=emr.instance_role.role_arn, description="EMR instance role ARN for OpenSearch mapping")
        CfnOutput(self, "S3BucketName", value=storage.bucket.bucket_name, description="S3 bucket name for data storage")
        CfnOutput(self, "OpenSearchEndpoint", value=opensearch.domain.domain_endpoint, description="OpenSearch domain endpoint")
        CfnOutput(self, "EMRClusterId", value=emr.cluster.ref, description="EMR cluster ID")