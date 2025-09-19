#!/usr/bin/env python3

from aws_cdk import (
    aws_iam as iam,
    aws_s3 as s3,
    aws_glue as glue,
    aws_opensearchservice as opensearch
)
from aws_cdk.aws_lakeformation import CfnPermissions as LF
from constructs import Construct

class PermissionsConstruct(Construct):
    def __init__(self, scope: Construct, construct_id: str, 
                 storage_construct, opensearch_construct, emr_construct):
        super().__init__(scope, construct_id)
        
        # Get database name from storage database object
        database_name = storage_construct.database.database_input.name
        
        # Get EMR role from EMR construct
        emr_role = emr_construct.instance_role
        
        # S3 permissions
        storage_construct.bucket.grant_read_write(emr_role)
        
        # Glue permissions
        emr_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "glue:GetDatabase", "glue:GetTable", "glue:CreateTable",
                "glue:UpdateTable", "glue:DeleteTable", "glue:GetPartitions"
            ],
            resources=[
                f"arn:aws:glue:{scope.region}:{scope.account}:catalog",
                f"arn:aws:glue:{scope.region}:{scope.account}:database/{database_name}",
                f"arn:aws:glue:{scope.region}:{scope.account}:table/{database_name}/*"
            ]
        ))
        
        # Sentinel-2 COGs access
        emr_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:ListBucket"],
            resources=["arn:aws:s3:::sentinel-cogs", "arn:aws:s3:::sentinel-cogs/*"]
        ))
        
        # SSM permissions
        emr_role.add_to_policy(iam.PolicyStatement(
            actions=["ssm:GetParameter", "ssm:GetParameters"],
            resources=[f"arn:aws:ssm:{scope.region}:{scope.account}:parameter/sentinel/*"]
        ))
        

        
        # OpenSearch permissions
        emr_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "es:ESHttpPost", "es:ESHttpPut", "es:ESHttpGet",
                "es:ESHttpDelete", "es:ESHttpHead"
            ],
            resources=[f"{opensearch_construct.domain.domain_arn}/*"]
        ))
        
        # Lake Formation permissions
        database_resource = LF.ResourceProperty(
            database_resource=LF.DatabaseResourceProperty(name=database_name)
        )
        table_resource = LF.ResourceProperty(
            table_resource=LF.TableResourceProperty(database_name=database_name, table_wildcard={})
        )
        
        emr_principal = LF.DataLakePrincipalProperty(
            data_lake_principal_identifier=emr_role.role_arn
        )
        
        LF(self, "EMRDatabasePermissions",
           data_lake_principal=emr_principal, 
           resource=database_resource, 
           permissions=["ALL"])
        
        LF(self, "EMRTablePermissions",
           data_lake_principal=emr_principal,
           resource=table_resource,
           permissions=["ALL"])