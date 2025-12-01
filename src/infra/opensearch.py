#!/usr/bin/env python3

from aws_cdk import (
    aws_opensearchservice as opensearch,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_ssm as ssm,
    CfnOutput,
    RemovalPolicy
)
from constructs import Construct

class OpenSearchConstruct(Construct):
    def __init__(self, scope: Construct, construct_id: str, vpc_construct, emr_construct):
        super().__init__(scope, construct_id)
        
        # OpenSearch domain
        self.domain = opensearch.Domain(
            self, "Domain",
            version=opensearch.EngineVersion.open_search("3.1"),
            capacity=opensearch.CapacityConfig(
                data_nodes=1,
                data_node_instance_type="r6g.large.search"
            ),
            ebs=opensearch.EbsOptions(
                volume_size=100,
                volume_type=ec2.EbsDeviceVolumeType.GP3
            ),
            zone_awareness=opensearch.ZoneAwarenessConfig(enabled=False),
            vpc=vpc_construct.vpc,
            vpc_subnets=[ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                availability_zones=[vpc_construct.vpc.availability_zones[0]]
            )],
            security_groups=[vpc_construct.opensearch_security_group],
            removal_policy=RemovalPolicy.DESTROY,
            enforce_https=True,
            node_to_node_encryption=True,
            encryption_at_rest=opensearch.EncryptionAtRestOptions(enabled=True),
            fine_grained_access_control=opensearch.AdvancedSecurityOptions(
                master_user_name="admin"
            ),
            access_policies=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[iam.AnyPrincipal()],
                    actions=["es:*"],
                    resources=[f"arn:aws:es:{scope.region}:{scope.account}:domain/*"]
                )
            ]
        )
        

        
