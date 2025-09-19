#!/usr/bin/env python3

from aws_cdk import aws_ec2 as ec2
from constructs import Construct

class VpcConstruct(Construct):
    def __init__(self, scope: Construct, construct_id: str, vpc_id: str = None):
        super().__init__(scope, construct_id)
        
        if vpc_id:
            # Use existing VPC
            self.vpc = ec2.Vpc.from_lookup(self, "VPC", vpc_id=vpc_id)
        else:
            # Create new VPC with defaults
            self.vpc = ec2.Vpc(self, "VPC", max_azs=2)
        
        # Security Groups
        self.emr_security_group = ec2.SecurityGroup(
            self, "EMRSecurityGroup",
            vpc=self.vpc,
            description="Security group for EMR GPU cluster",
            allow_all_outbound=True
        )
        
        self.emr_security_group.add_ingress_rule(
            peer=self.emr_security_group,
            connection=ec2.Port.all_traffic(),
            description="EMR internal communication"
        )
        
        self.opensearch_security_group = ec2.SecurityGroup(
            self, "OpenSearchSecurityGroup",
            vpc=self.vpc,
            description="Security group for OpenSearch domain",
            allow_all_outbound=False
        )
        
        self.opensearch_security_group.add_ingress_rule(
            peer=self.emr_security_group,
            connection=ec2.Port.tcp(443),
            description="HTTPS access from EMR cluster"
        )