#!/usr/bin/env python3
import aws_cdk as cdk
from src.infra.stack import SentinelStack

app = cdk.App()

env = cdk.Environment(account="121891376456", region="us-east-2")

# Sentinel stack with existing VPC
SentinelStack(
    app, "SentinelProcessingStack", 
    vpc_id=None, # provide existing ID or None to create new
    env=env
)

app.synth()