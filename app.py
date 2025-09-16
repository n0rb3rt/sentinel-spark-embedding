#!/usr/bin/env python3
import aws_cdk as cdk
from src.infra.sentinel_stack import SentinelProcessingStack
from src.infra.emr_gpu_stack import EMRGPUStack

app = cdk.App()

env = cdk.Environment(account="121891376456", region="us-east-2")

# Main processing stack
SentinelProcessingStack(app, "SentinelProcessingStack", env=env)

# EMR GPU stack (optional)
EMRGPUStack(app, "SentinelEMRGPUStack", env=env)

app.synth()