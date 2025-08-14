#!/usr/bin/env python3
import aws_cdk as cdk
from infra.sentinel_stack import SentinelProcessingStack

app = cdk.App()
SentinelProcessingStack(app, "SentinelProcessingStack")
app.synth()