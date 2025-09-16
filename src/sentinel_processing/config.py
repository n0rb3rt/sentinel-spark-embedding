"""
Configuration for Sentinel processing jobs using OmegaConf
"""
from omegaconf import OmegaConf
from pathlib import Path
import yaml
import boto3
from importlib import resources
from importlib.metadata import version



def load_config():
    """Load config with CLI overrides"""
    # Load base config from same package
    with resources.open_text('sentinel_processing', 'config.yaml') as f:
        base_config = OmegaConf.create(yaml.safe_load(f))
    
    # Set version from actual package version
    base_config.version = version('sentinel_processing')
    
    # Merge with CLI overrides (automatically parsed)
    cli_config = OmegaConf.from_cli()
    
    # Merge all sources (CLI takes highest precedence)
    config = OmegaConf.merge(base_config, cli_config)
    
    return config

def _flatten_config(config, prefix="spark"):
    """Flatten nested config to dot notation for Spark"""
    from collections.abc import Mapping
    items = []
    for key, value in config.items():
        if isinstance(value, Mapping):
            items.extend(_flatten_config(value, f"{prefix}.{key}"))
        else:
            items.append((f"{prefix}.{key}", str(value)))
    return items

def get_ssm_parameter(name):
    """Get parameter from SSM Parameter Store"""
    ssm = boto3.client('ssm')
    return ssm.get_parameter(Name=name)['Parameter']['Value']

# Global config instances
CONFIG = load_config()