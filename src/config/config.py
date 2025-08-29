"""
Configuration for Sentinel processing jobs using OmegaConf
"""
from omegaconf import OmegaConf
from pathlib import Path
import yaml
import boto3

def load_clay_metadata():
    """Load Clay model metadata"""
    metadata_path = Path(__file__).parent.parent.parent / "metadata.yaml"
    with open(metadata_path) as f:
        return OmegaConf.create(yaml.safe_load(f))

def load_config():
    """Load config with CLI overrides"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    # Load base config
    base_config = OmegaConf.load(config_path)
    
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
CLAY_METADATA = load_clay_metadata()

# Replace nested spark config with flattened version
CONFIG.spark = dict(_flatten_config(CONFIG.spark))