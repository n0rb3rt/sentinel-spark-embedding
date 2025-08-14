"""
Configuration for Sentinel processing jobs using OmegaConf
"""
from omegaconf import OmegaConf
from pathlib import Path

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

# Global config instance
CONFIG = load_config()

# Replace nested spark config with flattened version
CONFIG.spark = dict(_flatten_config(CONFIG.spark))