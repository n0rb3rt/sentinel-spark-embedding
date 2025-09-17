#!/usr/bin/env python3
"""
Clay compiled model utilities for Spark distribution
"""
import os
import torch
from pathlib import Path

def load_compiled_clay_model():
    """Load compiled Clay model with device placement"""
    # Auto-detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Find model file
    base_paths = [
        '/mnt1/opt/clay-model/',
        str(Path(__file__).parent.parent.parent.parent / ".cache/model/"),
    ]
    
    model_path = None
    for base_path in base_paths:
        for device_type in ['cuda', 'mps', 'cpu']:
            path = os.path.join(base_path, f"clay-v1.5-encoder-{device_type}.pt2")
            if os.path.exists(path):
                model_path = path
                break
        if model_path:
            break
    
    if not model_path:
        raise FileNotFoundError(f"Clay model not found in: {base_paths}")
    
    print(f"Loading Clay model from: {model_path} on {device}")
    return torch.export.load(str(model_path)).module().to(device)

def load_clay_metadata():
    """Load Clay metadata configuration"""
    from omegaconf import OmegaConf
    import yaml
    
    paths = [
        Path(__file__).parent.parent.parent.parent / ".cache/model/configs/metadata.yaml",
        '/mnt1/opt/clay-model/configs/metadata.yaml',
    ]
    
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                return OmegaConf.create(yaml.safe_load(f))
    
    raise FileNotFoundError("Clay metadata not found")

class ClayModelSingleton:
    """Singleton for Clay model - loads once per executor"""
    _instance = None
    
    @classmethod
    def get_model(cls):
        if cls._instance is None:
            cls._instance = load_compiled_clay_model()
        return cls._instance