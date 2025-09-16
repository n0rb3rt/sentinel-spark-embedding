#!/usr/bin/env python3
"""
Clay compiled model utilities for Spark distribution
"""
import os
import torch

class ClayModelSingleton:
    """Singleton pattern for compiled Clay model - loads once per executor"""
    _instance = None
    
    @classmethod
    def get_model(cls):
        if cls._instance is None:
            cls._instance = load_compiled_clay_model()
        return cls._instance

def load_compiled_clay_model():
    """Load compiled Clay model from distributed locations"""
    from pathlib import Path
    
    # Try device-specific models in preference order
    base_paths = [
        '/mnt1/opt/clay-model/',
        str(Path(__file__).parent.parent.parent.parent / ".cache/model/"),
    ]
    
    model_path = None
    for base_path in base_paths:
        cuda_path = os.path.join(base_path, "clay-v1.5-encoder-cuda.pt2")
        mps_path = os.path.join(base_path, "clay-v1.5-encoder-mps.pt2")
        cpu_path = os.path.join(base_path, "clay-v1.5-encoder-cpu.pt2")
        
        if os.path.exists(cuda_path):
            model_path = cuda_path
            break
        elif os.path.exists(mps_path):
            model_path = mps_path
            break
        elif os.path.exists(cpu_path):
            model_path = cpu_path
            break
    
    if model_path is None:
        raise FileNotFoundError(f"Compiled Clay model not found in any of: {base_paths}")
    
    print(f"Loading compiled Clay model from: {model_path}")
    model = torch.export.load(str(model_path)).module()
    print("Compiled Clay model loaded successfully")
    return model

def load_clay_metadata():
    """Load Clay metadata from distributed archive or local file"""
    from omegaconf import OmegaConf
    import yaml
    from pathlib import Path
    
    # Try multiple locations - cache first for local dev
    paths = [
        Path(__file__).parent.parent.parent.parent / ".cache/model/configs/metadata.yaml",  # Local cache
        '/mnt1/opt/clay-model/configs/metadata.yaml',  # Pre-staged from bootstrap
    ]
    
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                return OmegaConf.create(yaml.safe_load(f))
    
    raise FileNotFoundError("Clay metadata not found in archive or local directory")