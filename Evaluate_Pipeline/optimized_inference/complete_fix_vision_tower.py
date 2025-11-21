#!/usr/bin/env python3
"""
Complete fix for vision tower model directory structure
Ensures all necessary files are properly created and linked
"""

import os
import json
import shutil
from pathlib import Path

def create_preprocessor_config(snapshot_path):
    """Create preprocessor_config.json file"""
    preprocessor_config = {
        "do_resize": True,
        "size": {"shortest_edge": 336},
        "resample": 3,
        "do_center_crop": True,
        "crop_size": {"height": 336, "width": 336},
        "do_normalize": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "do_convert_rgb": True
    }
    
    preprocessor_path = snapshot_path / "preprocessor_config.json"
    if not preprocessor_path.exists():
        with open(preprocessor_path, 'w') as f:
            json.dump(preprocessor_config, f, indent=2)
        print("Created preprocessor_config.json")
    else:
        print("preprocessor_config.json already exists")

def create_config_json(snapshot_path):
    """Create config.json file"""
    config = {
        "model_type": "clip_vision_model",
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "image_size": 336,
        "patch_size": 14,
        "hidden_act": "quick_gelu",
        "layer_norm_eps": 1e-05,
        "attention_dropout": 0.0,
        "initializer_factor": 1.0,
        "initializer_range": 0.02
    }
    
    config_path = snapshot_path / "config.json"
    if not config_path.exists():
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("Created config.json")
    else:
        print("config.json already exists")

def create_adapter_config(snapshot_path):
    """Create adapter_config.json file"""
    adapter_config = {
        "model_name": "openai/clip-vit-large-patch14-336",
        "model_class": "CLIPVisionModel",
        "tokenizer_class": "CLIPTokenizer",
        "device": "cuda"
    }
    
    adapter_path = snapshot_path / "adapter_config.json"
    # Check if file is empty or doesn't exist
    if not adapter_path.exists() or adapter_path.stat().st_size == 0:
        with open(adapter_path, 'w') as f:
            json.dump(adapter_config, f, indent=2)
        print("Created adapter_config.json")
    else:
        print("adapter_config.json already exists and is not empty")

def fix_vision_tower_structure():
    """Fix vision tower model directory structure"""
    # Define paths
    base_path = Path("/data/model/Inference_VLM/.cache/models--openai--clip-vit-large-patch14-336")
    no_exist_path = base_path / ".no_exist" / "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    snapshot_path = base_path / "snapshots" / "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    
    # Create snapshot directory
    snapshot_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured snapshot directory exists: {snapshot_path}")
    
    # Move files from .no_exist directory if they exist
    if no_exist_path.exists():
        print(f"Found files in {no_exist_path}")
        for file_path in no_exist_path.iterdir():
            dest_path = snapshot_path / file_path.name
            # Only move if destination doesn't exist or is empty
            if not dest_path.exists() or (dest_path.exists() and dest_path.stat().st_size == 0):
                shutil.move(str(file_path), str(dest_path))
                print(f"Moved {file_path.name} to {snapshot_path}")
            else:
                print(f"File {file_path.name} already exists in destination")
        print("Files moved successfully!")
    else:
        print(f"No files found in {no_exist_path}")
    
    # Create refs directory and main reference
    refs_path = base_path / "refs"
    refs_path.mkdir(exist_ok=True)
    
    main_ref_path = refs_path / "main"
    if not main_ref_path.exists():
        with open(main_ref_path, 'w') as f:
            f.write("ce19dc912ca5cd21c8a653c79e251e808ccabcd1")
        print("Created main reference")
    else:
        print("Main reference already exists")
    
    # Create missing configuration files
    create_preprocessor_config(snapshot_path)
    create_config_json(snapshot_path)
    create_adapter_config(snapshot_path)
    
    # Verify all required files exist
    required_files = [
        "adapter_config.json", 
        "model.safetensors", 
        "model.safetensors.index.json", 
        "preprocessor_config.json", 
        "config.json"
    ]
    
    print("\nVerifying required files:")
    for file_name in required_files:
        file_path = snapshot_path / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✓ {file_name} ({size} bytes)")
        else:
            print(f"✗ {file_name} (MISSING)")

if __name__ == "__main__":
    fix_vision_tower_structure()