#!/usr/bin/env python3
"""
修复视觉塔模型目录结构的脚本
将.no_exist目录中的文件移动到正确的位置
"""

import os
import shutil
from pathlib import Path

def fix_vision_tower_structure():
    """修复视觉塔模型目录结构"""
    # 定义路径
    base_path = Path("/data/model/Inference_VLM/.cache/models--openai--clip-vit-large-patch14-336")
    no_exist_path = base_path / ".no_exist" / "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    snapshot_path = base_path / "snapshots" / "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    
    # 创建快照目录
    snapshot_path.mkdir(parents=True, exist_ok=True)
    
    # 检查.no_exist目录中的文件
    if no_exist_path.exists():
        print(f"Found files in {no_exist_path}")
        # 移动文件到快照目录
        for file_path in no_exist_path.iterdir():
            dest_path = snapshot_path / file_path.name
            if not dest_path.exists():
                shutil.move(str(file_path), str(dest_path))
                print(f"Moved {file_path.name} to {snapshot_path}")
            else:
                print(f"File {file_path.name} already exists in destination")
        
        print("Files moved successfully!")
    else:
        print(f"No files found in {no_exist_path}")
    
    # 创建refs目录和main引用
    refs_path = base_path / "refs"
    refs_path.mkdir(exist_ok=True)
    
    main_ref_path = refs_path / "main"
    if not main_ref_path.exists():
        with open(main_ref_path, 'w') as f:
            f.write("ce19dc912ca5cd21c8a653c79e251e808ccabcd1")
        print("Created main reference")
    
    # 创建preprocessor_config.json文件
    preprocessor_config = {
        "do_resize": True,
        "size": {"shortest_edge": 336},
        "resample": 3,
        "do_center_crop": True,
        "crop_size": 336,
        "do_normalize": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "do_convert_rgb": True
    }
    
    preprocessor_path = snapshot_path / "preprocessor_config.json"
    if not preprocessor_path.exists():
        import json
        with open(preprocessor_path, 'w') as f:
            json.dump(preprocessor_config, f, indent=2)
        print("Created preprocessor_config.json")
    
    # 创建config.json文件（如果不存在）
    config_path = snapshot_path / "config.json"
    if not config_path.exists():
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
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("Created config.json")
    
    # 检查是否所有必要的文件都存在
    required_files = ["adapter_config.json", "model.safetensors", "model.safetensors.index.json", "preprocessor_config.json", "config.json"]
    missing_files = []
    for file_name in required_files:
        if not (snapshot_path / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"Warning: Missing files: {missing_files}")
    else:
        print("All required files are present!")

if __name__ == "__main__":
    fix_vision_tower_structure()