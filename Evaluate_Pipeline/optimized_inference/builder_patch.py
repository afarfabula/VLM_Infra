"""
Patch module to replace the vision tower builder with a local version
"""

import os
import sys
from pathlib import Path

# 添加LLaVA路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLava"))

# 导入原始的CLIPVisionTower
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .local_clip_encoder import LocalCLIPVisionTower

def patched_build_vision_tower(vision_tower_cfg, **kwargs):
    """
    Patched version of build_vision_tower that uses LocalCLIPVisionTower
    """
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            # 使用我们的本地版本替代原始的CLIPVisionTower
            return LocalCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

def apply_builder_patch():
    """
    Apply the patch to replace the build_vision_tower function
    """
    # 直接修改模块级别的属性
    import llava.model.multimodal_encoder.builder as builder_module
    builder_module.build_vision_tower = patched_build_vision_tower
    print("Applied vision tower builder patch")
    
    # 同时修改全局命名空间中的引用（如果存在）
    import llava.model.multimodal_encoder.builder
    llava.model.multimodal_encoder.builder.build_vision_tower = patched_build_vision_tower