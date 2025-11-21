"""
Simple fix for LocalCLIPVisionTower initialization issue
Focuses on handling missing mm_vision_select_layer attribute
"""

import os
import sys
from pathlib import Path

# 添加LLaVA路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLava"))

# 导入LLaVA相关模块
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# 导入本地视觉塔补丁
from .local_clip_encoder import LocalCLIPVisionTower
from .builder_patch import apply_builder_patch

def test_local_vision_tower():
    """Test LocalCLIPVisionTower with proper configuration"""
    
    print("Testing LocalCLIPVisionTower initialization...")
    
    # 创建一个模拟的配置对象，包含必要的属性
    class MockConfig:
        def __init__(self):
            # 设置必要的属性
            self.mm_vision_tower = "openai/clip-vit-large-patch14-336"
            self.mm_vision_select_layer = -2  # 这是关键属性，之前缺失
            self.mm_vision_select_feature = "patch"
            self.s2 = False
            
            # 其他可能需要的属性
            self.hidden_size = 1024
            self.image_size = 336
            self.patch_size = 14
            self.num_attention_heads = 16
            self.num_hidden_layers = 24
            self.intermediate_size = 4096
    
    try:
        # 创建配置对象
        config = MockConfig()
        
        # 应用补丁
        apply_builder_patch()
        
        # 初始化 LocalCLIPVisionTower
        vision_tower = LocalCLIPVisionTower(
            vision_tower="openai/clip-vit-large-patch14-336",
            args=config,
            delay_load=True  # 使用延迟加载模式
        )
        
        print("✓ LocalCLIPVisionTower initialized successfully!")
        print(f"  - Select layer: {vision_tower.select_layer}")
        print(f"  - Select feature: {vision_tower.select_feature}")
        print(f"  - Vision tower name: {vision_tower.vision_tower_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize LocalCLIPVisionTower: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_existing_code():
    """Fix the existing LocalCLIPVisionTower code to handle missing attributes"""
    
    print("\nFixing LocalCLIPVisionTower code...")
    
    # 读取现有的 local_clip_encoder.py 文件
    local_clip_encoder_path = Path(__file__).parent / "local_clip_encoder.py"
    
    with open(local_clip_encoder_path, 'r') as f:
        content = f.read()
    
    # 修改 __init__ 方法，添加对缺失属性的处理
    old_init = """    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        # Support for local model path
        self.vision_tower_name = vision_tower
        self.local_model_path = self._get_local_model_path(vision_tower)
        
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')"""
    
    new_init = """    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        # Support for local model path
        self.vision_tower_name = vision_tower
        self.local_model_path = self._get_local_model_path(vision_tower)
        
        # 修复：处理缺失的 mm_vision_select_layer 属性
        self.select_layer = getattr(args, 'mm_vision_select_layer', -2)  # 默认值 -2
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')"""
    
    # 替换代码
    if old_init in content:
        content = content.replace(old_init, new_init)
        
        # 写回文件
        with open(local_clip_encoder_path, 'w') as f:
            f.write(content)
        
        print("✓ Fixed LocalCLIPVisionTower __init__ method")
        return True
    else:
        print("✗ Could not find the exact __init__ method to replace")
        return False

if __name__ == "__main__":
    print("=== LocalCLIPVisionTower Fix ===")
    
    # 首先测试当前状态
    success = test_local_vision_tower()
    
    if not success:
        # 如果测试失败，尝试修复代码
        fix_existing_code()
        
        # 再次测试
        print("\nTesting after fix...")
        test_local_vision_tower()
    
    print("\n=== Fix Complete ===")