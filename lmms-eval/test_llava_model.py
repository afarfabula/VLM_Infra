#!/usr/bin/env python3
"""
测试脚本，用于验证 LLaVA-NeXT 模型是否能够正常加载和推理
"""

import os
import sys
from pathlib import Path

# 设置正确的缓存路径
cache_path = "/data/model/Inference_VLM/.cache"
os.environ["HF_HOME"] = cache_path
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_DATASETS_CACHE"] = cache_path
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path

# 添加必要的路径
root_path = Path(__file__).parent.parent / "LLaVA-NeXT"
inference_path = root_path / "inference"
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(inference_path))

from PIL import Image
import requests
from llava_next_inference import LlavaNextInference

def test_model_loading():
    """测试模型加载"""
    print("开始测试模型加载...")
    try:
        infer = LlavaNextInference(
            model_path="lmms-lab/llama3-llava-next-8b",
            device="cuda",
            load_precision="4bit",
            use_flash_attn=True,
        )
        print("模型加载成功!")
        return infer
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

def test_inference(infer):
    """测试推理功能"""
    if infer is None:
        print("跳过推理测试，因为模型加载失败")
        return
        
    print("开始测试推理功能...")
    try:
        # 下载测试图片
        url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        
        # 进行推理
        prompt = "请描述这张图片的内容。"
        result = infer.generate_answer(
            prompt,
            image,
            max_new_tokens=100,
            temperature=0.2,
            top_p=0.9,
            min_new_tokens=5,
        )
        
        print(f"推理结果: {result}")
        print("推理测试完成!")
    except Exception as e:
        print(f"推理测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== LLaVA-NeXT 模型测试 ===")
    
    # 测试环境变量
    print("环境变量:")
    print(f"HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
    print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")
    
    # 测试模型加载和推理
    infer = test_model_loading()
    test_inference(infer)
    
    print("=== 测试完成 ===")