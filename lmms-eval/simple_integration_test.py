#!/usr/bin/env python3
"""
简单集成测试脚本，用于测试 llava_next_local.py 的功能
"""

import os
import sys
from PIL import Image
import requests
from io import BytesIO

# 添加项目路径
project_root = "/data/model/Inference_VLM/VLM_Infra/lmms-eval"
sys.path.insert(0, project_root)

# 导入模型类
from lmms_eval.models.chat.llava_next_local import LlavaNextLocalChat

def download_test_image(url="http://images.cocodataset.org/val2017/000000039769.jpg"):
    """下载测试图片"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def main():
    print("开始测试 LlavaNextLocalChat...")
    
    # 初始化模型
    model_path = "/data/model/Inference_VLM/models-LLava-NeXT"
    model = LlavaNextLocalChat(pretrained=model_path)
    
    # 下载测试图片
    print("下载测试图片...")
    test_image = download_test_image()
    
    # 创建一个模拟的请求对象
    class MockArgs:
        def __init__(self):
            self.args = (
                "Is there a person in the image?",  # ctx
                lambda x: [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Is there a person in the image?"}]}],  # doc_to_messages
                {
                    "max_new_tokens": 128,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "do_sample": True
                },  # gen_kwargs
                0,  # doc_id
                "pope",  # task
                "test"  # split
            )
    
    # 创建模拟实例
    mock_instances = [MockArgs()]
    
    # 运行生成
    print("运行模型生成...")
    try:
        result = model.generate_until(mock_instances)
        print(f"生成结果: {result}")
    except Exception as e:
        print(f"生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()