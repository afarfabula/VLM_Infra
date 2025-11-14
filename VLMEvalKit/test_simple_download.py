#!/usr/bin/env python3
"""
测试vlmeval的简单数据集下载功能
"""
import os
import sys

# 设置缓存目录到当前工作目录下
os.environ['HF_HOME'] = '/data/model/Inference_VLM/VLM_Infra/VLMEvalKit/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/data/model/Inference_VLM/VLM_Infra/VLMEvalKit/hf_cache'

from vlmeval.dataset import build_dataset

try:
    print("开始测试简单数据集下载功能...")
    
    # 尝试构建一个简单的图像数据集
    dataset = build_dataset('MMVet')
    print(f"数据集构建成功: {dataset}")
    
    # 检查数据集的基本信息
    print(f"数据集类型: {dataset.TYPE}")
    print(f"数据集大小: {len(dataset.data) if hasattr(dataset, 'data') else 'N/A'}")
    
    # 检查缓存目录
    if os.path.exists('/data/model/Inference_VLM/VLM_Infra/VLMEvalKit/hf_cache'):
        print("缓存目录已创建")
        
    print("测试完成！")
    
except Exception as e:
    print(f"数据集下载/构建失败: {e}")
    import traceback
    traceback.print_exc()