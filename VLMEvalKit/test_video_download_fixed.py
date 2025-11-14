#!/usr/bin/env python3
"""
测试vlmeval的视频数据集下载功能（修复版本）
"""
import os
import sys

# 设置缓存目录到当前工作目录下
os.environ['HF_HOME'] = '/data/model/Inference_VLM/VLM_Infra/VLMEvalKit/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/data/model/Inference_VLM/VLM_Infra/VLMEvalKit/hf_cache'

from vlmeval.dataset import build_dataset

try:
    print("开始测试视频数据集下载功能（修复版本）...")
    
    # 先测试一个简单的视频数据集
    print("测试MMBench_Video数据集...")
    dataset = build_dataset('MMBench_Video')
    print(f"视频数据集构建成功: {dataset}")
    
    # 检查数据集的基本信息
    print(f"数据集类型: {dataset.TYPE}")
    print(f"数据集大小: {len(dataset.data) if hasattr(dataset, 'data') else 'N/A'}")
    
    # 检查缓存目录
    if os.path.exists('/data/model/Inference_VLM/VLM_Infra/VLMEvalKit/hf_cache'):
        print("缓存目录已创建")
        
    print("视频数据集测试完成！")
    
except Exception as e:
    print(f"视频数据集下载/构建失败: {e}")
    import traceback
    traceback.print_exc()
    
    # 尝试使用modelscope下载
    print("\n尝试使用modelscope下载...")
    os.environ['VLMEVALKIT_USE_MODELSCOPE'] = '1'
    try:
        dataset = build_dataset('MMBench_Video')
        print(f"使用modelscope下载成功: {dataset}")
    except Exception as e2:
        print(f"modelscope下载也失败: {e2}")