#!/usr/bin/env python3
"""
测试vlmeval的数据集下载功能
"""
import os
from vlmeval.dataset import build_dataset

# 设置使用modelscope下载数据
os.environ['VLMEVALKIT_USE_MODELSCOPE'] = '1'

# 测试一个简单的数据集
try:
    print("开始测试数据集下载功能...")
    
    # 尝试构建一个数据集，这会触发数据下载
    dataset = build_dataset('MMVet')
    print(f"数据集构建成功: {dataset}")
    
    # 检查数据集的基本信息
    print(f"数据集类型: {dataset.TYPE}")
    print(f"数据集大小: {len(dataset.data) if hasattr(dataset, 'data') else 'N/A'}")
    
except Exception as e:
    print(f"数据集下载/构建失败: {e}")
    import traceback
    traceback.print_exc()