#!/usr/bin/env python3
"""
详细检查GQA benchmark数据集信息
"""
import os
import sys
import pandas as pd

# 设置缓存目录
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')
os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']

# 导入vlmeval
from vlmeval.dataset import build_dataset

def check_gqa_details():
    """详细检查GQA数据集信息"""
    print("=== 详细检查GQA benchmark数据集 ===")
    
    # 构建数据集
    dataset_name = 'GQA_TestDev_Balanced'
    dataset = build_dataset(dataset_name)
    
    if dataset is None:
        print("❌ 数据集构建失败")
        return
    
    print(f"✅ 数据集构建成功: {dataset_name}")
    print(f"\n📊 基本信息:")
    print(f"  数据集类: {dataset.__class__.__name__}")
    print(f"  类型: {dataset.TYPE}")
    print(f"  模态: {dataset.MODALITY}")
    print(f"  样本总数: {len(dataset.data):,}")
    
    # 检查数据列
    print(f"\n📋 数据列信息:")
    print(f"  列名: {list(dataset.data.columns)}")
    print(f"  列数: {len(dataset.data.columns)}")
    
    # 检查数据类型分布
    print(f"\n📈 数据类型分布:")
    if 'question' in dataset.data.columns:
        question_types = dataset.data['question'].str.len().describe()
        print(f"  问题长度统计:")
        print(f"    最短: {question_types['min']:.0f} 字符")
        print(f"    最长: {question_types['max']:.0f} 字符")
        print(f"    平均: {question_types['mean']:.1f} 字符")
    
    if 'answer' in dataset.data.columns:
        answer_stats = dataset.data['answer'].value_counts()
        print(f"  答案分布 (前10个):")
        for i, (answer, count) in enumerate(answer_stats.head(10).items()):
            print(f"    {i+1:2d}. {answer}: {count} 次")
    
    # 检查样本示例
    print(f"\n🔍 样本示例 (前5个):")
    for i in range(min(5, len(dataset.data))):
        sample = dataset.data.iloc[i]
        print(f"\n  样本 {i+1}:")
        print(f"    索引: {sample.get('index', 'N/A')}")
        print(f"    问题: {sample.get('question', 'N/A')}")
        if 'image' in sample or 'image_path' in sample:
            if 'image' in sample:
                print(f"    图像: [base64编码数据]")
            elif 'image_path' in sample:
                print(f"    图像路径: {sample.get('image_path', 'N/A')}")
        print(f"    答案: {sample.get('answer', 'N/A')}")
    
    # 检查prompt构建功能
    print(f"\n🛠️ 功能测试:")
    try:
        prompt = dataset.build_prompt(0)
        print(f"  ✅ Prompt构建功能正常")
        print(f"     构建的prompt类型: {type(prompt)}")
        if isinstance(prompt, list):
            print(f"     Prompt包含 {len(prompt)} 个元素")
            for i, item in enumerate(prompt[:2]):  # 只显示前2个元素
                print(f"      元素 {i+1}: {type(item).__name__}")
                if isinstance(item, dict):
                    print(f"        类型: {item.get('type', 'N/A')}")
                    content = item.get('content', '')
                    if content:
                        print(f"        内容预览: {content[:100]}...")
    except Exception as e:
        print(f"  ❌ Prompt构建失败: {e}")
    
    # 检查图像处理功能
    try:
        image_info = dataset.dump_image(dataset.data.iloc[0])
        print(f"  ✅ 图像处理功能正常")
        print(f"     图像信息: {image_info}")
    except Exception as e:
        print(f"  ❌ 图像处理失败: {e}")
    
    # 检查缓存文件
    print(f"\n💾 缓存文件检查:")
    cache_dir = os.environ['HF_HOME']
    if os.path.exists(cache_dir):
        hub_dir = os.path.join(cache_dir, 'hub')
        if os.path.exists(hub_dir):
            print(f"  Hub缓存目录: {hub_dir}")
            for item in os.listdir(hub_dir):
                item_path = os.path.join(hub_dir, item)
                if os.path.isdir(item_path):
                    size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                              for dirpath, dirnames, filenames in os.walk(item_path) 
                              for filename in filenames)
                    print(f"    📁 {item}: {size / (1024*1024):.1f} MB")
    
    print(f"\n🎯 GQA benchmark验证完成")

if __name__ == "__main__":
    check_gqa_details()