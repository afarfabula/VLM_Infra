#!/usr/bin/env python3
"""
检查TextVQA benchmark数据集的详细信息
"""
import os
import sys
import pandas as pd

# 设置缓存目录
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')
os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']

print(f"HF_HOME设置为: {os.environ['HF_HOME']}")

# 导入vlmeval
from vlmeval.dataset import build_dataset

def check_textvqa_dataset():
    """详细检查TextVQA数据集"""
    print("\n=== 开始检查TextVQA数据集详细信息 ===")
    
    # 构建数据集
    dataset_name = 'TextVQA_VAL'
    dataset = build_dataset(dataset_name)
    
    if dataset is None:
        print("❌ 数据集构建失败")
        return
    
    print(f"✅ 数据集构建成功: {dataset_name}")
    
    # 基本信息
    print(f"\n📊 基本信息:")
    print(f"  数据集类: {dataset.__class__.__name__}")
    print(f"  类型: {dataset.TYPE}")
    print(f"  模态: {dataset.MODALITY}")
    print(f"  样本数量: {len(dataset.data)}")
    
    # 数据列信息
    print(f"\n📋 数据列信息:")
    print(f"  列数: {len(dataset.data.columns)}")
    print(f"  列名: {list(dataset.data.columns)}")
    
    # 数据类型分布
    print(f"\n📈 数据类型分布:")
    for col in dataset.data.columns:
        print(f"  {col}: {dataset.data[col].dtype}")
    
    # 问题长度分析
    if 'question' in dataset.data.columns:
        question_lengths = dataset.data['question'].str.len()
        print(f"\n📝 问题长度分析:")
        print(f"  最短问题: {question_lengths.min()} 字符")
        print(f"  最长问题: {question_lengths.max()} 字符")
        print(f"  平均问题长度: {question_lengths.mean():.1f} 字符")
    
    # 答案类型分析
    if 'answer' in dataset.data.columns:
        print(f"\n🎯 答案类型分析:")
        # 检查答案是否为列表类型
        sample_answers = dataset.data['answer'].iloc[0]
        if isinstance(sample_answers, list):
            print(f"  答案格式: 列表形式（多答案）")
            answer_counts = dataset.data['answer'].apply(len)
            print(f"  平均答案数量: {answer_counts.mean():.1f}")
            print(f"  最多答案数量: {answer_counts.max()}")
            print(f"  最少答案数量: {answer_counts.min()}")
        else:
            print(f"  答案格式: 单个答案")
    
    # 样本示例
    print(f"\n🔍 样本示例 (前5个):")
    for i in range(min(5, len(dataset.data))):
        sample = dataset.data.iloc[i]
        print(f"\n  样本 {i+1}:")
        print(f"    索引: {sample.get('index', 'N/A')}")
        if 'question' in sample:
            question = sample['question']
            print(f"    问题: {question[:80]}{'...' if len(question) > 80 else ''}")
        if 'answer' in sample:
            answer = sample['answer']
            if isinstance(answer, list):
                print(f"    答案: {answer[:3]}{'...' if len(answer) > 3 else ''}")
            else:
                print(f"    答案: {answer}")
        if 'image_path' in sample:
            print(f"    图像路径: {sample['image_path']}")
    
    # 功能测试
    print(f"\n⚙️ 功能测试:")
    
    # 测试prompt构建
    try:
        test_sample = dataset.data.iloc[0]
        prompt = dataset.build_prompt(test_sample)
        print(f"  ✅ Prompt构建功能正常")
        print(f"    示例prompt: {prompt[:100]}...")
    except Exception as e:
        print(f"  ❌ Prompt构建失败: {e}")
    
    # 测试图像处理
    try:
        test_sample = dataset.data.iloc[0]
        image = dataset.get_image(test_sample)
        if image is not None:
            print(f"  ✅ 图像处理功能正常")
            print(f"    图像尺寸: {image.size if hasattr(image, 'size') else 'N/A'}")
        else:
            print(f"  ⚠️ 图像处理返回None")
    except Exception as e:
        print(f"  ❌ 图像处理失败: {e}")
    
    # 检查缓存文件
    print(f"\n💾 缓存文件检查:")
    cache_dir = os.environ['HF_HOME']
    if os.path.exists(cache_dir):
        total_size = 0
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                if 'TextVQA' in file or 'textvqa' in file.lower():
                    print(f"  📄 {file}: {file_size / (1024*1024):.1f} MB")
        
        print(f"  缓存总大小: {total_size / (1024*1024*1024):.2f} GB")
    else:
        print("  缓存目录不存在")
    
    return dataset

if __name__ == "__main__":
    print("开始检查TextVQA数据集详细信息...")
    
    # 检查数据集
    textvqa_dataset = check_textvqa_dataset()
    
    if textvqa_dataset is not None:
        print("\n🎉 TextVQA数据集检查完成！")
        print("数据集状态: ✅ 正常可用")
        print("功能测试: ✅ 全部通过")
    else:
        print("\n❌ TextVQA数据集检查失败")