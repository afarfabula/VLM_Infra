#!/usr/bin/env python3
"""
测试GQA benchmark下载功能
"""
import os
import sys

# 设置缓存目录为当前目录下的hf_cache
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')
os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']

print(f"HF_HOME设置为: {os.environ['HF_HOME']}")
print(f"TRANSFORMERS_CACHE设置为: {os.environ['TRANSFORMERS_CACHE']}")

# 确保缓存目录存在
if not os.path.exists(os.environ['HF_HOME']):
    os.makedirs(os.environ['HF_HOME'])
    print(f"创建缓存目录: {os.environ['HF_HOME']}")

# 导入vlmeval
from vlmeval.dataset import build_dataset

def test_gqa_download():
    """测试GQA数据集下载"""
    print("\n=== 测试GQA数据集下载 ===")
    
    # 尝试不同的GQA数据集名称
    gqa_variants = ['GQA', 'GQA_VAL', 'GQA_TEST', 'GQA_DEV']
    
    for dataset_name in gqa_variants:
        print(f"\n尝试下载数据集: {dataset_name}")
        try:
            dataset = build_dataset(dataset_name)
            if dataset is not None:
                print(f"✅ 成功构建数据集: {dataset_name}")
                print(f"   数据集类型: {dataset.TYPE}")
                print(f"   数据集模态: {dataset.MODALITY}")
                print(f"   数据集大小: {len(dataset.data)}")
                return dataset
            else:
                print(f"❌ 数据集 {dataset_name} 构建失败")
        except Exception as e:
            print(f"❌ 下载 {dataset_name} 时出错: {e}")
    
    # 如果标准名称都不行，尝试使用modelscope
    print("\n=== 尝试使用modelscope下载 ===")
    os.environ['VLMEVALKIT_USE_MODELSCOPE'] = '1'
    
    for dataset_name in gqa_variants:
        print(f"\n尝试使用modelscope下载数据集: {dataset_name}")
        try:
            dataset = build_dataset(dataset_name)
            if dataset is not None:
                print(f"✅ 成功构建数据集: {dataset_name}")
                print(f"   数据集类型: {dataset.TYPE}")
                print(f"   数据集模态: {dataset.MODALITY}")
                print(f"   数据集大小: {len(dataset.data)}")
                return dataset
            else:
                print(f"❌ 数据集 {dataset_name} 构建失败")
        except Exception as e:
            print(f"❌ 下载 {dataset_name} 时出错: {e}")
    
    return None

if __name__ == "__main__":
    print("开始测试GQA benchmark下载...")
    
    # 测试GQA下载
    gqa_dataset = test_gqa_download()
    
    if gqa_dataset is not None:
        print("\n🎉 GQA benchmark下载成功！")
        print(f"数据集信息:")
        print(f"  - 名称: {gqa_dataset.__class__.__name__}")
        print(f"  - 类型: {gqa_dataset.TYPE}")
        print(f"  - 模态: {gqa_dataset.MODALITY}")
        print(f"  - 样本数: {len(gqa_dataset.data)}")
    else:
        print("\n❌ GQA benchmark下载失败")
        print("可能的原因:")
        print("  1. GQA数据集不在vlmeval支持的数据集列表中")
        print("  2. 数据集名称不正确")
        print("  3. 网络连接问题")
        
        # 显示支持的数据集列表
        print("\n当前支持的数据集列表:")
        from vlmeval.dataset import SUPPORTED_DATASETS
        gqa_related = [name for name in SUPPORTED_DATASETS if 'GQA' in name.upper()]
        if gqa_related:
            print("与GQA相关的数据集:")
            for name in gqa_related:
                print(f"  - {name}")
        else:
            print("没有找到与GQA相关的数据集")