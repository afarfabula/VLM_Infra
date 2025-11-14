#!/usr/bin/env python3
"""
下载TextVQA benchmark数据集
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

def download_textvqa_dataset():
    """下载TextVQA数据集"""
    print("\n=== 开始下载TextVQA benchmark ===")
    
    # 使用官方支持的TextVQA数据集名称
    dataset_name = 'TextVQA_VAL'
    print(f"数据集名称: {dataset_name}")
    
    try:
        # 尝试通过HuggingFace下载
        print("尝试通过HuggingFace下载...")
        dataset = build_dataset(dataset_name)
        
        if dataset is not None:
            print(f"✅ 成功构建TextVQA数据集: {dataset_name}")
            print(f"   数据集类型: {dataset.TYPE}")
            print(f"   数据集模态: {dataset.MODALITY}")
            print(f"   数据集大小: {len(dataset.data)}")
            
            # 显示数据集的基本信息
            if hasattr(dataset, 'data') and len(dataset.data) > 0:
                print("\n数据集样本示例:")
                sample = dataset.data.iloc[0]
                print(f"  索引: {sample.get('index', 'N/A')}")
                print(f"  问题: {sample.get('question', 'N/A')[:100]}..." if 'question' in sample else "  问题字段不存在")
                print(f"  答案: {sample.get('answer', 'N/A')}")
                if 'image' in sample or 'image_path' in sample:
                    if 'image' in sample:
                        print(f"  图像: [base64编码数据]")
                    elif 'image_path' in sample:
                        print(f"  图像路径: {sample.get('image_path', 'N/A')}")
            
            return dataset
        else:
            print(f"❌ 数据集 {dataset_name} 构建失败")
            
    except Exception as e:
        print(f"❌ 下载 {dataset_name} 时出错: {e}")
        
        # 如果HuggingFace失败，尝试modelscope
        print("\n=== 尝试使用modelscope下载 ===")
        os.environ['VLMEVALKIT_USE_MODELSCOPE'] = '1'
        
        try:
            dataset = build_dataset(dataset_name)
            if dataset is not None:
                print(f"✅ 成功通过modelscope构建TextVQA数据集: {dataset_name}")
                print(f"   数据集类型: {dataset.TYPE}")
                print(f"   数据集模态: {dataset.MODALITY}")
                print(f"   数据集大小: {len(dataset.data)}")
                return dataset
            else:
                print(f"❌ modelscope下载 {dataset_name} 也失败")
        except Exception as e2:
            print(f"❌ modelscope下载 {dataset_name} 时出错: {e2}")
    
    return None

if __name__ == "__main__":
    print("开始下载TextVQA benchmark...")
    
    # 下载TextVQA数据集
    textvqa_dataset = download_textvqa_dataset()
    
    if textvqa_dataset is not None:
        print("\n🎉 TextVQA benchmark下载成功！")
        print(f"数据集信息:")
        print(f"  - 名称: {textvqa_dataset.__class__.__name__}")
        print(f"  - 类型: {textvqa_dataset.TYPE}")
        print(f"  - 模态: {textvqa_dataset.MODALITY}")
        print(f"  - 样本数: {len(textvqa_dataset.data)}")
        
        # 检查缓存目录内容
        cache_dir = os.environ['HF_HOME']
        print(f"\n缓存目录内容:")
        if os.path.exists(cache_dir):
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}")
                else:
                    print(f"  📄 {item}")
    else:
        print("\n❌ TextVQA benchmark下载失败")
        print("请检查:")
        print("  1. 网络连接是否正常")
        print("  2. 数据集名称是否正确")
        print("  3. 是否有足够的磁盘空间")