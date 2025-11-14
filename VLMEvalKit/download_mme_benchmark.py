#!/usr/bin/env python3
"""
使用VLMEvalKit下载MME benchmark的脚本

MME (Multimodal Evaluation) 是一个全面的多模态模型评估基准，
包含感知和认知两个维度的评估任务。

支持的MME数据集版本：
- MME_CoT: MME的思维链版本
- MMEReasoning: MME的推理版本
"""

import os
import sys
import argparse
from pathlib import Path

# 添加VLMEvalKit到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def setup_environment():
    """设置下载环境"""
    # 设置缓存目录
    cache_dir = Path(__file__).parent / "hf_cache"
    cache_dir.mkdir(exist_ok=True)
    
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)
    
    print(f"缓存目录设置为: {cache_dir}")
    return cache_dir

def download_mme_dataset(dataset_name, use_modelscope=False):
    """下载指定的MME数据集"""
    try:
        if use_modelscope:
            os.environ['VLMEVALKIT_USE_MODELSCOPE'] = '1'
            print(f"使用ModelScope下载 {dataset_name}...")
        else:
            os.environ['VLMEVALKIT_USE_MODELSCOPE'] = '0'
            print(f"使用HuggingFace下载 {dataset_name}...")
        
        from vlmeval.dataset import build_dataset
        
        print(f"开始构建数据集: {dataset_name}")
        dataset = build_dataset(dataset_name)
        
        print(f"数据集构建成功: {dataset}")
        print(f"数据集类型: {dataset.TYPE}")
        
        # 检查数据集大小
        if hasattr(dataset, 'data') and dataset.data is not None:
            print(f"数据集大小: {len(dataset.data)} 条记录")
        else:
            print("数据集大小: 未知")
        
        # 检查数据集的基本信息
        if hasattr(dataset, 'DATASET_URL'):
            print(f"数据集URL: {dataset.DATASET_URL}")
        
        return dataset
        
    except Exception as e:
        print(f"下载 {dataset_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='使用VLMEvalKit下载MME benchmark')
    parser.add_argument('--dataset', type=str, default='all', 
                       choices=['all', 'mme_cot', 'mme_reasoning'],
                       help='要下载的数据集 (默认: all)')
    parser.add_argument('--use-modelscope', action='store_true',
                       help='使用ModelScope下载 (默认使用HuggingFace)')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='自定义缓存目录路径')
    
    args = parser.parse_args()
    
    # 设置环境
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    else:
        cache_dir = setup_environment()
    
    print("=" * 60)
    print("MME Benchmark 下载脚本")
    print("=" * 60)
    
    # 定义要下载的数据集
    datasets_to_download = []
    
    if args.dataset == 'all' or args.dataset == 'mme_cot':
        datasets_to_download.append('MME_CoT')
    
    if args.dataset == 'all' or args.dataset == 'mme_reasoning':
        datasets_to_download.append('MMEReasoning')
    
    print(f"计划下载的数据集: {datasets_to_download}")
    
    # 下载数据集
    downloaded_datasets = {}
    
    for dataset_name in datasets_to_download:
        print(f"\n{'='*40}")
        print(f"下载数据集: {dataset_name}")
        print(f"{'='*40}")
        
        dataset = download_mme_dataset(dataset_name, args.use_modelscope)
        
        if dataset is not None:
            downloaded_datasets[dataset_name] = dataset
            print(f"✓ {dataset_name} 下载成功")
        else:
            print(f"✗ {dataset_name} 下载失败")
    
    # 总结结果
    print(f"\n{'='*60}")
    print("下载总结:")
    print(f"{'='*60}")
    
    if downloaded_datasets:
        print(f"成功下载 {len(downloaded_datasets)} 个数据集:")
        for name, dataset in downloaded_datasets.items():
            size_info = "未知大小"
            if hasattr(dataset, 'data') and dataset.data is not None:
                size_info = f"{len(dataset.data)} 条记录"
            print(f"  - {name}: {size_info}")
    else:
        print("没有成功下载任何数据集")
    
    print(f"\n缓存目录: {cache_dir}")
    print("下载完成!")

if __name__ == "__main__":
    main()