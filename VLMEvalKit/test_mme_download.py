#!/usr/bin/env python3
"""
测试MME数据集下载功能

这个脚本用于测试VLMEvalKit中MME数据集的下载功能，
验证数据集是否能够正确构建和加载。
"""

import os
import sys
from pathlib import Path

def test_mme_download():
    """测试MME数据集下载功能"""
    print("=" * 60)
    print("MME数据集下载功能测试")
    print("=" * 60)
    
    # 设置缓存目录
    cache_dir = Path(__file__).parent / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    
    print(f"测试缓存目录: {cache_dir}")
    
    try:
        # 导入VLMEvalKit
        from vlmeval.dataset import build_dataset
        
        # 测试MME_CoT数据集
        print("\n1. 测试MME_CoT数据集...")
        try:
            dataset = build_dataset('MME_CoT')
            print(f"   ✓ MME_CoT构建成功")
            print(f"     类型: {dataset.TYPE}")
            if hasattr(dataset, 'data') and dataset.data is not None:
                print(f"     大小: {len(dataset.data)} 条记录")
        except Exception as e:
            print(f"   ✗ MME_CoT构建失败: {e}")
        
        # 测试MMEReasoning数据集
        print("\n2. 测试MMEReasoning数据集...")
        try:
            dataset = build_dataset('MMEReasoning')
            print(f"   ✓ MMEReasoning构建成功")
            print(f"     类型: {dataset.TYPE}")
            if hasattr(dataset, 'data') and dataset.data is not None:
                print(f"     大小: {len(dataset.data)} 条记录")
        except Exception as e:
            print(f"   ✗ MMEReasoning构建失败: {e}")
        
        # 检查缓存文件
        print("\n3. 检查缓存文件...")
        cache_files = list(cache_dir.rglob("*"))
        if cache_files:
            print(f"   ✓ 发现 {len(cache_files)} 个缓存文件")
            for file in cache_files[:5]:  # 只显示前5个文件
                if file.is_file():
                    print(f"     - {file.relative_to(cache_dir)}")
        else:
            print("   ✗ 未发现缓存文件")
        
        print("\n" + "=" * 60)
        print("测试完成!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"导入VLMEvalKit失败: {e}")
        print("请确保VLMEvalKit已正确安装")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mme_download()