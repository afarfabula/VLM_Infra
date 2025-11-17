#!/usr/bin/env python3
"""
使用huggingface_hub下载MME数据集
"""

import os
from huggingface_hub import snapshot_download

def download_mme_dataset():
    """下载MME数据集"""
    print("开始下载MME数据集...")
    
    # 设置下载目录
    cache_dir = "./mme_data"
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # 下载数据集
        print(f"下载到目录: {cache_dir}")
        
        # 使用snapshot_download下载整个数据集
        local_dir = snapshot_download(
            repo_id="lmms-lab/MME",
            repo_type="dataset",
            local_dir=cache_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"✓ MME数据集下载完成!")
        print(f"数据保存位置: {local_dir}")
        
        # 列出下载的文件
        print("\n下载的文件列表:")
        for root, dirs, files in os.walk(local_dir):
            level = root.replace(local_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        return True
        
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("MME数据集下载脚本 (huggingface_hub)")
    print("=" * 60)
    
    if download_mme_dataset():
        print("\n✓ 下载任务完成")
        return 0
    else:
        print("\n✗ 下载任务失败")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())