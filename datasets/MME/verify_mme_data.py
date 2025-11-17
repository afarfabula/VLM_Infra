#!/usr/bin/env python3
"""
验证MME数据集下载结果
"""

import os
import pandas as pd
from pathlib import Path

def verify_mme_dataset():
    """验证MME数据集"""
    print("验证MME数据集下载结果...")
    
    # 数据目录
    data_dir = Path("./mme_data/data")
    
    if not data_dir.exists():
        print("✗ 数据目录不存在")
        return False
    
    # 检查parquet文件
    parquet_files = list(data_dir.glob("*.parquet"))
    print(f"找到 {len(parquet_files)} 个parquet文件:")
    
    for file in parquet_files:
        file_size = file.stat().st_size / (1024 * 1024)  # MB
        print(f"  - {file.name}: {file_size:.2f} MB")
    
    # 尝试读取第一个parquet文件
    if parquet_files:
        try:
            print("\n读取第一个parquet文件的内容...")
            df = pd.read_parquet(parquet_files[0])
            
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            
            # 显示前几行数据
            print("\n前3行数据:")
            print(df.head(3))
            
            # 显示数据类型
            print("\n数据类型:")
            print(df.dtypes)
            
            return True
            
        except Exception as e:
            print(f"✗ 读取parquet文件失败: {e}")
            return False
    else:
        print("✗ 没有找到parquet文件")
        return False

def check_readme():
    """检查README文件"""
    readme_path = Path("./mme_data/README.md")
    
    if readme_path.exists():
        print("\n读取README文件内容...")
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 显示前500个字符
                print(content[:500] + "..." if len(content) > 500 else content)
            return True
        except Exception as e:
            print(f"读取README失败: {e}")
            return False
    else:
        print("✗ README文件不存在")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("MME数据集验证脚本")
    print("=" * 60)
    
    success = True
    
    # 验证数据文件
    if not verify_mme_dataset():
        success = False
    
    # 检查README
    if not check_readme():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ MME数据集验证成功！")
        print(f"数据位置: {Path('./mme_data').absolute()}")
        return 0
    else:
        print("✗ MME数据集验证失败")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())