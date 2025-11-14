#!/usr/bin/env python3
"""
MME (Multimodal Evaluation) 数据集下载脚本

MME是一个全面的多模态模型评估基准，包含感知和认知两个维度的评估任务。
数据集包含14个子任务，涵盖图像识别、文本理解、推理等多个方面。
"""

import os
import sys
import argparse
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import hashlib

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def calculate_md5(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, destination, chunk_size=8192):
    """下载文件并显示进度条"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=f"下载 {Path(destination).name}") as pbar:
                for data in response.iter_content(chunk_size=chunk_size):
                    file.write(data)
                    pbar.update(len(data))
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def extract_archive(file_path, extract_to):
    """解压压缩文件"""
    try:
        if file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif file_path.suffix in ['.tar', '.gz', '.bz2']:
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"不支持的文件格式: {file_path.suffix}")
            return False
        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False

def download_mme_dataset(base_dir, force_download=False):
    """下载MME数据集"""
    # MME数据集信息
    mme_info = {
        "name": "MME",
        "description": "Multimodal Evaluation Benchmark",
        "tasks": [
            "艺术", "名人", "颜色", "计数", "场景", "OCR", "位置", "海报",
            "推理", "属性", "物体", "关系", "行为", "风格"
        ]
    }
    
    # 数据集下载链接（需要根据实际情况更新）
    dataset_urls = {
        "images": "https://huggingface.co/datasets/BradyFU/MME/resolve/main/images.zip",
        "annotations": "https://huggingface.co/datasets/BradyFU/MME/resolve/main/annotations.zip",
        "tsv_files": "https://huggingface.co/datasets/BradyFU/MME/resolve/main/mme.tsv"
    }
    
    # 预期的MD5哈希值（需要根据实际文件更新）
    expected_md5 = {
        "images.zip": "example_md5_hash_here",
        "annotations.zip": "example_md5_hash_here",
        "mme.tsv": "example_md5_hash_here"
    }
    
    # 创建目录结构
    base_path = Path(base_dir)
    images_dir = base_path / "images"
    annotations_dir = base_path / "annotations"
    tsv_dir = base_path / "tsv_files"
    
    for dir_path in [images_dir, annotations_dir, tsv_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载MME数据集到: {base_path}")
    print(f"数据集包含 {len(mme_info['tasks'])} 个子任务: {', '.join(mme_info['tasks'])}")
    
    # 下载图像数据
    images_zip = images_dir / "images.zip"
    if not images_zip.exists() or force_download:
        print("\n下载图像数据...")
        if download_file(dataset_urls["images"], images_zip):
            # 验证文件完整性
            actual_md5 = calculate_md5(images_zip)
            if expected_md5["images.zip"] and actual_md5 != expected_md5["images.zip"]:
                print("警告: 图像文件MD5校验失败")
            
            # 解压图像数据
            print("解压图像数据...")
            if extract_archive(images_zip, images_dir):
                # 删除压缩文件以节省空间
                images_zip.unlink()
                print("图像数据解压完成")
        else:
            print("图像数据下载失败")
    else:
        print("图像数据已存在，跳过下载")
    
    # 下载标注数据
    annotations_zip = annotations_dir / "annotations.zip"
    if not annotations_zip.exists() or force_download:
        print("\n下载标注数据...")
        if download_file(dataset_urls["annotations"], annotations_zip):
            # 验证文件完整性
            actual_md5 = calculate_md5(annotations_zip)
            if expected_md5["annotations.zip"] and actual_md5 != expected_md5["annotations.zip"]:
                print("警告: 标注文件MD5校验失败")
            
            # 解压标注数据
            print("解压标注数据...")
            if extract_archive(annotations_zip, annotations_dir):
                # 删除压缩文件以节省空间
                annotations_zip.unlink()
                print("标注数据解压完成")
        else:
            print("标注数据下载失败")
    else:
        print("标注数据已存在，跳过下载")
    
    # 下载TSV文件
    tsv_file = tsv_dir / "mme.tsv"
    if not tsv_file.exists() or force_download:
        print("\n下载TSV文件...")
        if download_file(dataset_urls["tsv_files"], tsv_file):
            # 验证文件完整性
            actual_md5 = calculate_md5(tsv_file)
            if expected_md5["mme.tsv"] and actual_md5 != expected_md5["mme.tsv"]:
                print("警告: TSV文件MD5校验失败")
            print("TSV文件下载完成")
        else:
            print("TSV文件下载失败")
    else:
        print("TSV文件已存在，跳过下载")
    
    # 统计下载结果
    print("\n" + "="*50)
    print("MME数据集下载完成!")
    
    # 统计文件数量
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    annotation_files = list(annotations_dir.glob("*.json"))
    
    print(f"图像文件数量: {len(image_files)}")
    print(f"标注文件数量: {len(annotation_files)}")
    print(f"TSV文件: {'存在' if tsv_file.exists() else '缺失'}")
    
    if tsv_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(tsv_file, sep='\t')
            print(f"TSV文件记录数: {len(df)}")
        except Exception as e:
            print(f"读取TSV文件失败: {e}")
    
    print(f"\n数据保存位置: {base_path}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='下载MME数据集')
    parser.add_argument('--base-dir', type=str, default='.', 
                       help='数据集保存的基础目录 (默认: 当前目录)')
    parser.add_argument('--force', action='store_true',
                       help='强制重新下载，即使文件已存在')
    
    args = parser.parse_args()
    
    # 设置基础目录
    if args.base_dir == '.':
        base_dir = Path(__file__).parent
    else:
        base_dir = Path(args.base_dir)
    
    download_mme_dataset(base_dir, args.force)

if __name__ == "__main__":
    main()