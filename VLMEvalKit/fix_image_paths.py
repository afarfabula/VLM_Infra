#!/usr/bin/env python3
"""
修复GQA和TextVQA数据集中的图像文件路径
"""
import os
import pandas as pd

def fix_gqa_image_paths():
    """修复GQA数据集的图像路径"""
    print("🔧 修复GQA数据集图像路径...")
    
    gqa_file = './LMUData/GQA_TestDev_Balanced.tsv'
    if not os.path.exists(gqa_file):
        print(f"❌ GQA文件不存在: {gqa_file}")
        return False
    
    # 读取GQA数据
    df = pd.read_csv(gqa_file, sep='\t')
    print(f"📊 读取GQA数据，共{len(df)}行")
    
    # 检查并修复图像路径
    if 'image_path' in df.columns:
        print("📷 检测到image_path列，开始修复路径...")
        
        # 统计原始路径分布
        original_paths = df['image_path'].unique()
        print(f"📁 原始路径模式: {original_paths[:3]}...")
        
        # 修复路径：将绝对路径改为相对路径
        def fix_path(path):
            if isinstance(path, str):
                # 如果路径包含原始目录结构，提取文件名
                if '/home/yanyi.qu/LMUData/images/' in path:
                    filename = os.path.basename(path)
                    return f'./images/{filename}'
                # 如果已经是相对路径，确保指向正确位置
                elif path.startswith('n') and path.endswith('.jpg'):
                    return f'./images/{path}'
                else:
                    return path
            return path
        
        df['image_path'] = df['image_path'].apply(fix_path)
        
        # 保存修复后的数据
        fixed_file = './LMUData/GQA_TestDev_Balanced_fixed.tsv'
        df.to_csv(fixed_file, sep='\t', index=False)
        print(f"✅ GQA图像路径修复完成，保存到: {fixed_file}")
        
        # 显示修复后的路径示例
        fixed_paths = df['image_path'].unique()
        print(f"📁 修复后路径模式: {fixed_paths[:3]}...")
        
        return True
    else:
        print("⚠️ 未找到image_path列，可能不需要修复")
        return True

def fix_textvqa_image_paths():
    """修复TextVQA数据集的图像路径"""
    print("\n🔧 修复TextVQA数据集图像路径...")
    
    textvqa_file = './LMUData/TextVQA_VAL.tsv'
    if not os.path.exists(textvqa_file):
        print(f"❌ TextVQA文件不存在: {textvqa_file}")
        return False
    
    # 读取TextVQA数据
    df = pd.read_csv(textvqa_file, sep='\t')
    print(f"📊 读取TextVQA数据，共{len(df)}行")
    
    # 检查并修复图像路径
    if 'image_path' in df.columns:
        print("📷 检测到image_path列，开始修复路径...")
        
        # 统计原始路径分布
        original_paths = df['image_path'].unique()
        print(f"📁 原始路径模式: {original_paths[:3]}...")
        
        # 修复路径：将绝对路径改为相对路径
        def fix_path(path):
            if isinstance(path, str):
                # 如果路径包含原始目录结构，提取文件名
                if '/home/yanyi.qu/LMUData/images/' in path:
                    filename = os.path.basename(path)
                    return f'./images/{filename}'
                # 如果已经是相对路径，确保指向正确位置
                elif path.startswith('train') or path.startswith('val'):
                    return f'./images/{path}'
                else:
                    return path
            return path
        
        df['image_path'] = df['image_path'].apply(fix_path)
        
        # 保存修复后的数据
        fixed_file = './LMUData/TextVQA_VAL_fixed.tsv'
        df.to_csv(fixed_file, sep='\t', index=False)
        print(f"✅ TextVQA图像路径修复完成，保存到: {fixed_file}")
        
        # 显示修复后的路径示例
        fixed_paths = df['image_path'].unique()
        print(f"📁 修复后路径模式: {fixed_paths[:3]}...")
        
        return True
    else:
        print("⚠️ 未找到image_path列，可能不需要修复")
        return True

def check_images_directory():
    """检查images目录是否存在"""
    print("\n📁 检查images目录...")
    
    images_dir = './LMUData/images/'
    if os.path.exists(images_dir):
        print(f"✅ Images目录存在: {images_dir}")
        
        # 统计图像文件数量
        image_files = []
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        print(f"📷 找到{len(image_files)}个图像文件")
        if image_files:
            print(f"📸 图像文件示例: {image_files[:3]}")
    else:
        print(f"❌ Images目录不存在: {images_dir}")
        print("⚠️ 需要下载图像文件才能完整使用数据集")

if __name__ == "__main__":
    print("开始修复图像文件路径...")
    
    # 检查images目录
    check_images_directory()
    
    # 修复GQA图像路径
    gqa_success = fix_gqa_image_paths()
    
    # 修复TextVQA图像路径
    textvqa_success = fix_textvqa_image_paths()
    
    if gqa_success and textvqa_success:
        print("\n🎉 图像路径修复完成！")
        print("✅ GQA数据集路径修复成功")
        print("✅ TextVQA数据集路径修复成功")
        print("\n📋 下一步：将修复后的数据集移动到datasets目录")
    else:
        print("\n❌ 图像路径修复失败")
        print("请检查数据文件是否存在")