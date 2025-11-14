#!/usr/bin/env python3
"""
GQA benchmark数据集下载脚本
"""
import os
import sys

# 添加vlmeval路径
sys.path.insert(0, '/data/model/Inference_VLM/VLM_Infra/VLMEvalKit')

# 设置缓存目录
os.environ['HF_HOME'] = '/data/model/Inference_VLM/VLM_Infra/datasets/GQA/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']

print(f"HF_HOME设置为: {os.environ['HF_HOME']}")
print(f"TRANSFORMERS_CACHE设置为: {os.environ['TRANSFORMERS_CACHE']}")

# 确保缓存目录存在
if not os.path.exists(os.environ['HF_HOME']):
    os.makedirs(os.environ['HF_HOME'])
    print(f"创建缓存目录: {os.environ['HF_HOME']}")

# 导入vlmeval
from vlmeval.dataset import build_dataset

def download_gqa():
    """下载GQA数据集"""
    print("\n=== 开始下载GQA benchmark数据集 ===")
    
    dataset_name = 'GQA_TestDev_Balanced'
    print(f"📥 下载数据集: {dataset_name}")
    
    try:
        # 尝试通过HuggingFace下载
        print("🔗 尝试通过HuggingFace下载...")
        dataset = build_dataset(dataset_name)
        
        if dataset is not None:
            print(f"✅ 数据集下载成功")
            print(f"   数据集类: {dataset.__class__.__name__}")
            print(f"   类型: {dataset.TYPE}")
            print(f"   模态: {dataset.MODALITY}")
            print(f"   样本数量: {len(dataset.data)}")
            
            # 检查数据文件位置
            data_file = '/data/model/Inference_VLM/VLM_Infra/datasets/GQA/GQA_TestDev_Balanced.tsv'
            if os.path.exists(data_file):
                file_size = os.path.getsize(data_file) / (1024*1024*1024)
                print(f"✅ 数据文件已保存: {data_file}")
                print(f"   文件大小: {file_size:.2f} GB")
            
            return True
        else:
            print(f"❌ 数据集下载失败")
            return False
            
    except Exception as e:
        print(f"❌ 下载过程中出错: {e}")
        print("🔄 尝试通过ModelScope下载...")
        
        # 设置ModelScope环境变量
        os.environ['VLMEVALKIT_USE_MODELSCOPE'] = '1'
        
        try:
            dataset = build_dataset(dataset_name)
            if dataset is not None:
                print(f"✅ 通过ModelScope下载成功")
                return True
            else:
                print(f"❌ ModelScope下载也失败")
                return False
        except Exception as e2:
            print(f"❌ ModelScope下载失败: {e2}")
            return False

if __name__ == "__main__":
    print("开始下载GQA benchmark数据集...")
    
    success = download_gqa()
    
    if success:
        print("\n🎉 GQA benchmark数据集下载完成！")
        print("📁 数据位置: /data/model/Inference_VLM/VLM_Infra/datasets/GQA/")
        print("📊 数据集信息: GQA_TestDev_Balanced (VQA类型, IMAGE模态)")
    else:
        print("\n❌ GQA benchmark数据集下载失败")
        print("请检查网络连接或尝试手动下载")