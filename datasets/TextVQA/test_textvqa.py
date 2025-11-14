#!/usr/bin/env python3
"""
TextVQA benchmark数据集测试脚本
"""
import os
import sys

# 添加vlmeval路径
sys.path.insert(0, '/data/model/Inference_VLM/VLM_Infra/VLMEvalKit')

# 设置缓存目录
os.environ['HF_HOME'] = '/data/model/Inference_VLM/datasets/TextVQA/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']

print(f"HF_HOME设置为: {os.environ['HF_HOME']}")
print(f"TRANSFORMERS_CACHE设置为: {os.environ['TRANSFORMERS_CACHE']}")

# 导入vlmeval
from vlmeval.dataset import build_dataset

def test_textvqa_dataset():
    """测试TextVQA数据集"""
    print("\n=== 开始测试TextVQA benchmark数据集 ===")
    
    # 检查TextVQA数据文件是否存在 - 使用相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    textvqa_file = os.path.join(script_dir, 'TextVQA_VAL.tsv')
    
    # 如果当前目录没有，检查VLMEvalKit目录
    if not os.path.exists(textvqa_file):
        vlmeval_dir = '/data/model/Inference_VLM/VLM_Infra/VLMEvalKit'
        textvqa_file = os.path.join(vlmeval_dir, 'LMUData', 'TextVQA_VAL.tsv')
    
    if os.path.exists(textvqa_file):
        file_size = os.path.getsize(textvqa_file) / (1024*1024*1024)
        print(f"✅ TextVQA数据文件存在: {textvqa_file}")
        print(f"   文件大小: {file_size:.2f} GB")
    else:
        print(f"❌ TextVQA数据文件不存在: {textvqa_file}")
        print("请检查数据文件是否已下载或路径配置是否正确")
        return False
    
    # 构建数据集
    dataset_name = 'TextVQA_VAL'
    print(f"\n📊 构建数据集: {dataset_name}")
    
    try:
        dataset = build_dataset(dataset_name)
        
        if dataset is not None:
            print(f"✅ 数据集构建成功")
            print(f"   数据集类: {dataset.__class__.__name__}")
            print(f"   类型: {dataset.TYPE}")
            print(f"   模态: {dataset.MODALITY}")
            print(f"   样本数量: {len(dataset.data)}")
            
            # 测试prompt构建
            print(f"\n⚙️ 测试prompt构建功能:")
            try:
                test_sample = dataset.data.iloc[0]
                prompt = dataset.build_prompt(test_sample)
                print(f"   ✅ Prompt构建正常")
                print(f"      示例prompt: {prompt[:100]}...")
                
                # 检查图像路径是否正确
                if 'image_path' in test_sample:
                    image_path = test_sample['image_path']
                    print(f"   📷 图像路径: {image_path}")
                    
                    # 检查图像文件是否存在
                    full_image_path = f'/data/model/Inference_VLM/datasets/TextVQA/{image_path}'
                    if os.path.exists(full_image_path):
                        print(f"   ✅ 图像文件存在")
                    else:
                        print(f"   ⚠️ 图像文件不存在，检查路径是否正确")
                
            except Exception as e:
                print(f"   ❌ Prompt构建失败: {e}")
                return False
            
            # 显示样本示例
            print(f"\n🔍 样本示例:")
            for i in range(min(3, len(dataset.data))):
                sample = dataset.data.iloc[i]
                print(f"   样本 {i+1}:")
                print(f"     索引: {sample.get('index', 'N/A')}")
                if 'question' in sample:
                    question = sample['question']
                    print(f"     问题: {question[:60]}...")
                if 'answer' in sample:
                    answer = sample['answer']
                    print(f"     答案: {answer}")
            
            return True
            
        else:
            print(f"❌ 数据集构建失败")
            return False
            
    except Exception as e:
        print(f"❌ 数据集构建时出错: {e}")
        return False

if __name__ == "__main__":
    print("开始测试TextVQA benchmark数据集...")
    
    # 测试TextVQA数据集
    success = test_textvqa_dataset()
    
    if success:
        print("\n🎉 TextVQA benchmark数据集测试完成！")
        print("数据集状态: ✅ 正常可用")
        print("功能测试: ✅ 全部通过")
        print("📁 数据位置: /data/model/Inference_VLM/datasets/TextVQA/")
    else:
        print("\n❌ TextVQA benchmark数据集测试失败")
        print("请检查数据文件位置和路径配置")