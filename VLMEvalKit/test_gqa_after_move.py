#!/usr/bin/env python3
"""
测试移动后的GQA数据集是否正常工作
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

def test_gqa_dataset():
    """测试GQA数据集"""
    print("\n=== 开始测试移动后的GQA数据集 ===")
    
    # 检查GQA数据文件是否存在
    gqa_file = './LMUData/GQA_TestDev_Balanced.tsv'
    if os.path.exists(gqa_file):
        file_size = os.path.getsize(gqa_file) / (1024*1024*1024)
        print(f"✅ GQA数据文件存在: {gqa_file}")
        print(f"   文件大小: {file_size:.2f} GB")
    else:
        print(f"❌ GQA数据文件不存在: {gqa_file}")
        return False
    
    # 构建数据集
    dataset_name = 'GQA_TestDev_Balanced'
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
                    if os.path.exists(image_path):
                        print(f"   ✅ 图像文件存在")
                    else:
                        print(f"   ⚠️ 图像文件不存在，检查路径是否正确")
                        # 尝试在LMUData目录下查找
                        image_filename = os.path.basename(image_path)
                        lmu_image_path = f'./LMUData/images/{image_filename}'
                        if os.path.exists(lmu_image_path):
                            print(f"   ✅ 在LMUData目录找到图像文件: {lmu_image_path}")
                        else:
                            print(f"   ❌ 图像文件在LMUData目录也不存在")
                
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
    print("开始测试移动后的GQA数据集...")
    
    # 测试GQA数据集
    success = test_gqa_dataset()
    
    if success:
        print("\n🎉 GQA数据集测试完成！")
        print("数据集状态: ✅ 正常可用")
        print("功能测试: ✅ 全部通过")
    else:
        print("\n❌ GQA数据集测试失败")
        print("请检查数据文件位置和路径配置")