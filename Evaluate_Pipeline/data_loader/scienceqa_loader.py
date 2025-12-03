"""
ScienceQA数据集加载器
使用本地problems.json文件和图片数据
"""

import os
import sys
import json
import random
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

# 添加ScienceQA目录到sys.path
scienceqa_path = '/data/model/Inference_VLM/VLM_Infra/datasets/ScienceQA/ScienceQA'
if scienceqa_path not in sys.path:
    sys.path.append(scienceqa_path)

# 导入build_prompt函数
from models.base_prompt import build_prompt

class Args:
    """模拟args参数对象"""
    def __init__(self, prompt_format, use_caption, options):
        self.prompt_format = prompt_format
        self.use_caption = use_caption
        self.options = options

class ScienceQADataLoader(Dataset):
    """
    ScienceQA数据集加载器
    使用本地problems.json文件和图片数据
    """
    
    def __init__(self, data_root=None, split='test', image_transform=None, 
                 prompt_format="QCM-A", use_caption=False, options=["A", "B", "C", "D", "E"], shot_number=3, is_test=False):
        """
        初始化ScienceQA数据加载器
        
        Args:
            data_root: 数据集根目录（默认为固定路径）
            split: 数据分割，可选'train', 'val', 'test'
            image_transform: 图像变换函数
            prompt_format: 提示格式
            use_caption: 是否使用图像标题
            options: 选项列表
            shot_number: few-shot数量
            is_test: 是否为测试模式（测试模式下不包含示例）
        """
        # 使用固定的本地路径
        if data_root is None:
            data_root = '/data/model/Inference_VLM/VLM_Infra/datasets/ScienceQA/ScienceQA'
            
        self.data_root = data_root
        self.split = split
        self.image_transform = image_transform
        self.prompt_format = prompt_format
        self.use_caption = use_caption
        self.options = options
        self.shot_number = shot_number
        self.is_test = is_test
        
        # 数据文件路径
        self.problems_file = os.path.join(data_root, 'data', 'scienceqa', 'problems.json')
        self.image_dir = os.path.join(data_root, 'tools', self.split)
        
        # 检查文件和目录是否存在
        if not os.path.exists(self.problems_file):
            raise FileNotFoundError(f"problems.json文件不存在: {self.problems_file}")
        
        if not os.path.exists(self.image_dir):
            print(f"警告: 图片目录不存在: {self.image_dir}，将只加载文本数据")
        
        # 加载问题数据
        self.problems = self._load_problems()
        self.samples = self._load_samples()
        print(f"加载完成，共包含 {len(self.samples)} 个{split}样本")
    
    def _load_problems(self):
        """
        加载所有问题数据
        
        Returns:
            dict: 所有问题数据
        """
        with open(self.problems_file, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        return problems
    
    def _load_samples(self):
        """
        加载problems.json文件中的样本数据
        
        Returns:
            样本列表
        """
        samples = []
        
        # 筛选指定分割的样本
        for qid, problem in self.problems.items():
            if problem['split'] == self.split:
                # 查找对应的图片
                image_paths = []
                img_dir = os.path.join(self.image_dir, qid)
                if os.path.exists(img_dir):
                    for img_file in os.listdir(img_dir):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            image_paths.append(os.path.join(img_dir, img_file))
                
                # 创建样本
                sample = {
                    'id': qid,
                    'question': problem.get('question', ''),
                    'choices': problem.get('choices', []),
                    'answer': problem.get('answer', -1),
                    'hint': problem.get('hint', ''),
                    'image_paths': image_paths,
                    'task': problem.get('task', ''),
                    'grade': problem.get('grade', ''),
                    'subject': problem.get('subject', ''),
                    'topic': problem.get('topic', ''),
                    'category': problem.get('category', ''),
                    'skill': problem.get('skill', ''),
                    'lecture': problem.get('lecture', ''),
                    'solution': problem.get('solution', ''),
                    'caption': problem.get('caption', '')
                }
                
                # 只添加有图像的样本
                if image_paths:
                    samples.append(sample)
        
        return samples
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含问题、选项、答案和图像的字典
        """
        # 获取样本数据
        sample = self.samples[idx]
        
        # 提取基本信息
        qid = sample['id']
        question = sample['question']
        choices = sample['choices']
        answer = sample['answer']
        image_paths = sample['image_paths']
        
        # 提取额外信息
        hint = sample.get('hint', '')
        lecture = sample.get('lecture', '')
        solution = sample.get('solution', '')
        caption = sample.get('caption', '')  # 如果没有caption字段，则为空字符串
        
        # 提取元数据
        subject = sample.get('subject', 'N/A')
        grade = sample.get('grade', 'N/A')
        topic = sample.get('topic', 'N/A')
        category = sample.get('category', 'N/A')
        
        # 加载图像（如果存在）
        image = None
        image_path = None
        if image_paths:
            image_path = image_paths[0]  # 使用第一个图像
            try:
                image = Image.open(image_path).convert('RGB')
                if self.image_transform:
                    image = self.image_transform(image)
            except Exception as e:
                print(f"警告: 加载图像失败 {image_path}: {e}")
        
        # 创建args对象用于build_prompt
        args_obj = Args(self.prompt_format, self.use_caption, self.options)
        
        # 随机选择训练示例
        train_qids = [q['id'] for q in self.samples if q['id'] != qid]
        shot_qids = random.sample(train_qids, min(self.shot_number, len(train_qids)))
        
        # 构建prompt_input
        prompt_input = build_prompt(self.problems, shot_qids, qid, args_obj, is_test=self.is_test)
        
        # 构建样本字典
        sample_dict = {
            'question_id': qid,
            'question': question,
            'choices': choices,
            'answer': answer,
            'image': image,
            'image_path': image_path,
            'subject': subject,
            'grade': grade,
            'topic': topic,
            'category': category,
            'hint': hint,
            'lecture': lecture,
            'solution': solution,
            'caption': caption,
            'prompt_input': prompt_input
        }
        
        return sample_dict
    
    def __len__(self):
        """
        返回样本总数
        """
        return len(self.samples)
    
    def get_dataset_info(self):
        """
        获取数据集信息
        
        Returns:
            数据集信息字典
        """
        subjects = {}
        grades = {}
        topics = {}
        
        for sample in self.samples:
            # 统计学科分布
            subject = sample.get('subject', 'unknown')
            subjects[subject] = subjects.get(subject, 0) + 1
            
            # 统计年级分布
            grade = sample.get('grade', 'unknown')
            grades[grade] = grades.get(grade, 0) + 1
            
            # 统计主题分布
            topic = sample.get('topic', 'unknown')
            topics[topic] = topics.get(topic, 0) + 1
        
        return {
            'total_samples': len(self.samples),
            'subject_distribution': subjects,
            'grade_distribution': grades,
            'topic_distribution': topics
        }
    
    def show_sample(self, idx=0):
        """
        显示数据集中的一个样本
        
        Args:
            idx: 样本索引
        """
        if idx < 0 or idx >= len(self.samples):
            raise ValueError(f"样本索引 {idx} 超出范围 [0, {len(self.samples)-1}]")
        
        sample = self.samples[idx]
        print("=== ScienceQA样本 ===")
        print(f"问题: {sample['question']}")
        print(f"选项: {sample['choices']}")
        print(f"答案索引: {sample['answer']}")
        if sample['answer'] >= 0 and sample['answer'] < len(sample['choices']):
            print(f"正确答案: {sample['choices'][sample['answer']]}")
        
        if sample['hint']:
            print(f"提示: {sample['hint']}")
        
        if sample['image_paths']:
            print(f"图像: 有 {len(sample['image_paths'])} 张图像")
            for img_path in sample['image_paths']:
                print(f"  - {img_path}")
        else:
            print("图像: 无")
        
        print(f"学科: {sample.get('subject', 'N/A')}")
        print(f"年级: {sample.get('grade', 'N/A')}")
        print(f"主题: {sample.get('topic', 'N/A')}")
        print(f"类别: {sample.get('category', 'N/A')}")
        
        return sample

    def load_image(self, image_path):
        """
        加载图像的方法，用于与VQAv2加载器保持一致的接口
        
        Args:
            image_path: 图像路径
            
        Returns:
            PIL Image对象或None
        """
        if image_path is None:
            return None
        
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"警告: 加载图像失败 {image_path}: {e}")
            return None

def _collate_fn(batch):
    """
    自定义批次处理函数，处理不同长度的选项列表
    
    Args:
        batch: 批次数据列表
        
    Returns:
        处理后的批次数据字典
    """
    # 提取批次中的各个字段
    result = {}
    
    # 获取所有字段名
    keys = batch[0].keys()
    
    for key in keys:
        # 处理每个字段
        values = [item[key] for item in batch]
        
        # 特殊处理选项字段（可能长度不一致）
        if key == 'choices':
            # 保持为列表的列表，不进行堆叠
            result[key] = values
        # 处理图像字段（可能为None）
        elif key == 'image':
            # 过滤None值，只对非None的图像进行堆叠
            non_none_images = [img for img in values if img is not None]
            if non_none_images:
                # 对于非None的图像，保持原样
                result[key] = values
            else:
                result[key] = None
        # 处理其他字段
        else:
            # 尝试进行正常的堆叠
            try:
                # 对于数字类型，转换为tensor
                if all(isinstance(v, (int, float)) for v in values):
                    result[key] = torch.tensor(values)
                # 对于字符串等其他类型，保持为列表
                else:
                    result[key] = values
            except Exception:
                # 如果堆叠失败，保持为列表
                result[key] = values
    
    return result

def scienceqa_collate_fn(batch):
    """
    自定义的collate函数，用于处理ScienceQA数据集中变长的数据
    
    Args:
        batch: 批次数据列表
        
    Returns:
        合并后的批次字典
    """
    # 创建一个空字典来存储合并后的批次
    collated_batch = {}
    
    # 获取第一个样本的所有键
    if not batch:
        return collated_batch
    
    # 对每个键进行处理
    for key in batch[0].keys():
        # 获取该键对应的所有值
        values = [item[key] for item in batch]
        
        # 处理不同类型的数据
        if key == 'choices':
            # choices是变长列表，不进行合并，保持为列表的列表
            collated_batch[key] = values
        elif key == 'image':
            # 如果图像存在，保留原始列表（None值表示没有图像）
            collated_batch[key] = values
        elif isinstance(values[0], (str, int, float, type(None))):
            # 对于基本类型，保持为列表
            collated_batch[key] = values
        else:
            # 对于其他类型，尝试使用默认合并方式
            try:
                import torch
                if all(isinstance(v, torch.Tensor) for v in values):
                    collated_batch[key] = torch.stack(values)
                else:
                    collated_batch[key] = values
            except:
                collated_batch[key] = values
    
    return collated_batch

def create_scienceqa_dataloader(data_root=None, batch_size=32, num_workers=4, num_samples=None, split='test',
                               prompt_format="QCM-A", use_caption=False, options=["A", "B", "C", "D", "E"], shot_number=3, is_test=False):
    """
    创建ScienceQA数据加载器的工厂函数
    
    Args:
        data_root: 数据集根目录
        batch_size: 批次大小
        num_workers: 工作线程数
        num_samples: 限制样本数量
        split: 数据分割，可选'train', 'val', 'test'
        prompt_format: 提示格式
        use_caption: 是否使用图像标题
        options: 选项列表
        shot_number: few-shot数量
        is_test: 是否为测试模式（测试模式下不包含示例）
        
    Returns:
        DataLoader实例
    """
    # 创建数据集实例
    dataset = ScienceQADataLoader(
        data_root=data_root,
        split=split,
        image_transform=None,  # 可以根据需要添加图像变换
        prompt_format=prompt_format,
        use_caption=use_caption,
        options=options,
        shot_number=shot_number,
        is_test=is_test
    )
    
    # 如果指定了样本数量，截取数据集
    if num_samples is not None and num_samples < len(dataset):
        # 创建一个子数据集
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(num_samples)))
    
    # 创建DataLoader，使用自定义的collate函数
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # 评估时通常不需要打乱
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=scienceqa_collate_fn  # 使用自定义的collate函数
    )
    
    return dataloader

if __name__ == "__main__":
    # 测试数据加载器
    print("=== ScienceQA数据加载器测试 ===")
    
    # 数据集根目录 - 使用本地路径
    data_root = "/data/model/Inference_VLM/VLM_Infra/datasets/ScienceQA/ScienceQA"
    
    # 创建数据加载器
    dataloader = create_scienceqa_dataloader(
        data_root=data_root,
        split="test",
        batch_size=2,
        num_workers=0,
        prompt_format="QCM-A",
        use_caption=False,
        options=["A", "B", "C", "D", "E"],
        shot_number=3
    )
    
    # 显示数据集信息
    dataset = dataloader.dataset
    info = dataset.get_dataset_info()
    print(f"\n数据集信息:")
    print(f"总样本数: {info['total_samples']}")
    print(f"学科分布: {info['subject_distribution']}")
    
    # 显示单个样本
    print("\n显示第一个样本:")
    dataset.show_sample(0)
    
    # 迭代一个批次
    print("\n迭代第一个批次:")
    for i, batch in enumerate(dataloader):
        print(f"\n批次 {i+1}:")
        print(f"问题ID: {batch['question_id']}")
        print(f"问题: {batch['question']}")
        print(f"选项: {batch['choices']}")
        print(f"答案索引: {batch['answer']}")
        print(f"图像路径: {batch['image_path']}")
        print(f"学科: {batch['subject']}")
        print(f"Prompt Input: {batch['prompt_input'][0][:1000]}...")  # 显示前100个字符
        
        # 只显示第一个批次
        break
    
    print("\n数据加载器测试完成!")