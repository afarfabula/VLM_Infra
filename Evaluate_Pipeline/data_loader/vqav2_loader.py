#!/usr/bin/env python3
"""
VQAv2数据集加载器
support distributed data loading
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image


class VQAv2DataLoader:
    """VQAv2数据集加载器"""
    
    def __init__(self, data_root: str, split: str = "val", num_samples: int = 100):
        self.data_root = Path(data_root)
        self.split = split
        self.num_samples = num_samples
        
        # 根据实际数据集结构设置图像目录
        if split == "val":
            self.images_dir = self.data_root / "val2014" / "val2014"
            self.questions_file = self.data_root / "annotations" / "v2_OpenEnded_mscoco_val2014_questions.json"
            self.annotations_file = self.data_root / "annotations" / "v2_mscoco_val2014_annotations.json"
        elif split == "train":
            self.images_dir = self.data_root / "train2014" / "train2014"
            self.questions_file = self.data_root / "annotations" / "v2_OpenEnded_mscoco_train2014_questions.json"
            self.annotations_file = self.data_root / "annotations" / "v2_mscoco_train2014_annotations.json"
        else:
            self.images_dir = self.data_root / "test2015"
            self.questions_file = None
            self.annotations_file = None
        
        # 分布式配置
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # 加载真实的问题和标注数据
        self.questions = self._load_real_questions()
        self.annotations = self._load_annotations()
        self.question_to_annotations = self._build_question_to_annotations_map()
        
        # 分布式数据划分
        self.indices = self._distribute_indices()
        
        print(f"VQAv2数据加载器初始化完成: rank={self.rank}, world_size={self.world_size}, "
              f"样本数={len(self.indices)}")
    
    def _load_real_questions(self) -> List[Dict]:
        """加载真实的问题数据"""
        if not self.questions_file or not self.questions_file.exists():
            raise FileNotFoundError(f"问题文件不存在: {self.questions_file}")
        
        print(f"加载问题文件: {self.questions_file}")
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取问题列表
        questions = data.get('questions', [])
        
        # 限制样本数量
        if self.num_samples > 0:
            questions = questions[:self.num_samples]
        
        print(f"加载了 {len(questions)} 个问题 (限制为 {self.num_samples} 个样本)")
        return questions
    
    def _load_annotations(self) -> List[Dict]:
        """加载真实的标注数据"""
        if not self.annotations_file or not self.annotations_file.exists():
            raise FileNotFoundError(f"标注文件不存在: {self.annotations_file}")
        
        print(f"加载标注文件: {self.annotations_file}")
        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = data.get('annotations', [])
        print(f"加载了 {len(annotations)} 个标注")
        return annotations
    
    def _build_question_to_annotations_map(self) -> Dict[int, Dict]:
        """构建question_id到标注的映射"""
        qid_to_anno = {}
        for anno in self.annotations:
            qid_to_anno[anno['question_id']] = anno
        return qid_to_anno
    
    def _distribute_indices(self) -> List[int]:
        """分布式数据划分"""
        total_samples = len(self.questions)
        
        # 计算每个进程的样本范围
        samples_per_rank = total_samples // self.world_size
        start_idx = self.rank * samples_per_rank
        end_idx = start_idx + samples_per_rank
        
        # 最后一个进程处理剩余样本
        if self.rank == self.world_size - 1:
            end_idx = total_samples
        
        indices = list(range(start_idx, end_idx))
        
        print(f"进程 {self.rank}: 处理样本 {start_idx}-{end_idx-1} (共{len(indices)}个样本)")
        return indices
    
    def __len__(self) -> int:
        """返回当前进程的样本数量"""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        # 转换为全局索引
        global_idx = self.indices[idx]
        question_data = self.questions[global_idx]
        
        # 获取对应的标注数据
        question_id = question_data['question_id']
        annotation = self.question_to_annotations.get(question_id, {})
        
        # 提取ground truth（使用multiple_choice_answer作为主要答案）
        ground_truth = annotation.get('multiple_choice_answer', '')
        
        # 构建样本
        sample = {
            'question_id': question_id,
            'image_id': question_data['image_id'],
            'question': question_data['question'],
            'ground_truth': ground_truth,
            'image_path': self._get_image_path(question_data['image_id']),
            'global_index': global_idx,
            'local_index': idx
        }
        
        return sample
    
    def _get_image_path(self, image_id: int) -> Path:
        """获取图像文件路径"""
        # 根据实际图像文件名格式查找图像文件
        # 实际图像文件名格式: COCO_val2014_000000000001.jpg
        
        # 尝试标准格式
        image_filename = f"COCO_{self.split}2014_{image_id:012d}.jpg"
        image_path = self.images_dir / image_filename
        
        if image_path.exists():
            return image_path
        
        # 尝试其他可能的命名格式
        for pattern in [f"{image_id:012d}.jpg", f"{image_id}.jpg", f"COCO_{self.split}2014_{image_id}.jpg"]:
            alt_path = self.images_dir / pattern
            if alt_path.exists():
                return alt_path
        
        # 如果找不到特定图像，尝试使用第一个可用的图像文件
        image_files = list(self.images_dir.glob("*.jpg"))
        if image_files:
            return image_files[0]
        
        raise FileNotFoundError(f"在 {self.images_dir} 中找不到图像文件")
    
    def load_image(self, image_path: Path) -> Image.Image:
        """加载图像"""
        return Image.open(image_path).convert('RGB')
    
    def get_batch(self, batch_indices: List[int]) -> List[Dict]:
        """批量获取样本"""
        return [self[i] for i in batch_indices]
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        return {
            'total_samples': len(self.questions),
            'local_samples': len(self.indices),
            'rank': self.rank,
            'world_size': self.world_size,
            'split': self.split
        }


def create_vqav2_dataloader(data_root: str, batch_size: int = 8, num_workers: int = 4, num_samples: int = 100):
    """创建VQAv2数据加载器"""
    
    # 创建数据集
    dataset = VQAv2DataLoader(data_root, num_samples=num_samples)
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 评估时不需要shuffle
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: x  # 自定义collate函数
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试数据加载器
    data_root = "/data/model/Inference_VLM/VLM_Infra/datasets/VQAv2"
    
    try:
        loader = VQAv2DataLoader(data_root)
        print(f"数据加载器创建成功: {loader.get_statistics()}")
        
        # 测试第一个样本
        sample = loader[0]
        print(f"样本信息: {sample}")
        
    except Exception as e:
        print(f"数据加载器测试失败: {e}")
        print("请确保VQAv2数据集已正确下载并放置在指定路径")