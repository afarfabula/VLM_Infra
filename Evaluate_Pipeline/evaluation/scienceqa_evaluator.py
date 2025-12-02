"""
ScienceQA评估器
用于评估模型在ScienceQA数据集上的性能
"""

import os
import json
from pathlib import Path
import numpy as np


class ScienceQAEvaluator:
    """
    ScienceQA评估器类
    计算准确率、F1分数等评估指标
    """
    
    def __init__(self, result_dir='./results'):
        """
        初始化评估器
        
        Args:
            result_dir: 结果保存目录
        """
        self.result_dir = result_dir
        # 创建结果目录（如果不存在）
        os.makedirs(result_dir, exist_ok=True)
        
    def evaluate(self, results):
        """
        评估模型预测结果
        
        Args:
            results: 模型预测结果列表，每个元素包含question_id、model_prediction、ground_truth等
            
        Returns:
            dict: 评估指标字典
        """
        print("开始评估ScienceQA结果...")
        
        # 统计变量
        total_samples = len(results)
        correct_count = 0
        image_samples = 0
        no_image_samples = 0
        correct_image_samples = 0
        correct_no_image_samples = 0
        
        # 详细结果记录
        detailed_results = []
        
        for result in results:
            question_id = result.get('question_id', '')
            prediction = result.get('model_prediction', '').strip()
            ground_truth = result.get('ground_truth', '')
            
            # 确保ground_truth是字符串类型后再调用.strip()
            if isinstance(ground_truth, str):
                ground_truth = ground_truth.strip()
            else:
                ground_truth = str(ground_truth).strip() if ground_truth is not None else ""
                
            has_image = 'image_path' in result and result['image_path']
            
            # 判断是否正确（简单的文本匹配）
            is_correct = self._check_correctness(prediction, ground_truth)
            
            # 更新统计
            if is_correct:
                correct_count += 1
                if has_image:
                    correct_image_samples += 1
                else:
                    correct_no_image_samples += 1
            
            if has_image:
                image_samples += 1
            else:
                no_image_samples += 1
            
            # 记录详细结果
            detailed_results.append({
                'question_id': question_id,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'is_correct': is_correct,
                'has_image': has_image
            })
        
        # 计算准确率
        overall_accuracy = correct_count / total_samples if total_samples > 0 else 0.0
        image_accuracy = correct_image_samples / image_samples if image_samples > 0 else 0.0
        no_image_accuracy = correct_no_image_samples / no_image_samples if no_image_samples > 0 else 0.0
        
        # 构建评估指标
        metrics = {
            'overall_accuracy': overall_accuracy,
            'image_accuracy': image_accuracy,
            'no_image_accuracy': no_image_accuracy,
            'total_samples': total_samples,
            'correct_samples': correct_count,
            'image_samples': image_samples,
            'no_image_samples': no_image_samples,
            'correct_image_samples': correct_image_samples,
            'correct_no_image_samples': correct_no_image_samples
        }
        
        # 保存详细结果
        detailed_file = Path(self.result_dir) / 'scienceqa_detailed_evaluation.json'
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics,
                'detailed_results': detailed_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"评估完成，详细结果已保存到: {detailed_file}")
        return metrics
    
    def _check_correctness(self, prediction, ground_truth):
        """
        检查预测是否正确
        
        Args:
            prediction: 模型预测结果
            ground_truth: 真实答案
            
        Returns:
            bool: 是否正确
        """
        # 确保prediction和ground_truth都是字符串类型
        prediction_str = str(prediction) if prediction is not None else ""
        ground_truth_str = str(ground_truth) if ground_truth is not None else ""
        
        # 转换为小写进行比较
        prediction_lower = prediction_str.lower()
        ground_truth_lower = ground_truth_str.lower()
        
        # 简单的字符串匹配
        # 如果真实答案包含在预测中，或者预测包含在真实答案中
        if ground_truth_lower in prediction_lower or prediction_lower in ground_truth_lower:
            return True
        
        # 处理选择题的情况（答案可能是A/B/C/D或数字索引）
        if len(ground_truth_str) == 1 and ground_truth_str.isalpha():
            # 检查是否包含字母答案
            return ground_truth_lower in prediction_lower
        
        if ground_truth_str.isdigit():
            # 检查是否包含数字答案
            return ground_truth_str in prediction_str
        
        # 更复杂的匹配逻辑可以在这里添加
        return False
    
    def save_metrics(self, metrics, filename='scienceqa_metrics.json'):
        """
        保存评估指标到文件
        
        Args:
            metrics: 评估指标字典
            filename: 文件名
        """
        filepath = Path(self.result_dir) / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return filepath


def create_scienceqa_evaluator(result_dir='./results'):
    """
    创建ScienceQA评估器的工厂函数
    
    Args:
        result_dir: 结果保存目录
        
    Returns:
        ScienceQAEvaluator实例
    """
    return ScienceQAEvaluator(result_dir=result_dir)


if __name__ == "__main__":
    # 示例使用
    evaluator = ScienceQAEvaluator('./example_results')
    
    # 示例数据
    example_results = [
        {
            'question_id': '1',
            'model_prediction': 'The answer is A',
            'ground_truth': 'A',
            'image_path': '/path/to/image.jpg'
        },
        {
            'question_id': '2',
            'model_prediction': 'The answer is 3',
            'ground_truth': '3',
            'image_path': ''
        }
    ]
    
    # 运行评估
    metrics = evaluator.evaluate(example_results)
    print(f"评估指标: {metrics}")
