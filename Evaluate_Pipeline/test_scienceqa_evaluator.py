#!/usr/bin/env python3
"""
测试ScienceQA评估器
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.scienceqa_evaluator import ScienceQAEvaluator

def test_scienceqa_evaluator():
    """测试ScienceQA评估器"""
    print("测试ScienceQA评估器...")
    
    # 创建评估器实例
    evaluator = ScienceQAEvaluator('./test_results')
    
    # 示例数据
    example_results = [
        {
            'question_id': '1',
            'model_prediction': 'The answer is A',
            'ground_truth': '0',  # A对应0
            'image_path': '/path/to/image.jpg'
        },
        {
            'question_id': '2',
            'model_prediction': 'The answer is 1',
            'ground_truth': '1',
            'image_path': ''
        },
        {
            'question_id': '3',
            'model_prediction': 'B',
            'ground_truth': '1',  # B对应1
            'image_path': '/path/to/image2.jpg'
        }
    ]
    
    # 进行评估
    metrics = evaluator.evaluate(example_results)
    
    print("评估结果:")
    print(f"总体准确率: {metrics['overall_accuracy']:.4f}")
    print(f"总样本数: {metrics['total_samples']}")
    print(f"正确样本数: {metrics['correct_samples']}")
    
    if 'official_scores' in metrics:
        print("官方评估分数:")
        for key, value in metrics['official_scores'].items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    test_scienceqa_evaluator()