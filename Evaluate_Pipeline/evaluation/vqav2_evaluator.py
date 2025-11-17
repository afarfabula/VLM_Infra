#!/usr/bin/env python3
"""
VQAv2评估器
支持分布式评估结果合并
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class VQAv2Evaluator:
    """VQAv2评估器"""
    
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # 分布式配置
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # 评估结果
        self.predictions = {}
        self.metrics = {}
    
    def add_prediction(self, question_id: int, prediction: str, ground_truth: Optional[str] = None):
        """添加预测结果"""
        self.predictions[question_id] = {
            'prediction': prediction,
            'ground_truth': ground_truth,
            'rank': self.rank
        }
    
    def calculate_accuracy(self) -> float:
        """计算准确率"""
        if not self.predictions:
            return 0.0
        
        correct = 0
        total = 0
        
        for qid, pred_data in self.predictions.items():
            if pred_data['ground_truth'] is not None:
                total += 1
                # 简单的字符串匹配（实际VQAv2使用更复杂的评估）
                if pred_data['prediction'].lower().strip() == pred_data['ground_truth'].lower().strip():
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        self.metrics['accuracy'] = accuracy
        self.metrics['total_samples'] = total
        self.metrics['correct_samples'] = correct
        
        return accuracy
    
    def save_predictions(self, filename: str = "vqav2_predictions.json"):
        """保存预测结果"""
        output_file = self.result_dir / f"rank_{self.rank}_{filename}"
        
        with open(output_file, 'w') as f:
            json.dump({
                'predictions': self.predictions,
                'metrics': self.metrics,
                'rank': self.rank,
                'world_size': self.world_size
            }, f, indent=2)
        
        print(f"进程 {self.rank} 预测结果已保存: {output_file}")
    
    def merge_distributed_results(self) -> Dict:
        """合并分布式评估结果"""
        if self.rank != 0:
            # 只有rank 0进程执行合并
            return {}
        
        # 收集所有进程的结果
        all_predictions = {}
        all_metrics = defaultdict(list)
        
        # 读取所有进程的预测文件
        for rank in range(self.world_size):
            result_file = self.result_dir / f"rank_{rank}_vqav2_predictions.json"
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # 合并预测
                all_predictions.update(data['predictions'])
                
                # 合并指标
                for key, value in data['metrics'].items():
                    all_metrics[key].append(value)
            else:
                print(f"警告: 进程 {rank} 的结果文件不存在")
        
        # 计算总体指标
        final_metrics = {}
        
        if 'accuracy' in all_metrics:
            final_metrics['overall_accuracy'] = sum(all_metrics['accuracy']) / len(all_metrics['accuracy'])
        
        if 'total_samples' in all_metrics:
            final_metrics['total_samples'] = sum(all_metrics['total_samples'])
        
        if 'correct_samples' in all_metrics:
            final_metrics['correct_samples'] = sum(all_metrics['correct_samples'])
        
        # 保存合并结果
        merged_file = self.result_dir / "merged_vqav2_results.json"
        with open(merged_file, 'w') as f:
            json.dump({
                'predictions': all_predictions,
                'metrics': final_metrics,
                'total_ranks': self.world_size
            }, f, indent=2)
        
        # 生成Excel报告
        self._generate_excel_report(all_predictions, final_metrics)
        
        print(f"分布式评估结果合并完成: {merged_file}")
        return final_metrics
    
    def _generate_excel_report(self, predictions: Dict, metrics: Dict):
        """生成Excel格式的报告"""
        # 创建DataFrame
        rows = []
        
        for qid, pred_data in predictions.items():
            rows.append({
                'question_id': qid,
                'prediction': pred_data['prediction'],
                'ground_truth': pred_data['ground_truth'],
                'rank': pred_data['rank'],
                'is_correct': pred_data['prediction'].lower().strip() == pred_data['ground_truth'].lower().strip() 
                    if pred_data['ground_truth'] else None
            })
        
        df = pd.DataFrame(rows)
        
        # 保存Excel文件
        excel_file = self.result_dir / "vqav2_evaluation_report.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 预测结果表
            df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # 指标汇总表
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # 统计信息表
            stats = {
                'total_questions': len(predictions),
                'world_size': self.world_size,
                'accuracy': metrics.get('overall_accuracy', 0)
            }
            stats_df = pd.DataFrame([stats])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        print(f"Excel报告已生成: {excel_file}")
    
    def get_evaluation_summary(self) -> Dict:
        """获取评估摘要"""
        return {
            'rank': self.rank,
            'local_predictions': len(self.predictions),
            'metrics': self.metrics,
            'result_dir': str(self.result_dir)
        }


def create_vqav2_evaluator(result_dir: str) -> VQAv2Evaluator:
    """创建VQAv2评估器"""
    return VQAv2Evaluator(result_dir)


if __name__ == "__main__":
    # 测试评估器
    result_dir = "/tmp/vqav2_test"
    
    try:
        evaluator = VQAv2Evaluator(result_dir)
        
        # 添加测试预测
        evaluator.add_prediction(1, "cat", "cat")
        evaluator.add_prediction(2, "dog", "cat")  # 错误预测
        
        # 计算准确率
        accuracy = evaluator.calculate_accuracy()
        print(f"准确率: {accuracy:.4f}")
        
        # 保存结果
        evaluator.save_predictions()
        
        # 获取摘要
        summary = evaluator.get_evaluation_summary()
        print(f"评估摘要: {summary}")
        
    except Exception as e:
        print(f"评估器测试失败: {e}")