#!/usr/bin/env python3
"""
ScienceQA评估器模块
"""

import json
import os
import sys
import traceback

class ScienceQAEvaluator:
    """ScienceQA评估器类"""
    
    def __init__(self, output_dir=None):
        """初始化评估器
        
        Args:
            output_dir: 输出目录，用于保存评估结果
        """
        self.output_dir = output_dir or './output'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def evaluate(self, results):
        """评估模型在ScienceQA上的表现
        
        Args:
            results: 推理结果列表，每项包含question_id, model_prediction, ground_truth等
            
        Returns:
            dict: 评估指标
        """
        print(f"开始评估ScienceQA结果，共{len(results)}个样本")
        
        # 初始化统计变量
        total_samples = 0
        correct_samples = 0
        error_count = 0
        skipped_samples = 0
        error_logs = []
        
        # 定义选项映射
        option_map = {
            '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F',
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F'
        }
        
        # 处理每个结果
        for i, result in enumerate(results):
            try:
                # 打印调试信息
                print(f"\n处理样本 {i+1}/{len(results)}")
                print(f"结果类型: {type(result)}")
                print(f"结果键: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                
                # 提取必要字段
                question_id = str(result.get('question_id', f'unknown_{i}'))
                
                # 获取模型预测和真实答案
                model_prediction = self._get_field(result, 'model_prediction')
                ground_truth = self._get_field(result, 'ground_truth')
                
                print(f"问题ID: {question_id}")
                print(f"模型预测: {model_prediction}")
                print(f"真实答案: {ground_truth}")
                
                # 标准化预测和答案格式
                pred_option = self._normalize_option(model_prediction, option_map)
                gt_option = self._normalize_option(ground_truth, option_map)
                
                print(f"标准化后预测: {pred_option}")
                print(f"标准化后答案: {gt_option}")
                
                # 检查预测是否正确
                is_correct = self._check_correctness(pred_option, gt_option)
                
                if is_correct:
                    correct_samples += 1
                    print(f"✅ 样本 {question_id} 预测正确")
                else:
                    print(f"❌ 样本 {question_id} 预测错误")
                
                total_samples += 1
                
            except Exception as e:
                error_count += 1
                error_info = {
                    'sample_index': i,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'sample_data': str(result)[:500]  # 限制错误日志大小
                }
                error_logs.append(error_info)
                
                print(f"❌ 处理样本时出错: {str(e)}")
                print(f"错误堆栈: {traceback.format_exc()}")
                
                # 跳过错误样本
                skipped_samples += 1
                continue
        
        # 计算准确率
        accuracy = correct_samples / total_samples if total_samples > 0 else 0.0
        
        # 生成评估报告
        evaluation_report = {
            'total_samples': total_samples,
            'correct_samples': correct_samples,
            'accuracy': accuracy,
            'error_count': error_count,
            'skipped_samples': skipped_samples,
            'error_logs': error_logs
        }
        
        # 保存评估报告
        self._save_report(evaluation_report)
        
        # 打印评估结果
        print(f"\n===== ScienceQA评估结果 =====")
        print(f"总样本数: {total_samples}")
        print(f"正确样本数: {correct_samples}")
        print(f"准确率: {accuracy:.4f}")
        print(f"错误样本数: {error_count}")
        print(f"跳过样本数: {skipped_samples}")
        
        return evaluation_report
    
    def _get_field(self, result, field_name, default=''):
        """安全获取字段值
        
        Args:
            result: 结果字典或对象
            field_name: 字段名
            default: 默认值
            
        Returns:
            获取到的值或默认值
        """
        try:
            if isinstance(result, dict):
                return str(result.get(field_name, default))
            elif hasattr(result, field_name):
                return str(getattr(result, field_name))
            else:
                return str(default)
        except Exception as e:
            print(f"获取字段 {field_name} 时出错: {e}")
            return str(default)
    
    def _normalize_option(self, option_str, option_map):
        """标准化选项格式
        
        Args:
            option_str: 选项字符串
            option_map: 选项映射字典
            
        Returns:
            标准化后的选项
        """
        if not option_str:
            return ''
        
        # 转换为字符串并去除空格
        option_str = str(option_str).strip().upper()
        
        # 尝试直接映射
        if option_str in option_map:
            return option_map[option_str]
        
        # 尝试从字符串中提取选项
        for char in option_str:
            if char in option_map:
                return option_map[char]
        
        # 如果是数字，直接映射
        if option_str.isdigit():
            return option_map.get(option_str, '')
        
        return option_str  # 返回原始字符串作为回退
    
    def _check_correctness(self, prediction, ground_truth):
        """检查预测是否正确
        
        Args:
            prediction: 模型预测
            ground_truth: 真实答案
            
        Returns:
            bool: 是否正确
        """
        try:
            # 转换为字符串进行比较
            pred_str = str(prediction).strip().upper()
            gt_str = str(ground_truth).strip().upper()
            
            print(f"检查正确性 - 预测: '{pred_str}', 答案: '{gt_str}'")
            
            # 直接字符串匹配
            if pred_str == gt_str:
                return True
            
            # 检查预测是否包含正确答案
            if gt_str and pred_str and gt_str in pred_str:
                return True
            
            # 数字索引匹配
            try:
                pred_idx = int(pred_str) if pred_str.isdigit() else -1
                gt_idx = int(gt_str) if gt_str.isdigit() else -1
                if pred_idx >= 0 and gt_idx >= 0:
                    return pred_idx == gt_idx
            except Exception:
                pass
            
            return False
            
        except Exception as e:
            print(f"检查正确性时出错: {e}")
            print(f"预测类型: {type(prediction)}, 答案类型: {type(ground_truth)}")
            # 发生错误时返回False，避免整个评估失败
            return False
    
    def _save_report(self, report):
        """保存评估报告到文件
        
        Args:
            report: 评估报告字典
        """
        try:
            report_path = os.path.join(self.output_dir, 'scienceqa_evaluation_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"评估报告已保存到: {report_path}")
        except Exception as e:
            print(f"保存评估报告时出错: {e}")

def create_scienceqa_evaluator(output_dir=None):
    """创建ScienceQA评估器
    
    Args:
        output_dir: 输出目录
        
    Returns:
        ScienceQAEvaluator: 评估器实例
    """
    return ScienceQAEvaluator(output_dir=output_dir)