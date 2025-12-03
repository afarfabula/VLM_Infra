"""
ScienceQA评估器
用于评估模型在ScienceQA数据集上的性能
"""

import os
import json
import sys
from pathlib import Path
import numpy as np

# 添加ScienceQA工具目录到系统路径
sys.path.append('/data/model/Inference_VLM/VLM_Infra/datasets/ScienceQA/ScienceQA/tools')
try:
    from evaluate_acc import get_scores, print_scores
except ImportError:
    print("警告: 无法导入ScienceQA官方评估模块，将使用默认评估方法")
    get_scores = None
    print_scores = None


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
        
        # 如果可以使用官方评估方法，则优先使用
        if get_scores is not None:
            return self._evaluate_with_official_method(results)
        
        # 否则使用原有的评估方法
        return self._evaluate_with_default_method(results)
    
    def _evaluate_with_official_method(self, results):
        """
        使用ScienceQA官方评估方法进行评估
        
        Args:
            results: 模型预测结果列表
            
        Returns:
            dict: 评估指标字典
        """
        print("使用ScienceQA官方评估方法进行评估...")
        
        # 创建临时结果文件供官方评估脚本使用
        temp_result_file = Path(self.result_dir) / 'temp_scienceqa_results.json'
        
        # 转换结果格式以适配官方评估脚本
        official_results = {}
        official_results["results"] = {}
        
        # 详细结果记录
        detailed_results = []
        
        for result in results:
            question_id = result.get('question_id', '')
            prediction = result.get('model_prediction', '').strip()
            ground_truth = result.get('ground_truth', '')
            
            # 确保ground_truth是字符串类型
            if isinstance(ground_truth, str):
                ground_truth = ground_truth.strip()
            else:
                ground_truth = str(ground_truth).strip() if ground_truth is not None else ""
                
            has_image = 'image_path' in result and result['image_path']
            
            # 提取预测的答案（数字格式）
            pred_answer = self._extract_answer_as_number(prediction)
            
            # 记录到官方格式
            official_results["results"][str(question_id)] = int(pred_answer)
            
            # 记录详细结果
            detailed_results.append({
                'question_id': question_id,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'is_correct': str(pred_answer) == str(ground_truth),
                'has_image': has_image
            })
        
        # 保存临时结果文件
        with open(temp_result_file, 'w', encoding='utf-8') as f:
            json.dump(official_results, f, indent=2, ensure_ascii=False)
        
        # 使用官方评估方法计算指标
        try:
            # 调用官方评估函数
            scores = get_scores(
                str(temp_result_file), 
                "/data/model/Inference_VLM/VLM_Infra/datasets/ScienceQA/ScienceQA/data/scienceqa/problems.json"
            )
            
            # 打印分数
            print_scores(scores)
            
            # 转换为我们的指标格式
            metrics = {
                'overall_accuracy': float(scores.get('acc_average', 0)) / 100.0 if scores.get('acc_average') != -1 else 0.0,
                'image_accuracy': 0.0,  # 官方脚本不直接提供此指标
                'no_image_accuracy': 0.0,  # 官方脚本不直接提供此指标
                'total_samples': len(results),
                'correct_samples': 0,  # 需要重新计算
                'image_samples': 0,
                'no_image_samples': 0,
                'correct_image_samples': 0,
                'correct_no_image_samples': 0
            }
            
            # 重新计算正确样本数
            correct_count = 0
            image_samples = 0
            no_image_samples = 0
            correct_image_samples = 0
            correct_no_image_samples = 0
            
            for result in detailed_results:
                is_correct = result['is_correct']
                has_image = result['has_image']
                
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
            
            # 更新指标
            metrics.update({
                'correct_samples': correct_count,
                'image_samples': image_samples,
                'no_image_samples': no_image_samples,
                'correct_image_samples': correct_image_samples,
                'correct_no_image_samples': correct_no_image_samples,
                'official_scores': scores  # 保留原始官方分数
            })
            
            # 保存详细结果
            detailed_file = Path(self.result_dir) / 'scienceqa_detailed_evaluation.json'
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metrics': metrics,
                    'detailed_results': detailed_results
                }, f, indent=2, ensure_ascii=False)
            
            print(f"评估完成，详细结果已保存到: {detailed_file}")
            return metrics
            
        except Exception as e:
            print(f"使用官方评估方法时出错: {e}")
            # 回退到原有方法
            return self._evaluate_with_default_method(results)
    
    def _extract_answer_as_number(self, prediction):
        """
        从预测文本中提取答案并转换为数字格式
        
        Args:
            prediction: 模型预测文本
            
        Returns:
            int: 数字格式的答案
        """
        # 尝试直接转换为整数
        if prediction.isdigit():
            return int(prediction)
        
        # 匹配 "The answer is X" 格式
        import re
        match = re.search(r"[Tt]he answer is ([A-Z])", prediction)
        if match:
            letter = match.group(1)
            # 将字母转换为数字 (A=0, B=1, C=2, D=3, E=4)
            return ord(letter.upper()) - ord('A')
        
        # 匹配单独的字母
        match = re.match(r"^([A-Z])$", prediction.strip(), re.IGNORECASE)
        if match:
            letter = match.group(1)
            # 将字母转换为数字 (A=0, B=1, C=2, D=3, E=4)
            return ord(letter.upper()) - ord('A')
        
        # 默认返回0
        return 0
    
    def _evaluate_with_default_method(self, results):
        """
        使用默认评估方法进行评估
        
        Args:
            results: 模型预测结果列表
            
        Returns:
            dict: 评估指标字典
        """
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
        
        # 提取预测中的答案
        def extract_answer(text):
            # 尝试直接转换为整数
            if text.isdigit():
                return int(text)
            
            # 匹配 "The answer is X" 格式
            import re
            match = re.search(r"[Tt]he answer is ([A-Za-z])", text)
            if match:
                letter = match.group(1)
                # 将字母转换为数字 (A=0, B=1, C=2, D=3, E=4)
                return ord(letter.upper()) - ord('A')
            
            # 匹配单独的字母
            match = re.match(r"^([A-Za-z])$", text.strip(), re.IGNORECASE)
            if match:
                letter = match.group(1)
                # 将字母转换为数字 (A=0, B=1, C=2, D=3, E=4)
                return ord(letter.upper()) - ord('A')
            
            # 默认返回原始字符串
            return text
        
        # 提取预测和真实答案
        pred_answer = extract_answer(prediction_str)
        gt_answer = extract_answer(ground_truth_str)
        
        # 如果都能转换为数字，则比较数字
        if isinstance(pred_answer, int) and isinstance(gt_answer, int):
            return pred_answer == gt_answer
        
        # 否则进行字符串比较
        return str(pred_answer).lower() == str(gt_answer).lower()
    
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
    import sys
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python scienceqa_evaluator.py <predictions_file> [result_dir]")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    result_dir = sys.argv[2] if len(sys.argv) > 2 else './results'
    
    # 创建评估器
    evaluator = ScienceQAEvaluator(result_dir)
    
    # 读取预测结果
    results = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    
    # 运行评估
    metrics = evaluator.evaluate(results)
    print(f"评估指标: {metrics}")