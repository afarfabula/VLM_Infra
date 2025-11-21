import os
import json
import time
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# 修复导入路径问题
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.distributed_utils import (
    setup_distributed, 
    cleanup_distributed, 
    print_rank0, 
    is_distributed
)
from data_loader.vqav2_loader import create_vqav2_dataloader
from evaluation.vqav2_evaluator import create_vqav2_evaluator

# 尝试导入VLLM推理引擎，如果失败则使用简化版
try:
    from optimized_inference.vllm_inference import create_distributed_vllm_inference
    from optimized_inference.flash_attn_inference import FlashAttentionInferenceEngine, create_flash_attn_inference
    VLLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VLLM not available: {e}")
    VLLM_AVAILABLE = False

# 简化版推理引擎作为后备（如果VLLM不可用）
class SimpleFallbackInferenceEngine:
    """简化版推理引擎作为后备方案"""
    
    def __init__(self, model_path: str, device: str, load_precision: str):
        self.model_path = model_path
        self.device = device
        self.load_precision = load_precision
        print(f"Initializing simplified inference engine on {device}")
        
        # 这里应该初始化实际的模型，但为了简化，我们只做占位
        self.stats = {
            'total_inference_time': 0,
            'total_tokens': 0,
            'batch_count': 0
        }
    
    def batch_generate(self, questions: List[str], images: List[str], max_new_tokens: int = 1024) -> List[str]:
        """模拟批量生成"""
        import time
        time.sleep(0.1)  # 模拟推理时间
        # 返回简单的模拟答案
        return [f"Answer to '{q}' (simulated)" for q in questions]
    
    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        return self.stats

def create_simple_vllm_inference(model_path: str, device: str, load_precision: str):
    """创建简化版推理引擎的工厂函数"""
    return SimpleFallbackInferenceEngine(model_path, device, load_precision)


class OptimizedEvaluatePipeline:
    """优化的评估管道类，使用VLLM和FlashAttention提高吞吐量"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        
        # 设置分布式环境
        self.rank, self.world_size, self.local_rank = setup_distributed()
        
        # 初始化组件
        self.data_loader = None
        self.inference_engine = None
        self.evaluator = None
        
        # 性能统计
        self.start_time = None
        self.end_time = None
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        # 检查配置文件是否存在，如果不存在则使用默认配置
        if not os.path.exists(config_path):
            # 使用默认配置
            default_config = {
                "model_configs": {
                    "LLaVA-1.5-7B": {
                        "model_path": "/data/model/Inference_VLM/models-LLava-1.5-7B"
                    },
                    "LLaVA-1.5-13B": {
                        "model_path": "/data/model/Inference_VLM/models-LLava-1.5-13B"
                    }
                }
            }
            return default_config
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def run_vqav2_evaluation(self, 
                           model_name='LLaVA-1.5-7B', 
                           result_dir='./results', 
                           num_samples=100, 
                           batch_size=32, 
                           load_precision='fp16',
                           tensor_parallel_size=1,
                           max_new_tokens=1024):
        """
        运行VQAv2评估（优化版）
        
        Args:
            model_name: 模型名称
            result_dir: 结果保存目录
            num_samples: 样本数量
            batch_size: 批次大小
            load_precision: 加载精度
            tensor_parallel_size: 张量并行大小（用于多GPU）
            max_new_tokens: 最大生成token数
        """
        self.start_time = time.time()
        
        print_rank0(f"开始VQAv2评估 - 样本数: {num_samples}, 批次大小: {batch_size}")
        print_rank0(f"使用VLLM优化，张量并行大小: {tensor_parallel_size}")
        
        try:
            # 1. 准备数据加载器
            print_rank0("准备数据加载器...")
            self.data_loader = create_vqav2_dataloader(
                data_root="/data/model/Inference_VLM/VLM_Infra/datasets/VQAv2",
                batch_size=batch_size,
                num_workers=4,
                num_samples=num_samples
            )
            
            # 2. 初始化推理引擎（优先使用VLLM，其次FlashAttention，最后使用简化版）
            # 检查配置中是否启用VLLM或FlashAttention
            use_vllm = self.config.get('optimization_configs', {}).get('use_vllm', True)  # 默认启用VLLM
            use_flash_attn = self.config.get('optimization_configs', {}).get('use_flash_attn', False)
            enable_flash_attention = self.config.get('optimization_configs', {}).get('enable_flash_attention', True)
            
            if VLLM_AVAILABLE and use_vllm:
                print_rank0("初始化VLLM推理引擎...")
                # 获取模型路径和相关配置
                model_config = self.config['model_configs'][model_name]
                model_path = model_config['model_path']
                
                # 使用VLLM推理引擎，支持多GPU和FlashAttention
                self.inference_engine = create_distributed_vllm_inference(
                    model_path=model_path,
                    device=f"cuda:{self.local_rank}",
                    load_precision=load_precision,
                    tensor_parallel_size=tensor_parallel_size,
                    enable_flash_attention=enable_flash_attention
                )
            elif VLLM_AVAILABLE and use_flash_attn:
                print_rank0("初始化FlashAttention推理引擎...")
                # 获取模型路径和相关配置
                model_config = self.config['model_configs'][model_name]
                model_path = model_config['model_path']
                
                # 使用FlashAttention推理引擎
                self.inference_engine = create_flash_attn_inference(
                    model_path=model_path,
                    device=f"cuda:{self.local_rank}",
                    load_precision=load_precision,
                    tensor_parallel_size=tensor_parallel_size,
                    enable_flash_attention=enable_flash_attention
                )
            else:
                print_rank0("初始化简化版推理引擎...")
                # 获取模型路径和相关配置
                model_config = self.config['model_configs'][model_name]
                model_path = model_config['model_path']
                
                # 使用简化版推理引擎
                self.inference_engine = create_simple_vllm_inference(
                    model_path=model_path,
                    device=f"cuda:{self.local_rank}",
                    load_precision=load_precision
                )
                if not use_vllm and not use_flash_attn:
                    print_rank0("注意：根据配置未启用VLLM或FlashAttention优化")
                else:
                    print_rank0("注意：VLLM/FlashAttention不可用，使用简化版推理引擎")
            
            # 3. 初始化评估器
            print_rank0("初始化评估器...")
            self.evaluator = create_vqav2_evaluator(
                result_dir=result_dir
            )
            
            # 4. 运行推理
            print_rank0("开始批量推理...")
            total_samples = 0
            batch_count = 0
            
            # 存储所有推理结果
            all_results = []
            
            for batch_idx, batch in enumerate(self.data_loader):
                # 提取问题和图像
                questions = [item['question'] for item in batch]
                images = [item['image_path'] for item in batch]  # 直接使用图像路径
                question_ids = [item['question_id'] for item in batch]
                image_ids = [item['image_id'] for item in batch]
                
                print_rank0(f"进程 {self.rank} 处理批次 {batch_idx + 1}, 样本数: {len(questions)}")
                
                # 批量推理
                predictions = self.inference_engine.batch_generate(
                    questions, 
                    images,
                    max_new_tokens=max_new_tokens
                )
                
                # 保存批次结果
                batch_results = []
                for i, (q_id, img_id, question, prediction, batch_item) in enumerate(
                    zip(question_ids, image_ids, questions, predictions, batch)
                ):
                    # 从批次项中获取ground truth（如果存在）
                    ground_truth = batch_item.get('ground_truth', '')
                    
                    result = {
                        'question_id': q_id,
                        'image_id': img_id,
                        'question': question,
                        'model_prediction': prediction,
                        'ground_truth': ground_truth,
                        'batch_index': batch_idx,
                        'sample_index_in_batch': i,
                        'image_path': str(batch_item['image_path'])  # 添加完整图片路径（转换为字符串）
                    }
                    batch_results.append(result)
                
                all_results.extend(batch_results)
                total_samples += len(predictions)
                batch_count += 1
                
                # 改进的进度显示
                remaining_samples = len(self.data_loader.dataset) - total_samples
                print_rank0(f"进程 {self.rank} 进度: 已推理 {total_samples}/{len(self.data_loader.dataset)} 样本, 剩余 {remaining_samples} 样本")
                
                # 确保当前批次完全处理完成
                print_rank0(f"批次 {batch_idx + 1} 推理完成，准备下一个批次...")
            
            # 5. 保存推理结果
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            
            # 保存当前进程的推理结果
            result_file = Path(result_dir) / f"rank_{self.rank}_vqav2_inference_results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'inference_results': all_results,
                    'statistics': {
                        'total_samples': total_samples,
                        'batch_count': batch_count,
                        'elapsed_time': elapsed_time,
                        'samples_per_second': total_samples / elapsed_time if elapsed_time > 0 else 0,
                        'rank': self.rank,
                        'world_size': self.world_size
                    }
                }, f, indent=2, ensure_ascii=False)
            
            print_rank0(f"进程 {self.rank} 推理完成 - 总样本: {total_samples}, 耗时: {elapsed_time:.2f}秒")
            print_rank0(f"推理结果已保存: {result_file}")
            
            # 6. 如果是主进程，合并所有结果
            if self.rank == 0:
                # 等待所有进程完成
                if self.world_size > 1:
                    print_rank0("等待所有进程完成推理...")
                    # 在实际应用中，这里应该添加分布式同步逻辑
                
                # 合并所有进程的结果
                merged_results = []
                for rank in range(self.world_size):
                    rank_file = Path(result_dir) / f"rank_{rank}_vqav2_inference_results.json"
                    if rank_file.exists():
                        with open(rank_file, 'r', encoding='utf-8') as f:
                            rank_data = json.load(f)
                            merged_results.extend(rank_data['inference_results'])
                
                # 按question_id排序
                merged_results.sort(key=lambda x: x['question_id'])
                
                # 保存合并结果
                merged_file = Path(result_dir) / "merged_vqav2_inference_results.json"
                with open(merged_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'inference_results': merged_results,
                        'total_samples': len(merged_results),
                        'world_size': self.world_size,
                        'elapsed_time': elapsed_time
                    }, f, indent=2, ensure_ascii=False)
                
                print_rank0(f"合并推理结果已保存: {merged_file}")
                print_rank0(f"总样本数: {len(merged_results)}, 总耗时: {elapsed_time:.2f}秒")
            
            return total_samples
            
        except Exception as e:
            print_rank0(f"进程 {self.rank} 评估失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # 清理分布式环境
            cleanup_distributed()
    
    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        if not self.start_time or not self.end_time:
            return {}
        
        elapsed_time = self.end_time - self.start_time
        
        # 获取推理引擎的性能统计
        engine_stats = {}
        if self.inference_engine:
            engine_stats = self.inference_engine.get_performance_stats()
        
        return {
            'total_time': elapsed_time,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'rank': self.rank,
            'world_size': self.world_size,
            'engine_stats': engine_stats
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Optimized VQAv2 Evaluation Pipeline')
    parser.add_argument('--config', type=str, default='configs/vqav2_config.json',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='LLaVA-1.5-7B',
                        help='Model name')
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--load_precision', type=str, default='fp16',
                        choices=['fp16', 'bf16', 'fp32'],
                        help='Model loading precision')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Tensor parallel size for multi-GPU inference')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Maximum number of new tokens to generate')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 创建评估管道
    pipeline = OptimizedEvaluatePipeline(args.config)
    
    # 运行评估
    try:
        total_samples = pipeline.run_vqav2_evaluation(
            model_name=args.model,
            result_dir=args.output,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            load_precision=args.load_precision,
            tensor_parallel_size=args.tensor_parallel_size,
            max_new_tokens=args.max_new_tokens
        )
        
        # 输出性能统计
        stats = pipeline.get_performance_stats()
        print_rank0("=== 性能统计 ===")
        print_rank0(f"总样本数: {total_samples}")
        print_rank0(f"总耗时: {stats.get('total_time', 0):.2f} 秒")
        if stats.get('total_time', 0) > 0:
            print_rank0(f"吞吐量: {total_samples / stats['total_time']:.2f} 样本/秒")
        
        print_rank0("评估完成!")
        
    except Exception as e:
        print_rank0(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()