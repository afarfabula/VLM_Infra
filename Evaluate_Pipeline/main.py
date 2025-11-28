#!/usr/bin/env python3
"""
VQAv2评估管道主入口
支持分布式推理
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from data_loader.vqav2_loader import create_vqav2_dataloader
from inference.visionzip_inference import create_distributed_inference
from evaluation.vqav2_evaluator import create_vqav2_evaluator
from utils.distributed_utils import setup_distributed, cleanup_distributed, get_rank, get_world_size, print_rank0


class EvaluatePipeline:
    """评估管道类"""
    
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
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def run_vqav2_evaluation(self, model_name='LLaVA-1.5-7B', visionzip_enabled=False, result_dir='./results', num_samples=100, batch_size=32, load_precision='4bit', dominant=54, contextual=10, use_flash_attn=False, use_shared_model=False, num_workers=4):
        """运行VQAv2评估"""
        self.start_time = time.time()
        
        print_rank0(f"开始VQAv2评估 - 样本数: {num_samples}, 批次大小: {batch_size}")
        
        try:
            # 1. 准备数据加载器
            print_rank0("准备数据加载器...")
            self.data_loader = create_vqav2_dataloader(
                data_root="/data/model/Inference_VLM/VLM_Infra/datasets/VQAv2",
                batch_size=batch_size,  # 使用传入的批次大小
                num_workers=4,  # 默认工作进程数
                num_samples=num_samples  # 使用传入的样本数量
            )
            
            # 2. 初始化推理引擎
            print_rank0("初始化推理引擎...")
            
            if use_shared_model:
                # 使用共享模型优化推理
                from optimized_inference.shared_model_inference import create_shared_model_inference
                print_rank0("使用共享模型优化推理")
                self.inference_engine = create_shared_model_inference(
                    model_path=self.config['model_configs'][model_name]['model_path'],
                    num_workers=num_workers,
                    batch_size=batch_size,
                    use_visionzip=visionzip_enabled
                )
            else:
                # 获取模型路径
                model_path = self.config['model_configs'][model_name]['model_path']
                
                # 根据是否启用VisionZip选择不同的推理器
                if visionzip_enabled:
                    # 使用VisionZip推理器
                    from inference.visionzip_inference import VisionZipInference
                    self.inference_engine = VisionZipInference(
                        model_path=model_path,
                        device=f"cuda:{self.local_rank}",
                        load_precision=load_precision,
                        dominant=dominant,
                        contextual=contextual,
                        use_flash_attn=use_flash_attn
                    )
                else:
                    # 使用标准推理器
                    from inference.visionzip_inference import VisionZipInference
                    self.inference_engine = VisionZipInference(
                        model_path=model_path,
                        device=f"cuda:{self.local_rank}",
                        load_precision=load_precision,
                        use_flash_attn=use_flash_attn
                    )
            
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
            
            for batch in self.data_loader:
                # 提取问题和图像
                questions = [item['question'] for item in batch]
                images = [self.data_loader.dataset.load_image(item['image_path']) for item in batch]
                question_ids = [item['question_id'] for item in batch]
                image_ids = [item['image_id'] for item in batch]
                
                print(f"进程 {self.rank} 处理批次 {batch_count + 1}, 样本数: {len(questions)}")
                
                # 批量推理
                predictions = self.inference_engine.batch_generate(questions, images)
                
                # 保存批次结果
                batch_results = []
                for i, (q_id, img_id, question, prediction, batch_item) in enumerate(zip(question_ids, image_ids, questions, predictions, batch)):
                    # 从批次项中获取ground truth（如果存在）
                    ground_truth = batch_item.get('ground_truth', '')
                    
                    result = {
                        'question_id': q_id,
                        'image_id': img_id,
                        'question': question,
                        'model_prediction': prediction,
                        'ground_truth': ground_truth,  # 从数据加载器获取真实答案
                        'batch_index': batch_count,
                        'sample_index_in_batch': i,
                        'image_path': str(batch_item['image_path'])  # 添加完整图片路径（转换为字符串）
                    }
                    batch_results.append(result)
                
                all_results.extend(batch_results)
                total_samples += len(predictions)
                batch_count += 1
                
                # 改进的进度显示
                remaining_samples = len(self.data_loader.dataset) - total_samples
                print(f"进程 {self.rank} 进度: 已推理 {total_samples}/{len(self.data_loader.dataset)} 样本, 剩余 {remaining_samples} 样本")
                
                # 确保当前批次完全处理完成
                print(f"批次 {batch_count} 推理完成，准备下一个批次...")
            
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
                        'samples_per_second': total_samples / elapsed_time,
                        'rank': self.rank,
                        'world_size': self.world_size
                    }
                }, f, indent=2, ensure_ascii=False)
            
            print(f"进程 {self.rank} 推理完成 - 总样本: {total_samples}, 耗时: {elapsed_time:.2f}秒")
            print(f"推理结果已保存: {result_file}")
            
            # 6. 如果是主进程，合并所有结果
            if self.rank == 0:
                # 等待所有进程完成
                if self.world_size > 1:
                    print_rank0("等待所有进程完成推理...")
                    # 这里可以添加分布式同步逻辑
                
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
            print(f"进程 {self.rank} 评估失败: {e}")
            raise
        
        finally:
            # 清理分布式环境
            cleanup_distributed()
    
    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        if not self.start_time or not self.end_time:
            return {}
        
        elapsed_time = self.end_time - self.start_time
        
        return {
            'total_time': elapsed_time,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'rank': self.rank,
            'world_size': self.world_size
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VQAv2评估管道')
    parser.add_argument('--config', type=str, required=True, 
                       help='配置文件路径')
    parser.add_argument('--model', type=str, default='LLaVA-1.5-7B',
                       help='模型名称')
    parser.add_argument('--visionzip', action='store_true',
                       help='启用VisionZip优化')
    parser.add_argument('--output', type=str, default='./results',
                       help='输出目录')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='样本数量 (默认: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小 (默认: 32)')
    parser.add_argument('--load_precision', type=str, default='4bit',
                       help='模型加载精度 (默认: 4bit)')
    parser.add_argument('--use-flash-attn', action='store_true',
                       help='使用Flash Attention 2加速推理')
    parser.add_argument('--use-shared-model', action='store_true',
                       help='使用共享模型优化推理')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='预处理工作线程数 (默认: 4)')
    
    args = parser.parse_args()
    
    # 验证配置文件存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 创建评估管道
    pipeline = EvaluatePipeline(args.config)
    
    # 运行评估
    try:
        pipeline.run_vqav2_evaluation(
            model_name=args.model,
            visionzip_enabled=args.visionzip,
            result_dir=args.output,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            load_precision=args.load_precision,
            use_flash_attn=args.use_flash_attn,
            use_shared_model=args.use_shared_model,
            num_workers=args.num_workers
        )
        print_rank0("评估管道执行成功")
        
        # 输出性能统计
        stats = pipeline.get_performance_stats()
        if stats:
            print_rank0(f"性能统计: {stats}")
            
    except Exception as e:
        print(f"评估管道执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()