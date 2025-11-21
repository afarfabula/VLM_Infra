"""
VLLM推理引擎实现
支持多GPU推理和FlashAttention加速
"""

import os
import torch
from typing import List, Union
from pathlib import Path
import json
from PIL import Image
import time

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VLLM not available: {e}")
    VLLM_AVAILABLE = False

class VLLMInferenceEngine:
    """基于VLLM的推理引擎，支持多GPU和FlashAttention优化"""
    
    def __init__(self, 
                 model_path: str, 
                 device: str = "cuda",
                 load_precision: str = "fp16",
                 tensor_parallel_size: int = 1,
                 enable_flash_attention: bool = True,
                 max_model_len: int = 2048):
        """
        初始化VLLM推理引擎
        
        Args:
            model_path: 模型路径
            device: 设备类型
            load_precision: 加载精度 (fp16, bf16, fp32)
            tensor_parallel_size: 张量并行大小（用于多GPU）
            enable_flash_attention: 是否启用FlashAttention
            max_model_len: 最大模型长度
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError("VLLM is not available. Please install vllm package.")
        
        self.model_path = model_path
        self.device = device
        self.load_precision = load_precision
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_flash_attention = enable_flash_attention
        self.max_model_len = max_model_len
        
        # 性能统计
        self.stats = {
            'total_inference_time': 0,
            'total_tokens': 0,
            'batch_count': 0
        }
        
        # 根据精度设置dtype
        if load_precision == "fp16":
            dtype = "float16"
        elif load_precision == "bf16":
            dtype = "bfloat16"
        else:
            dtype = "float32"
        
        print(f"Initializing VLLM engine with:")
        print(f"  - Model: {model_path}")
        print(f"  - Device: {device}")
        print(f"  - Precision: {dtype}")
        print(f"  - Tensor Parallel Size: {tensor_parallel_size}")
        print(f"  - Flash Attention: {enable_flash_attention}")
        print(f"  - Max Model Length: {max_model_len}")
        
        # 初始化VLLM引擎
        try:
            # 构建引擎参数
            engine_args = {
                "model": model_path,
                "tokenizer": model_path,
                "tensor_parallel_size": tensor_parallel_size,
                "dtype": dtype,
                "max_model_len": max_model_len,
                "trust_remote_code": True,
                "gpu_memory_utilization": 0.85,  # 使用85%的GPU内存
            }
            
            # 如果启用了FlashAttention，则添加相关参数
            if enable_flash_attention:
                engine_args["enable_prefix_caching"] = True
                engine_args["enforce_eager"] = False  # 允许使用CUDA图优化
            
            # 创建LLM实例
            self.llm = LLM(**engine_args)
            
            # 设置采样参数
            self.sampling_params = SamplingParams(
                temperature=0.0,  # 贪婪解码
                max_tokens=1024,
                stop_token_ids=[self.llm.llm_engine.tokenizer.eos_token_id]
            )
            
            print("VLLM engine initialized successfully!")
            
        except Exception as e:
            print(f"Failed to initialize VLLM engine: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> str:
        """
        预处理图像（简单实现）
        
        Args:
            image_path: 图像路径
            
        Returns:
            图像描述或路径
        """
        try:
            # 对于视觉语言模型，通常需要特殊的提示格式
            # 这里返回图像路径，在实际使用中可能需要更复杂的处理
            return f"<image>\n{image_path}"
        except Exception as e:
            print(f"Warning: Failed to process image {image_path}: {e}")
            return "<image>"
    
    def batch_generate(self, 
                       questions: List[str], 
                       images: List[str],
                       max_new_tokens: int = 1024) -> List[str]:
        """
        批量生成答案
        
        Args:
            questions: 问题列表
            images: 图像路径列表
            max_new_tokens: 最大新token数
            
        Returns:
            生成的答案列表
        """
        start_time = time.time()
        
        try:
            # 更新采样参数
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_new_tokens,
                stop_token_ids=[self.llm.llm_engine.tokenizer.eos_token_id]
            )
            
            # 构造输入提示
            prompts = []
            for question, image_path in zip(questions, images):
                # 对于视觉语言模型，构造包含图像的提示
                image_prompt = self.preprocess_image(image_path)
                prompt = f"{image_prompt}\n{question}"
                prompts.append(prompt)
            
            # 批量推理
            outputs = self.llm.generate(prompts, sampling_params)
            
            # 提取生成的文本
            predictions = [output.outputs[0].text.strip() for output in outputs]
            
            # 更新性能统计
            end_time = time.time()
            batch_time = end_time - start_time
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            
            self.stats['total_inference_time'] += batch_time
            self.stats['total_tokens'] += total_tokens
            self.stats['batch_count'] += 1
            
            print(f"Batch inference completed in {batch_time:.2f}s, {len(predictions)} samples")
            
            return predictions
            
        except Exception as e:
            print(f"Batch generation failed: {e}")
            # 返回空答案作为后备
            return [""] * len(questions)
    
    def get_performance_stats(self) -> dict:
        """获取性能统计信息"""
        stats = self.stats.copy()
        if stats['total_inference_time'] > 0:
            stats['tokens_per_second'] = stats['total_tokens'] / stats['total_inference_time']
        else:
            stats['tokens_per_second'] = 0
        return stats

def create_distributed_vllm_inference(model_path: str,
                                    device: str = "cuda",
                                    load_precision: str = "fp16",
                                    tensor_parallel_size: int = 1,
                                    enable_flash_attention: bool = True,
                                    max_model_len: int = 2048):
    """
    创建分布式VLLM推理引擎的工厂函数
    
    Args:
        model_path: 模型路径
        device: 设备类型
        load_precision: 加载精度
        tensor_parallel_size: 张量并行大小
        enable_flash_attention: 是否启用FlashAttention
        max_model_len: 最大模型长度
        
    Returns:
        VLLMInferenceEngine实例
    """
    return VLLMInferenceEngine(
        model_path=model_path,
        device=device,
        load_precision=load_precision,
        tensor_parallel_size=tensor_parallel_size,
        enable_flash_attention=enable_flash_attention,
        max_model_len=max_model_len
    )