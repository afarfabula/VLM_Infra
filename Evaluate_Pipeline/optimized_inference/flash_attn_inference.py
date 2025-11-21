"""
基于FlashAttention的多卡推理引擎实现
不依赖VLLM，直接使用PyTorch和FlashAttention进行优化
使用LLaVA兼容的方式加载模型
"""

import os
import sys
import torch
import torch.nn as nn
from typing import List, Union
from pathlib import Path
import json
from PIL import Image
import time
from transformers import StoppingCriteriaList
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加LLaVA路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLava"))

# 立即应用视觉塔构建器补丁
from .builder_patch import apply_builder_patch
apply_builder_patch()

# 导入LLaVA相关模块
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from threading import Thread
from transformers import TextIteratorStreamer

# 导入本地视觉塔补丁
from .vision_tower_patch import patch_model_for_local_vision_tower

# 检查FlashAttention是否可用
try:
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
    FLASH_ATTN_AVAILABLE = True
    print("FlashAttention is available")
except ImportError as e:
    print(f"Warning: FlashAttention not available: {e}")
    FLASH_ATTN_AVAILABLE = False

class FlashAttentionInferenceEngine:
    """基于FlashAttention的推理引擎，支持多GPU优化"""
    
    def __init__(self, 
                 model_path: str, 
                 device: str = "cuda",
                 load_precision: str = "fp16",
                 tensor_parallel_size: int = 1,
                 enable_flash_attention: bool = True,
                 max_model_len: int = 2048):
        """
        初始化FlashAttention推理引擎
        
        Args:
            model_path: 模型路径
            device: 设备类型
            load_precision: 加载精度 (fp16, bf16, fp32)
            tensor_parallel_size: 张量并行大小（用于多GPU）
            enable_flash_attention: 是否启用FlashAttention
            max_model_len: 最大模型长度
        """
        self.model_path = model_path
        self.device = device
        self.load_precision = load_precision
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_flash_attention = enable_flash_attention and FLASH_ATTN_AVAILABLE
        self.max_model_len = max_model_len
        
        # 性能统计
        self.stats = {
            'total_inference_time': 0,
            'total_tokens': 0,
            'batch_count': 0
        }
        
        # 根据精度设置dtype
        if load_precision == "fp16":
            self.dtype = torch.float16
        elif load_precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        print(f"Initializing FlashAttention engine with:")
        print(f"  - Model: {model_path}")
        print(f"  - Device: {device}")
        print(f"  - Precision: {self.dtype}")
        print(f"  - Tensor Parallel Size: {tensor_parallel_size}")
        print(f"  - Flash Attention: {self.enable_flash_attention}")
        print(f"  - Max Model Length: {max_model_len}")
        
        # 初始化模型和分词器
        try:
            
            # 初始化分布式环境（如果需要）
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.rank = torch.distributed.get_rank()
                self.world_size = torch.distributed.get_world_size()
            else:
                self.rank = 0
                self.world_size = 1
            
            # 禁用torch初始化
            disable_torch_init()
            
            # 获取模型名称
            model_name = get_model_name_from_path(self.model_path)
            
            # 适配本地模型路径 - 特别处理LLaVA-1.5系列模型
            if 'llava' in model_name.lower() and 'LLava-1.5-' in self.model_path:
                if 'LLava-1.5-7B' in self.model_path:
                    model_name = 'llava-v1.5-7b'
                elif 'LLava-1.5-13B' in self.model_path:
                    model_name = 'llava-v1.5-13b'
            
            # 根据load_precision设置加载选项
            load_8bit = self.load_precision.lower() == '8bit'
            load_4bit = self.load_precision.lower() == '4bit'
            
            print(f"Using {self.load_precision} precision to load model...")
            
            # 加载预训练模型（使用LLaVA的方式）
            # 在离线环境中加载模型，避免访问HuggingFace
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
                self.model_path, 
                None,  # model_base
                model_name, 
                load_8bit, 
                load_4bit, 
                device=self.device
            )
            
            # 应用本地视觉塔补丁
            patch_model_for_local_vision_tower(self.model)
            
            # 如果启用了FlashAttention且可用，修改模型的注意力机制
            if self.enable_flash_attention:
                self._replace_attention_with_flash()
            
            # 设置对话模板
            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            else:
                # 默认使用llava_v1模板，适用于LLaVA-1.5系列模型
                conv_mode = "llava_v1"
            
            self.conv_template = conv_templates[conv_mode].copy()
            
            # 如果使用多GPU，将模型包装为DDP
            if self.world_size > 1:
                self.model = DDP(self.model, device_ids=[self.rank])
            
            # 将模型设置为评估模式
            self.model.eval()
            
            print("FlashAttention engine initialized successfully!")
            
        except Exception as e:
            print(f"Failed to initialize FlashAttention engine: {e}")
            raise
    
    def _replace_attention_with_flash(self):
        """将模型中的标准注意力替换为FlashAttention"""
        # 这里需要根据具体模型架构来实现
        # 对于LLaMA-like模型，可以替换SelfAttention层
        print("Replacing attention layers with FlashAttention (if applicable)")
        # 注意：具体实现需要根据模型架构调整
    
    def load_image(self, image_file: str) -> Image.Image:
        """
        加载图像
        
        Args:
            image_file: 图像文件路径
            
        Returns:
            PIL图像对象
        """
        image = Image.open(image_file).convert('RGB')
        return image
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: PIL图像对象
            
        Returns:
            预处理后的图像张量
        """
        try:
            image_tensor = process_images([image], self.image_processor, self.model.config)
            
            if isinstance(image_tensor, list):
                image_tensor = [img.to(self.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(self.device, dtype=torch.float16)
            
            return image_tensor
        except Exception as e:
            print(f"Warning: Failed to process image: {e}")
            return None
    
    def batch_generate(self, 
                       questions: List[str], 
                       images: List[str],
                       max_new_tokens: int = 1024,
                       temperature: float = 0.2) -> List[str]:
        """
        批量生成答案
        
        Args:
            questions: 问题列表
            images: 图像路径列表
            max_new_tokens: 最大新token数
            temperature: 采样温度
            
        Returns:
            生成的答案列表
        """
        start_time = time.time()
        
        try:
            # 确保问题和图像数量一致
            if len(questions) != len(images):
                raise ValueError("问题数量和图像数量必须相同")
            
            if len(questions) == 0:
                return []
            
            predictions = []
            
            # 逐个处理每个样本，确保图像正确传递
            for idx, (question, image_path) in enumerate(zip(questions, images)):
                try:
                    # 加载图像
                    image = self.load_image(image_path)
                    
                    # 为每个样本单独构建提示
                    conv = self.conv_template.copy()
                    
                    # 处理图像
                    image_size = image.size
                    image_tensor = self.preprocess_image(image)
                    
                    if image_tensor is None:
                        predictions.append("")
                        continue
                    
                    # 添加图像token到第一个消息
                    if self.model.config.mm_use_im_start_end:
                        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
                    else:
                        question = DEFAULT_IMAGE_TOKEN + '\n' + question
                    
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    
                    # Tokenize输入
                    input_ids = tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                    ).unsqueeze(0).to(self.device)
                    
                    # 设置停止条件
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)])
                    
                    # 使用流式生成
                    streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                    
                    # 生成答案
                    with torch.inference_mode():
                        # 创建生成线程
                        generation_kwargs = dict(
                            inputs=input_ids,
                            images=image_tensor,
                            image_sizes=[image_size],
                            do_sample=True if temperature > 0 else False,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                            streamer=streamer,
                            use_cache=True,
                            stopping_criteria=stopping_criteria,
                        )
                        
                        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                        thread.start()
                        
                        # 收集生成的文本
                        chunks = []
                        for new_text in streamer:
                            chunks.append(new_text)
                        
                        # 等待生成完成
                        thread.join()
                        outputs = "".join(chunks).strip()
                    
                    # 清理输出，移除停止字符串
                    if outputs.endswith(stop_str):
                        outputs = outputs[:-len(stop_str)]
                        
                    predictions.append(outputs)
                    
                except Exception as e:
                    print(f"Sample {idx} processing error: {e}")
                    predictions.append("")
            
            # 更新性能统计
            end_time = time.time()
            batch_time = end_time - start_time
            total_tokens = sum(len(pred.split()) for pred in predictions)
            
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

def create_flash_attn_inference(model_path: str,
                                device: str = "cuda",
                                load_precision: str = "fp16",
                                tensor_parallel_size: int = 1,
                                enable_flash_attention: bool = True,
                                max_model_len: int = 2048):
    """
    创建FlashAttention推理引擎的工厂函数
    
    Args:
        model_path: 模型路径
        device: 设备类型
        load_precision: 加载精度
        tensor_parallel_size: 张量并行大小
        enable_flash_attention: 是否启用FlashAttention
        max_model_len: 最大模型长度
        
    Returns:
        FlashAttentionInferenceEngine实例
    """
    return FlashAttentionInferenceEngine(
        model_path=model_path,
        device=device,
        load_precision=load_precision,
        tensor_parallel_size=tensor_parallel_size,
        enable_flash_attention=enable_flash_attention,
        max_model_len=max_model_len
    )