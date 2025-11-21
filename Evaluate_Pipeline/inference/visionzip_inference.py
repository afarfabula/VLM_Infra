#!/usr/bin/env python3
"""
VisionZip优化的模型推理模块
支持分布式推理
"""

# 关键：设置环境变量来禁用自动加载适配器，避免inject_adapter_in_model错误
import os
os.environ['TRANSFORMERS_NO_ADAPTERS'] = '1'

# 确保使用正确的缓存目录
os.environ['HF_HOME'] = '/data/model/Inference_VLM/.cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/data/model/Inference_VLM/.cache'
os.environ['TRANSFORMERS_CACHE'] = '/data/model/Inference_VLM/.cache'

import sys
import torch
import time
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

# 在导入transformers之前进行monkey patching，这是关键的修复
import types

# 先导入transformers
import transformers
print(f"Transformers模块导入成功，版本: {transformers.__version__}")

# 重写load_adapter方法，让它什么都不做，避免inject_adapter_in_model错误
def no_op_load_adapter(self, *args, **kwargs):
    pass

# 应用monkey patch
transformers.modeling_utils.PreTrainedModel.load_adapter = no_op_load_adapter

# 导入transformers组件
from transformers import StoppingCriteriaList, TextIteratorStreamer
from threading import Thread

# 设置LLaVA路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLava"))

# 直接导入LLaVA相关模块，与visionzip_cli.py保持一致
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# 设置VisionZip路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "VisionZip"))

# 直接导入VisionZip，与visionzip_cli.py保持一致
from visionzip import visionzip
print("VisionZip模块导入成功")


class VisionZipInference:
    """VisionZip优化的模型推理器"""
    
    def __init__(self, model_path: str, device: Optional[str] = None, load_precision: str = '4bit', dominant: int = 54, contextual: int = 10):
        self.model_path = model_path
        self.load_precision = load_precision
        self.dominant = dominant
        self.contextual = contextual
        
        # 分布式配置
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # 设备设置
        if device is None:
            self.device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 模型组件
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.conv_template = None
        
        # 性能统计
        self.inference_times = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        print(f"进程 {self.rank} 正在初始化VisionZip模型...")
        
        # 禁用torch初始化
        disable_torch_init()
        
        # 获取模型名称
        model_name = get_model_name_from_path(self.model_path)
        
        # 适配本地模型路径
        if 'llava' not in model_name.lower():
            if 'LLava-1.5-7B' in self.model_path:
                model_name = 'llava-v1.5-7b'
            elif 'LLava-1.5-13B' in self.model_path:
                model_name = 'llava-v1.5-13b'
            elif 'llava' in self.model_path.lower():
                model_name = 'llava'
        
        # 根据load_precision设置加载选项
        load_8bit = self.load_precision.lower() == '8bit'
        load_4bit = self.load_precision.lower() == '4bit'
        
        print(f"进程 {self.rank} 使用{self.load_precision}精度加载模型...")
        
        # 对于本地已下载的完整LLaVA模型，直接使用None作为model_base
        model_base = None
        
        # 加载预训练模型
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
            self.model_path, 
            model_base,
            model_name, 
            load_8bit, 
            load_4bit, 
            device=self.device
        )
        
        # 在VisionZip注入前进行PEFT相关检查，避免inject_adapter_in_model错误
        # 检查并处理模型的PEFT相关属性
        if hasattr(self.model, 'base_model'):
            if hasattr(self.model.base_model, 'peft_config'):
                if self.model.base_model.peft_config is None:
                    print(f"进程 {self.rank} 移除了None的peft_config以避免注入错误")
                    delattr(self.model.base_model, 'peft_config')
        
        # 注入VisionZip补丁
        print(f"进程 {self.rank} 开始注入VisionZip补丁...")
        self.model = visionzip(self.model, dominant=self.dominant, contextual=self.contextual)
        print(f"进程 {self.rank} VisionZip补丁注入完成")
        
        # 设置对话模板
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        
        self.conv_template = conv_templates[conv_mode].copy()
        
        print(f"进程 {self.rank} 模型初始化完成，设备: {self.device}")
    
    def load_image(self, image_file):
        """加载图像"""
        image = Image.open(image_file).convert('RGB')
        return image
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """预处理图像"""
        image_tensor = process_images([image], self.image_processor, self.model.config)
        
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        
        return image_tensor
    
    def generate_answer(self, question: str, image: Image.Image, 
                       max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        """生成答案"""
        start_time = time.time()
        
        try:
            # 预处理图像
            image_tensor = self.preprocess_image(image)
            image_size = image.size
            
            # 构建对话
            conv = self.conv_template.copy()
            
            # 添加图像token
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
            
            # 生成答案
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True
                )
            
            # 解码答案
            answer = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # 记录推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return answer
            
        except Exception as e:
            print(f"推理失败: {e}")
            return ""
    
    def batch_generate(self, questions: List[str], images: List[Image.Image],
                       max_new_tokens: int = 512, temperature: float = 0.2) -> List[str]:
        """批量生成答案 - 参考visionzip_cli.py的稳定实现"""
        if len(questions) != len(images):
            raise ValueError("问题数量和图像数量必须相同")
        
        if len(questions) == 0:
            return []
        
        answers = []
        start_time = time.time()
        
        # 逐个处理每个样本，确保图像正确传递
        for idx, (question, image) in enumerate(zip(questions, images)):
            try:
                # 为每个样本单独构建提示
                conv = self.conv_template.copy()
                
                # 处理图像
                image_size = image.size
                image_tensor = process_images([image], self.image_processor, self.model.config)
                
                if type(image_tensor) is list:
                    image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
                else:
                    image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
                
                # 添加图像token到第一个消息
                if self.model.config.mm_use_im_start_end:
                    question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
                else:
                    question = DEFAULT_IMAGE_TOKEN + '\n' + question
                
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                # Tokenize输入
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
                
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
                    
                answers.append(outputs)
                
            except Exception as e:
                print(f"样本 {idx} 处理错误: {e}")
                answers.append("")
        
        # 记录推理时间
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        print(f"进程 {self.rank} 批量推理完成: {len(questions)} 个样本, 耗时: {inference_time:.2f}秒")
        
        return answers
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if not self.inference_times:
            return {}
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        total_time = sum(self.inference_times)
        
        return {
            'total_inferences': len(self.inference_times),
            'average_time_per_inference': avg_time,
            'total_inference_time': total_time,
            'rank': self.rank,
            'device': self.device
        }


def create_distributed_inference(model_path: str, num_processes: int = 4):
    """创建分布式推理器"""
    
    # 检查分布式环境
    rank = int(os.environ.get('RANK', 0))
    
    if rank >= num_processes:
        print(f"进程 {rank} 超出范围，跳过初始化")
        return None
    
    # 创建推理器
    inference = VisionZipInference(model_path)
    
    return inference


if __name__ == "__main__":
    # 测试推理器
    model_path = "/data/model/Inference_VLM/models-LLava-1.5-7B"
    
    try:
        inference = VisionZipInference(model_path)
        print("推理器创建成功")
        
        # 测试性能统计
        stats = inference.get_performance_stats()
        print(f"性能统计: {stats}")
        
    except Exception as e:
        print(f"推理器测试失败: {e}")