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
from transformers import StoppingCriteria, StoppingCriteriaList

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


class VisionZipInference:
    """VisionZip优化的模型推理器"""
    
    def __init__(self, model_path: str, device: Optional[str] = None, load_precision: str = '4bit', dominant: int = 54, contextual: int = 10, use_flash_attn: bool = True):
        self.model_path = model_path
        self.load_precision = load_precision
        self.dominant = dominant
        self.contextual = contextual
        self.use_flash_attn = use_flash_attn
        
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
            device=self.device,
            use_flash_attn=self.use_flash_attn
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
        # 处理图像为None的情况
        if image is None:
            return None
            
        image_tensor = process_images([image], self.image_processor, self.model.config)
        
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        
        return image_tensor
    
    def generate_answer(self, question: str, image: Image.Image, 
                       max_new_tokens: int = 512, temperature: float = 2) -> str:
        """生成答案"""
        start_time = time.time()
        
        try:
            # 预处理图像
            image_tensor = self.preprocess_image(image)
            # 处理图像为None的情况
            image_size = image.size if image is not None else None
            
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
            
            # 生成答案（使用流式+停用词，避免空输出）
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)])
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                thread = Thread(target=self.model.generate, kwargs=dict(
                    inputs=input_ids,
                    images=image_tensor,
                    image_sizes=[image_size] if image_size is not None else None,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    cache_position=None,
                    streamer=streamer,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                ))
                thread.start()
                chunks = []
                for new_text in streamer:
                    chunks.append(new_text)
                thread.join()
                answer = ''.join(chunks).strip()
                if answer.endswith(stop_str):
                    answer = answer[:-len(stop_str)]

            # 记录推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return answer
            
        except Exception as e:
            print(f"推理失败: {e}")
            return ""
    
    
    
    @torch.inference_mode()
    def batch_generate(self,
                    questions: List[str],
                    images: List[Image.Image],
                    max_new_tokens: int = 512,
                    temperature: float = 0.2,
                    batch_size: int = 8) -> List[str]:
        """
        并行 batch 推理，权重只加载一次
        batch_size: 一次喂给 GPU 的样本数，根据显存调整
        """
        if len(questions) != len(images):
            raise ValueError("questions 与 images 数量不一致")
        n = len(questions)
        if n == 0:
            return []

        answers = [''] * n
        idx = 0
        while idx < n:
            # 1. 取出 mini-batch
            batch_q = questions[idx: idx + batch_size]
            batch_img = images[idx: idx + batch_size]
            b = len(batch_q)

            # 2. 一次性处理图像
            image_tensor = process_images(batch_img, self.image_processor, self.model.config)
            if isinstance(image_tensor, list):
                image_tensor = torch.stack([img.to(self.model.device, dtype=torch.float16) for img in image_tensor])
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

            #print("image_tensor shape:",image_tensor.shape)

            # 3. 一次性构建 prompts 并 tokenize
            input_ids_list, stop_str_list = [], []
            for q in batch_q:
                conv = self.conv_template.copy()
                q = DEFAULT_IMAGE_TOKEN + '\n' + q
                conv.append_message(conv.roles[0], q)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                input_ids_list.append(input_ids)
                stop_str_list.append(conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2)
            #print("input_ids_list shape:",input_ids_list[0].shape)
            # 4. 右 padding 到同一长度
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            ).to(self.model.device)
            #print("input_ids shape:",input_ids.shape)
            # 4. 逐条生成（复用老代码的 streamer + StoppingCriteria）
            for i in range(b):
                conv = self.conv_template.copy()
                q = DEFAULT_IMAGE_TOKEN + '\n' + batch_q[i]
                conv.append_message(conv.roles[0], q)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, self.tokenizer,
                                                IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                stopping_criteria = StoppingCriteriaList(
                    [KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)]
                )
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

                with torch.inference_mode():
                    # 处理图像为None的情况
                    image_size = batch_img[i].size if batch_img[i] is not None else None
                    thread = Thread(target=self.model.generate,
                                    kwargs=dict(inputs=input_ids,
                                                images=image_tensor[i:i+1],   # 单图
                                                image_sizes=[image_size],
                                                do_sample=temperature > 0,
                                                temperature=temperature,
                                                max_new_tokens=max_new_tokens,
                                                streamer=streamer,
                                                stopping_criteria=stopping_criteria,
                                                use_cache=True))
                    thread.start()
                    chunks = []
                    for new_text in streamer:
                        chunks.append(new_text)
                    thread.join()
                    text = ''.join(chunks).strip()
                    if text.endswith(stop_str):
                        text = text[:-len(stop_str)]
                answers[idx + i] = text

            idx += b

        return answers


class VisionZipInference_Batch:
    """VisionZip优化的模型推理器"""
    
    def __init__(self, model_path: str, device: Optional[str] = None, load_precision: str = '4bit', dominant: int = 54, contextual: int = 10, use_flash_attn: bool = True):
        self.model_path = model_path
        self.load_precision = load_precision
        self.dominant = dominant
        self.contextual = contextual
        self.use_flash_attn = use_flash_attn
        
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
            device=self.device,
            use_flash_attn=self.use_flash_attn
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
        #self.model = visionzip(self.model, dominant=self.dominant, contextual=self.contextual)
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
        # 处理图像为None的情况
        if image is None:
            return None
            
        image_tensor = process_images([image], self.image_processor, self.model.config)
        
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        
        return image_tensor
    
    def generate_answer(self, question: str, image: Image.Image, 
                       max_new_tokens: int = 512, temperature: float = 2) -> str:
        """生成答案"""
        start_time = time.time()
        
        try:
            # 预处理图像
            image_tensor = self.preprocess_image(image)
            # 处理图像为None的情况
            image_size = image.size if image is not None else None
            
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
    
    
    
    
    
    @torch.inference_mode()
    def batch_generate(self,
                    questions: List[str],
                    images: List[Image.Image],
                    max_new_tokens: int = 512,
                    temperature: float = 0.2,
                    batch_size: int = 8) -> List[str]:
        """
        并行 batch 推理，权重只加载一次
        batch_size: 一次喂给 GPU 的样本数，根据显存调整
        """
        if len(questions) != len(images):
            raise ValueError("questions 与 images 数量不一致")
        n = len(questions)
        if n == 0:
            return []

        answers = [''] * n
        idx = 0
        while idx < n:
            # 1. 取出 mini-batch
            batch_q = questions[idx: idx + batch_size]
            batch_img = images[idx: idx + batch_size]
            b = len(batch_q)

            # 2. 一次性处理图像
            image_tensor = process_images(batch_img, self.image_processor, self.model.config)
            if isinstance(image_tensor, list):
                image_tensor = [img.to(self.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(self.device, dtype=torch.float16)

            # 3. 一次性构建 prompts 并 tokenize
            input_ids_list, stop_str_list = [], []
            for q in batch_q:
                conv = self.conv_template.copy()
                # 添加图像token，考虑mm_use_im_start_end配置
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    q = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + q
                else:
                    q = DEFAULT_IMAGE_TOKEN + '\n' + q
                conv.append_message(conv.roles[0], q)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                input_ids_list.append(input_ids)
                stop_str_list.append(conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2)

            # 4. 右 padding 到同一长度
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            ).to(self.device)

            # 5. 批量生成
            generation_config = dict(
                inputs=input_ids,
                images=image_tensor,
                image_sizes=[img.size if img is not None else None for img in batch_img],
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            outputs_ids = self.model.generate(**generation_config)  # [B, seq_len]

            # 6. 解码 & 去停止符
            for i in range(b):
                # 计算正确的解码起始位置
                start_idx = input_ids_list[i].shape[0]  # 使用shape[0]而不是shape[-1]
                # 确保start_idx不超过outputs_ids的长度
                start_idx = min(start_idx, outputs_ids.shape[1])
                
                # 解码答案
                gen = self.tokenizer.decode(outputs_ids[i, start_idx:], skip_special_tokens=True)
                
                # 处理停止符 - 更健壮的方式
                if stop_str_list[i]:
                    # 查找停止符的位置
                    if stop_str_list[i] in gen:
                        gen = gen[:gen.find(stop_str_list[i])]
                answers[idx + i] = gen.strip()

            idx += b

        return answers


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


# 添加这一行，使得外部调用VisionZipInference时实际上是调用VisionZipInference_Batch
#VisionZipInference = VisionZipInference_Batch

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
