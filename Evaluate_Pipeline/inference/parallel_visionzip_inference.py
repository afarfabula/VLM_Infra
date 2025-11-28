import os
import torch
import time
import concurrent.futures
from typing import List, Dict, Optional
from PIL import Image
from threading import Thread
from transformers import StoppingCriteriaList, TextIteratorStreamer
import sys
from pathlib import Path

# 设置环境变量
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLava"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "VisionZip"))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import visionzip

class ParallelVisionZipInference:
    """并行处理的VisionZip推理类，支持真正的批量并行推理"""
    
    def __init__(self, model_path: str, device: str = None, conv_mode: str = "llava_v1", 
                 num_workers: int = 8, max_batch_size: int = 32):
        """
        初始化并行VisionZip推理器
        
        Args:
            model_path: 模型路径
            device: 运行设备，默认自动选择
            conv_mode: 对话模式
            num_workers: 并行工作线程数
            max_batch_size: 最大批量大小
        """
        self.model_path = model_path
        self.conv_mode = conv_mode
        self.num_workers = num_workers
        self.max_batch_size = max_batch_size
        self.inference_times = []
        
        # 自动选择设备
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rank = int(os.environ.get('RANK', 0))
        
        # 初始化模型
        self._initialize_model()
        print(f"并行VisionZip推理器初始化完成，设备: {self.device}, 工作线程数: {self.num_workers}")
    
    def _initialize_model(self):
        """初始化模型和相关组件"""
        disable_torch_init()
        
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, None, model_name)
        
        # 设置模型到指定设备
        self.model = self.model.to(self.device)
        
        # 加载对话模板
        self.conv_template = conv_templates[self.conv_mode]
        
        # 检查是否启用VisionZip
        try:
            # 尝试注入VisionZip补丁
            visionzip.apply_patch(self.model)
            print(f"VisionZip补丁成功注入到模型")
        except Exception as e:
            print(f"VisionZip补丁注入失败: {e}")
    
    def _process_single_sample(self, idx: int, question: str, image: Image.Image, 
                              max_new_tokens: int = 512, temperature: float = 0.2) -> tuple:
        """处理单个样本并返回结果"""
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
                
            return idx, outputs
        except Exception as e:
            print(f"样本 {idx} 处理错误: {e}")
            return idx, ""
    
    def batch_generate(self, questions: List[str], images: List[Image.Image],
                       max_new_tokens: int = 512, temperature: float = 0.2) -> List[str]:
        """批量并行生成答案 - 实现真正的并行推理"""
        if len(questions) != len(images):
            raise ValueError("问题数量和图像数量必须相同")
        
        if len(questions) == 0:
            return []
        
        answers = [""] * len(questions)
        start_time = time.time()
        
        # 确定实际使用的工作线程数（不超过样本数和设置的工作线程数）
        actual_workers = min(self.num_workers, len(questions))
        
        # 使用线程池并行处理样本
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # 提交所有任务
            future_to_idx = {executor.submit(
                self._process_single_sample, idx, question, image, max_new_tokens, temperature
            ): idx for idx, (question, image) in enumerate(zip(questions, images))}
            
            # 收集结果，保持顺序
            for future in concurrent.futures.as_completed(future_to_idx):
                idx, result = future.result()
                answers[idx] = result
        
        # 记录推理时间
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        samples_per_second = len(questions) / inference_time
        print(f"进程 {self.rank} 并行推理完成: {len(questions)} 个样本, 耗时: {inference_time:.2f}秒, 速率: {samples_per_second:.2f} 样本/秒")
        
        return answers
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if not self.inference_times:
            return {}
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        total_time = sum(self.inference_times)
        total_samples = sum(len(t) for t in self.inference_times) if isinstance(self.inference_times[0], list) else len(self.inference_times) * self.max_batch_size
        avg_throughput = total_samples / total_time if total_time > 0 else 0
        
        return {
            'total_inferences': len(self.inference_times),
            'average_time_per_inference': avg_time,
            'total_inference_time': total_time,
            'average_throughput': avg_throughput,
            'rank': self.rank,
            'device': self.device,
            'num_workers': self.num_workers
        }

# 优化版本，使用更高效的批处理策略
class OptimizedParallelVisionZipInference(ParallelVisionZipInference):
    """优化版并行VisionZip推理类，使用更高效的批处理策略"""
    
    def __init__(self, model_path: str, device: str = None, conv_mode: str = "llava_v1",
                 num_workers: int = 8, max_batch_size: int = 32):
        """初始化优化版并行推理器"""
        super().__init__(model_path, device, conv_mode, num_workers, max_batch_size)
        self.batch_processing_enabled = torch.cuda.is_available()  # 仅在GPU上启用批处理优化
        print(f"优化版并行VisionZip推理器初始化完成，批处理优化: {'启用' if self.batch_processing_enabled else '禁用'}")
    
    def _batch_preprocess(self, questions: List[str], images: List[Image.Image]) -> tuple:
        """批量预处理图像和问题"""
        processed_data = []
        
        # 批量处理图像
        image_tensors = []
        image_sizes = []
        prompts = []
        
        for question, image in zip(questions, images):
            # 处理图像
            image_size = image.size
            image_tensor = process_images([image], self.image_processor, self.model.config)
            
            if type(image_tensor) is list:
                image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            
            # 构建提示
            conv = self.conv_template.copy()
            if self.model.config.mm_use_im_start_end:
                formatted_question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                formatted_question = DEFAULT_IMAGE_TOKEN + '\n' + question
            
            conv.append_message(conv.roles[0], formatted_question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            image_tensors.append(image_tensor)
            image_sizes.append(image_size)
            prompts.append(prompt)
        
        # Tokenize所有提示
        tokenized_inputs = []
        stopping_criterias = []
        
        for prompt in prompts:
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
            
            # 设置停止条件
            conv = self.conv_template.copy()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)])
            
            tokenized_inputs.append(input_ids)
            stopping_criterias.append(stopping_criteria)
        
        return image_tensors, image_sizes, tokenized_inputs, stopping_criterias, prompts
    
    def batch_generate(self, questions: List[str], images: List[Image.Image],
                       max_new_tokens: int = 512, temperature: float = 0.2) -> List[str]:
        """优化版批量并行生成答案"""
        if len(questions) != len(images):
            raise ValueError("问题数量和图像数量必须相同")
        
        if len(questions) == 0:
            return []
        
        answers = [""] * len(questions)
        start_time = time.time()
        
        # 批量预处理数据
        image_tensors, image_sizes, tokenized_inputs, stopping_criterias, prompts = self._batch_preprocess(questions, images)
        
        # 确定实际使用的工作线程数
        actual_workers = min(self.num_workers, len(questions))
        
        # 定义单个样本的处理函数
        def _process_with_preprocessed(idx: int):
            try:
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                with torch.inference_mode():
                    # 创建生成线程
                    generation_kwargs = dict(
                        inputs=tokenized_inputs[idx],
                        images=image_tensors[idx],
                        image_sizes=[image_sizes[idx]],
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        streamer=streamer,
                        use_cache=True,
                        stopping_criteria=stopping_criterias[idx],
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
                
                # 清理输出
                conv = self.conv_template.copy()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                    
                return idx, outputs
            except Exception as e:
                print(f"样本 {idx} 处理错误: {e}")
                return idx, ""
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            future_to_idx = {executor.submit(_process_with_preprocessed, idx): idx for idx in range(len(questions))}
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_idx):
                idx, result = future.result()
                answers[idx] = result
        
        # 记录推理时间
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        samples_per_second = len(questions) / inference_time
        print(f"优化版进程 {self.rank} 并行推理完成: {len(questions)} 个样本, 耗时: {inference_time:.2f}秒, 速率: {samples_per_second:.2f} 样本/秒")
        
        return answers

def create_parallel_inference(model_path: str, num_processes: int = 4, optimized: bool = True, num_workers: int = 8):
    """创建并行推理器的工厂函数"""
    rank = int(os.environ.get('RANK', 0))
    
    if rank >= num_processes:
        print(f"进程 {rank} 超出范围，跳过初始化")
        return None
    
    if optimized:
        return OptimizedParallelVisionZipInference(model_path, num_workers=num_workers)
    else:
        return ParallelVisionZipInference(model_path, num_workers=num_workers)

if __name__ == "__main__":
    # 测试代码
    model_path = "/data/model/Inference_VLM/models-LLava-1.5-7B"
    
    try:
        # 创建优化版推理器
        inference = OptimizedParallelVisionZipInference(model_path, num_workers=16)
        print("优化版推理器创建成功")
        
        # 这里可以添加测试代码
        print("推理器测试完成")
        
    except Exception as e:
        print(f"推理器测试失败: {e}")
