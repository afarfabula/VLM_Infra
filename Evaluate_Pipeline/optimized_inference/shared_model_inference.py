import os
import torch
import time
import threading
import psutil
from typing import List, Dict, Optional
from PIL import Image
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

class MemoryMonitor:
    """
    显存和内存监控类，实时跟踪GPU和CPU内存使用情况
    """
    
    def __init__(self, log_interval: int = 1):
        """初始化内存监控器"""
        self.log_interval = log_interval
        self.start_time = None
        self.memory_stats = []
        self.active = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.active = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        """停止监控"""
        self.active = False
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """监控循环，定期收集内存使用数据"""
        while not self.stop_event.is_set():
            timestamp = time.time() - (self.start_time or time.time())
            
            # 获取GPU显存使用情况
            gpu_memory = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        allocated = torch.cuda.memory_allocated(i)
                        reserved = torch.cuda.memory_reserved(i)
                        gpu_memory[f"cuda:{i}"] = {
                            "allocated_gb": allocated / (1024 ** 3),
                            "reserved_gb": reserved / (1024 ** 3)
                        }
                    except:
                        pass
            
            # 获取CPU内存使用情况
            process = psutil.Process()
            cpu_memory = process.memory_info().rss / (1024 ** 3)
            
            self.memory_stats.append({
                "timestamp": timestamp,
                "gpu_memory": gpu_memory,
                "cpu_memory_gb": cpu_memory
            })
            
            time.sleep(self.log_interval)
    
    def get_report(self):
        """获取内存使用报告"""
        if not self.memory_stats:
            return {}
        
        report = {
            "max_cpu_memory_gb": max(stats["cpu_memory_gb"] for stats in self.memory_stats),
            "avg_cpu_memory_gb": sum(stats["cpu_memory_gb"] for stats in self.memory_stats) / len(self.memory_stats)
        }
        
        # 分析每个GPU的显存使用
        gpu_ids = set()
        for stats in self.memory_stats:
            gpu_ids.update(stats["gpu_memory"].keys())
        
        for gpu_id in gpu_ids:
            allocated_values = []
            reserved_values = []
            for stats in self.memory_stats:
                if gpu_id in stats["gpu_memory"]:
                    allocated_values.append(stats["gpu_memory"][gpu_id]["allocated_gb"])
                    reserved_values.append(stats["gpu_memory"][gpu_id]["reserved_gb"])
            
            if allocated_values:
                report[f"{gpu_id}_max_allocated_gb"] = max(allocated_values)
                report[f"{gpu_id}_avg_allocated_gb"] = sum(allocated_values) / len(allocated_values)
                report[f"{gpu_id}_max_reserved_gb"] = max(reserved_values)
                report[f"{gpu_id}_avg_reserved_gb"] = sum(reserved_values) / len(reserved_values)
        
        return report


class SharedModelInference:
    """
    显存优化的并行推理类，实现模型权重共享和私有激活/KV-Cache管理
    
    核心特性：
    1. 模型权重只在GPU上加载一次并共享
    2. 每个推理请求只申请私有的激活/KV-Cache
    3. 计算完成后立即释放私有资源
    4. 支持批量预处理和顺序推理以优化显存使用
    """
    
    def __init__(self, model_path: str, device: str = None, conv_mode: str = "llava_v1", 
                 batch_size: int = 8, use_visionzip: bool = True):
        """
        初始化共享模型推理器
        
        Args:
            model_path: 模型路径
            device: 运行设备，默认自动选择
            conv_mode: 对话模式
            batch_size: 处理批次大小（不是并行度，而是预处理批次）
            use_visionzip: 是否使用VisionZip优化
        """
        self.model_path = model_path
        self.conv_mode = conv_mode
        self.batch_size = batch_size
        self.use_visionzip = use_visionzip
        self.inference_times = []
        self.memory_usage = []
        
        # 初始化内存监控器
        self.memory_monitor = MemoryMonitor()
        
        # 自动选择设备
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rank = int(os.environ.get('RANK', 0))
        
        # 线程安全锁，保护模型访问
        self.model_lock = threading.Lock()
        
        # 初始化共享模型
        self._initialize_shared_model()
        print(f"共享模型推理器初始化完成，设备: {self.device}, 批次大小: {self.batch_size}")
    
    def _initialize_shared_model(self):
        """
        初始化共享模型和相关组件
        只加载一次模型权重并放在GPU上
        """
        disable_torch_init()
        
        print(f"加载模型到 {self.device}...")
        start_time = time.time()
        
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, None, model_name)
        
        # 将模型移至指定设备并设置为评估模式
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载对话模板
        self.conv_template = conv_templates[self.conv_mode]
        
        # 应用VisionZip优化（如果启用）
        if self.use_visionzip:
            try:
                visionzip.apply_patch(self.model)
                print(f"VisionZip补丁成功注入到模型")
            except Exception as e:
                print(f"VisionZip补丁注入失败: {e}")
        
        # 记录模型加载时间和显存使用
        load_time = time.time() - start_time
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            print(f"模型加载完成，耗时: {load_time:.2f}秒，显存使用: {memory_used:.2f}GB")
            self.memory_usage.append({
                'stage': 'model_loading',
                'memory_gb': memory_used,
                'time': load_time
            })
    
    def _process_single_sample(self, idx: int, question: str, image: Image.Image, 
                              max_new_tokens: int = 512, temperature: float = 0.2) -> tuple:
        """
        处理单个样本并返回结果
        这个方法在模型锁的保护下执行，确保模型权重共享但每次只处理一个样本
        """
        try:
            # 为每个样本单独构建提示
            conv = self.conv_template.copy()
            
            # 处理图像（私有资源）
            image_size = image.size
            image_tensor = process_images([image], self.image_processor, self.model.config)
            
            if isinstance(image_tensor, list):
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
            
            # Tokenize输入（私有资源）
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
            
            # 设置停止条件
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)])
            
            # 使用流式生成
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # 在模型锁的保护下执行推理
            with self.model_lock:
                # 记录开始推理前的显存使用
                if torch.cuda.is_available():
                    start_memory = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
                
                with torch.inference_mode():
                    # 创建生成参数（私有KV-Cache将在generate方法中创建）
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
                    
                    # 执行生成
                    thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    # 收集生成的文本
                    chunks = []
                    for new_text in streamer:
                        chunks.append(new_text)
                    
                    # 等待生成完成
                    thread.join()
                    outputs = "".join(chunks).strip()
                
                # 记录推理后的显存使用
                if torch.cuda.is_available():
                    end_memory = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
                    # 记录此次推理的显存使用增量（KV-Cache等）
                    memory_increment = end_memory - start_memory
                    self.memory_usage.append({
                        'stage': f'inference_sample_{idx}',
                        'memory_gb': memory_increment,
                        'total_memory': end_memory
                    })
            
            # 清理输出，移除停止字符串
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
                
            # 清理私有资源引用，帮助垃圾回收
            del image_tensor, input_ids, stopping_criteria, conv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return idx, outputs
        except Exception as e:
            print(f"样本 {idx} 处理错误: {e}")
            # 出错时也要尝试清理资源
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return idx, ""
    
    def _batch_preprocess(self, questions: List[str], images: List[Image.Image]) -> List[dict]:
        """
        批量预处理样本，将图像和问题转换为模型输入格式
        预处理过程在CPU上并行进行，以减少GPU占用
        
        Args:
            questions: 问题列表
            images: 图像列表
            
        Returns:
            预处理后的样本列表
        """
        preprocessed_samples = []
        
        # 批量处理图像 - 这里使用多线程进行CPU预处理
        from concurrent.futures import ThreadPoolExecutor
        
        def process_sample(idx, q, img):
            # 处理图像
            image_size = img.size
            try:
                # 在CPU上预处理图像
                image_tensor = process_images([img], self.image_processor, self.model.config)
                
                # 构建提示
                if self.model.config.mm_use_im_start_end:
                    processed_q = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + q
                else:
                    processed_q = DEFAULT_IMAGE_TOKEN + '\n' + q
                
                return {
                    'idx': idx,
                    'question': processed_q,
                    'image_tensor': image_tensor,
                    'image_size': image_size
                }
            except Exception as e:
                print(f"预处理样本 {idx} 失败: {e}")
                return {
                    'idx': idx,
                    'question': q,
                    'image_tensor': None,
                    'image_size': image_size,
                    'error': str(e)
                }
        
        # 使用线程池并行预处理
        with ThreadPoolExecutor(max_workers=min(16, len(questions))) as executor:
            futures = [executor.submit(process_sample, idx, q, img) for idx, (q, img) in enumerate(zip(questions, images))]
            for future in futures:
                result = future.result()
                preprocessed_samples.append(result)
        
        return preprocessed_samples
    
    def batch_generate(self, questions: List[str], images: List[Image.Image],
                       max_new_tokens: int = 512, temperature: float = 0.2) -> List[str]:
        """
        批量生成答案 - 使用共享模型权重，优化的批量处理以提高推理效率
        
        Args:
            questions: 问题列表
            images: 图像列表
            max_new_tokens: 最大生成token数
            temperature: 生成温度
            
        Returns:
            生成的答案列表，与输入顺序一致
        """
        if len(questions) != len(images):
            raise ValueError("问题数量和图像数量必须相同")
        
        if len(questions) == 0:
            return []
        
        answers = [""] * len(questions)
        start_time = time.time()
        
        # 开始内存监控
        self.memory_monitor.start()
        
        print(f"开始批量推理，共 {len(questions)} 个样本，使用共享模型权重...")
        
        # 第一步：批量预处理（在CPU上并行进行）
        preprocess_start = time.time()
        preprocessed_samples = self._batch_preprocess(questions, images)
        preprocess_time = time.time() - preprocess_start
        print(f"预处理完成，耗时: {preprocess_time:.2f}秒")
        
        # 第二步：批量推理 - 分批处理以优化内存使用
        batch_size = self.batch_size
        for batch_start in range(0, len(preprocessed_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(preprocessed_samples))
            current_batch = preprocessed_samples[batch_start:batch_end]
            
            # 处理当前批次中的每个样本
            for sample in current_batch:
                if 'error' in sample:
                    # 跳过预处理失败的样本
                    continue
                    
                # 为每个样本单独构建对话
                conv = self.conv_template.copy()
                conv.append_message(conv.roles[0], sample['question'])
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                # 处理单个样本
                idx, answer = self._process_single_sample(
                    sample['idx'], 
                    sample['question'], 
                    images[sample['idx']],  # 注意：这里我们需要原始图像，因为_process_single_sample需要重新处理
                    max_new_tokens, 
                    temperature
                )
                answers[idx] = answer
            
            # 批次间清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 打印批次进度
            processed = batch_end
            elapsed_time = time.time() - start_time
            samples_per_second = processed / elapsed_time if elapsed_time > 0 else 0
            print(f"进度: {processed}/{len(questions)}, 速率: {samples_per_second:.2f} 样本/秒")
        
        # 记录整体推理时间
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        total_samples_per_second = len(questions) / inference_time if inference_time > 0 else 0
        print(f"批量推理完成: {len(questions)} 个样本, 总耗时: {inference_time:.2f}秒, 平均速率: {total_samples_per_second:.2f} 样本/秒")
        
        # 最终清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            print(f"推理完成后显存使用: {final_memory:.2f}GB")
        
        # 停止内存监控并生成报告
        self.memory_monitor.stop()
        memory_report = self.memory_monitor.get_report()
        
        # 打印内存使用报告
        print("\n====== 内存使用报告 ======")
        print(f"最大CPU内存使用: {memory_report.get('max_cpu_memory_gb', 0):.2f} GB")
        print(f"平均CPU内存使用: {memory_report.get('avg_cpu_memory_gb', 0):.2f} GB")
        
        # 打印GPU显存使用
        for key, value in memory_report.items():
            if "cuda" in key and "max_allocated" in key:
                gpu_id = key.split("_")[0]
                print(f"{gpu_id} 最大显存分配: {value:.2f} GB")
                print(f"{gpu_id} 平均显存分配: {memory_report.get(f'{gpu_id}_avg_allocated_gb', 0):.2f} GB")
        print("========================\n")
        
        return answers
    
    def get_performance_stats(self) -> Dict:
        """
        获取性能和内存统计信息
        
        Returns:
            包含性能指标和内存使用统计的字典
        """
        if not self.inference_times:
            return {}
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        total_time = sum(self.inference_times)
        total_samples = len(self.inference_times) * self.batch_size  # 估算总样本数
        avg_throughput = total_samples / total_time if total_time > 0 else 0
        
        stats = {
            'total_inferences': len(self.inference_times),
            'average_time_per_inference': avg_time,
            'total_inference_time': total_time,
            'average_throughput': avg_throughput,
            'rank': self.rank,
            'device': self.device,
            'batch_size': self.batch_size,
            'model_path': self.model_path
        }
        
        # 添加显存使用统计
        if self.memory_usage:
            stats['memory_usage'] = self.memory_usage
            # 计算平均显存增量
            inference_memory = [m['memory_gb'] for m in self.memory_usage if m['stage'].startswith('inference')]
            if inference_memory:
                stats['average_inference_memory_increment_gb'] = sum(inference_memory) / len(inference_memory)
        
        # 添加内存监控报告
        memory_report = self.memory_monitor.get_report()
        if memory_report:
            stats['memory_monitor_report'] = memory_report
        
        return stats

# 支持有限并行度的版本，使用队列控制并发数量
class OptimizedParallelInference(SharedModelInference):
    """
    优化的并行推理类，使用队列控制并发数量以平衡性能和显存使用
    """
    
    def __init__(self, model_path: str, device: str = None, conv_mode: str = "llava_v1", 
                 batch_size: int = 8, num_workers: int = 4, use_visionzip: bool = True):
        """
        初始化优化的并行推理器
        
        Args:
            model_path: 模型路径
            device: 运行设备
            conv_mode: 对话模式
            batch_size: 批次大小
            num_workers: 并行工作线程数（控制并发度）
            use_visionzip: 是否使用VisionZip优化
        """
        super().__init__(model_path, device, conv_mode, batch_size, use_visionzip)
        self.num_workers = num_workers
        print(f"优化的并行推理器初始化完成，并行工作线程数: {self.num_workers}")
    
    def batch_generate(self, questions: List[str], images: List[Image.Image],
                       max_new_tokens: int = 512, temperature: float = 0.2) -> List[str]:
        """
        使用有限并行度的批量生成，通过控制并发数量优化显存使用
        
        Args:
            questions: 问题列表
            images: 图像列表
            max_new_tokens: 最大生成token数
            temperature: 生成温度
            
        Returns:
            生成的答案列表
        """
        if len(questions) != len(images):
            raise ValueError("问题数量和图像数量必须相同")
        
        if len(questions) == 0:
            return []
        
        answers = [""] * len(questions)
        start_time = time.time()
        
        # 计算实际使用的工作线程数（不超过样本数）
        actual_workers = min(self.num_workers, len(questions))
        
        print(f"开始优化并行推理，共 {len(questions)} 个样本，并行工作线程数: {actual_workers}...")
        
        # 第一步：批量预处理（在CPU上并行进行）
        preprocess_start = time.time()
        preprocessed_samples = self._batch_preprocess(questions, images)
        preprocess_time = time.time() - preprocess_start
        print(f"预处理完成，耗时: {preprocess_time:.2f}秒")
        
        # 按批次处理样本，每个批次内部使用线程池
        batch_size = self.batch_size
        for batch_start in range(0, len(preprocessed_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(preprocessed_samples))
            current_batch = preprocessed_samples[batch_start:batch_end]
            
            print(f"处理批次 {batch_start//batch_size + 1}: 样本 {batch_start} 到 {batch_end-1}")
            
            # 使用线程池处理当前批次，但控制并发数
            from concurrent.futures import ThreadPoolExecutor
            results = []
            
            # 实际使用的线程数不超过配置的工作线程数和当前批次大小
            workers_for_batch = min(actual_workers, len(current_batch))
            
            # 提交任务到线程池
            with ThreadPoolExecutor(max_workers=workers_for_batch) as executor:
                futures = []
                for sample in current_batch:
                    if 'error' in sample:
                        continue
                    
                    # 提交处理任务
                    future = executor.submit(
                        self._process_single_sample,
                        sample['idx'],
                        sample['question'],
                        images[sample['idx']],  # 使用原始图像
                        max_new_tokens,
                        temperature
                    )
                    futures.append(future)
                
                # 收集结果
                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"处理任务异常: {e}")
            
            # 更新答案数组
            for idx, answer in results:
                answers[idx] = answer
            
            # 批次间清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                current_memory = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
                print(f"批次完成后显存使用: {current_memory:.2f}GB")
            
            # 打印进度
            processed = batch_end
            elapsed_time = time.time() - start_time
            samples_per_second = processed / elapsed_time if elapsed_time > 0 else 0
            print(f"进度: {processed}/{len(questions)}, 速率: {samples_per_second:.2f} 样本/秒")
        
        # 记录整体推理时间
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        total_samples_per_second = len(questions) / inference_time if inference_time > 0 else 0
        print(f"优化并行推理完成: {len(questions)} 个样本, 总耗时: {inference_time:.2f}秒, 平均速率: {total_samples_per_second:.2f} 样本/秒")
        
        return answers

def create_shared_model_inference(model_path: str, optimized: bool = True, 
                                 num_workers: int = 4, batch_size: int = 8, 
                                 use_visionzip: bool = True):
    """
    创建共享模型推理器的工厂函数
    
    Args:
        model_path: 模型路径
        optimized: 是否使用优化的并行版本
        num_workers: 并行工作线程数
        batch_size: 批次大小
        use_visionzip: 是否使用VisionZip优化
        
    Returns:
        推理器实例
    """
    if optimized:
        return OptimizedParallelInference(
            model_path, 
            num_workers=num_workers, 
            batch_size=batch_size,
            use_visionzip=use_visionzip
        )
    else:
        return SharedModelInference(
            model_path,
            batch_size=batch_size,
            use_visionzip=use_visionzip
        )

if __name__ == "__main__":
    # 测试代码
    model_path = "/data/model/Inference_VLM/models-LLava-1.5-7B"
    
    try:
        # 创建优化的并行推理器
        inference = OptimizedParallelInference(model_path, num_workers=2, batch_size=4)
        print("优化的并行推理器创建成功")
        
        # 这里可以添加测试代码
        print("推理器测试完成")
        
    except Exception as e:
        print(f"推理器测试失败: {e}")
