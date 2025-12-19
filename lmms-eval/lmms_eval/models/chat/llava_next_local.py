# lmms_eval/models/chat/llava_next_local.py
import sys
from pathlib import Path
import torch
import warnings
from typing import List, Tuple
from PIL import Image

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.api.instance import Instance
from lmms_eval.protocol import ChatMessages

# 确保llava模块总是从LLaVA-NeXT加载，阻止从旧LLava目录加载
def _ensure_llava_from_correct_path():
    """确保llava模块从正确的路径加载"""
    # 移除sys.path中任何指向旧LLava目录的路径
    old_llava_paths = [p for p in sys.path if "/data/model/Inference_VLM/VLM_Infra/LLava" in p and "/data/model/Inference_VLM/VLM_Infra/LLaVA-NeXT" not in p]
    for path in old_llava_paths:
        if path in sys.path:
            sys.path.remove(path)
    
    # 清理已加载的模块缓存中来自被阻止路径的模块
    modules_to_remove = []
    for module_name, module in sys.modules.items():
        try:
            module_file = getattr(module, '__file__', None)
            if module_file:
                # 检查模块是否来自旧的LLava目录但不是LLaVA-NeXT
                if "/data/model/Inference_VLM/VLM_Infra/LLava/" in module_file and "/data/model/Inference_VLM/VLM_Infra/LLaVA-NeXT/" not in module_file:
                    modules_to_remove.append(module_name)
        except:
            pass
    
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]

# 执行安全检查
_ensure_llava_from_correct_path()

# 添加从LLaVA-NeXT inference模块导入的常量
# 使用绝对路径而不是相对路径，避免因工作目录不同而导致路径错误
inference_path = Path("/data/model/Inference_VLM/VLM_Infra/LLaVA-NeXT/inference")
if str(inference_path) not in sys.path:
    sys.path.insert(0, str(inference_path))

# 确保能正确导入所需的模块
try:
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava_next_inference import LlavaNextInference
except ImportError as e:
    print(f"Failed to import from LLaVA-NeXT: {e}")
    # 尝试备用导入路径
    backup_path = Path(__file__).parent.parent.parent.parent.parent / "LLaVA-NeXT" / "inference"
    if str(backup_path) not in sys.path:
        sys.path.insert(0, str(backup_path))
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava_next_inference import LlavaNextInference


@register_model("llava_next_local_chat")
class LlavaNextLocalChat(lmms):
   

    def __init__(self,
                 pretrained: str = "/data/model/Inference_VLM/models-LLava-NeXT",
                 device: str = "cuda",
                 batch_size: int = 1,
                 load_precision: str = "4bit",
                 use_flash_attn: bool = True,
                 **kwargs) -> None:
        super().__init__()
        # 保存框架传来的参数
        self.batch_size = batch_size
        
        # 实例化推理器（使用简化的初始化）
        self.infer = LlavaNextInference()
        
       

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        每个 request 对应一条 (图+文 -> 文) 任务
        """
        res = []
        for req in requests:
            ctx, doc_to_messages, gen_kwargs, doc_id, task, split = req.args
            doc = self.task_dict[task][split][doc_id]
            
            # 根据实际情况构建消息列表
            # 从文档中提取问题和图像
            question = doc.get('question', '')
            image_from_doc = doc.get('image', None)
            
            # 确保图片是标准的PIL.Image.Image格式
            if image_from_doc is not None:
                # 强制转换为标准的Image.Image格式
                if hasattr(image_from_doc, 'convert'):
                    # 创建新的Image.Image实例，复制原图内容
                    temp_image = image_from_doc.copy()
                    image_from_doc = Image.new(temp_image.mode, temp_image.size)
                    image_from_doc.paste(temp_image)
                    image_from_doc = image_from_doc.convert("RGB")
                else:
                    # 如果无法直接转换，创建一个新的RGB图像
                    image_from_doc = Image.new("RGB", image_from_doc.size)
            
            # 构建符合 ChatMessages 格式的消息列表
            messages_raw = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}]
                }
            ]
            
            # 如果有图像，添加图像信息
            if image_from_doc is not None:
                messages_raw[0]["content"].insert(0, {"type": "image"})
            
            chat_messages = ChatMessages(**{"messages": messages_raw})
            visuals, videos, _ = chat_messages.extract_media()
            # 使用从文档中提取的图像，如果ChatMessages未能提取则回退到原始图像
            image = visuals[0] if visuals and visuals[0] is not None else image_from_doc
            
            # 再次确保最终使用的图片是标准的PIL.Image.Image格式
            if image is not None:
                # 强制转换为标准的Image.Image格式
                if hasattr(image, 'convert'):
                    # 创建新的Image.Image实例，复制原图内容
                    temp_image = image.copy()
                    image = Image.new(temp_image.mode, temp_image.size)
                    image.paste(temp_image)
                    image = image.convert("RGB")
                else:
                    # 如果无法直接转换，创建一个新的RGB图像
                    image = Image.new("RGB", image.size)
            
            # 处理 gen_kwargs，它可能是一个函数而不是字典
            if callable(gen_kwargs):
                # 如果 gen_kwargs 是一个函数，尝试调用它来获取参数
                try:
                    # 首先尝试直接调用
                    gen_kwargs_dict = gen_kwargs()
                except TypeError:
                    # 如果直接调用失败，尝试传递doc参数
                    try:
                        gen_kwargs_dict = gen_kwargs(doc)
                    except Exception:
                        gen_kwargs_dict = doc_to_messages  # 回退到doc_to_messages作为参数字典
                except Exception:
                    gen_kwargs_dict = doc_to_messages  # 回退到doc_to_messages作为参数字典
            else:
                gen_kwargs_dict = gen_kwargs
            
            # Use more appropriate default values that work well with LLaVA-NeXT
            # These values match those used in direct execution which produces correct output
            max_new_tokens = gen_kwargs_dict.get("max_new_tokens", 512) if isinstance(gen_kwargs_dict, dict) else 512
            temperature = gen_kwargs_dict.get("temperature", 0.2) if isinstance(gen_kwargs_dict, dict) else 0.2
            top_p = gen_kwargs_dict.get("top_p", 0.9) if isinstance(gen_kwargs_dict, dict) else 0.9
            do_sample = gen_kwargs_dict.get("do_sample", True if temperature > 0.0 else False) if isinstance(gen_kwargs_dict, dict) else (True if temperature > 0.0 else False)
            
            # 提取问题文本
            prompt_text = question
            print('llava_next_local prompt_text:',prompt_text)
            
            # 调用推理器生成答案
            ans = self.infer.generate_answer(
                prompt=prompt_text,
                image=image,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
            print('llava_next_local ans:',ans)
            res.append(ans)
        
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for LlavaNextLocalChat")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("generate_until_multi_round is not implemented for LlavaNextLocalChat")


if __name__ == "__main__":
    import requests
    tokenizer, model, image_processor, max_length = None, None, None, None
    inf = LlavaNextInference()
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    prompt = "请描述这张图片的内容，用一句话回答"
    out = inf.generate_answer(prompt, image, max_new_tokens=512, temperature=0.2, top_p=0.9, min_new_tokens=5)
    print(out)
