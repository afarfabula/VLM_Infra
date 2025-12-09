# lmms_eval/models/chat/visionzip_llava.py
import sys
from pathlib import Path
import torch
from typing import List, Tuple
from PIL import Image

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.api.instance import Instance
from lmms_eval.protocol import ChatMessages

# 确保拼接出正确的路径
inference_path = Path(__file__).parent.parent.parent.parent.parent / "Evaluate_Pipeline" / "inference"
#print("Trying to insert path:", inference_path.resolve())
sys.path.insert(0, str(inference_path))

from visionzip_inference import VisionZipInference


@register_model("visionzip_llava")          # 命令行 --model visionzip_llava
class VisionZipLlavaForLmmsEval(lmms):
    is_simple = False                       # 走 chat 分支

    def __init__(self,
                 pretrained: str,            # 对应你本地模型目录
                 device: str = "cuda",
                 batch_size: int = 8,
                 **kwargs) -> None:
        super().__init__()
        # 把框架传过来的 batch_size / device 等参数存下来
        self.batch_size = batch_size
        # 实例化你自己的推理器（已包含 4bit/8bit/flash-attn 等逻辑）
        self.infer = VisionZipInference(
            model_path=pretrained,
            device=device,
            load_precision=kwargs.get("load_precision", "4bit"),
            use_flash_attn=kwargs.get("use_flash_attn", True)
        )

    # ---------- 框架唯一关心的两个接口 ----------
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        每个 request 对应一条 (图+文 -> 文) 任务
        框架已经把 doc_to_messages 处理好，我们直接取图和文本
        """
        res = []
        for req in requests:
            ctx, doc_to_messages, gen_kwargs, doc_id, task, split = req.args
            doc = self.task_dict[task][split][doc_id]
            messages_raw = doc_to_messages(doc)
            chat_messages = ChatMessages(**{"messages": messages_raw})
            visuals, videos, _ = chat_messages.extract_media()
            image = visuals[0] if visuals else None
            max_new_tokens = gen_kwargs.get("max_new_tokens", 512)
            temperature = gen_kwargs.get("temperature", 0.0)
            ans = self.infer.generate_answer(ctx, image, max_new_tokens=max_new_tokens, temperature=temperature)
            res.append(ans)
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        多选 VQA 需要算候选答案的 log-prob
        你的 LLaVA 没提供 p(continuation) 接口，先抛异常跳过这类任务
        """
        raise NotImplementedError("VisionZip LLaVA 暂未实现 loglikelihood 接口")

    def generate_until_multi_round(self, requests):
        # 这里可以根据需要实现具体的逻辑
        raise NotImplementedError("generate_until_multi_round is not implemented")