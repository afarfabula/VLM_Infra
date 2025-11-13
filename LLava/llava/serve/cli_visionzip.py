import argparse
import os
import sys
import time
from threading import Thread

import torch
from PIL import Image
import requests
from io import BytesIO

# 让 VisionZip 在当前 monorepo 中可导入：
# - 计算当前文件所在目录 `_THIS_DIR`
# - 回到仓库根目录 `_REPO_ROOT`
# - 拼接 VisionZip 子项目路径 `_VISIONZIP_DIR`
# - 若未在 `sys.path` 中则追加，保证 `import visionzip` 可用
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_VISIONZIP_DIR = os.path.join(_REPO_ROOT, "VisionZip")
if _VISIONZIP_DIR not in sys.path:
    sys.path.append(_VISIONZIP_DIR)

from visionzip import visionzip  # noqa: E402

# Transformers 相关工具（流式输出 / 停止准则）与日志模块
from transformers import TextIteratorStreamer, StoppingCriteriaList  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402
import warnings  # noqa: E402

# 首次加载时屏蔽冗余/噪声警告：
# - 忽略 HuggingFace Hub 的 FutureWarning（例如 `resume_download` 即将移除）
# - 将 Transformers 的日志级别设为 error，避免类型匹配等非致命警告打断首轮对话
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
hf_logging.set_verbosity_error()

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)


def safe_input(prompt: str) -> str:
    # 兼容交互与管道场景：优先从 TTY 读取，回退到标准输入
    try:
        if sys.stdin.isatty():
            return input(prompt)
        else:
            try:
                # 明确从 /dev/tty 读取，避免管道/重定向导致阻塞或 EOF
                with open('/dev/tty', 'r') as tty:
                    print(prompt, end='', flush=True)
                    line = tty.readline()
                    return '' if not line else line.rstrip('\n')
            except Exception:
                return input(prompt)
    except EOFError:
        # 捕获 EOF，返回空字符串用于外层退出逻辑
        return ""


def load_image(image_file: str) -> Image.Image:
    # 支持 URL 与本地路径；统一转为 RGB
    if image_file.startswith("http://") or image_file.startswith("https://"):
        resp = requests.get(image_file)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(image_file).convert("RGB")
    return img





def main(args):
    # 模型初始化：禁用冗余权重随机化以缩短启动时间
    disable_torch_init()

    # 根据路径推断模型名，兼容 hub 别名/镜像路径
    model_name = get_model_name_from_path(args.model_path)
    if 'llava' not in model_name.lower():
        if 'liuhaotian--llava-v1.6-7b' in args.model_path:
            model_name = 'llava-v1.6-7b'
        elif 'liuhaotian--llava-v1.5-7b' in args.model_path:
            model_name = 'llava-v1.5-7b'
        elif 'llava' in args.model_path.lower():
            model_name = 'llava'

    # 加载预训练模型与处理器；显式传入 device 以避免默认 CPU
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device
    )

    # 注入 VisionZip 补丁：替换视觉塔 forward 与多模态准备逻辑
    model = visionzip(model, dominant=args.dominant, contextual=args.contextual)

    # 会话模版选择：不同模型使用不同 prompt 格式
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

    # 若用户指定 `--conv-mode` 与推断不一致，优先使用用户指定
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        conv_mode = args.conv_mode
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[conv_mode].copy()
    # MPT 使用 ('user','assistant')，其余沿用模版默认角色
    roles = ('user', 'assistant') if "mpt" in model_name.lower() else conv.roles


    # 单图预处理：保留原始尺寸用于 anyres 还原；将像素值移至模型设备并转为 fp16
    image = load_image(args.image_file)
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [im.to(model.device, dtype=torch.float16) for im in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)


    # 对话主循环：首轮注入图像 token，后续仅文本
    while True:
        try:
            user_inp = safe_input(f"{roles[0]}: ")
        except EOFError:
            user_inp = ""
        if not user_inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        # 首轮：将图像 token 插入到用户消息前（根据配置决定是否使用 IM_START/IM_END）
        if image is not None:
            if model.config.mm_use_im_start_end:
                user_inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + user_inp
            else:
                user_inp = DEFAULT_IMAGE_TOKEN + '\n' + user_inp
            image = None

        conv.append_message(conv.roles[0], user_inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 构建多模态 input_ids：把图像特殊 token 替换为 `IMAGE_TOKEN_INDEX`
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        # 基于分隔符的停止准则，避免输出越界到下一轮历史
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)])

        # 流式生成：迭代输出 + 统计 prefill（建立 KV cache）耗时
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        t0 = time.perf_counter()
        prefill_time = None
        chunks = []

        def _generate():
            # 推理模式下禁用梯度；传入 `images`/`image_sizes` 触发 VisionZip 编码路径
            with torch.inference_mode():
                model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                )

        # 子线程执行生成，主线程消费流式输出
        thread = Thread(target=_generate)
        thread.start()

        for new_text in streamer:
            # 首个 token 到达时记录 prefill 时间
            if prefill_time is None:
                prefill_time = time.perf_counter() - t0
            print(new_text, end="", flush=True)
            chunks.append(new_text)

        thread.join()
        print()
        outputs = ("".join(chunks)).strip()

        # 去掉尾部分隔符，保证历史干净
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]

        conv.messages[-1][-1] = outputs

        # 时间统计：总耗时 / prefill / 平均解码时延（s/token）
        total_time = time.perf_counter() - t0
        decoding_time = total_time - (prefill_time or 0.0)
        gen_token_ids = tokenizer(outputs).input_ids
        avg_decoding_time = decoding_time / max(len(gen_token_ids), 1)
        print(f"[Timing] total={total_time:.3f}s, prefill={prefill_time or 0.0:.3f}s, avg_decoding={avg_decoding_time:.4f}s/token")


        if args.debug:
            # 打印最近几条消息用于调试 prompt 组装
            print("\n", {"messages": conv.messages[-4:]}, "\n")
            # 提示下一条输入
            print(f"{roles[0]}: ", end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 模型与设备配置
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    # VisionZip 配置：dominant 表示保留显著 patch 数量；contextual 表示上下文补齐数量
    parser.add_argument("--dominant", type=int, default=54, help="VisionZip dominant tokens")
    parser.add_argument("--contextual", type=int, default=10, help="VisionZip contextual tokens")

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)