import argparse
import torch
import sys
import time
from threading import Thread

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextIteratorStreamer, StoppingCriteriaList

# VisionZip injection
from visionzip import visionzip

# ======================= VisionZip CLI 注释 =======================
# 本脚本展示如何在 LLaVA 模型上注入 VisionZip 的视觉 token 压缩补丁：
# - 通过 visionzip(model, dominant, contextual) 将补丁挂载到 CLIP 视觉塔与 LLaVA 的多模态对接逻辑；
# - dominant 表示显著视觉 token 数（含 CLS），contextual 表示上下文聚合 token 数；
# - 生成阶段依旧使用 model.generate，只是图像侧会走 VisionZip 的压缩/还原流程以降低视觉 token 数。
#
# 启动示例（在仓库根目录运行）：
# - 默认（建议加量化以降显存）：
#   python VisionZip/visionzip_cli.py --model-path liuhaotian/llava-v1.6-7b-hf --image-file VisionZip/sample_dog.png --load-4bit
# - 指定物理 GPU（不设置 CUDA_VISIBLE_DEVICES 时）：
#   python VisionZip/visionzip_cli.py --model-path liuhaotian/llava-v1.6-7b-hf --image-file VisionZip/sample_dog.png --device cuda:2
# - 指定 GPU4（只暴露一张卡，进程内索引从 0 开始）：
#   CUDA_VISIBLE_DEVICES=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
#   python VisionZip/visionzip_cli.py --model-path liuhaotian/llava-v1.6-7b-hf --image-file VisionZip/sample_dog.png --device cuda:0 --load-4bit --max-new-tokens 64
# - 远程图片 URL：
#   python VisionZip/visionzip_cli.py --model-path liuhaotian/llava-v1.6-7b-hf --image-file https://images.cocodataset.org/val2017/000000039769.jpg --load-4bit
# - CPU 运行（很慢，不推荐）：
#   python VisionZip/visionzip_cli.py --model-path liuhaotian/llava-v1.6-7b-hf --image-file VisionZip/sample_dog.png --device cpu
# 说明与建议：
# - 使用 CUDA_VISIBLE_DEVICES 只暴露某张卡时，请在本脚本内用 --device cuda:0；不设置该环境变量时可直接用物理索引 --device cuda:<gpu_id>。
# - --dominant 与 --contextual 控制 VisionZip 压缩强度（默认 54/10），可按需调整：如 --dominant 32 --contextual 8。
# - 如果显存紧张，建议 --load-4bit，并适当降低 --max-new-tokens；可加 PYTORCH_ALLOC_CONF=expandable_segments:True 降低碎片化。
# - 使用本地权重路径（避免远程下载）：
#   python VisionZip/visionzip_cli.py --model-path /home_ext/quyanyi/Models/liuhaotian--llava-v1.6-7b-hf/snapshots/<commit_id> \
#     --image-file VisionZip/sample_dog.png --load-4bit
#   或者（v1.5-7B 本地快照示例）：
#   python VisionZip/visionzip_cli.py --model-path /home_ext/quyanyi/Models/liuhaotian--llava-v1.5-7b/snapshots/<commit_id> \
#     --image-file VisionZip/sample_dog.png --load-4bit
#   说明：HuggingFace 本地缓存通常位于 ~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<commit_id>；
#   请选择具体的 snapshots/<commit_id> 目录作为 --model-path（该目录需包含 tokenizer_config.json、config.json、model.safetensors 等）。
#
# 启动示例（在仓库根目录运行）：
# 1) 使用 LLaVA v1.6-7B（默认 GPU）：
#    python VisionZip/visionzip_cli.py --model-path liuhaotian/llava-v1.6-7b-hf --image-file VisionZip/sample_dog.png --dominant 54 --contextual 10
# 2) 指定 GPU：
#    CUDA_VISIBLE_DEVICES=0 python VisionZip/visionzip_cli.py --model-path liuhaotian/llava-v1.6-7b-hf --image-file /path/to/your_image.jpg
# 3) 低显存 4bit 量化：
#    python VisionZip/visionzip_cli.py --model-path liuhaotian/llava-v1.6-7b-hf --image-file VisionZip/sample_dog.png --load-4bit
# 4) CPU 运行（很慢，不推荐）：
#    python VisionZip/visionzip_cli.py --model-path liuhaotian/llava-v1.6-7b-hf --image-file VisionZip/sample_dog.png --device cpu
# 说明：
# - --model-path 可以是 HuggingFace 的模型名（如 liuhaotian/llava-v1.6-7b-hf），也可以是本地快照路径。
# - --dominant 与 --contextual 控制 VisionZip 的压缩强度，默认分别为 54 与 10。
# - 如果遇到推理显存压力，可加 --load-4bit 或 --load-8bit。

def safe_input(prompt: str) -> str:
    try:
        # If stdin is a TTY, normal interactive input
        if sys.stdin.isatty():
            return input(prompt)
        # If stdin is piped, read from /dev/tty to require an actual Enter
        else:
            try:
                with open('/dev/tty', 'r') as tty:
                    print(prompt, end='', flush=True)
                    line = tty.readline()
                    return '' if not line else line.rstrip('\n')
            except Exception:
                # Fallback: still try input(), but this may consume piped content
                return input(prompt)
    except EOFError:
        return ""


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    # If using local HF snapshot path, ensure model_name contains 'llava'
    if 'llava' not in model_name.lower():
        if 'liuhaotian--llava-v1.6-7b' in args.model_path:
            model_name = 'llava-v1.6-7b'
        elif 'liuhaotian--llava-v1.5-7b' in args.model_path:
            model_name = 'llava-v1.5-7b'
        elif 'llava' in args.model_path.lower():
            model_name = 'llava'
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device
    )


    # Inject VisionZip
    # 将 VisionZip 补丁注入到模型：
    # - 视觉塔会替换为 VisionZip 的 forward，实现 Dominant/Contextual 两阶段压缩；
    # - LLaVA 的 prepare_inputs_labels_for_multimodal 会改为 VisionZip 版本，支持 anyres 的 unpad 还原；
    # - 生成调用保持不变，图像特征经过压缩后再映射到 LLM 空间参与对齐与生成。
    model = visionzip(model, dominant=args.dominant, contextual=args.contextual)

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

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = safe_input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(keywords, tokenizer, input_ids)])

        # Timing: measure prefill until first chunk, then decoding
        t0 = time.perf_counter()
        prefill_time = None
        chunks = []

        def _generate():
            with torch.inference_mode():
                # VisionZip 生效点：generate 内部会触发多模态准备，
                # 其中图像侧的编码调用 VisionZip 的视觉塔 forward，
                # 压缩为 Dominant+Contextual tokens 并（在需要时）按 anyres 还原后插入到文本序列。
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

        thread = Thread(target=_generate)
        thread.start()
        for new_text in streamer:
            if prefill_time is None:
                prefill_time = time.perf_counter() - t0
            print(new_text, end="", flush=True)
            chunks.append(new_text)

        thread.join()
        print()
        outputs = ("".join(chunks)).strip()
        # Trim trailing stop string if present to avoid artifacts in history
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        conv.messages[-1][-1] = outputs

        total_time = time.perf_counter() - t0
        decoding_time = total_time - (prefill_time or 0.0)
        gen_token_ids = tokenizer(outputs).input_ids
        avg_decoding_time = decoding_time / max(len(gen_token_ids), 1)
        print(f"[Timing] total={total_time:.3f}s, prefill={prefill_time or 0.0:.3f}s, avg_decoding={avg_decoding_time:.4f}s/token")

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default = "/datadisk2/quyanyi/cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--dominant", type=int, default=54, help="VisionZip dominant tokens")
    parser.add_argument("--contextual", type=int, default=10, help="VisionZip contextual tokens")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)