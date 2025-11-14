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
# conda activate LLava && cd /data/model/Inference_VLM/VLM_Infra/VisionZip
# HF_HOME=/data/model/Inference_VLM/.cache HUGGINGFACE_HUB_CACHE=/data/model/Inference_VLM/.cache TRANSFORMERS_CACHE=/data/model/Inference_VLM/.cache python visionzip_cli.py --model-path /data/model/Inference_VLM/models-LLava-1.5-7B --image-file /data/model/Inference_VLM/sample_dog.png --load-4bit --max-new-tokens 512

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