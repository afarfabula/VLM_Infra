#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image

# Make sure we can import local packages (llava, VisionZip)
_THIS_DIR = Path(__file__).resolve().parent
_LLAVA_DIR = _THIS_DIR.parent
_REPO_ROOT = _LLAVA_DIR.parent
_VISIONZIP_DIR = _REPO_ROOT / "VisionZip"
import sys
for p in [str(_LLAVA_DIR), str(_VISIONZIP_DIR)]:
    if p not in sys.path:
        sys.path.append(p)

from visionzip import visionzip  # noqa: E402
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
from transformers import StoppingCriteriaList


def load_image(image_file: str) -> Image.Image:
    img = Image.open(image_file).convert("RGB")
    return img


def coco_val_name(image_id: int) -> str:
    return f"COCO_val2014_{image_id:012d}.jpg"


def pick_conv_mode(model_name: str) -> str:
    name = model_name.lower()
    if "llama-2" in name:
        return "llava_llama_2"
    elif "mistral" in name:
        return "mistral_instruct"
    elif "v1.6-34b" in name:
        return "chatml_direct"
    elif "v1" in name:
        return "llava_v1"
    elif "mpt" in name:
        return "mpt"
    else:
        return "llava_v0"


def main():
    parser = argparse.ArgumentParser(description="Randomly sample VQAv2 val questions and evaluate with VisionZip LLaVA")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dominant", type=int, default=54, help="VisionZip dominant tokens")
    parser.add_argument("--contextual", type=int, default=10, help="VisionZip contextual tokens")
    parser.add_argument("--output", type=str, default=str(_LLAVA_DIR / "evaluate" / "results" / "vqav2_random10.jsonl"))
    parser.add_argument("--vqav2-root", type=str, default=str(_LLAVA_DIR / "Benchmark" / "VQAv2"))
    args = parser.parse_args()

    random.seed(args.seed)

    vqa_root = Path(args.vqav2_root)
    q_path = vqa_root / "Questions" / "v2_OpenEnded_mscoco_val2014_questions.json"
    a_path = vqa_root / "Annotations" / "v2_mscoco_val2014_annotations.json"  # optional
    img_dir = vqa_root / "Images" / "mscoco" / "val2014"

    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found: {q_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {img_dir}")

    with open(q_path, "r") as f:
        q_data = json.load(f)
    questions = q_data.get("questions", [])
    print(f"Loaded questions: {len(questions)}")

    # Filter to those with existing images
    def has_img(rec):
        iid = rec.get("image_id")
        try:
            name = coco_val_name(int(iid))
        except Exception:
            return False
        return (img_dir / name).exists()

    valid_qs = [rec for rec in questions if has_img(rec)]
    print(f"Valid questions with images: {len(valid_qs)}")

    if len(valid_qs) < args.num_samples:
        raise RuntimeError("Not enough valid questions to sample.")

    picks = random.sample(valid_qs, args.num_samples)

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
    model = visionzip(model, dominant=args.dominant, contextual=args.contextual)

    conv_mode = pick_conv_mode(model_name)
    out_dir = Path(args.output).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)

    print(f"Writing results to: {out_path}")
    with open(out_path, "w") as fout:
        for rec in picks:
            image_id = int(rec["image_id"])  # type: ignore
            qid = rec.get("question_id")
            qtext = rec.get("question")
            img_name = coco_val_name(image_id)
            img_path = img_dir / img_name

            # Prepare image
            image = load_image(str(img_path))
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [im.to(model.device, dtype=torch.float16) for im in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            # Compose prompt
            conv = conv_templates[conv_mode].copy()
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qtext
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qtext
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(model.device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            # Wrap custom stopping criteria into StoppingCriteriaList to satisfy HF generate API
            stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)])

            t0 = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                )
            gen_time = time.perf_counter() - t0
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]

            record = {
                "question_id": qid,
                "image_id": image_id,
                "image": str(img_path),
                "question": qtext,
                "pred": outputs,
                "time_sec": round(gen_time, 4),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"qid={qid} -> {outputs}")

    print("Done.")


if __name__ == "__main__":
    main()