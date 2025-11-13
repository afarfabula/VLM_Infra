#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import re

LLAVA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = LLAVA_DIR / "evaluate" / "results" / "vqav2_random10.jsonl"
DEFAULT_VQAV2 = LLAVA_DIR / "Benchmark" / "VQAv2"


def _canonicalize_device(device_str: str) -> str:
    """Normalize CUDA device index when CUDA_VISIBLE_DEVICES limits visibility.
    - If only one visible GPU, remap any non-zero index to cuda:0.
    - If requested index is out of visible range, fallback to cuda:0 with a warning.
    """
    try:
        if isinstance(device_str, str) and device_str.startswith('cuda'):
            if ':' in device_str:
                _, idx_str = device_str.split(':', 1)
                idx = int(idx_str)
                visible = torch.cuda.device_count()
                if visible <= 0:
                    return device_str
                if visible == 1 and idx != 0:
                    print(f"[提示] 仅 1 个可见 GPU，映射 {device_str} 为 cuda:0")
                    return 'cuda:0'
                if idx >= visible:
                    print(f"[警告] 设备索引 {idx} 超出可见范围 [0, {visible-1}]，改用 cuda:0")
                    return 'cuda:0'
    except Exception:
        pass
    return device_str


def _coco_val_name(image_id: int) -> str:
    return f"COCO_val2014_{image_id:012d}.jpg"


def _load_vqav2(q_root: Path):
    q_path = q_root / "Questions" / "v2_OpenEnded_mscoco_val2014_questions.json"
    a_path = q_root / "Annotations" / "v2_mscoco_val2014_annotations.json"
    img_dir = q_root / "Images" / "mscoco" / "val2014"
    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found: {q_path}")
    if not a_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {a_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {img_dir}")
    questions = json.load(open(q_path, 'r')).get('questions', [])
    anns = json.load(open(a_path, 'r')).get('annotations', [])
    ann_by_qid = {a['question_id']: a for a in anns}
    return questions, ann_by_qid, img_dir


def _pick_questions(questions, img_dir: Path, num_samples: int, seed: int):
    def has_img(rec):
        try:
            name = _coco_val_name(int(rec.get('image_id')))
        except Exception:
            return False
        return (img_dir / name).exists()
    valid = [q for q in questions if has_img(q)]
    if seed is not None and seed >= 0:
        random.seed(seed)
    if len(valid) < num_samples:
        raise RuntimeError(f"Not enough valid questions ({len(valid)}) for sampling {num_samples}")
    return random.sample(valid, num_samples)


def _majority_answer(answers):
    from collections import Counter
    if not answers:
        return ""
    counts = Counter([a.get('answer', '') for a in answers])
    return counts.most_common(1)[0][0]


def _write_jsonl_atomic(out_path: Path, records: list):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out_path)


def _pick_conv_mode(model_name: str) -> str:
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


def _generate_with_model(
    picks,
    ann_by_qid,
    img_dir: Path,
    model_path: str,
    model_base: Optional[str],
    device: str,
    load_8bit: bool,
    load_4bit: bool,
    temperature: float,
    max_new_tokens: int,
    dominant: int,
    contextual: int,
    batch_size: int = 1,
    use_flash_attn: bool = False,
    disable_visionzip: bool = False,
):
    import sys
    from PIL import Image
    repo_root = LLAVA_DIR.parent
    visionzip_dir = repo_root / "VisionZip"
    for p in [str(LLAVA_DIR), str(visionzip_dir)]:
        if p not in sys.path:
            sys.path.append(p)
    if disable_visionzip:
        def visionzip_func(model, dominant: int = 0, contextual: int = 0):
            print("[提示] 已禁用 VisionZip，使用原始模型进行生成。")
            return model
    else:
        try:
            from visionzip import visionzip as visionzip_func
        except Exception:
            def visionzip_func(model, dominant: int = 0, contextual: int = 0):
                print("[提示] 未找到 VisionZip，使用原始模型进行生成。")
                return model

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

    disable_torch_init()
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, load_8bit, load_4bit, device=device, use_flash_attn=use_flash_attn
    )
    model = visionzip_func(model, dominant=dominant, contextual=contextual)
    conv_mode = _pick_conv_mode(model_name)

    records = []
    # 固定停止字符串（所有样本的对话模板一致）
    _conv_tmp = conv_templates[conv_mode].copy()
    stop_str_const = _conv_tmp.sep if _conv_tmp.sep_style != SeparatorStyle.TWO else _conv_tmp.sep2
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pbar = tqdm(total=len(picks), desc='生成进度', unit='样本')
    for i in range(0, len(picks), max(1, batch_size)):
        batch = picks[i:i+max(1, batch_size)]
        # Build images and prompts for the batch
        images = []
        image_sizes = []
        prompts = []
        metas = []
        for rec in batch:
            qid = rec.get('question_id')
            qtext = rec.get('question')
            image_id = int(rec.get('image_id'))
            img_name = _coco_val_name(image_id)
            img_path = img_dir / img_name
            image = Image.open(str(img_path)).convert('RGB')
            images.append(image)
            image_sizes.append(image.size)

            conv = conv_templates[conv_mode].copy()
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qtext
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qtext
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
            metas.append((qid, image_id, str(img_path), qtext, conv))

        # Process images into tensor (supports anyres)
        image_tensor = process_images(images, image_processor, model.config)
        dtype = torch.float16 if model.device.type == 'cuda' else torch.float32
        if isinstance(image_tensor, list):
            image_tensor = [im.to(model.device, dtype=dtype) for im in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=dtype)

        # Tokenize prompts and pad
        id_list = [tokenizer_image_token(pmpt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for pmpt in prompts]
        input_lengths = [ids.shape[0] for ids in id_list]
        attn_masks = [torch.ones_like(ids, dtype=torch.long) for ids in id_list]
        batch_input_ids = pad_sequence(id_list, batch_first=True, padding_value=pad_id).to(model.device)
        batch_attention_mask = pad_sequence(attn_masks, batch_first=True, padding_value=0).to(model.device)

        stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria([stop_str_const], tokenizer, batch_input_ids)])
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=batch_input_ids,
                attention_mask=batch_attention_mask,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max(1, min(4, max_new_tokens)),
                use_cache=True,
                pad_token_id=pad_id,
                stopping_criteria=stopping_criteria,
            )
        # 显式同步，确保本批次推理完全结束后再进入后续处理/下一批
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        # 首次解码：完整序列，随后按停止符裁剪和清洗
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        cleaned = []
        invalid_idx = []
        for idx, ((qid, image_id, img_path, qtext, conv)) in enumerate(metas):
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            out = outputs[idx].strip()
            if stop_str and out.endswith(stop_str):
                out = out[:-len(stop_str)].strip()
            # 基本清洗：去掉前导空白
            out = out.lstrip()
            # 判空或近空（仅空白/标点/非字母数字）
            if (not out) or (not any(ch.isalnum() for ch in out)):
                invalid_idx.append(idx)
            cleaned.append(out)

        # 二次生成：对近空样本进行贪心、低温度、较长最小长度的再生成
        if len(invalid_idx) > 0:
            sub_ids = batch_input_ids[invalid_idx]
            sub_masks = batch_attention_mask[invalid_idx]
            if isinstance(image_tensor, torch.Tensor):
                sub_images = image_tensor[invalid_idx]
            else:
                sub_images = [image_tensor[j] for j in invalid_idx]
            sub_sizes = [image_sizes[j] for j in invalid_idx]
            with torch.inference_mode():
                sub_out_ids = model.generate(
                    inputs=sub_ids,
                    attention_mask=sub_masks,
                    images=sub_images,
                    image_sizes=sub_sizes,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=max(max_new_tokens, 32),
                    min_new_tokens=max(8, min(16, max_new_tokens)),
                    use_cache=True,
                    pad_token_id=pad_id,
                    stopping_criteria=stopping_criteria,
                )
            sub_decoded = tokenizer.batch_decode(sub_out_ids, skip_special_tokens=True)
            for k, idx in enumerate(invalid_idx):
                stop_str = metas[idx][4].sep if metas[idx][4].sep_style != SeparatorStyle.TWO else metas[idx][4].sep2
                out2 = sub_decoded[k].strip()
                if stop_str and out2.endswith(stop_str):
                    out2 = out2[:-len(stop_str)].strip()
                cleaned[idx] = out2 if out2 else cleaned[idx]

        # 写入记录
        for (qid, image_id, img_path, qtext, _conv), out in zip(metas, cleaned):
            records.append({
                'question_id': qid,
                'image_id': image_id,
                'image': img_path,
                'question': qtext,
                'pred': out,
                'source': 'model'
            })
        pbar.update(len(batch))
    pbar.close()
    return records


def _generate_with_gt_stub(picks, ann_by_qid, img_dir: Path):
    records = []
    for rec in picks:
        qid = rec.get('question_id')
        qtext = rec.get('question')
        image_id = int(rec.get('image_id'))
        img_name = _coco_val_name(image_id)
        img_path = img_dir / img_name
        ann = ann_by_qid.get(qid, {})
        pred = _majority_answer(ann.get('answers', []))
        records.append({
            'question_id': qid,
            'image_id': image_id,
            'image': str(img_path),
            'question': qtext,
            'pred': pred,
            'source': 'gt_stub'
        })
    return records


def _generate_with_random_stub(picks, ann_by_qid, img_dir: Path):
    records = []
    for rec in picks:
        qid = rec.get('question_id')
        qtext = rec.get('question')
        image_id = int(rec.get('image_id'))
        img_name = _coco_val_name(image_id)
        img_path = img_dir / img_name
        ann = ann_by_qid.get(qid, {})
        qt = (ann.get('question_type') or '').lower()
        at = (ann.get('answer_type') or '').lower()
        pred = 'unknown'
        if at == 'yes/no' or qt.startswith('is '):
            pred = random.choice(['yes', 'no'])
        elif at == 'number' or qt.startswith('how many'):
            pred = str(random.choice([0,1,2,3,4,5]))
        elif 'color' in qt:
            pred = random.choice(['black','white','red','green','blue'])
        records.append({
            'question_id': qid,
            'image_id': image_id,
            'image': str(img_path),
            'question': qtext,
            'pred': pred,
            'source': 'random_stub'
        })
    return records


def main():
    parser = argparse.ArgumentParser(description='Generate random VQAv2 JSONL predictions (with optional auto-scoring)')
    parser.add_argument('--vqav2-root', type=str, default=str(DEFAULT_VQAV2))
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUT))
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=-1, help='-1 for different random each run')
    parser.add_argument('--mode', type=str, default='gt_stub', choices=['gt_stub','random_stub','model'])
    # model options
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:4')
    parser.add_argument('--load-8bit', action='store_true')
    parser.add_argument('--load-4bit', action='store_true')
    parser.add_argument('--quant', type=str, choices=['4bit','8bit','fp16'], default='4bit', help='权重量化选项，默认4bit')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--dominant', type=int, default=54)
    parser.add_argument('--contextual', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1, help='并行生成的 batch size')
    parser.add_argument('--use-flash-attn', action='store_true', help='启用 FlashAttention-2（需已安装）')
    parser.add_argument('--disable-visionzip', action='store_true', help='禁用 VisionZip 加成，降低显存占用')
    # auto score options
    parser.add_argument('--auto-score', action='store_true', help='生成结束后自动对结果进行VQAv2软准确率评分')
    parser.add_argument('--annotations-json', type=str, default=None, help='可选，显式指定 v2_mscoco_val2014_annotations.json 路径')
    parser.add_argument('--score-output', type=str, default=None, help='可选，评分详情输出路径（默认在 results 同名加 _scored.json）')
    # output alias for compatibility
    parser.add_argument('--out', dest='output', type=str, default=None, help='预测输出文件路径（--output 的别名）')
    args = parser.parse_args()
    # Normalize device to avoid invalid ordinal when CUDA_VISIBLE_DEVICES is set
    args.device = _canonicalize_device(args.device)
    # Apply quantization preference
    if getattr(args, 'quant', None):
        if args.quant == '4bit':
            args.load_4bit = True
            args.load_8bit = False
        elif args.quant == '8bit':
            args.load_8bit = True
            args.load_4bit = False
        else:
            args.load_8bit = False
            args.load_4bit = False
    # Normalize device to avoid invalid ordinal when CUDA_VISIBLE_DEVICES is set
    args.device = _canonicalize_device(args.device)

    vqav2_root = Path(args.vqav2_root)
    out_path = Path(args.output)
    print(f"将生成 {args.num_samples} 条样本到: {out_path}")
    print(f"VQAv2 根目录: {vqav2_root} | 模式: {args.mode}")

    questions, ann_by_qid, img_dir = _load_vqav2(vqav2_root)
    picks = _pick_questions(questions, img_dir, args.num_samples, args.seed)

    if args.mode == 'model':
        quant_desc = '4bit' if args.load_4bit else ('8bit' if args.load_8bit else 'fp16/bf16')
        print(f"加载模型: {args.model_path} | 设备: {args.device} | 量化: {quant_desc} | 温度: {args.temperature} | max_new_tokens: {args.max_new_tokens} | batch_size: {args.batch_size}")
        records = _generate_with_model(
            picks, ann_by_qid, img_dir,
            model_path=args.model_path,
            model_base=args.model_base,
            device=args.device,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            dominant=args.dominant,
            contextual=args.contextual,
            batch_size=args.batch_size,
            use_flash_attn=args.use_flash_attn,
            disable_visionzip=args.disable_visionzip,
        )
    elif args.mode == 'random_stub':
        records = _generate_with_random_stub(picks, ann_by_qid, img_dir)
    else:
        records = _generate_with_gt_stub(picks, ann_by_qid, img_dir)

    _write_jsonl_atomic(out_path, records)
    print(f"已写入: {out_path} ({len(records)} 行)")
    print("示例记录:")
    for r in records[:3]:
        print(json.dumps(r, ensure_ascii=False))

    # Auto scoring: call score_vqav2_subset.py with correct flags
    if args.auto_score:
        score_script = LLAVA_DIR / 'evaluate' / 'score_vqav2_subset.py'
        # Resolve annotations path: prefer CLI override, else under vqav2_root
        if args.annotations_json:
            ann_path = Path(args.annotations_json)
        else:
            ann_path = vqav2_root / 'Annotations' / 'v2_mscoco_val2014_annotations.json'
        if not ann_path.exists():
            print(f"[警告] 未找到标注文件: {ann_path}，跳过自动评分。可使用 --annotations-json 显式指定路径。")
            return
        # Default score output path
        if args.score_output:
            score_out = Path(args.score_output)
        else:
            score_out = out_path.with_name(out_path.stem + '_scored.json')
        cmd = [
            'python', str(score_script),
            '--pred-jsonl', str(out_path),
            '--annotations-json', str(ann_path),
            '--output', str(score_out)
        ]
        print(f"开始自动评分…\n命令: {' '.join(cmd)}")
        import subprocess
        try:
            res = subprocess.run(cmd, cwd=str(LLAVA_DIR.parent), check=True, capture_output=True, text=True)
            # Show summarized stdout from scorer
            print(res.stdout)
            print(f"[完成] 自动评分写入: {score_out}")
        except subprocess.CalledProcessError as e:
            print("[错误] 自动评分失败:")
            print(e.stdout or '')
            print(e.stderr or '')
        except Exception as e:
            print(f"[错误] 自动评分异常: {e}")


if __name__ == '__main__':
    main()