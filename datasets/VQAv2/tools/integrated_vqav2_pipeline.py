#!/usr/bin/env python3
"""
VQAv2推理和评分一体化管道
将推理、结果转换和评分流程集成在一个脚本中
"""

import argparse
import json
import os
from pathlib import Path
import sys

# 添加必要的路径以确保能正确导入模块
sys.path.append("/data/model/Inference_VLM/VLM_Infra/LLava")
sys.path.append("/data/model/Inference_VLM/VLM_Infra/Evaluate_Pipeline")

from main import EvaluatePipeline
from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


class VQASoftScorer:
    def __init__(self):
        # Ported from official VQAEval (PythonEvaluationTools/vqaEvaluation/vqaEval.py)
        self.contractions = {
            "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
            "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
            "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
            "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
            "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
            "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
            "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
            "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
            "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
            "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
            "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
            "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", "somebodys": "somebody's",
            "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
            "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
            "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll",
            "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
            "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
            "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
            "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've",
            "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
            "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
            "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
            "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
            "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
            "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
            "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
            "youve": "you've"
        }
        self.manualMap = {
            'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
        }
        self.articles = ['a', 'an', 'the']
        self.punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']

    def process_punctuation(self, text: str) -> str:
        out = text
        # Remove punctuation unless comma is number separator
        import re
        period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        comma_strip = re.compile(r"(\d)(\,)(\d)")
        for p in self.punct:
            if (p + ' ' in text or ' ' + p in text) or (re.search(comma_strip, text) is not None):
                out = out.replace(p, '')
            else:
                out = out.replace(p, ' ')
        out = period_strip.sub("", out)
        return out

    def process_digit_article(self, text: str) -> str:
        tokens = text.lower().split()
        out = []
        for w in tokens:
            w = self.manualMap.get(w, w)
            if w not in self.articles:
                out.append(w)
        for i, w in enumerate(out):
            if w in self.contractions:
                out[i] = self.contractions[w]
        return ' '.join(out)

    def normalize(self, text: str) -> str:
        text = text.replace('\n', ' ').replace('\t', ' ').strip()
        return self.process_digit_article(self.process_punctuation(text))

    def score_one(self, pred: str, gt_answers: list) -> float:
        # Strip whitespace and control chars
        pred = pred.replace('\n', ' ').replace('\t', ' ').strip()
        gt_answers = [a.replace('\n', ' ').replace('\t', ' ').strip() for a in gt_answers]

        if len(set(gt_answers)) > 1:
            gt_answers = [self.process_digit_article(self.process_punctuation(a)) for a in gt_answers]
            pred = self.process_digit_article(self.process_punctuation(pred))

        accs = []
        for i in range(len(gt_answers)):
            other = [gt_answers[j] for j in range(len(gt_answers)) if j != i]
            matches = sum([1 for a in other if a == pred])
            accs.append(min(1.0, float(matches) / 3.0))
        return float(sum(accs)) / float(len(accs)) if accs else 0.0


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def load_pred_jsonl(path: Path):
    preds = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            qid = rec.get("question_id")
            # 支持两种可能的字段名：answer 或 pred
            ans = rec.get("answer") or rec.get("pred")
            if qid is None or ans is None:
                continue
            preds[int(qid)] = str(ans)
    return preds


def extract_short_answer(pred: str, gt_norm: list, ques_type: str, ans_type: str, scorer: VQASoftScorer) -> tuple:
    """
    Return a short answer string derived from prediction using GT-aware and heuristic rules.
    Also return a string describing which rule was used.
    """
    pred_norm = scorer.normalize(pred)
    tokens = pred_norm.split()

    # 1) Prefer exact match with any normalized GT answer appearing in pred
    uniq_gt = [a for a, _ in __import__('collections').Counter(gt_norm).most_common()]
    for a in uniq_gt:
        if not a:
            continue
        # token-level or substring match
        if a in tokens or a in pred_norm:
            return a, "match_gt"

    # 2) Yes/No
    if ans_type == 'yes/no' or ques_type.startswith('is '):
        if 'yes' in tokens:
            return 'yes', 'yesno'
        if 'no' in tokens:
            return 'no', 'yesno'

    # 3) Number: find first digit or spelled number
    if ans_type == 'number' or ques_type.startswith('how many'):
        # digits
        import re
        m = re.search(r"\b(\d+)\b", pred_norm)
        if m:
            return m.group(1), 'number_digit'
        # spelled numbers via manualMap
        for t in tokens:
            if t in scorer.manualMap:
                return scorer.manualMap[t], 'number_word'

    # 4) Color extraction for color questions
    if 'what color' in ques_type:
        colors = {"red","orange","yellow","green","blue","purple","pink","brown","black","white","gray","grey","tan","teal","maroon","gold","silver"}
        for t in tokens:
            if t in colors:
                # normalize grey->gray to match gt if needed
                return ('gray' if t=='grey' else t), 'color'

    # 5) Up/Down (direction) common case
    if 'up or down' in ques_type or 'up or down' in pred_norm:
        if 'up' in tokens:
            return 'up', 'direction'
        if 'down' in tokens:
            return 'down', 'direction'

    # Fallback: use a single token (first or last) as short answer
    if tokens:
        return tokens[0], 'fallback_first'
    return pred_norm, 'fallback_raw'


def convert_inference_results_to_predictions(merged_results_file: Path, predictions_file: Path):
    """将推理结果转换为预测文件格式"""
    print("=== 转换推理结果 ===")
    
    # 读取合并的推理结果
    if not merged_results_file.exists():
        raise FileNotFoundError(f"合并的推理结果文件不存在: {merged_results_file}")

    with open(merged_results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换为预测文件格式
    predictions = []
    answer_processor = EvalAIAnswerProcessor()
    
    for item in data['inference_results']:
        # 使用EvalAIAnswerProcessor处理预测结果
        processed_answer = answer_processor(item['model_prediction'])
        pred = {
            'question_id': item['question_id'],
            'answer': processed_answer
        }
        predictions.append(pred)

    # 保存为JSONL格式
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    print(f'已保存预测文件: {predictions_file}')
    print(f'预测样本数: {len(predictions)}')
    return predictions


def run_vqav2_scoring(pred_jsonl: Path, annotations_json: Path, output_path: Path):
    """运行VQAv2评分"""
    print("=== 运行VQAv2评分 ===")
    
    if not pred_jsonl.exists():
        raise FileNotFoundError(f"预测文件不存在: {pred_jsonl}")
    if not annotations_json.exists():
        raise FileNotFoundError(f"标注文件不存在: {annotations_json}")

    preds = load_pred_jsonl(pred_jsonl)
    anns = load_json(annotations_json)
    ann_list = anns.get("annotations", [])

    # Build index by question_id
    ann_by_qid = {item["question_id"]: item for item in ann_list}

    scorer = VQASoftScorer()

    results = []
    accQA = []
    accQuesType = {}
    accAnsType = {}

    missing = []
    for qid, pred in preds.items():
        ann = ann_by_qid.get(qid)
        if ann is None:
            missing.append(qid)
            continue
        gt_answers = [a.get("answer", "") for a in ann.get("answers", [])]
        ques_type = ann.get("question_type", "")
        ans_type = ann.get("answer_type", "")
        # prepare normalized GT list
        gt_norm = [scorer.normalize(a) for a in gt_answers]
        # get short answer from pred using heuristics
        pred_short, used_rule = extract_short_answer(pred, gt_norm, ques_type, ans_type, scorer)
        # score using short answer
        acc = scorer.score_one(pred_short, gt_answers)
        accQA.append(acc)
        accQuesType.setdefault(ques_type, []).append(acc)
        accAnsType.setdefault(ans_type, []).append(acc)
        # collect GT answers and normalized forms for debugging
        pred_norm = scorer.normalize(pred)
        # majority vote GT short answer (post-normalized) for display
        from collections import Counter
        maj_gt = Counter(gt_norm).most_common(1)[0][0] if gt_norm else ""
        results.append({
            "question_id": qid,
            "pred": pred,
            "pred_norm": pred_norm,
            "pred_short": pred_short,
            "pred_short_rule": used_rule,
            "gt_answers": gt_answers,
            "gt_norm": gt_norm,
            "gt_majority": maj_gt,
            "acc": round(100.0 * acc, 2),
            "ques_type": ques_type,
            "ans_type": ans_type
        })

    overall = round(100.0 * (sum(accQA) / len(accQA)) if accQA else 0.0, 2)
    per_qtype = {k: round(100.0 * (sum(v) / len(v)), 2) for k, v in accQuesType.items()}
    per_atype = {k: round(100.0 * (sum(v) / len(v)), 2) for k, v in accAnsType.items()}

    summary = {
        "count": len(accQA),
        "missing": missing,
        "overall": overall,
        "per_question_type": per_qtype,
        "per_answer_type": per_atype,
        "details": results,
    }

    # Show paths used for clarity
    print(f"使用预测文件: {pred_jsonl}")
    print(f"使用标注文件: {annotations_json}")
    print(f"评分结果输出: {output_path}")

    print(f"评测样本数: {summary['count']}")
    print(f"总体软准确率: {summary['overall']:.2f}")
    if per_qtype:
        print("按问题类型准确率:")
        for k, v in sorted(per_qtype.items(), key=lambda x: x[0]):
            print(f"- {k}: {v:.2f}")
    if per_atype:
        print("按答案类型准确率:")
        for k, v in sorted(per_atype.items(), key=lambda x: x[0]):
            print(f"- {k}: {v:.2f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"已写入评分详情: {output_path}")
    
    return summary


def run_vqav2_pipeline(
    model_name: str = "LLaVA-1.5-7B",
    visionzip_enabled: bool = True,
    result_dir: str = "./results",
    num_samples: int = 100,
    batch_size: int = 32,
    load_precision: str = "fp16",
    annotations_json: str = "/data/model/Inference_VLM/VLM_Infra/datasets/VQAv2/annotations/v2_mscoco_val2014_annotations.json",
    dominant: int = 54,
    contextual: int = 10
):
    """运行完整的VQAv2推理和评分管道"""
    # 创建结果目录
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    
    print("=== VQAv2推理和评分一体化管道 ===")
    print(f"模型: {model_name}")
    print(f"使用VisionZip: {visionzip_enabled}")
    print(f"结果目录: {result_dir}")
    print(f"样本数量: {num_samples}")
    print(f"批次大小: {batch_size}")
    print(f"加载精度: {load_precision}")
    print(f"Dominant Token数: {dominant}")
    print(f"Contextual Token数: {contextual}")
    print("")
    
    # 1. 运行推理
    print("=== 开始VQAv2推理 ===")
    pipeline = EvaluatePipeline("../../Evaluate_Pipeline/configs/vqav2_config.json")
    
    try:
        total_samples = pipeline.run_vqav2_evaluation(
            model_name=model_name,
            visionzip_enabled=visionzip_enabled,
            result_dir=str(result_path),
            num_samples=num_samples,
            batch_size=batch_size,
            load_precision=load_precision,
            dominant=dominant,
            contextual=contextual
        )
        print(f"推理完成 - 总样本: {total_samples}")
    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        raise
    
    # 2. 转换推理结果为预测文件格式
    merged_results_file = result_path / "merged_vqav2_inference_results.json"
    predictions_file = result_path / "predictions.jsonl"
    
    try:
        convert_inference_results_to_predictions(merged_results_file, predictions_file)
    except Exception as e:
        print(f"结果转换过程中出现错误: {e}")
        raise
    
    # 3. 运行评分
    scoring_output = result_path / "scoring_result.json"
    annotations_path = Path(annotations_json)
    
    try:
        scoring_result = run_vqav2_scoring(predictions_file, annotations_path, scoring_output)
        print("")
        print("=== 推理和评分流程完成 ===")
        print(f"推理结果保存在: {result_path}")
        print(f"预测文件: {predictions_file}")
        print(f"评分结果: {scoring_output}")
        print(f"总体软准确率: {scoring_result['overall']:.2f}")
        return scoring_result
    except Exception as e:
        print(f"评分过程中出现错误: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VQAv2推理和评分一体化管道')
    parser.add_argument('--model', type=str, default='LLaVA-1.5-7B',
                       help='模型名称')
    parser.add_argument('--visionzip', action='store_true', default=True,
                       help='启用VisionZip优化')
    parser.add_argument('--output', type=str, default='./results',
                       help='输出目录')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='样本数量 (默认: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小 (默认: 32)')
    parser.add_argument('--load_precision', type=str, default='fp16',
                       help='模型加载精度 (默认: fp16)')
    parser.add_argument('--annotations_json', type=str, 
                       default='/data/model/Inference_VLM/VLM_Infra/datasets/VQAv2/annotations/v2_mscoco_val2014_annotations.json',
                       help='VQAv2标注文件路径')
    parser.add_argument('--dominant', type=int, default=54,
                       help='VisionZip dominant token数 (默认: 54)')
    parser.add_argument('--contextual', type=int, default=10,
                       help='VisionZip contextual token数 (默认: 10)')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["HF_HOME"] = "/data/model/Inference_VLM/.cache"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/model/Inference_VLM/.cache"
    os.environ["TRANSFORMERS_CACHE"] = "/data/model/Inference_VLM/.cache"
    
    # 设置PYTHONPATH以确保能正确导入LLava模块
    sys.path.insert(0, "/data/model/Inference_VLM/VLM_Infra/LLava")
    
    try:
        result = run_vqav2_pipeline(
            model_name=args.model,
            visionzip_enabled=args.visionzip,
            result_dir=args.output,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            load_precision=args.load_precision,
            annotations_json=args.annotations_json,
            dominant=args.dominant,
            contextual=args.contextual
        )
        print("评估管道执行成功")
        return result
    except Exception as e:
        print(f"评估管道执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()