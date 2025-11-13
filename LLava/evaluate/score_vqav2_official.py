#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
LLAVA_DIR = THIS_DIR.parent
VQAV2_DIR = LLAVA_DIR / "Benchmark" / "VQAv2"


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
            ans = rec.get("pred", rec.get("text"))
            if qid is None or ans is None:
                continue
            preds[int(qid)] = str(ans)
    return preds


def import_official_modules(vqav2_root: Path):
    vqa_tools = vqav2_root / "PythonHelperTools" / "vqaTools"
    vqa_eval_parent = vqav2_root / "PythonEvaluationTools"
    for p in [str(vqa_tools), str(vqa_eval_parent)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from vqa import VQA  # type: ignore
    from vqaEvaluation.vqaEval import VQAEval  # type: ignore
    return VQA, VQAEval


def main():
    parser = argparse.ArgumentParser(description="Score VQAv2 predictions using official VQA/VQAEval on a subset")
    parser.add_argument("--pred-jsonl", type=str, required=True)
    parser.add_argument("--vqav2-root", type=str, default=str(VQAV2_DIR))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    pred_path = Path(args.pred_jsonl)
    vqav2_root = Path(args.vqav2_root)
    q_path = vqav2_root / "Questions" / "v2_OpenEnded_mscoco_val2014_questions.json"
    a_path = vqav2_root / "Annotations" / "v2_mscoco_val2014_annotations.json"

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    if not q_path.exists():
        raise FileNotFoundError(f"Questions JSON not found: {q_path}")
    if not a_path.exists():
        raise FileNotFoundError(f"Annotations JSON not found: {a_path}")

    VQA, VQAEval = import_official_modules(vqav2_root)

    preds = load_pred_jsonl(pred_path)

    # Build official VQA objects
    vqa = VQA(str(a_path), str(q_path))
    res = VQA()

    import copy as _copy
    res.questions = json.load(open(str(q_path), 'r'))
    res.dataset['info'] = _copy.deepcopy(vqa.questions['info'])
    res.dataset['task_type'] = _copy.deepcopy(vqa.questions['task_type'])
    res.dataset['data_type'] = _copy.deepcopy(vqa.questions['data_type'])
    res.dataset['data_subtype'] = _copy.deepcopy(vqa.questions['data_subtype'])
    res.dataset['license'] = _copy.deepcopy(vqa.questions['license'])

    # Fill annotations with our subset predictions (bypass loadRes' full-coverage assert)
    anns = []
    for qid, ans in preds.items():
        qa = vqa.qa[int(qid)]
        anns.append({
            'question_id': int(qid),
            'answer': ans,
            'image_id': qa['image_id'],
            'question_type': qa['question_type'],
            'answer_type': qa['answer_type'],
        })
    res.dataset['annotations'] = anns
    res.createIndex()

    vqaEval = VQAEval(vqa, res, n=2)
    vqaEval.evaluate(quesIds=list(preds.keys()))

    overall = vqaEval.accuracy.get('overall', 0.0)
    per_qtype = vqaEval.accuracy.get('perQuestionType', {})
    per_atype = vqaEval.accuracy.get('perAnswerType', {})

    print(f"评测样本数: {len(preds)}")
    print(f"总体软准确率: {overall:.2f}")
    if per_qtype:
        print("按问题类型准确率:")
        for k in sorted(per_qtype.keys()):
            print(f"- {k}: {per_qtype[k]:.2f}")
    if per_atype:
        print("按答案类型准确率:")
        for k in sorted(per_atype.keys()):
            print(f"- {k}: {per_atype[k]:.2f}")

    if args.output:
        out = {
            'count': len(preds),
            'overall': overall,
            'per_question_type': per_qtype,
            'per_answer_type': per_atype,
            'used_official': True,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"已写入评分摘要: {out_path}")


if __name__ == "__main__":
    main()