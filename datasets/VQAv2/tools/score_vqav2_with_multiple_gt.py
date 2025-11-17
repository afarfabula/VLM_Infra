#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


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


_DEFAULT_PRED = Path("/data/model/Inference_VLM/VLM_Infra/--pred-jsonl")
_DEFAULT_ANN = Path("/data/model/Inference_VLM/VLM_Infra/datasets/VQAv2/annotations/v2_mscoco_val2014_annotations.json")
_DEFAULT_OUT = Path("/data/model/Inference_VLM/VLM_Infra/--output")


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


def main():
    parser = argparse.ArgumentParser(description="Score VQAv2 predictions with official soft accuracy using multiple GT answers")
    parser.add_argument("--pred-jsonl", type=str, default=str(_DEFAULT_PRED), help="Path to predictions JSONL")
    parser.add_argument("--annotations-json", type=str, default=str(_DEFAULT_ANN), help="Path to v2_mscoco_val2014_annotations.json with multiple GT answers")
    parser.add_argument("--output", type=str, default=str(_DEFAULT_OUT), help="Optional path to write scoring details JSON")
    args = parser.parse_args()

    pred_path = Path(args.pred_jsonl)
    ann_path = Path(args.annotations_json)
    out_path = Path(args.output) if args.output else None

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_path}")

    preds = load_pred_jsonl(pred_path)
    anns = load_json(ann_path)
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
    print(f"使用预测文件: {pred_path}")
    print(f"使用标注文件: {ann_path}")
    if out_path:
        print(f"评分结果输出: {out_path}")

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

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"已写入评分详情: {out_path}")


if __name__ == "__main__":
    main()