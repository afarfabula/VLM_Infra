import os
import os.path as osp
import json
import argparse
import pandas as pd
import re

# ---------------------------
# Minimal VQAv2 soft accuracy
# ---------------------------
def process_punctuation(inText):
    outText = inText
    punct = [';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
    commaStrip  = re.compile(r'(\d)(,)(\d)')
    periodStrip = re.compile(r'(?<!\d)\.(?!\d)')
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText)
    return outText


def _process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    articles = ['a', 'an', 'the']
    manualMap = {
        'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    contractions = {
        'aint': "ain't", 'arent': "aren't", 'cant': "can't", 'couldve': "could've",
        'couldnt': "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
        'didnt': "didn't", 'doesnt': "doesn't", 'dont': "don't", 'hadnt': "hadn't",
        "hadnt've": "hadn't've", "hadn'tve": "hadn't've", 'hasnt': "hasn't",
        'havent': "haven't", 'hed': "he'd", "hed've": "he'd've", "he'dve": "he'd've",
        'hes': "he's", 'howd': "how'd", 'howll': "how'll", 'hows': "how's",
        "Id've": "I'd've", "I'dve": "I'd've", 'Im': "I'm", 'Ive': "I've", 'isnt': "isn't",
        'itd': "it'd", "itd've": "it'd've", "it'dve": "it'd've", 'itll': "it'll",
        "let's": "let's", 'maam': "ma'am", 'mightnt': "mightn't", "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've", 'mightve': "might've", 'mustnt': "mustn't", 'mustve': "must've",
        'neednt': "needn't", 'notve': "not've", 'oclock': "o'clock", 'oughtnt': "oughtn't",
        "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", 'shant': "shan't",
        "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", 'shouldve': "should've",
        'shouldnt': "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
        "somebody'd": 'somebodyd', "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
        'somebodyll': "somebody'll", 'somebodys': "somebody's", 'someoned': "someone'd",
        "someoned've": "someone'd've", "someone'dve": "someone'd've", 'someonell': "someone'll",
        'someones': "someone's", 'somethingd': "something'd", "somethingd've": "something'd've",
        "something'dve": "something'd've", 'somethingll': "something'll", 'thats': "that's",
        'thered': "there'd", "thered've": "there'd've", "there'dve": "there'd've",
        'therere': "there're", 'theres': "there's", 'theyd': "they'd", "theyd've": "they'd've",
        "they'dve": "they'd've", 'theyll': "they'll", 'theyre': "they're", 'theyve': "they've",
        'twas': "'twas", 'wasnt': "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
        'weve': "we've", 'werent': "weren't", 'whatll': "what'll", 'whatre': "what're",
        'whats': "what's", 'whatve': "what've", 'whens': "when's", 'whered': "where'd",
        'wheres': "where's", 'whereve': "where've", 'whod': "who'd", "whod've": "who'd've",
        "who'dve": "who'd've", 'wholl': "who'll", 'whos': "who's", 'whove': "who've",
        'whyll': "why'll", 'whyre': "why're", 'whys': "why's", 'wont': "won't",
        'wouldve': "would've", 'wouldnt': "wouldn't", "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've", 'yall': "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
        "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
        'youd': "you'd", "youd've": "you'd've", "you'dve": "you'd've", 'youll': "you'll",
        'youre': "you're", 'youve': "you've",
    }
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for i, w in enumerate(outText):
        if w in contractions:
            outText[i] = contractions[w]
    return ' '.join(outText)


def process_answer(answer):
    answer = answer.replace('\n', ' ').replace('\t', ' ').strip()
    answer = process_punctuation(answer)
    answer = _process_digit_article(answer)
    return answer


def process_line_vqa(line):
    # Expect line to have keys 'answer' (list of gt answers) and 'prediction' (str)
    answers = line['answer'] if isinstance(line['answer'], list) else [line['answer']]
    gt = [process_answer(x) for x in answers]
    pred = process_answer(str(line.get('prediction', '')))
    match = []
    for current_idx, gtAns in enumerate(gt):
        otherGTAns = [item for j, item in enumerate(gt) if j != current_idx]
        matchingAns = [item for item in otherGTAns if item == pred]
        acc = min(1, float(len(matchingAns)) / 3)
        match.append(acc)
    return {'gt': gt, 'pred': pred, 'match': match}


def hit_calculate_vqa(result):
    # Default for VQAv2: mean of match
    return [float(pd.Series(x['match']).mean()) for x in result]


def read_jsonl(pth):
    rows = []
    with open(pth, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_pred_row(r):
    # Accept keys: pred/prediction, answer/answers, question_id/qid
    pred = r.get('pred', r.get('prediction', ''))
    ans = r.get('answers', r.get('answer', None))
    qid = r.get('question_id', r.get('qid', None))
    return dict(prediction=pred, answer=ans if ans is not None else '', question_id=qid)


def load_vqa_annotations(annot_path):
    # Official VQAv2 annotations json
    with open(annot_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Flatten to {question_id: [ans1,...,ans10]}
    ann_map = {}
    for item in data.get('annotations', data):
        qid = item['question_id']
        if 'answers' in item:
            ann_map[qid] = [a['answer'] for a in item['answers']]
        else:
            # fallback single answer
            ann_map[qid] = [item.get('multiple_choice_answer', item.get('answer', ''))]
    return ann_map


def score(pred_rows, ann_map):
    # Build dataframe with prediction, answer, question_id
    norm = [normalize_pred_row(r) for r in pred_rows]
    df = pd.DataFrame(norm)
    # Align answers using question_id
    def _to_int(x):
        try:
            return int(x)
        except Exception:
            return None
    if 'question_id' in df.columns:
        qids = df['question_id'].tolist()
        df['answer'] = [ann_map.get(_to_int(qid), ['']) if (qid is not None and pd.notna(qid) and _to_int(qid) in ann_map) else [''] for qid in qids]
    # Compute vqa soft accuracy per row
    result = [process_line_vqa(row) for _, row in df.iterrows()]
    # Overall soft accuracy (mean of hit_calculate)
    scores = hit_calculate_vqa(result)
    overall = float(pd.Series(scores).mean()) * 100.0
    return overall, result, df


def main():
    parser = argparse.ArgumentParser(description='Score VisionZip predictions on VQAv2 (soft accuracy).')
    parser.add_argument('--pred-jsonl', type=str, required=True, help='Prediction jsonl file from VisionZip.')
    parser.add_argument('--vqa-anno', type=str, required=True, help='Path to VQAv2 annotations json (val2014).')
    parser.add_argument('--out-json', type=str, default=None, help='Output summary json path.')
    args = parser.parse_args()

    assert osp.exists(args.pred_jsonl), f'Prediction file not found: {args.pred_jsonl}'
    assert osp.exists(args.vqa_anno), f'Annotation file not found: {args.vqa_anno}'

    pred_rows = read_jsonl(args.pred_jsonl)
    ann_map = load_vqa_annotations(args.vqa_anno)
    overall, result, df = score(pred_rows, ann_map)

    summary = {
        'num_samples': len(df),
        'overall_soft_accuracy': overall,
        'pred_file': osp.realpath(args.pred_jsonl),
        'anno_file': osp.realpath(args.vqa_anno),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out_json is not None:
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()