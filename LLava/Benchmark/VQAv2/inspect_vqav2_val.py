#!/usr/bin/env python3
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
QUESTIONS_DIR = BASE / "Questions"
ANNOTATIONS_DIR = BASE / "Annotations"
IMAGES_DIR = BASE / "Images" / "mscoco" / "val2014"

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    print("[Paths]")
    print(f"Questions dir: {QUESTIONS_DIR}")
    print(f"Annotations dir: {ANNOTATIONS_DIR}")
    print(f"Images dir: {IMAGES_DIR}")

    # Detect question and annotation files
    q_files = sorted(QUESTIONS_DIR.glob("*val2014*.json"))
    a_files = sorted(ANNOTATIONS_DIR.glob("*val2014*.json"))
    print(f"Found questions JSON: {[p.name for p in q_files]}")
    print(f"Found annotations JSON: {[p.name for p in a_files]}")

    # Basic stats for questions
    if q_files:
        q = load_json(q_files[0])
        questions = q.get("questions", [])
        print(f"Questions count: {len(questions)}")
        if questions:
            sample_q = questions[:3]
            print("Sample questions:")
            for item in sample_q:
                img_id = item.get("image_id")
                qid = item.get("question_id")
                text = item.get("question")
                # COCO val filename format
                img_name = f"COCO_val2014_{img_id:012d}.jpg" if isinstance(img_id, int) else str(img_id)
                img_path = IMAGES_DIR / img_name
                print(f"- qid={qid}, image_id={img_id}, exists={img_path.exists()}, image={img_name}, question={text}")

    # Basic stats for annotations
    if a_files:
        a = load_json(a_files[0])
        anns = a.get("annotations", [])
        print(f"Annotations count: {len(anns)}")
        if anns:
            sample_a = anns[:3]
            print("Sample annotations:")
            for item in sample_a:
                aid = item.get("question_id")
                img_id = item.get("image_id")
                answers = item.get("answers", [])
                ans_texts = [ans.get("answer") for ans in answers[:3]]
                print(f"- qid={aid}, image_id={img_id}, answers={ans_texts}")

    # Image directory sanity
    if IMAGES_DIR.exists():
        files = list(IMAGES_DIR.iterdir())[:5]
        print(f"Image samples ({len(files)} shown): {[p.name for p in files]}")
    else:
        print("[Warn] Images directory not found.")

if __name__ == "__main__":
    main()