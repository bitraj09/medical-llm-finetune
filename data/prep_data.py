"""Build the fine-tuning dataset from the real MedQA-USMLE-4-options data.

Downloads the official train split from Hugging Face
(GBaker/MedQA-USMLE-4-options, 10,178 USMLE-style questions), formats each
item as a prompt/completion pair for supervised fine-tuning, and writes:
  - medical_qa_train.jsonl : 1,000 examples for LoRA SFT
  - medical_qa_eval.jsonl  : 200 held-out examples for accuracy eval

Usage:
    python data/prep_data.py [--train 1000] [--eval 200]
"""
import argparse
import json
import random
import urllib.request
from pathlib import Path

HF_URL = (
    "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options"
    "/resolve/main/phrases_no_exclude_train.jsonl"
)
HERE = Path(__file__).resolve().parent

PROMPT_TEMPLATE = (
    "You are a medical AI assistant. Answer the following USMLE-style "
    "question by choosing the best option.\n\n"
    "Question: {question}\n\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
    "Answer with the letter and the option text."
)


def format_example(item):
    opts = item["options"]
    prompt = PROMPT_TEMPLATE.format(
        question=item["question"],
        a=opts["A"], b=opts["B"], c=opts["C"], d=opts["D"],
    )
    letter = item["answer_idx"]
    completion = f"The correct answer is {letter}: {item['answer']}."
    return {"prompt": prompt, "completion": completion, "answer_idx": letter}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--eval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_path = HERE / "medqa_train.jsonl"
    if not raw_path.exists():
        print(f"Downloading MedQA train split from {HF_URL} ...")
        urllib.request.urlretrieve(HF_URL, raw_path)

    items = [json.loads(line) for line in open(raw_path)]
    print(f"Loaded {len(items)} MedQA train examples")

    rng = random.Random(args.seed)
    rng.shuffle(items)
    train_items = items[: args.train]
    eval_items = items[args.train : args.train + args.eval]

    for name, subset in [("medical_qa_train.jsonl", train_items),
                         ("medical_qa_eval.jsonl", eval_items)]:
        out = HERE / name
        with open(out, "w") as f:
            for item in subset:
                f.write(json.dumps(format_example(item)) + "\n")
        print(f"Wrote {len(subset)} examples -> {out}")


if __name__ == "__main__":
    main()
