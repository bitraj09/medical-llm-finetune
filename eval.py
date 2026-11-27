"""Evaluate the base model vs the LoRA fine-tuned model on held-out MedQA.

Generates a short answer for each eval question (greedy decoding), parses
the predicted option letter, and reports accuracy. Run twice — once with
--adapter omitted (base model) and once with --adapter pointing at the
trained LoRA adapter — to show the fine-tune's effect.

Usage:
    python eval.py --data data/medical_qa_eval.jsonl
    python eval.py --data data/medical_qa_eval.jsonl --adapter outputs/lora-adapter
"""
import argparse
import json
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTER = re.compile(r"\b([A-D])\b")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def predict_letter(model, tokenizer, prompt, max_new_tokens=32):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    match = LETTER.search(reply)
    return (match.group(1) if match else "?"), reply.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/medical_qa_eval.jsonl")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    rows = load_jsonl(args.data)[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"Loaded LoRA adapter from {args.adapter}")
    model.eval()

    correct = 0
    results = []
    for i, row in enumerate(rows):
        letter, reply = predict_letter(model, tokenizer, row["prompt"])
        ok = letter == row["answer_idx"]
        correct += ok
        results.append({
            "answer_idx": row["answer_idx"],
            "predicted": letter,
            "correct": ok,
            "reply": reply,
        })
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)} done, running accuracy: {correct / (i + 1):.3f}")

    acc = correct / len(rows)
    print(f"\nAccuracy: {correct}/{len(rows)} = {acc:.3f}")
    tag = "adapter" if args.adapter else "base"
    with open(f"results_{tag}.json", "w") as f:
        json.dump({"accuracy": acc, "n": len(rows), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
