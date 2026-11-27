"""LoRA fine-tune of a small base model on MedQA (USMLE-style medical QA).

Base model: Qwen/Qwen2.5-0.5B-Instruct (0.5B params — trains on a free
Colab T4 GPU in ~15 minutes for 1,000 examples).
Method: LoRA (r=16) supervised fine-tuning with TRL SFTTrainer, bf16.

Usage (Colab / any GPU machine):
    pip install -r requirements.txt
    python train.py --data data/medical_qa_train.jsonl --output outputs/lora-adapter

Swap the base model with --model (e.g. microsoft/BioGPT, a biomedical
pretrained model, also works with this script).
"""
import argparse

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_jsonl(path):
    import json

    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/medical_qa_train.jsonl")
    parser.add_argument("--output", default="outputs/lora-adapter")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    print(f"Loaded {len(rows)} training examples from {args.data}")
    dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=args.max_length,
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA adapter saved to {args.output}")


if __name__ == "__main__":
    main()
