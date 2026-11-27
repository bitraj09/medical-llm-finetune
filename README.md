# Medical LLM Fine-Tune

LoRA fine-tuning of a small language model on real USMLE-style medical exam questions — so it learns to answer clinical multiple-choice questions in the right format.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bitraj09/medical-llm-finetune/blob/main/notebooks/medical_llm_finetune.ipynb)

## What this is

- **Base model:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — 0.5B params, small enough to fine-tune on a free Colab T4 GPU in ~15 minutes.
- **Dataset:** [MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) — 10,178 real USMLE-style questions (question + 4 options + correct answer). This repo ships 300 formatted training examples and 200 held-out eval examples; `data/prep_data.py` rebuilds the full 1,000-example training set from the official source.
- **Method:** LoRA supervised fine-tuning (r=16, alpha=32) with TRL `SFTTrainer`, bf16, 1 epoch. Only the adapter weights are trained — the base model stays frozen.
- **Eval:** greedy-decoded accuracy on 200 held-out MedQA questions, base model vs fine-tuned adapter (`eval.py`).

## Quickstart (Colab — recommended)

Open the notebook in Colab and run all cells — it installs deps, trains the LoRA adapter, evaluates base vs fine-tuned, and tries a fresh medical question:

👉 [notebooks/medical_llm_finetune.ipynb](notebooks/medical_llm_finetune.ipynb)

## Quickstart (local GPU)

```bash
git clone https://github.com/bitraj09/medical-llm-finetune.git
cd medical-llm-finetune
pip install -r requirements.txt

# Train (1 epoch, ~15 min on a T4)
python train.py --data data/medical_qa_train.jsonl --output outputs/lora-adapter

# Evaluate base vs fine-tuned on 200 held-out questions
python eval.py --data data/medical_qa_eval.jsonl
python eval.py --data data/medical_qa_eval.jsonl --adapter outputs/lora-adapter
```

## Results

Run the notebook (or the commands above) to generate `results_base.json` and `results_adapter.json` — accuracy on the 200-question held-out set. The LoRA adapter is expected to beat the base model at answering in the correct letter format and picking the right option; exact numbers depend on the run.

## Example

**Question:** A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination… Which of the following is the best treatment?

- A. Ampicillin · B. Ceftriaxone · C. Doxycycline · D. Nitrofurantoin

**Fine-tuned answer:** `The correct answer is D: Nitrofurantoin.`

## Repo layout

```
├── train.py                    # LoRA SFT training script
├── eval.py                     # Base vs adapter accuracy eval
├── requirements.txt
├── notebooks/
│   └── medical_llm_finetune.ipynb   # One-click Colab training
└── data/
    ├── prep_data.py            # Rebuilds the dataset from MedQA on HF
    ├── medical_qa_train.jsonl  # 300 formatted SFT examples (prep_data.py builds 1,000)
    └── medical_qa_eval.jsonl   # 200 held-out eval examples
```

## Alternatives

Swap the base model with `--model`: `microsoft/BioGPT` (355M, pretrained on PubMed biomedical literature) also works with `train.py` — a biomedical-domain base model instead of a general one.

## Disclaimer

Educational project. The model is trained on exam-style questions and is **not** a medical device — never use it for real diagnosis or treatment decisions.
