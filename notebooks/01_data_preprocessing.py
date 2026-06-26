# ============================================================
# 01_data_preprocessing.py
# MedLLM: Domain-Adaptive LLM Fine-Tuning Pipeline
# ── Runs on CPU (local machine) ──
# ============================================================

"""
WHAT THIS DOES:
  - Downloads MedQA dataset from HuggingFace
  - Formats into Mistral instruction format
  - Creates train/val/test splits
  - Saves processed data locally

DATASET: medalpaca/medical_meadow_medqa
  - Medical question-answer pairs
  - ~10,000 QA samples from USMLE-style questions
"""

import os
import json
import random
import pandas as pd
from datasets import load_dataset
from collections import Counter

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── 1. Load Dataset ──────────────────────────────────────────
print("=" * 65)
print("STEP 1: Loading MedQA Dataset from HuggingFace")
print("=" * 65)

dataset = load_dataset("medalpaca/medical_meadow_medqa", trust_remote_code=True)
print(f"Dataset splits: {list(dataset.keys())}")
print(f"Train size    : {len(dataset['train']):,}")
print(f"Columns       : {dataset['train'].column_names}")
print(f"\nSample record:")
sample = dataset["train"][0]
for k, v in sample.items():
    print(f"  {k}: {str(v)[:120]}")

# ── 2. Format into Mistral Instruction Template ──────────────
print("\n" + "=" * 65)
print("STEP 2: Formatting into Mistral Instruction Format")
print("=" * 65)

"""
Mistral instruction format:
<s>[INST] {instruction} [/INST] {output}</s>

We format: instruction = question + context (if any)
           output      = answer
"""

def format_medqa_prompt(example):
    """
    Format a MedQA example into Mistral chat template.
    The dataset has 'input' (question), 'output' (answer),
    and sometimes 'instruction' field.
    """
    instruction = example.get("instruction", "Answer the following medical question accurately and concisely.")
    question    = example.get("input", "")
    answer      = example.get("output", "")

    # Skip empty examples
    if not question.strip() or not answer.strip():
        return None

    # Mistral instruction format
    prompt = f"<s>[INST] {instruction}\n\nQuestion: {question} [/INST] {answer}</s>"

    return {
        "prompt"     : prompt,
        "instruction": instruction,
        "question"   : question,
        "answer"     : answer,
        "prompt_len" : len(prompt.split()),
    }

print("Formatting dataset...")
formatted = []
skipped   = 0

for example in dataset["train"]:
    result = format_medqa_prompt(example)
    if result:
        formatted.append(result)
    else:
        skipped += 1

print(f"Formatted : {len(formatted):,} examples")
print(f"Skipped   : {skipped} (empty fields)")

# ── 3. Analyze Token Lengths ─────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Token Length Analysis")
print("=" * 65)

lengths = [ex["prompt_len"] for ex in formatted]
df_len  = pd.DataFrame({"length": lengths})

print(f"  Mean length  : {df_len['length'].mean():.0f} words")
print(f"  Median length: {df_len['length'].median():.0f} words")
print(f"  Max length   : {df_len['length'].max()} words")
print(f"  Min length   : {df_len['length'].min()} words")
print(f"  > 512 words  : {(df_len['length'] > 512).sum()} examples (will be truncated)")
print(f"  > 1024 words : {(df_len['length'] > 1024).sum()} examples")

# Filter out extremely long examples (> 800 words)
filtered = [ex for ex in formatted if ex["prompt_len"] <= 800]
print(f"\nAfter filtering (<=800 words): {len(filtered):,} examples")

# ── 4. Train / Val / Test Split ──────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Train / Validation / Test Split")
print("=" * 65)

random.seed(42)
random.shuffle(filtered)

total     = len(filtered)
train_end = int(total * 0.80)
val_end   = int(total * 0.90)

train_data = filtered[:train_end]
val_data   = filtered[train_end:val_end]
test_data  = filtered[val_end:]

print(f"  Train : {len(train_data):,} ({len(train_data)/total*100:.1f}%)")
print(f"  Val   : {len(val_data):,}  ({len(val_data)/total*100:.1f}%)")
print(f"  Test  : {len(test_data):,}  ({len(test_data)/total*100:.1f}%)")

# ── 5. Save Datasets ─────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: Saving Processed Datasets")
print("=" * 65)

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} records → {path}")

save_jsonl(train_data, "data/train.jsonl")
save_jsonl(val_data,   "data/val.jsonl")
save_jsonl(test_data,  "data/test.jsonl")

# Save a small sample for quick testing
save_jsonl(test_data[:50], "data/test_sample.jsonl")

# Save stats
stats = {
    "total_examples"  : total,
    "train_size"      : len(train_data),
    "val_size"        : len(val_data),
    "test_size"       : len(test_data),
    "avg_prompt_words": round(df_len["length"].mean(), 1),
    "max_prompt_words": int(df_len["length"].max()),
    "dataset"         : "medalpaca/medical_meadow_medqa",
    "model_target"    : "mistralai/Mistral-7B-Instruct-v0.3",
    "format"          : "mistral_instruction"
}
with open("data/dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"  Saved stats → data/dataset_stats.json")

# ── 6. Preview ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: Sample Formatted Prompts")
print("=" * 65)

for i, ex in enumerate(train_data[:2]):
    print(f"\n--- Example {i+1} ---")
    print(ex["prompt"][:400] + "...")
    print()

print(f"""
╔══════════════════════════════════════════════════════╗
║         DATA PREPROCESSING COMPLETE                 ║
╠══════════════════════════════════════════════════════╣
║  Dataset    : MedQA (Medical Meadow)                ║
║  Train      : {len(train_data):>6,} examples                    ║
║  Val        : {len(val_data):>6,} examples                    ║
║  Test       : {len(test_data):>6,} examples                    ║
║  Format     : Mistral [INST] template               ║
║  Next Step  : Upload 02_qlora_finetuning.py         ║
║               to Google Colab with T4 GPU           ║
╚══════════════════════════════════════════════════════╝
""")
