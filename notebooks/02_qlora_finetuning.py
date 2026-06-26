# ============================================================
# 02_qlora_finetuning.py
# MedLLM: QLoRA Fine-Tuning on Medical QA
# ── RUN THIS ON GOOGLE COLAB / KAGGLE (T4/A100 GPU) ──
# ============================================================

"""
SETUP ON GOOGLE COLAB:
  1. Runtime → Change runtime type → T4 GPU
  2. Run in first cell:
     !pip install -q transformers datasets peft trl bitsandbytes accelerate wandb
  3. Upload this file or paste into cells
  4. Upload data/train.jsonl and data/val.jsonl

WHAT THIS DOES:
  - Loads Mistral-7B-Instruct in 4-bit quantization (QLoRA)
  - Attaches LoRA adapters (r=16, alpha=32)
  - Fine-tunes with TRL SFTTrainer
  - Tracks experiments with W&B
  - Saves adapter weights locally
"""

import os
import json
import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# ── 0. Config ────────────────────────────────────────────────
print("=" * 65)
print("MedLLM — QLoRA Fine-Tuning Pipeline")
print("=" * 65)

# ── GPU Check ────────────────────────────────────────────────
print(f"\nGPU Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name      : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# CONFIGURATION — Edit these values
# ============================================================
CONFIG = {
    # Model
    "model_id"       : "mistralai/Mistral-7B-Instruct-v0.3",
    "output_dir"     : "./adapter",           # where LoRA weights are saved

    # LoRA hyperparameters
    "lora_r"         : 16,       # rank — higher = more params, better quality
    "lora_alpha"     : 32,       # scaling factor (usually 2x rank)
    "lora_dropout"   : 0.05,
    "lora_target"    : ["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],  # Mistral attention layers

    # Training
    "max_seq_length" : 512,      # max tokens per example
    "num_epochs"     : 3,
    "batch_size"     : 4,        # per device
    "grad_accum"     : 4,        # effective batch = 4 * 4 = 16
    "learning_rate"  : 2e-4,
    "warmup_ratio"   : 0.03,
    "lr_scheduler"   : "cosine",
    "weight_decay"   : 0.001,

    # Logging
    "wandb_project"  : "medllm-finetuning",
    "logging_steps"  : 25,
    "eval_steps"     : 100,
    "save_steps"     : 200,
}
print("\nConfig loaded:")
for k, v in CONFIG.items():
    if k != "lora_target":
        print(f"  {k:<20}: {v}")

# ── 1. Login ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 1: Authentication")
print("=" * 65)

# HuggingFace token — get from https://huggingface.co/settings/tokens
# In Colab: from google.colab import userdata; HF_TOKEN = userdata.get('HF_TOKEN')
HF_TOKEN = os.environ.get("HF_TOKEN", "")
WB_TOKEN = os.environ.get("WANDB_API_KEY", "")

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("HuggingFace: logged in")
else:
    print("HuggingFace: set HF_TOKEN env var or call login() manually")

if WB_TOKEN:
    wandb.login(key=WB_TOKEN)
    print("W&B: logged in")
else:
    print("W&B: set WANDB_API_KEY env var or call wandb.login() manually")

# ── 2. Load & Prepare Dataset ────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Loading Dataset")
print("=" * 65)

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

train_data = load_jsonl("data/train.jsonl")
val_data   = load_jsonl("data/val.jsonl")

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_list([{"text": ex["prompt"]} for ex in train_data])
val_dataset   = Dataset.from_list([{"text": ex["prompt"]} for ex in val_data])

print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
print(f"Sample: {train_dataset[0]['text'][:200]}...")

# ── 3. 4-Bit Quantization Config (QLoRA) ────────────────────
print("\n" + "=" * 65)
print("STEP 3: Setting up 4-bit Quantization (QLoRA)")
print("=" * 65)

"""
QLoRA key ideas:
  - NF4 (NormalFloat4): 4-bit quantization optimized for normally distributed weights
  - double_quant: quantizes the quantization constants (saves ~0.4 bits/param extra)
  - compute_dtype: float16 for actual compute (GPU arithmetic stays in fp16)

Result: 7B model uses ~4.5GB VRAM instead of ~14GB (fp16)
"""

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",          # NormalFloat4
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,            # nested quantization
)
print("BitsAndBytes config: 4-bit NF4 + double quantization")

# ── 4. Load Base Model + Tokenizer ───────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Loading Mistral-7B (4-bit)")
print("=" * 65)
print("This takes ~3-5 min on first run (downloading ~4.5GB)...")

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["model_id"],
    trust_remote_code=True,
)
tokenizer.pad_token     = tokenizer.eos_token  # Mistral has no pad token
tokenizer.padding_side  = "right"              # pad right for causal LM

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_id"],
    quantization_config = bnb_config,
    device_map          = "auto",              # auto-place on GPU
    trust_remote_code   = True,
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False                 # disable KV-cache during training

total_params    = sum(p.numel() for p in model.parameters())
print(f"\nModel loaded successfully!")
print(f"Total parameters: {total_params/1e9:.2f}B")

# ── 5. LoRA Configuration ────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: Attaching LoRA Adapters")
print("=" * 65)

"""
LoRA key ideas:
  - Instead of updating W (large), learn W + ΔW = W + BA
  - B is (d × r), A is (r × k) — r << d, k (low rank)
  - Only B and A are trained — everything else frozen
  - alpha/r scales the update: effective_lr = alpha/r * lr

With r=16 on Mistral-7B:
  Trainable params ≈ 40M out of 7,241M (0.55%)
"""

lora_config = LoraConfig(
    task_type    = TaskType.CAUSAL_LM,
    r            = CONFIG["lora_r"],
    lora_alpha   = CONFIG["lora_alpha"],
    lora_dropout = CONFIG["lora_dropout"],
    target_modules = CONFIG["lora_target"],
    bias         = "none",
)

model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params : {trainable:,}  ({trainable/total*100:.3f}%)")
print(f"Frozen params    : {total-trainable:,}")
print(f"LoRA rank (r)    : {CONFIG['lora_r']}")
print(f"LoRA alpha       : {CONFIG['lora_alpha']}")
print(f"Target modules   : {CONFIG['lora_target']}")

# ── 6. Training Arguments ────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: Training Configuration")
print("=" * 65)

wandb.init(
    project = CONFIG["wandb_project"],
    name    = f"mistral7b-medqa-lora-r{CONFIG['lora_r']}",
    config  = CONFIG,
    tags    = ["qlora", "mistral", "medical-qa"]
)

training_args = SFTConfig(
    output_dir                  = CONFIG["output_dir"],
    num_train_epochs            = CONFIG["num_epochs"],
    per_device_train_batch_size = CONFIG["batch_size"],
    per_device_eval_batch_size  = CONFIG["batch_size"],
    gradient_accumulation_steps = CONFIG["grad_accum"],
    learning_rate               = CONFIG["learning_rate"],
    warmup_ratio                = CONFIG["warmup_ratio"],
    lr_scheduler_type           = CONFIG["lr_scheduler"],
    weight_decay                = CONFIG["weight_decay"],
    fp16                        = True,             # mixed precision
    bf16                        = False,
    max_grad_norm               = 1.0,              # gradient clipping
    logging_steps               = CONFIG["logging_steps"],
    eval_steps                  = CONFIG["eval_steps"],
    save_steps                  = CONFIG["save_steps"],
    evaluation_strategy         = "steps",
    save_strategy               = "steps",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    report_to                   = "wandb",
    run_name                    = f"medqa-r{CONFIG['lora_r']}",
    dataloader_num_workers      = 2,
    group_by_length             = True,             # speeds up training
    max_seq_length              = CONFIG["max_seq_length"],
    dataset_text_field          = "text",
    packing                     = False,
)

print(f"Effective batch size : {CONFIG['batch_size'] * CONFIG['grad_accum']}")
print(f"Learning rate        : {CONFIG['learning_rate']}")
print(f"Epochs               : {CONFIG['num_epochs']}")
print(f"Max seq length       : {CONFIG['max_seq_length']}")

# ── 7. Train ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7: Training (this takes 2-4 hours on T4 GPU)")
print("=" * 65)
print("Monitor at: https://wandb.ai\n")

trainer = SFTTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    tokenizer       = tokenizer,
)

train_result = trainer.train()

print("\n Training complete!")
print(f"  Train loss     : {train_result.training_loss:.4f}")
print(f"  Train runtime  : {train_result.metrics.get('train_runtime', 0)/60:.1f} min")
print(f"  Samples/sec    : {train_result.metrics.get('train_samples_per_second', 0):.2f}")

# ── 8. Save Adapter ──────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8: Saving LoRA Adapter")
print("=" * 65)

trainer.model.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])

print(f"Adapter saved to: {CONFIG['output_dir']}/")
print("Files saved:")
for f in os.listdir(CONFIG["output_dir"]):
    print(f"  - {f}")

# Log final metrics to W&B
wandb.log({
    "final_train_loss": train_result.training_loss,
    "trainable_params": trainable,
    "trainable_pct"   : trainable/total*100,
})
wandb.finish()

print(f"""
╔══════════════════════════════════════════════════════╗
║         FINE-TUNING COMPLETE                        ║
╠══════════════════════════════════════════════════════╣
║  Base model      : Mistral-7B-Instruct-v0.3         ║
║  Trainable params: {trainable/1e6:>6.1f}M ({trainable/total*100:.3f}%)         ║
║  Train loss      : {train_result.training_loss:.4f}                       ║
║  Adapter saved   : ./adapter/                       ║
║                                                     ║
║  NEXT: Download ./adapter/ folder                   ║
║        Then run 03_evaluation.py locally            ║
╚══════════════════════════════════════════════════════╝
""")
