# ============================================================
# 05_merge_and_export.py
# MedLLM: Merge LoRA Adapter + Export for Deployment
# ── Run on GPU (Colab) or CPU for small models ──
# ============================================================

"""
WHAT THIS DOES:
  - Merges LoRA adapter weights INTO the base model
  - Creates a single standalone model (no PEFT dependency)
  - Exports in HuggingFace format for FastAPI deployment
  - Optionally pushes to HuggingFace Hub

WHY MERGE?
  During training, adapter weights (B, A matrices) are separate.
  For deployment, merging them means:
    - No PEFT library needed at inference
    - Faster inference (no adapter overhead)
    - Easier to serve with vLLM, TGI, etc.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("=" * 65)
print("MedLLM — Merge Adapter & Export")
print("=" * 65)

MODEL_ID    = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_DIR = "adapter"
MERGED_DIR  = "merged_model"
PUSH_TO_HUB = False   # Set True to push to HuggingFace Hub
HUB_REPO    = "your-username/medllm-mistral7b"   # change this

os.makedirs(MERGED_DIR, exist_ok=True)

print(f"Base model  : {MODEL_ID}")
print(f"Adapter dir : {ADAPTER_DIR}")
print(f"Merged dir  : {MERGED_DIR}")

# ── 1. Load base model in fp16 (NOT 4-bit) for merging ──────
print("\n" + "=" * 65)
print("STEP 1: Loading Base Model (fp16 for clean merge)")
print("=" * 65)
print("NOTE: Merging requires fp16, not 4-bit. Needs ~14GB VRAM or CPU.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype   = torch.float16,
    device_map    = "auto" if torch.cuda.is_available() else "cpu",
    trust_remote_code = True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
print("Base model loaded.")

# ── 2. Load adapter ──────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Loading LoRA Adapter")
print("=" * 65)

model = PeftModel.from_pretrained(model, ADAPTER_DIR)
print("Adapter loaded. Merging weights...")

# ── 3. Merge and Unload ──────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Merging LoRA into Base Weights")
print("=" * 65)

"""
merge_and_unload():
  - Computes W_merged = W_base + (alpha/r) * B @ A
  - Removes the PEFT wrapper
  - Returns a plain HuggingFace model
"""

model = model.merge_and_unload()
print("Merge complete! Model is now a standard HuggingFace model.")

# Verify no PEFT dependency
print(f"Model type: {type(model).__name__}")  # Should be MistralForCausalLM
total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total/1e9:.2f}B")

# ── 4. Save Merged Model ─────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Saving Merged Model")
print("=" * 65)

model.save_pretrained(MERGED_DIR, safe_serialization=True)  # saves as .safetensors
tokenizer.save_pretrained(MERGED_DIR)

print(f"Merged model saved to: {MERGED_DIR}/")
print("\nFiles:")
for f in sorted(os.listdir(MERGED_DIR)):
    size = os.path.getsize(f"{MERGED_DIR}/{f}") / 1e6
    print(f"  {f:<45} {size:.1f} MB")

# ── 5. Quick Inference Test ──────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: Quick Inference Test")
print("=" * 65)

test_prompt = "<s>[INST] Answer the following medical question accurately.\n\nQuestion: What is the first-line treatment for Type 2 Diabetes Mellitus in a newly diagnosed patient with no contraindications? [/INST]"

model.eval()
with torch.no_grad():
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens     = 150,
        temperature        = 0.1,
        do_sample          = True,
        pad_token_id       = tokenizer.eos_token_id,
        repetition_penalty = 1.1,
    )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(f"Question : What is the first-line treatment for T2DM?")
print(f"Response : {response.strip()}")

# ── 6. Optional: Push to Hub ─────────────────────────────────
if PUSH_TO_HUB:
    print("\n" + "=" * 65)
    print("STEP 6: Pushing to HuggingFace Hub")
    print("=" * 65)
    from huggingface_hub import login
    login()
    model.push_to_hub(HUB_REPO, safe_serialization=True)
    tokenizer.push_to_hub(HUB_REPO)
    print(f"Model pushed to: https://huggingface.co/{HUB_REPO}")
else:
    print("\nSkipping HuggingFace Hub push (PUSH_TO_HUB=False)")

print(f"""
╔══════════════════════════════════════════════════════╗
║         MERGE & EXPORT COMPLETE                     ║
╠══════════════════════════════════════════════════════╣
║  Merged model : ./merged_model/                     ║
║  Format       : HuggingFace SafeTensors             ║
║  No PEFT dep  : ✅ Standalone model                 ║
║                                                     ║
║  Next steps:                                        ║
║  1. uvicorn api.main:app --reload                   ║
║  2. streamlit run dashboard/app.py                  ║
╚══════════════════════════════════════════════════════╝
""")
