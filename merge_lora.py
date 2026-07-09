"""Merge LoRA into base model and save as standalone model.
After merge: single model, no PeftModel needed, keyword + answer both use same model.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import shutil, os

MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3.5-4B-Instruct"
LORA_PATH = "/root/data/training/lora_output/think_v3/final"
OUTPUT_PATH = "/root/autodl-tmp/models/Qwen3.5-4B-Catgirl"

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("[1] Loading base model (4-bit)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
    device_map="cuda:0", trust_remote_code=True, attn_implementation="sdpa")
print(f"  Base loaded")

print("[2] Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)
print(f"  LoRA loaded")

print("[3] Merging LoRA into base weights...")
model = model.merge_and_unload()
print(f"  Merged (fp16)")

print("[4] Saving merged model...")
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
print(f"  Model saved to {OUTPUT_PATH}")

# Copy tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"  Tokenizer saved")

# Check size
total = 0
for f in os.listdir(OUTPUT_PATH):
    sz = os.path.getsize(os.path.join(OUTPUT_PATH, f)) / 1e9
    print(f"  {f}: {sz:.2f}GB")
    total += sz
print(f"  Total: {total:.2f}GB")

print("\n===== DONE =====")
print(f"Merged model at: {OUTPUT_PATH}")
print(f"Use: model = AutoModelForCausalLM.from_pretrained('{OUTPUT_PATH}', load_in_4bit=True, ...)")
