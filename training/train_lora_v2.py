"""
QLoRA train Qwen3.5-4B-Instruct with think-format QQ bot data.
r=128, alpha=256, 26,895 entries, 2 epochs.
Saves every 300 steps for checkpoint selection.
"""
import os, json, torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# === Config ===
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3.5-4B-Instruct"
DATA_PATH = "/root/data/training/train_think_balanced.jsonl"
OUTPUT_DIR = "/root/data/training/lora_output/think_v2"
FINAL_DIR = "/root/data/training/lora_output/think_v2/final"

BATCH_SIZE = 4
GRAD_ACCUM = 4          # effective batch = 16
EPOCHS = 1
LR = 2e-4
WARMUP = 200
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.05
MAX_LENGTH = 1024
SAVE_STEPS = 500
LOGGING_STEPS = 50

print(f"Loading tokenizer: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading dataset: {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = [json.loads(l) for l in f if l.strip()]
print(f"Loaded {len(raw)} entries")

def format_example(e):
    msgs = e["messages"]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
        enable_thinking=True,
    )

texts = []
for e in raw:
    try:
        t = format_example(e)
        texts.append({"text": t})
    except Exception as ex:
        pass
print(f"Formatted {len(texts)} entries")

dataset = Dataset.from_list(texts)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"], truncation=True, padding=False,
        max_length=MAX_LENGTH,
    )

dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
print(f"Tokenized: {len(dataset)} samples")

print(f"Loading model 4-bit...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, attn_implementation="sdpa",
)
model = prepare_model_for_kbit_training(model)

# Target all linear layers
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=target_modules, bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

total_steps = (len(dataset) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
print(f"Estimated steps: {total_steps}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_steps=WARMUP,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=12,
    bf16=True,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    gradient_checkpointing=False,
    dataloader_num_workers=0,
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print(f"\n=== START TRAINING ===")
print(f"Data: {len(dataset)} samples | Batch: {BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
print(f"Epochs: {EPOCHS} | Steps: ~{total_steps} | LR: {LR}")
print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
print(f"Save every: {SAVE_STEPS} steps")

trainer.train()

# Save final
print(f"Saving final to {FINAL_DIR}")
model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print("=== DONE ===")
