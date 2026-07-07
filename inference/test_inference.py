import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "/root/autodl-tmp/models/Qwen/Qwen3.5-4B-Instruct"
LORA = "/root/data/training/lora_output/think_v3/final"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

print("Loading base model 4bit...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, attn_implementation="sdpa",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA)
model.eval()

tests = [
    ("缇酱(喵)", "你好呀，今天天气不错呢~"),
    ("灯酱(にゃ)", "にゃ！新番更新了你知道吗？"),
    ("灯酱(にゃ)", "帮我查一下Python列表推导式怎么写"),
    ("缇酱(喵)", "哼，我才不是在关心你呢"),
]

for role, user_msg in tests:
    msgs = [
        {"role": "system", "content": f"你是QQ群里的猫娘助手。当前角色：{role}。用猫娘口吻回复，简短活泼。"},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.8, top_p=0.9, do_sample=True)
    elapsed = time.time() - t0

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"[{role}] 用户: {user_msg}")
    print(f"回复: {response[:400]}")
    print(f"耗时: {elapsed:.1f}s")

print("\n===== DONE =====")
