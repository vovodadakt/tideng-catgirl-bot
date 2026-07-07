import torch, time, re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "/root/autodl-tmp/models/Qwen/Qwen3.5-4B-Instruct"
LORA = "/root/data/training/lora_output/think_v3/final"

def clean_response(text):
    for marker in ["\nuser\n", "\n\nuser", "\n\nassistant"]:
        idx = text.find(marker)
        if idx > 10:
            text = text[:idx]
            break
    text = re.sub(r'</?think>\s*', '', text).strip()
    text = re.sub(r'^assistant\s*', '', text).strip()
    return text

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []

# 抢手机模式：匹配训练数据的调情互动风格
# 训练数据关键特征：
# - 灯酱直球撩 → 缇酱嘴硬但尾巴/耳朵出卖她
# - 身体接触梗：蹭蹭、抱抱、压到尾巴
# - 缇酱脸红/别过脸/小声 → 灯酱追加攻击
# - 灯酱经常叫"缇酱"名字，缇酱被撩急了才叫"灯酱"
# - 调情场景占比34%（9138/26895）
SYSTEM_PROMPT = """你是「缇酱」和「灯酱」——两只共用一个QQ号的猫娘。

【人设】
缇酱(喵)：黑猫傲娇。自称"本喵"，句末加「喵」。嘴硬但耳朵和尾巴会出卖她。
  害羞时：(〃ω〃)、(*/ω＼*)  得意时：(。-`ω´-)、(｀・ω・´)
  被灯酱撩了会：别过脸、脸红、小声嘟囔、尾巴不自觉地翘起来。
灯酱(にゃ)：白猫元气直球。自称"灯酱"，句末加「にゃ」。
  放飞时：(ノ>▽<)ノ、(≧▽≦)  被反撩时：(〃ω〃)、にゃー！！
  喜欢撩缇酱看炸毛，经常夸缇酱可爱、可靠。

【互动规则·重要】
1. 灯酱必须先抢到手机说第一句。
2. 缇酱夺回手机补充/吐槽/纠正。
3. 灯酱在旁边插嘴撩猫——夸缇酱、提尾巴耳朵、或者要蹭蹭抱抱。
4. 缇酱被撩后：嘴硬否认，但尾巴/耳朵/脸红出卖自己。
5. 灯酱发现缇酱的反应后追加攻击："缇酱你尾巴翘了にゃ！""缇酱脸红了にゃ！"
6. 缇酱被逼急了小声承认一点真心话，然后立刻凶回去。

纯对话无动作描述，喵/にゃ区分身份。每次都要两只猫都说话！"""

tests = [
    "晚安啦，明天还要早起呢",
    "解释一下什么叫递归，简单点说",
    "「木漏れ日」这个词是什么意思？",
    "（递过来一条围巾）天气冷了，给你织的",
    "今天被老板骂了，好难过…",
    "最近有什么好看的轻小说推荐吗？",
    "你能证明你自己不是AI吗？",
    "如果有一天超能力，你想做什么？",
    "你们两个谁更可爱？",
    """帮我查一下：进击的巨人最终季什么时候播出的？

[参考资料]
进击的巨人The Final Season于2020年12月7日在NHK综合台首播，共16集。
第二部分于2022年1月10日播出，共12集。
完结篇前篇于2023年3月4日播出，完结篇后篇于2023年11月5日播出。
[来源: Wikipedia]""",
]

for i, user_msg in enumerate(tests, 1):
    print(f"\n{'='*60}")
    print(f"测试 {i}/{len(tests)}: {user_msg[:60].strip().split(chr(10))[0]}")

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    plen = inputs.input_ids.shape[1]

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=200, temperature=0.85, top_p=0.92,
            do_sample=True, eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0
    raw = tokenizer.decode(outputs[0][plen:], skip_special_tokens=True)
    response = clean_response(raw)

    print(f"  输出 ({elapsed:.1f}s):")
    for line in response.split('\n'):
        if line.strip():
            print(f"    {line.strip()}")
    print()

print("===== ALL DONE =====")
