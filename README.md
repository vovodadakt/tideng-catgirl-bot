# 灯酱/提酱 — 猫娘 AI 模型

基于 Qwen3.5-4B-Instruct + QLoRA (think_v3) 微调的猫娘 QQ 机器人模型。

## 下载

**ModelScope（国内推荐）：** https://modelscope.cn/models/vovodadakt/Qwen3.5-4B-Catgirl

```bash
# 方法 1: Git LFS
git clone https://modelscope.cn/vovodadakt/Qwen3.5-4B-Catgirl.git

# 方法 2: Python SDK
from modelscope import snapshot_download
model_dir = snapshot_download("vovodadakt/Qwen3.5-4B-Catgirl")
```

## 模型

- **基座**: Qwen3.5-4B-Instruct
- **微调**: LoRA r=64 alpha=128
- **文件大小**: 3.12GB（fp16 safetensors，不含量化）
- **格式**: 标准 HuggingFace 格式，合并 LoRA 后的全量权重

### 加载方式

**4-bit 量化加载（推荐）** — 约 3.3GB VRAM：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen3.5-4B-Catgirl",
    quantization_config=bnb,
    device_map="cuda:0",
    trust_remote_code=True,
    attn_implementation="sdpa",
)
```

**fp16 全精度加载** — 约 8GB VRAM：
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen3.5-4B-Catgirl",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
)
```

**CPU 加载** — 约 8GB 内存，无需 GPU：
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen3.5-4B-Catgirl",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
```

## 快速使用

```python
from catgirl_bot import CatgirlBot

bot = CatgirlBot()
bot.load()

# 一次性输出
answer = bot.ask("鬼灭之刃里水之呼吸的使用者有哪些？")
print(answer)

# 流式输出（适合 QQ 机器人分段发送）
for chunk in bot.ask_stream("木漏れ日是什么意思？"):
    print(chunk, end="", flush=True)
    # 或者: await send_qq_message(group_id, chunk)

bot.close()
```

### 一行调用

```python
from catgirl_bot import ask
print(ask("药屋少女的呢喃里猫猫的声优是谁？"))
```

### 命令行

```bash
python catgirl_bot.py 进击的巨人最终季什么时候播出
```

## API

### `CatgirlBot(model_path=..., lora_path=None, load_in_4bit=True, headless=True)`

| 方法 | 说明 |
|------|------|
| `load()` | 加载模型和浏览器（首次 ask 自动调用） |
| `ask(question) -> str` | 输入问题，返回完整回答 |
| `ask_stream(question)` | 流式输出，逐段 yield 文本，适合 QQ 分段发送 |
| `close()` | 释放 GPU 和浏览器资源 |

### 搜索管线

```
用户问题 -> 关键词提取 -> 百度百科(Browser) + 萌娘百科(API) -> <500c? -> Bing深度搜索 -> 猫娘回答
```

## 能力

- 猫娘角色扮演（提酱傲娇 / 灯酱元气）
- 自动搜索：百度百科 + 萌娘百科 + Bing
- 诚实：材料没有就说不知道，不编造

## 项目结构

```
tideng-catgirl-bot/
├── catgirl_bot.py                 # 接口（引用这个即可）
├── inference/search/              # 搜索管线
├── data_gen/                      # 训练数据生成
├── training/                      # 训练脚本
└── merge_lora.py                  # LoRA 合并脚本
```

## 硬件

- GPU: NVIDIA RTX 4080 SUPER (16GB)
- VRAM: ~3.3GB（4-bit 加载）
