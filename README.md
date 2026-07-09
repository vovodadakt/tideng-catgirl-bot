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

## 能力

- 猫娘角色扮演（提酱傲娇 / 灯酱元气）
- 关键词提取 + 搜索回答
- 百度百科 + 萌娘百科 + Bing 多源搜索
- 诚实：材料没有就说不知道，不编造

## 搜索管线

见 `inference/search/search_pipeline.py`

```
用户问题 -> 关键词提取 -> 百度百科 + 萌娘百科 -> <500c? -> Bing -> 猫娘回答
```

## 项目结构

```
tideng-catgirl-bot/
├── inference/
│   └── search/
│       ├── search_pipeline.py    # 搜索管线
│       └── version_history/
├── data_gen/                     # 训练数据生成
├── training/                     # 训练脚本
└── merge_lora.py                 # LoRA 合并脚本
```

## 硬件

- GPU: NVIDIA RTX 4080 SUPER (16GB)
- VRAM: ~3.3GB（单模型加载）
