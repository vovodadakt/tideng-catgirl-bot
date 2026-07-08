# 缇灯猫娘 QQ 机器人 🐱

为双猫娘 QQ 机器人 QLoRA 微调的 Qwen3.5-4B-Instruct。

## 角色

- **缇酱（黑猫）**：傲娇知识担当，句末加「喵」。被夸了嘴硬但耳朵和尾巴会出卖她。
  颜文字：(。-`ω´-)、(*/ω＼*)、(｀・ω・´)
- **灯酱（白猫）**：元气直球担当，句末加「にゃ」。喜欢撩缇酱看炸毛。
  颜文字：(ノ>▽<)ノ、(≧▽≦)、(๑>◡<๑)

## 模型

- 基座：Qwen/Qwen3.5-4B-Instruct
- 方法：QLoRA（4-bit nf4, r=64, alpha=128, 全线性层）
- 训练数据：26,895 条带 think 块的训练数据
- 最终 Loss：2.40 → 0.51（3,362 步，双 RTX 5090 DDP）

## 推理模式

### 抢手机模式（推荐）
一次生成两只猫娘抢手机互动的完整对话，灯酱先抢 → 缇酱夺回 → 灯酱插嘴撩猫。

### 独立回复模式
分别调用两个 system prompt 生成各自回复，代码层面加前缀区分角色。

## 项目结构

```
training/        - LoRA 训练脚本
inference/       - 推理脚本（抢手机 & 独立模式）
data_gen/        - 训练数据生成脚本
```

## 快速开始

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-4B-Instruct',
    quantization_config=bnb, device_map='auto',
    trust_remote_code=True, attn_implementation='sdpa',
)
model = PeftModel.from_pretrained(model, 'path/to/lora/final')
```

## 许可

MIT
