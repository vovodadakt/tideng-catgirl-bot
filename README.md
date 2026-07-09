# 灯酱/提酱 — 猫娘 AI 模型

基于 Qwen3.5-4B-Instruct + QLoRA (think_v3) 微调的猫娘 QQ 机器人模型。

## 模型

- **基座**: Qwen3.5-4B-Instruct
- **微调**: LoRA r=64 alpha=128, 4-bit nf4 量化
- **合并后大小**: 3.12GB (safetensors), 加载后 ~3.3GB VRAM
- **路径**: 



## 能力

- 猫娘角色扮演（提酱傲娇 / 灯酱元气）
- 关键词提取 + 搜索回答
- 百度百科 + 萌娘百科 + Bing 多源搜索
- 诚实：材料没有就说不知道，不编造

## 搜索管线

见 



## 项目结构



## 硬件

- GPU: NVIDIA RTX 4080 SUPER (16GB)
- VRAM: ~3.3GB（单模型加载）

