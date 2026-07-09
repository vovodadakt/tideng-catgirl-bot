# Catgirl QQ Bot — 搜索管线

基于 Qwen3.5-4B-Instruct + QLoRA (think_v3) 的搜索回答管线。

## 架构

用户问题 → 基膜提取关键词 → 百度百科(Browser) + 萌娘百科(API) → 材料<500c? → Bing深度搜索 → LoRA模型回答

## 搜索源

| 来源 | 方式 | 说明 |
|------|------|------|
| 百度百科 | Playwright 浏览器 | 绕过服务器 403，提取 meta/正文 |
| 萌娘百科 | API (opensearch + query) | ACG 知识 |
| Bing | Playwright 深度阅读 | 仅在 R1 材料不足时触发 |

## 关键设计

- **多关键字搜索**：基膜从问题提取 1-3 个核心词，每个词分别搜百度和萌娘
- **Bing 兜底**：R1 总材料 < 500 字符时才触发
- **无清洗步骤**：LoRA 训练数据格式就是 [参考资料]，无需基膜预清洗
- **单次推理**：搜索 → 回答一气呵成

## 版本历史

| 版本 | 里程碑 |
|------|--------|
| V4 | Playwright 绕过百度 403 |
| V15 | Bing 深度搜索 fallback |
| V20 | 百度百科回归 + 多关键字搜索 |
| V22 | 最终稳定版：500c Bing 阈值 |

## 服务器

- GPU: NVIDIA RTX 4080 SUPER (16GB)
- 模型: Qwen3.5-4B-Instruct (4-bit nf4) + LoRA r=64 α=128
- VRAM: ~10GB
