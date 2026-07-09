# Catgirl QQ Bot — 搜索管线

基于 Qwen3.5-4B-Instruct + QLoRA (think_v3) 的搜索回答管线。

## 架构



单模型运行：基膜处理搜索阶段，然后加载 LoRA 适配器做回答。VRAM ~5GB。

## 搜索源

| 来源 | 方式 | 说明 |
|------|------|------|
| 百度百科 | Playwright 浏览器 | 绕过服务器 403 |
| 萌娘百科 | API (opensearch + query) | ACG 知识 |
| Bing | Playwright 深度阅读 | R1 < 500c 时触发 |

## 性能 (4题测试)

| 阶段 | 耗时 |
|------|------|
| 模型加载 | 4.3s |
| 搜索 (4题) | 171.6s |
| LoRA 加载 | 4.7s |
| 回答 (4题) | 129.9s |
| **总计** | **310.8s** |

## 服务器

- GPU: NVIDIA RTX 4080 SUPER (16GB)
- 模型: Qwen3.5-4B-Instruct (4-bit nf4) + LoRA r=64 α=128
- VRAM: ~5GB
