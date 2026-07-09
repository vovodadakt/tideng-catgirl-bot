# Catgirl QQ Bot — Search Pipeline

Qwen3.5-4B-Instruct + QLoRA (think_v3) search + answer pipeline.

## Architecture

Phase 1: Base model extracts keywords -> Baidu Baike (Browser) + Moegirl (API) -> if <500c, Bing deep search
Phase 2: Load LoRA adapter -> Answer all questions

Single model: ~5GB VRAM (vs ~10GB for dual-model approach).

## Search Sources

| Source | Method | Description |
|--------|--------|-------------|
| Baidu Baike | Playwright browser | Bypasses server IP 403 |
| Moegirl | API (opensearch + query) | ACG knowledge |
| Bing | Playwright deep-read | Only triggered when R1 < 500 chars |

## Performance (4 questions)

| Phase | Time |
|-------|------|
| Base model load | 4.3s |
| Search (4 questions) | 171.6s |
| LoRA load | 4.7s |
| Answer (4 questions) | 129.9s |
| **Total** | **310.8s** |

vs V22 dual-model: ~390s — 20% faster, 50% less VRAM.

## Server

- GPU: NVIDIA RTX 4080 SUPER (16 GB)
- Model: Qwen3.5-4B-Instruct (4-bit nf4) + LoRA r=64 alpha=128
- VRAM: ~5 GB

