# Tideng Catgirl QQ Bot

QLoRA fine-tuned Qwen3.5-4B-Instruct for a dual-catgirl QQ bot.

## Characters

- **Ti (Ti-chan)**: Black cat, tsundere knowledge expert. Ends sentences with .
- **Tou (Tou-chan)**: White cat, genki straight-shooter. Ends sentences with .

## Model

- Base: Qwen/Qwen3.5-4B-Instruct
- Method: QLoRA (4-bit nf4, r=64, alpha=128, all linear layers)
- Training data: 26,895 entries with think blocks
- Final loss: 2.40 -> 0.51 (3,362 steps, dual RTX 5090 DDP)

## Project Structure



## Quick Start



## License

MIT
