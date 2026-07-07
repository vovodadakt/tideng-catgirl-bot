"""Prepare indices for 1-round regeneration. Runs on AutoDL."""
import json, random

with open('/root/data/training/train_data_balanced.jsonl', 'r', encoding='utf-8') as f:
    entries = [json.loads(l) for l in f if l.strip()]

def round_count(content):
    paras = [p.strip() for p in content.split('\n') if p.strip() and p not in ['<think>', '</think>']]
    return max(1, len(paras))

# Find real 1-round (single speaker)
real_one = 0
multi_pool = []
for i, e in enumerate(entries):
    content = e['messages'][2]['content']
    rc = round_count(content)
    if rc == 1:
        has_nya = 'にゃ' in content
        has_miao = '喵' in content
        if has_nya and has_miao:
            multi_pool.append(i)  # fake, needs redo
        else:
            real_one += 1
    elif rc >= 3:
        multi_pool.append(i)

print(f'Total: {len(entries)}')
print(f'Real 1-round: {real_one}')
print(f'Multi-round pool: {len(multi_pool)}')

need = int(len(entries) * 0.10) - real_one
print(f'Need: {need}')

random.seed(99)
random.shuffle(multi_pool)
indices = multi_pool[:need * 3]
print(f'Indices: {len(indices)}')

with open('/root/data/training/_convert_indices.json', 'w') as f:
    json.dump(indices, f)
print('Saved _convert_indices.json')
