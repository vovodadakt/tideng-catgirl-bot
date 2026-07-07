"""
Convert 1731 multi-round entries to 1-round.
1-round = single catgirl speaks (can be either 灯酱 or 提酱).
Uses DeepSeek API, dual-key, 20 concurrent.
"""
import json, sys, time, random, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.stdout.reconfigure(encoding='utf-8')
import requests

DATA_DIR = Path(__file__).resolve().parent
API_KEYS = [
    "sk-fd1ca01f3df041a29cecc64961210d94",
    "sk-885daf0d60454669ae2b753bb566f66f",
]
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"
PARALLEL = 20
RATE_LIMIT = 0.02

key_index = 0
key_lock = Lock()
_key_last_call = {}
write_lock = Lock()
_success = 0
_lock = Lock()

def next_key():
    global key_index
    with key_lock:
        k = API_KEYS[key_index]
        key_index = (key_index + 1) % len(API_KEYS)
    return k

def call_api(system_prompt, user_prompt, max_retries=3):
    api_key = next_key()
    for attempt in range(max_retries):
        now = time.time()
        last = _key_last_call.get(api_key, 0)
        wait = RATE_LIMIT - (now - last)
        if wait > 0: time.sleep(wait)
        _key_last_call[api_key] = time.time()
        try:
            resp = requests.post(API_URL, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }, json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.85, "max_tokens": 1024, "top_p": 0.95,
            }, timeout=120)
            data = resp.json()
            if "choices" in data and len(data["choices"]) > 0:
                c = data["choices"][0]["message"]["content"]
                if c and len(c) > 40: return c
            else:
                err = str(data.get("error", ""))[:100]
                if "rate" in err.lower() or resp.status_code == 429:
                    time.sleep(10 * (attempt + 1))
            if attempt < max_retries - 1: time.sleep(3 * (attempt + 1))
        except:
            if attempt < max_retries - 1: time.sleep(5 * (attempt + 1))
    return None

def generate_one_round(entry):
    """Rewrite a multi-round entry as 1-round (single catgirl speaks)."""
    msgs = entry["messages"]
    original = msgs[2]["content"]
    user_msg = msgs[1]["content"]
    cat = entry.get("_category", "?")

    # Decide which catgirl speaks (50/50 for variety)
    speaker = random.choice(["提酱(喵)", "灯酱(にゃ)"])

    gen_prompt = f"""请把下面这条多轮猫娘回复，改写成一条单轮简短回复。

【用户消息】{user_msg}

【原多轮回复（片段）】{original[:400]}

【改写要求】
1. 只有{speaker}一个人说话，另外一只不出现
2. 直接回答问题或回应用户，不要辩论格式
3. 保持猫娘语气（{speaker}的专属句末+颜文字风格）
4. 简短，80~180字
5. 纯对话，无角色标记，无emoji

直接输出改写的单轮回复，不要前缀。"""

    sys_prompt = "你是训练数据改写器。把多轮猫娘对话压缩为单轮，保持猫娘语气。"
    result = call_api(sys_prompt, gen_prompt)
    if not result: return None

    new_content = result.strip()
    if len(new_content) < 30: return None

    # Must have ONLY the speaker's marker, not the other catgirl's
    if "喵" in speaker:
        if "喵" not in new_content: return None
        if "にゃ" in new_content: return None   # 灯酱 must NOT appear
    if "にゃ" in speaker:
        if "にゃ" not in new_content: return None
        if "喵" in new_content: return None     # 提酱 must NOT appear

    # Must NOT be multi-round (no back-and-forth)
    if new_content.count('\n') > 4: return None  # too many lines = still multi-round

    return new_content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--target", type=int, default=1731)
    args = parser.parse_args()
    n = args.parallel or PARALLEL

    # Load data
    source = DATA_DIR / "train_data_balanced.jsonl"
    with open(source, 'r', encoding='utf-8') as f:
        entries = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded: {len(entries)} entries")

    # Load indices to convert
    with open(DATA_DIR / "_convert_indices.json", 'r', encoding='utf-8') as f:
        indices = json.load(f)
    print(f"Indices to convert: {len(indices)}")

    # Extract entries to convert
    to_convert = []
    for idx in indices:
        if idx < len(entries):
            to_convert.append((idx, entries[idx]))

    target = min(args.target, len(to_convert))
    # 3x oversample by cycling through pool
    tasks = to_convert * 3
    random.shuffle(tasks)
    print(f"Target: {target} | Pool: {len(tasks)} (3x of {len(to_convert)}) | Parallel: {n}")

    # Save to temp file so we can resume if killed
    tmp_file = DATA_DIR / "_one_round_tmp.json"
    if tmp_file.exists():
        with open(tmp_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results from tmp")
    else:
        results = {}

    def gen(idx, entry):
        global _success
        if _success >= target:
            return
        new_content = generate_one_round(entry)
        if new_content:
            with write_lock:
                results[str(idx)] = new_content
                if len(results) % 50 == 0:
                    with open(tmp_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False)
            with _lock:
                _success += 1
                print(f"[{_success}/{target}] OK  idx={idx}  cat={entry.get('_category','?')}", flush=True)
        else:
            with _lock:
                print(f"[{_success}/{target}] FAIL  idx={idx}", flush=True)

    with ThreadPoolExecutor(max_workers=n) as executor:
        futs = []
        for idx, entry in tasks:
            if _success >= target:
                executor.shutdown(wait=False, cancel_futures=True)
                break
            futs.append(executor.submit(gen, idx, entry))
            time.sleep(0.02)
        for f in as_completed(futs):
            try: f.result()
            except: pass

    # Save final tmp
    with open(tmp_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

    # Apply results: replace entries
    applied = 0
    for idx_str, new_content in results.items():
        if applied >= target: break
        idx = int(idx_str)
        entries[idx]["messages"][2]["content"] = new_content
        applied += 1

    # Write final file
    out = DATA_DIR / "train_data_balanced.jsonl"
    with open(out, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nDone! Generated: {len(results)}, Applied: {applied} -> {out}")

    # Verify
    def round_count(content):
        paras = [p.strip() for p in content.split('\n') if p.strip()]
        return max(1, len(paras))

    one_r = sum(1 for e in entries if round_count(e['messages'][2]['content']) == 1)
    print(f"Final 1-round: {one_r}/{len(entries)} ({one_r*100/len(entries):.1f}%)")

if __name__ == "__main__":
    main()
