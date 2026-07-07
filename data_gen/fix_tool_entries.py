"""
Fix 398 tool entries that lack <tool_call>.
Rewrite them to include proper dict/search tool calls.
Runs on DeepSeek API, 20 concurrent.
"""
import json, sys, time, random, argparse, os
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
                "temperature": 0.85, "max_tokens": 2048, "top_p": 0.95,
            }, timeout=120)
            data = resp.json()
            if "choices" in data and len(data["choices"]) > 0:
                c = data["choices"][0]["message"]["content"]
                if c and len(c) > 60: return c
            else:
                err = str(data.get("error",""))[:100]
                if "rate" in err.lower() or resp.status_code == 429:
                    time.sleep(10 * (attempt + 1))
            if attempt < max_retries - 1: time.sleep(3 * (attempt + 1))
        except:
            if attempt < max_retries - 1: time.sleep(5 * (attempt + 1))
    return None

def detect_tool_type(user_msg):
    """Heuristic: what tool(s) does this question need?"""
    u = user_msg.lower()
    # Dict-only patterns
    dict_patterns = ['怎么读', '什么意思', '读音', '用法', '词典', '解释一下', '是什么意思',
                     '查一下「', '这个词', '什么场合用', '常见吗', '声调', '音调']
    # Search patterns
    search_patterns = ['搜一下', '语源', '流行', '最近', '来源', '出处', '历史',
                       '为什么叫', '怎么来的', '由来', '文化背景']

    needs_dict = any(p in user_msg for p in dict_patterns)
    needs_search = any(p in user_msg for p in search_patterns)

    if needs_dict and needs_search:
        return "both"
    elif needs_search:
        return "search"
    else:
        return "dict"

def fix_one(entry):
    """Rewrite a tool-less entry to include proper <tool_call>."""
    user_msg = entry["messages"][1]["content"]
    cat = detect_tool_type(user_msg)

    # Determine tool setup
    if cat == "dict":
        # Extract likely word from user query
        word = user_msg
        for marker in ['「', '」', '"', '"']:
            if marker in word:
                parts = word.split(marker)
                if len(parts) >= 3:
                    word = parts[1]
                    break
        tool_str = f'<tool_call>{{"name": "lookup_dict", "arguments": {{"word": "{word}"}}}}</tool_call>'
        tool_desc = "先调 lookup_dict 查词典"
    elif cat == "search":
        tool_str = '<tool_call>{"name": "search_web", "arguments": {"query": "搜索关键词"}}</tool_call>'
        tool_desc = "先调 search_web 搜索"
    else:
        tool_str = '<tool_call>{"name": "lookup_dict", "arguments": {"word": "XXX"}}</tool_call>'
        tool_desc = "先调 lookup_dict 再调 search_web"

    prompt = f"""请为猫娘翻译bot生成一条带工具调用的回复。

【用户消息】{user_msg}

【要求】
1. {tool_desc}
2. 灯酱(にゃ)先抢手机喊"にゃ！我来查！"然后输出工具调用
3. 提酱(喵)补充解释，引用工具返回的信息
4. 灯酱最后插嘴撩猫
5. 工具调用格式: {tool_str}
6. 纯对话，喵/にゃ区分，无角色标记，无emoji
7. 150~350字

直接输出完整猫娘回复，不要前缀。"""

    result = call_api("你是训练数据生成器。为猫娘bot生成带tool_call的翻译查询回复。", prompt)
    if not result: return None

    content = result.strip()
    if len(content) < 60: return None
    if "<tool_call>" not in content: return None
    if "にゃ" not in content or "喵" not in content: return None
    if "【提酱】" in content or "【灯酱】" in content: return None

    return content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=None)
    args = parser.parse_args()
    n = args.parallel or PARALLEL

    data_file = DATA_DIR / "train_data_balanced.jsonl"
    with open(data_file, 'r', encoding='utf-8') as f:
        entries = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded: {len(entries)}")

    # Find bad tool entries (no <tool_call>)
    bad_indices = []
    for i, e in enumerate(entries):
        if e.get('_category') == 'tool' and '<tool_call>' not in e['messages'][2]['content']:
            bad_indices.append(i)

    print(f"Bad tool entries (no <tool_call>): {len(bad_indices)}")

    target = len(bad_indices)
    tasks = [(idx, entries[idx]) for idx in bad_indices]
    # 3x oversample for failures
    tasks = tasks * 3
    random.shuffle(tasks)

    # Resume from checkpoint
    ckpt_file = DATA_DIR / "_tool_fix_tmp.json"
    if ckpt_file.exists():
        with open(ckpt_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Resume: {len(results)} cached")
    else:
        results = {}

    def gen_and_save(idx, entry):
        global _success
        if _success >= target: return
        new_content = fix_one(entry)
        if new_content:
            with write_lock:
                results[str(idx)] = new_content
                if len(results) % 30 == 0:
                    with open(ckpt_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False)
            with _lock:
                _success += 1
                print(f"[{_success}/{target}] OK  idx={idx}", flush=True)
        else:
            with _lock:
                print(f"[{_success}/{target}] FAIL  idx={idx}", flush=True)

    print(f"Target: {target} | Pool: {len(tasks)} | Parallel: {n}")

    with ThreadPoolExecutor(max_workers=n) as executor:
        futs = []
        for idx, entry in tasks:
            if _success >= target:
                executor.shutdown(wait=False, cancel_futures=True)
                break
            futs.append(executor.submit(gen_and_save, idx, entry))
            time.sleep(0.03)
        for f in as_completed(futs):
            try: f.result()
            except: pass

    # Save final results
    with open(ckpt_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

    # Apply
    applied = 0
    for idx_str, new_content in results.items():
        idx = int(idx_str)
        entries[idx]["messages"][2]["content"] = new_content
        applied += 1

    with open(data_file, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nDone! Fixed: {applied}/{target} -> {data_file}")

if __name__ == "__main__":
    main()
