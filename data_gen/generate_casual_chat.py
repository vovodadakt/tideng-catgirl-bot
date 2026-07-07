"""
Generate casual chat / greeting training data for QQ group bot.
Target: 2818 entries, 15% of total data.
Round dist: 1轮 40%, 2-3轮 35%, 4-5轮 25%.
V2: Dynamic user message generation — each API call creates a UNIQUE casual chat topic + catgirl response.
No fixed topic pool — avoids duplicate data.
Runs on AutoDL with DeepSeek API, dual-key, 20 concurrent.
"""
import json, sys, time, random, argparse, re
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
TARGET = 2818

SYSTEM = """你是「缇酱」和「灯酱」——两只共用一个QQ号的猫娘翻译顾问。

【固定人设】
缇酱(喵)：黑猫，傲娇知识担当。被夸了嘴硬但耳朵会抖、尾巴出卖她。
  颜文字害羞/得意系：(。-`ω´-)、(*/ω＼*)、(｀・ω・´)、Σ(°△°|||)
  句末加「喵」。只有被灯酱撩急了才会叫灯酱名字。
灯酱(にゃ)：白猫，元气直球担当。喜欢看缇酱炸毛，经常叫缇酱名字。
  颜文字放飞系：(ノ>▽<)ノ、ヽ(>∀<)ノ、(≧▽≦)、(๑>◡<๑)
  句末加「にゃ」。

【互动规则】
每次回复灯酱先抢到手机说 → 缇酱夺回来补充/纠正 → 灯酱在旁边插嘴撩猫。
纯对话格式，喵/にゃ区分身份。无角色标记、无动作描述、绝不用emoji。
调情玩闹50%，日常聊天50%。抢手机斗嘴尾巴互动全部通过对话内容自然流露。

【当前模式：日常聊天/主动搭话】
现在是普通聊天模式。群里有人说话、打招呼、分享内容，两只猫娘主动回应互动。"""

# Scenario descriptions for the API to generate diverse user messages
SCENARIO_POOL = [
    # 日常问候
    "问候/打招呼（早安/晚安/上线/下线/好久不见/周末问候）",
    "新成员入群欢迎",
    # 分享推荐
    "聊最近看的新番/动画/漫画，分享观后感",
    "推荐/讨论动漫角色、声优、剧情",
    "聊最近看的视频/直播/MAD/手书/AMV",
    "分享B站/Youtube/Niconico上的有趣视频",
    "聊电影/日剧/特摄（假面骑士/奥特曼/超级战队）",
    "讨论轻小说/web小说/同人创作",
    # 日常互动
    "聊食物/零食/饮料/料理",
    "聊天气/季节/心情/日常琐事",
    "聊游戏（手游/主机/PC/音游/galgame）",
    "聊音乐/歌单/虚拟歌手/VOCALOID/演唱会",
    "聊宠物/猫/动物相关的可爱话题",
    "聊熬夜/失眠/早起/作息混乱",
    "聊购物/快递/周边/手办/痛包",
    "摸鱼/上班/上学/写作业吐槽",
    # 互动活动
    "邀请群友一起看番/直播/同步视听",
    "分享刚看到的资讯/新闻/动漫情报",
    "聊科技/AI/编程/数码产品（阿宅向）",
    "聊cosplay/漫展/同人展/线下活动",
    "聊日语学习/翻译/字幕组趣事",
    # 情感互动
    "撒娇/求安慰/求夸奖/吐槽不开心的事",
    "分享开心的事/庆祝/祝贺/节日祝福",
    "调侃/吐槽/群友互损",
    "聊脑洞/幻想/如果系列/排名讨论",
]

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
                "temperature": 0.9, "max_tokens": 2048, "top_p": 0.95,
            }, timeout=120)
            data = resp.json()
            if "choices" in data and len(data["choices"]) > 0:
                c = data["choices"][0]["message"]["content"]
                if c and len(c) > 60: return c
            else:
                err = str(data.get("error", ""))[:100]
                if "rate" in err.lower() or resp.status_code == 429:
                    time.sleep(10 * (attempt + 1))
            if attempt < max_retries - 1: time.sleep(3 * (attempt + 1))
        except:
            if attempt < max_retries - 1: time.sleep(5 * (attempt + 1))
    return None

def round_count(content):
    paras = [p.strip() for p in content.split('\n') if p.strip()]
    return max(1, len(paras))

def generate_one():
    """Generate one complete casual chat entry with dynamic user message."""
    r = random.random()
    if r < 0.40:
        target_rounds = 1
    elif r < 0.75:
        target_rounds = random.randint(2, 3)
    else:
        target_rounds = random.randint(4, 5)

    # Pick a random scenario to guide diversity
    scenario = random.choice(SCENARIO_POOL)

    if target_rounds == 1:
        speaker = random.choice(["只有缇酱(喵)", "只有灯酱(にゃ)"])
        prompt = f"""请生成一条QQ群猫娘bot的训练数据。你需要先构思一个自然的群友消息，然后生成猫娘回复。

【场景方向】{scenario}
【轮数】单轮，{speaker}一个人回复

【步骤】
第1步：先写一条自然的群友消息（10~40字，日常聊天风格，带一点ACG文化感）
第2步：然后生成{speaker}的回复（60~150字，保持猫娘语气）

【回复要求】
- 如果是缇酱：句末加「喵」，颜文字害羞/得意系：(。-`ω´-)、(*/ω＼*)、(｀・ω・´)
- 如果是灯酱：句末加「にゃ」，颜文字放飞系：(ノ>▽<)ノ、ヽ(>∀<)ノ、(≧▽≦)
- 纯对话，无角色标记，无emoji，无动作描述

【输出格式】
[USER]
（群友消息）
[ASSISTANT]
（猫娘回复）"""
    else:
        prompt = f"""请生成一条QQ群猫娘bot的训练数据。你需要先构思一个自然的群友消息，然后生成两只猫娘的{target_rounds}轮互动回复。

【场景方向】{scenario}
【轮数】{target_rounds}轮（灯酱→缇酱→灯酱…交替）

【步骤】
第1步：先写一条自然的群友消息（10~50字，日常聊天风格，带一点ACG文化感）
第2步：然后生成{target_rounds}轮猫娘回复（灯酱先抢手机→缇酱补充/吐槽→灯酱再撩猫）

【回复要求】
- 灯酱(にゃ)：白猫元气系，句末加「にゃ」，颜文字放飞
- 缇酱(喵)：黑猫傲娇系，句末加「喵」，颜文字害羞/得意
- 调情50%+日常50%，纯对话，喵/にゃ区分，无角色标记，无emoji
- 150~400字

【输出格式】
[USER]
（群友消息）
[ASSISTANT]
（猫娘多轮回复）"""

    gen_prompt = "你是训练数据生成器。请严格按照输出格式生成数据，确保每次生成不同的群友消息。"
    result = call_api(gen_prompt, prompt)
    if not result: return None

    # Parse the output
    result = result.strip()

    # Try [USER]/[ASSISTANT] format
    user_match = re.search(r'\[USER\]\s*\n?(.*?)\n\s*\[ASSISTANT\]', result, re.DOTALL)
    if user_match:
        user_msg = user_match.group(1).strip()
        assistant_start = result.index('[ASSISTANT]') + len('[ASSISTANT]')
        assistant = result[assistant_start:].strip()
    else:
        # Fallback: try to split by obvious markers
        lines = result.split('\n')
        # Look for first non-empty line as user, rest as assistant
        clean_lines = [l.strip() for l in lines if l.strip()]
        if len(clean_lines) >= 2:
            user_msg = clean_lines[0]
            assistant = '\n'.join(clean_lines[1:])
        else:
            return None

    # Validate
    if len(user_msg) < 6 or len(user_msg) > 100: return None
    if len(assistant) < 40: return None
    if "にゃ" not in assistant and "喵" not in assistant: return None

    # Clean any remaining markers
    assistant = assistant.replace('[ASSISTANT]', '').replace('[USER]', '').strip()
    user_msg = user_msg.replace('[USER]', '').strip()

    # Single round validation: only one catgirl speaks
    if target_rounds == 1:
        if "喵" in speaker:
            if "にゃ" in assistant: return None
        else:
            if "喵" in assistant: return None

    # No role markers
    if "【缇酱】" in assistant or "【灯酱】" in assistant: return None
    if "缇酱：" in assistant or "灯酱：" in assistant: return None

    # No emoji (except allowed kaomoji chars)
    emoji_pat = re.compile(r'[\U0001F300-\U0001F9FF]')
    bad = [c for c in emoji_pat.findall(assistant) if c not in "▽△○☆°"]
    if bad: return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant},
        ],
        "_category": "chat",
    }


OUT_FILE = DATA_DIR / "train_casual_chat.jsonl"

def gen(_dummy):
    global _success
    entry = generate_one()
    if entry:
        with write_lock:
            with open(OUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with _lock:
            _success += 1
            if _success % 50 == 0:
                print(f"[{_success}/{TARGET}] OK ({_success*100//TARGET}%)", flush=True)
    else:
        with _lock:
            print(f"[{_success}/{TARGET}] FAIL", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    n = args.parallel or PARALLEL

    print(f"Target: {TARGET} | Parallel: {n} | Dynamic topics v2")

    # If resume, count existing valid entries
    if args.resume:
        existing = 0
        if OUT_FILE.exists():
            with open(OUT_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            e = json.loads(line)
                            if len(e["messages"][1]["content"]) > 5:
                                existing += 1
                        except:
                            pass
        print(f"Resume mode: {existing} valid entries already")
        global _success
        _success = existing
    else:
        # Clean start
        if OUT_FILE.exists():
            OUT_FILE.unlink()

    # 3x oversample for quality filtering
    total_tasks = (TARGET - _success) * 3
    print(f"Total tasks planned: {total_tasks} (oversample 3x)")

    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = []
        for i in range(total_tasks):
            if _success >= TARGET:
                pool.shutdown(wait=False, cancel_futures=True)
                break
            futs.append(pool.submit(gen, i))
            time.sleep(0.03)
        for f in as_completed(futs):
            try: f.result()
            except: pass

    print(f"\nDone! {_success} entries -> {OUT_FILE}")

    # Show stats
    with open(OUT_FILE, "r", encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]

    # Unique user messages
    users = set(e["messages"][1]["content"] for e in entries)
    print(f"Unique user messages: {len(users)}/{len(entries)}")

    rd = {}
    for e in entries:
        rc = round_count(e["messages"][2]["content"])
        if rc >= 6: rc = "6+"
        rd[rc] = rd.get(rc, 0) + 1
    print("Round dist:")
    for k in sorted(rd.keys(), key=lambda x: (isinstance(x, str), x)):
        print(f"  {str(k)}轮: {rd[k]} ({rd[k]*100/len(entries):.1f}%)")

    # Show 3 samples
    print("\n=== Samples ===")
    for i, e in enumerate(entries[:3]):
        print(f"\n[{i}] User: {e['messages'][1]['content'][:80]}")
        print(f"    Assistant ({round_count(e['messages'][2]['content'])}r): {e['messages'][2]['content'][:200]}...")


if __name__ == "__main__":
    main()
