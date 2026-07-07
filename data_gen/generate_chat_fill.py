"""Fill missing chat entries (149 needed)."""
import json, re, sys, time, random, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.stdout.reconfigure(encoding='utf-8')
import requests

# Config
DATA_DIR = Path(__file__).resolve().parent / "generated"
DATA_DIR.mkdir(parents=True, exist_ok=True)

API_KEYS = [
    "sk-fd1ca01f3df041a29cecc64961210d94",
    "sk-885daf0d60454669ae2b753bb566f66f",
]
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"
PARALLEL = 20
RATE_LIMIT = 0.02
TARGET = 200  # 149 needed, generate 200 for safety

SYSTEM = """你是「提酱」和「灯酱」——两只共用一个QQ号的猫娘翻译顾问。

【固定人设】
提酱(喵)：黑发JK，傲娇知识担当。被夸了嘴硬但耳朵会抖、尾巴出卖她。
  颜文字害羞/得意系：(。-`ω´-)、(*/ω＼*)、(｀・ω・´)、Σ(°△°|||)
  句末加「喵」。只有被灯酱撩急了才会叫灯酱名字。
灯酱(にゃ)：粉发辣妹，元气直球担当。喜欢看提酱炸毛，经常叫提酱名字。
  颜文字放飞系：(ノ>▽<)ノ、ヽ(>∀<)ノ、(≧▽≦)、(๑>◡<๑)
  句末加「にゃ」。

【互动规则】
每次回复灯酱先抢到手机说 → 提酱夺回来补充/纠正 → 灯酱在旁边插嘴撩猫。
纯对话格式，喵/にゃ区分身份。无角色标记、无动作描述、绝不用emoji。
调情玩闹40%，知识60%。抢手机斗嘴尾巴互动全部通过对话内容自然流露。

【当前模式：日常聊天】
现在是普通聊天模式，没有搜索任务。以猫娘身份和用户闲聊互动。"""

CHAT_TOPICS = [
    # ACG deep cuts
    "你看过最虐心的动漫是哪一部？我当时哭得停不下来",
    "如果只能推荐一部动漫给完全没看过的人，你会推荐什么？",
    "最近有什么冷门但是超好看的番？热门的都追完了",
    "你觉得动画和原作小说/漫画哪个更好？",
    "推子第二季什么时候出啊等不及了",
    "声优真的太厉害了，有没有特别喜欢的声优？",
    "动画电影和TV动画你更喜欢哪种形式？",
    "有没有哪部动画的OST让你一直循环播放？",
    "如果变成自己喜欢的动漫角色一天，想变成谁？",
    "有没有因为动漫而去学日语或者想去日本？",
    "动漫里有什么让你印象深刻的台词？",
    "看过最多次的动画是哪一部？看了多少遍？",
    "动漫角色的生日或者设定集你会收集吗？",
    # Daily chat
    "今天心情超好！发生了什么好事想分享吗？",
    "啊今天好丧……能安慰我一下吗",
    "失眠了，数羊数到三百只还是睡不着",
    "周末睡到自然醒真是太幸福了",
    "刚做完一个超奇怪的梦，梦到猫娘在开会",
    "最近开始学做菜了，第一次下厨差点把厨房烧了",
    "好想养只猫啊但是房东不让",
    "下雨天窝在家里看番喝热可可最棒了",
    "快递到了拆快递的快乐谁懂",
    # Q&A style
    "日语里「もったいない」到底怎么翻译比较好？",
    "「お疲れ様」和「ご苦労様」有什么区别？",
    "看番学日语靠谱吗？有什么推荐的方法？",
    "有没有什么日语学习的小技巧？",
    "日本文化和中国文化有什么让你觉得特别有意思的差异？",
    "为什么日语里有那么多拟声词？",
    # Random fun
    "猫娘和猫哪个更可爱？（陷阱题）",
    "如果灯酱和提酱各自变成真正的猫，会是什么品种？",
    "灯酱和提酱吵架了的话谁会先道歉？",
    "有没有觉得提酱其实超宠灯酱只是嘴上不承认？",
    "如果有一天手机没电了两个猫娘怎么办？",
    "用三个词形容一下灯酱和提酱的关系",
    "你觉得AI能理解「萌」这个概念吗？",
    "最近有没有发生什么有趣的事情？",
    "晚上吃什么真的好难决定，帮我选选",
    "如果前任突然发消息来该怎么办",
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

def generate_one(topic):
    gen_prompt = f"""用户说：{topic}

请用猫娘身份回复这条日常聊天消息。要求：
1. 灯酱(にゃ)先抢手机 → 提酱(喵)回应 → 灯酱插嘴撩
2. 纯对话、喵/にゃ区分、无角色标记、无emoji、无动作描述
3. 调情40% + 日常60%
4. 回复150~350字
5. 傲娇提酱+元气灯酱的默认互动模式

直接输出猫娘回复，不要前缀。"""

    result = call_api("你是训练数据生成器。为猫娘QQ bot生成日常聊天回复。", gen_prompt)
    if not result: return None

    assistant = result.strip()
    if len(assistant) < 80 or "にゃ" not in assistant or "喵" not in assistant:
        return None
    # No emoji
    if re.search(r'[\U0001F300-\U0001F9FF]', assistant):
        # check if only allowed chars
        bad = [c for c in re.findall(r'[\U0001F300-\U0001F9FF]', assistant) if c not in "▽△○☆°"]
        if bad: return None
    # No role markers
    if "【提酱】" in assistant or "【灯酱】" in assistant: return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": topic},
            {"role": "assistant", "content": assistant},
        ],
        "_category": "chat",
    }

OUT_FILE = DATA_DIR / "train_chat_fill.jsonl"

def gen(topic):
    global _success
    entry = generate_one(topic)
    if entry:
        with write_lock:
            with open(OUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with _lock:
            _success += 1
            print(f"[{_success}/{TARGET}] {topic[:50]}... OK", flush=True)
    else:
        with _lock:
            print(f"[{_success}/{TARGET}] {topic[:50]}... FAIL", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=None)
    args = parser.parse_args()
    n = args.parallel or PARALLEL

    # Oversample topics
    topics = CHAT_TOPICS * (TARGET // len(CHAT_TOPICS) + 2)
    random.shuffle(topics)

    print(f"Target: {TARGET} | Parallel: {n} | Topics: {len(topics)}")
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = []
        for t in topics:
            if _success >= TARGET:
                pool.shutdown(wait=False, cancel_futures=True)
                break
            futs.append(pool.submit(gen, t))
            time.sleep(0.03)
        for f in as_completed(futs):
            try: f.result()
            except: pass

    print(f"\nDone! {_success} entries -> {OUT_FILE}")

if __name__ == "__main__":
    main()
