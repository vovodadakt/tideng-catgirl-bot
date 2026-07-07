"""
Fill missing tool entries (396 needed → generate 500).
Tool entries: user asks about JP word/slang → catgirls call dict/search → explain.

Format: <tool_call>{"name": "lookup_dict", "arguments": {"word": "xxx"}}</tool_call>
"""
import json, re, sys, time, random, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.stdout.reconfigure(encoding='utf-8')
import requests

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
TARGET = 500

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
调情玩闹40%，知识60%。

【工具使用】
你可以调用以下工具：
- lookup_dict: 查日语词典，获取读音、释义。参数: word (string)
- search_web: 搜索网络获取语言用例、语源。参数: query (string)
工具调用格式: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
调用工具后，根据返回的结果解释给用户。"""

# Japanese word pools for dict lookup
JP_WORDS_DICT = [
    "尊い", "エモい", "ワンチャン", "それな", "ぴえん", "すこ", "映える",
    "メンヘラ", "リア充", "厨二病", "ツンデレ", "ヤンデレ", "メスガキ",
    "詰んだ", "萎える", "キモい", "ウザい", "ダサい", "ヤバい", "マジで",
    "なう", "乙", "ガチで", "パクる", "サボる", "ダブる", "トラブる",
    "ググる", "スタバる", "タピる", "ディスる", "オワコン", "フロリダ",
    "陽キャ", "陰キャ", "コミュ症", "ガチ勢", "エンジョイ勢",
    "ワイ", "おk", "りょ", "おつ", "あけおめ", "メリクリ", "あざす",
    "イキる", "チキる", "ビビる", "モテる", "キレる", "イケる", "ダマる",
    "半端ない", "かっこいい", "しょうがない", "もちろん", "だいじょうぶ",
    "おもてなし", "頑張って", "めんどくさい", "はずかしい", "なつかしい",
    "お世辞", "建前", "本音", "空気を読む", "KY", "付和雷同",
    "ごちそうさま", "おじゃまします", "お世話になります", "恐れ入ります",
]

# Web search topics for slangs/terms
JP_WEB_TOPICS = [
    "若者言葉 流行り 2024", "今年の新語 ネットスラング", "Z世代 日本語 造語",
    "インスタ映え 語源", "TikTok バズり言葉", "ツイッター 流行語 語源",
    "2ちゃんねる 発祥 ネット用語", "オタク用語 最近",
    "ビジネス用語 カタカナ 新語", "和製英語 一覧 例",
    "業界用語 意味 わかりやすく", "方言 全国的 広がり",
    "JK語 流行り 一覧", "ギャル語 一覧 意味",
    "オノマトペ 日本語 面白い", "日本の美しい言葉 意味",
    "四字熟語 面白い 珍しい", "ことわざ 勘違い 誤用",
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
                if c and len(c) > 80: return c
            else:
                err = str(data.get("error",""))[:100]
                if "rate" in err.lower() or resp.status_code == 429:
                    time.sleep(10 * (attempt + 1))
            if attempt < max_retries - 1: time.sleep(3 * (attempt + 1))
        except:
            if attempt < max_retries - 1: time.sleep(5 * (attempt + 1))
    return None

def build_user_message(cat, word_or_topic):
    if cat == "dict":
        templates = [
            "「%s」这个词怎么读？什么意思？",
            "刚在群里看到「%s」，不懂什么意思",
            "查一下「%s」",
            "「%s」这个词的用法是什么？",
            "有人对我说「%s」，这什么意思啊？",
            "「%s」这个词常见吗？什么场合用？",
        ]
        return random.choice(templates) % word_or_topic
    elif cat == "web":
        templates = [
            "帮我搜一下%s是怎么回事",
            "最近经常看到%s，帮我查查",
            "群里在讨论%s，能帮我搜一下吗？",
            "%s是最近流行起来的吗？",
            "科普一下%s",
        ]
        return random.choice(templates) % word_or_topic
    elif cat == "multi":
        word, web = word_or_topic
        templates = [
            "查了词典里「%s」感觉信息不够，再帮我搜一下用法",
            "「%s」词典释义有点少，能搜一下网上怎么用的吗？",
            "刚才聊天出现了「%s」，查了词典感觉不全，搜一下%s",
        ]
        return random.choice(templates) % (word, web)

def build_gen_prompt(cat, user_msg, tool_config):
    if cat == "dict":
        tool_part = '<tool_call>{"name": "lookup_dict", "arguments": {"word": "XXX"}}</tool_call>'
    elif cat == "web":
        tool_part = '<tool_call>{"name": "search_web", "arguments": {"query": "搜索关键词"}}</tool_call>'
    elif cat == "multi":
        tool_part = '<tool_call>{"name": "lookup_dict", "arguments": {"word": "XXX"}}</tool_call>\n然后是\n<tool_call>{"name": "search_web", "arguments": {"query": "XXX"}}</tool_call>'

    return f"""用户消息：{user_msg}

请生成猫娘回复。必须包含{tool_config['count']}个工具调用。

工具调用格式：{tool_part}

【要求】
1. 灯酱(にゃ)先抢手机喊"にゃ！我来查！"然后输出工具调用
2. 如果是查词：先 <tool_call>lookup_dict</tool_call>，然后根据"词典结果"解释读音和意思
3. 如果是搜索：先 <tool_call>search_web</tool_call>，然后根据"搜索结果"说明语源和用例
4. 如果是两个工具：先dict后web，分别给出结果
5. 提酱(喵)补充/确认，灯酱最后插嘴撩猫
6. 纯对话、喵/にゃ区分、无角色标记、无emoji
7. 回复150~400字
8. 工具调用的word/query参数要和用户问的词一致！

直接输出完整猫娘回复，不要前缀。"""

def generate_one():
    # Randomly pick single tool or multi
    r = random.random()
    if r < 0.45:
        cat = "dict"
        word = random.choice(JP_WORDS_DICT)
        user_msg = build_user_message(cat, word)
        tool_config = {"count": 1, "tool": "lookup_dict", "param": word}
    elif r < 0.80:
        cat = "web"
        topic = random.choice(JP_WEB_TOPICS)
        user_msg = build_user_message(cat, topic)
        tool_config = {"count": 1, "tool": "search_web", "param": topic}
    else:
        cat = "multi"
        word = random.choice(JP_WORDS_DICT)
        web = "%s %s" % (word, random.choice(["語源","使い方","意味","例文"]))
        user_msg = build_user_message(cat, (word, web))
        tool_config = {"count": 2, "tool": "both", "param": (word, web)}

    gen_prompt = build_gen_prompt(cat, user_msg, tool_config)
    gen_system = "你是训练数据生成器。为猫娘翻译bot生成日语词汇查询回复。必须包含tool_call格式。"

    result = call_api(gen_system, gen_prompt)
    if not result: return None

    assistant = result.strip()
    if len(assistant) < 80: return None
    if "にゃ" not in assistant or "喵" not in assistant: return None
    if "<tool_call>" not in assistant: return None  # MUST have tool call

    # Emoji check
    emoji_pat = re.compile(r'[\U0001F300-\U0001F9FF]')
    bad = [c for c in emoji_pat.findall(assistant) if c not in "▽△○☆°"]
    if bad: return None
    if "【提酱】" in assistant or "【灯酱】" in assistant: return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant},
        ],
        "_category": "tool",
    }

OUT_FILE = DATA_DIR / "train_tool_fill.jsonl"

def gen(_dummy):
    global _success
    entry = generate_one()
    if entry:
        with write_lock:
            with open(OUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with _lock:
            _success += 1
            print(f"[{_success}/{TARGET}] OK", flush=True)
    else:
        print(f"[{_success}/{TARGET}] FAIL", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=None)
    args = parser.parse_args()
    n = args.parallel or PARALLEL

    print(f"Target: {TARGET} | Parallel: {n}")
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = []
        for i in range(TARGET * 3):  # 3x oversample
            if _success >= TARGET: break
            futs.append(pool.submit(gen, i))
            time.sleep(0.03)
        for f in as_completed(futs):
            try: f.result()
            except: pass

    print(f"\nDone! {_success} entries -> {OUT_FILE}")

if __name__ == "__main__":
    main()
