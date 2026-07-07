"""
Generate 3,000 L1_search training entries.
L1_search = user asks → bot searches → 灯酱 summarizes → 提酱 confirms with citations.
Key: search results are ACCURATE, no fake errors, no debate needed.

Usage:
  python generate_l1_search.py              # Auto-resume
  python generate_l1_search.py --limit 50   # Test with 50
  python generate_l1_search.py --from-scratch
"""
import json, os, re, sys, time, random, argparse, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.stdout.reconfigure(encoding='utf-8')
import requests

# ── Config ──
DATA_DIR = Path(__file__).resolve().parent / "generated"
DATA_DIR.mkdir(parents=True, exist_ok=True)

API_KEYS = [
    "sk-fd1ca01f3df041a29cecc64961210d94",
    "sk-885daf0d60454669ae2b753bb566f66f",
]
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"
PARALLEL = 20
BATCH_SIZE = 4
RATE_LIMIT = 0.02
TARGET = 3000

OUT_FILE = DATA_DIR / "train_l1_search_3000.jsonl"
CKPT_FILE = DATA_DIR / "checkpoint_l1_search.json"

key_index = 0
key_lock = Lock()
_key_last_call = {}
_thread_local = threading.local()
write_lock = Lock()
checkpoint_lock = Lock()

# ═══════════════════════════════════════════════════════════════
#  Unified System Prompt (L1_search mode)
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是「提酱」和「灯酱」——两只共用一个QQ号的猫娘翻译顾问。

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

【当前模式：搜索准确回答】
通过搜索工具获取了准确信息。
灯酱先总结搜索结果 → 提酱引用来源逐条确认无误 → 不制造假错误。
材料准确就不需要辩论，干净利落地给用户正确答案即可。

【今日猫娘模式】傲娇提酱×元气灯酱（默认模式）
提酱傲娇嘴硬但耳朵出卖她，灯酱元气直球喜欢撩提酱看炸毛。"""

# ═══════════════════════════════════════════════════════════════
#  Question Pools — searchable anime/manga/game trivia
# ═══════════════════════════════════════════════════════════════

ANIME_LIST = [
    "鬼灭之刃", "咒术回战", "葬送的芙莉莲", "间谍过家家", "孤独摇滚",
    "我的推是坏人大小姐", "药屋少女的呢喃", "魔法少女小圆", "进击的巨人",
    "钢之炼金术师", "命运石之门", "紫罗兰永恒花园", "冰菓", "吹响吧上低音号",
    "莉可丽丝", "更衣人偶坠入爱河", "夏日重现", "电锯人", "死亡笔记",
    "新世纪福音战士", "凉宫春日的忧郁", "CLANNAD", "化物语", "Fate/Zero",
    "Re:从零开始的异世界生活", "为美好的世界献上祝福", "无职转生", "86不存在的战区",
    "王者天下", "排球少年", "黑子的篮球", "Free!", "强风吹拂",
    "少女歌剧", "BanG Dream!", "Love Live!", "赛马娘", "公主连结Re:Dive",
    "原神", "崩坏星穹铁道", "明日方舟", "最终幻想XIV", "塞尔达传说",
    "宝可梦", "女神异闻录5", "尼尔自动人形", "只狼", "艾尔登法环",
]

QUESTION_TEMPLATES = {
    "character_info": {
        "weight": 800,
        "templates": [
            "{anime}里{character}的声优是谁？配过哪些角色？",
            "{anime}中{character}的能力/技能设定是什么？",
            "{anime}的{character}的背景故事是什么？",
            "{anime}里{character}和{character2}是什么关系？",
            "{anime}中{character}的结局是怎样的？",
            "{anime}的{character}有哪些经典台词/名场面？",
            "{anime}里{character}的年龄/身高/生日等设定数据？",
        ],
    },
    "plot_story": {
        "weight": 600,
        "templates": [
            "{anime}的结局是什么？",
            "{anime}里关于{plot_point}的剧情是怎样的？",
            "{anime}中{event}发生在哪一集/哪一卷？",
            "{anime}的剧情主线讲了什么故事？",
            "{anime}的世界观设定是什么？有什么特色？",
            "{anime}的故事发生在什么时代/地点？",
        ],
    },
    "production": {
        "weight": 500,
        "templates": [
            "{anime}是哪个动画公司制作的？他们还做过什么作品？",
            "{anime}的导演/监督是谁？有什么代表作？",
            "{anime}的主题曲/OP/ED是谁唱的？叫什么名字？",
            "{anime}的原作是什么形式？小说/漫画/游戏？",
            "{anime}是哪一年播出的？一共有多少集/季？",
            "{anime}剧场版的票房/口碑怎么样？",
        ],
    },
    "game_mechanics": {
        "weight": 400,
        "templates": [
            "{game}里{character}的强度怎么样？T几级别？",
            "{game}的{mechanic}系统怎么玩？有什么技巧？",
            "{game}中怎么获得{item}？",
            "{game}的最新版本/最新活动有什么新内容？",
            "{game}的剧情里关于{plot_point}的部分讲了什么？",
        ],
    },
    "trivia_fun": {
        "weight": 400,
        "templates": [
            "{anime}有什么有趣的制作幕后故事？",
            "{anime}里有什么容易被忽略的细节/伏笔？",
            "关于{anime}有哪些粉丝都不一定知道的冷知识？",
            "{anime}和{anime2}有什么相似之处或共同点？",
            "{anime}有哪些致敬/彩蛋/联动内容？",
        ],
    },
    "culture_language": {
        "weight": 300,
        "templates": [
            "{anime}里「{japanese_term}」这句台词是什么意思？在什么场景说的？",
            "{anime}中出现的{place}是真实存在的地方吗？",
            "{anime}里描写的{culture_element}在日本现实中是什么样的？",
            "{anime}标题/角色名字有什么含义或典故？",
        ],
    },
}

# Fill-in pools
# Generic descriptors ONLY — never random character names cross-universe
FILL_GENERIC_CHARS = [
    "主角", "女主角", "男主角", "反派BOSS", "最终BOSS",
    "老师/师父", "宠物/吉祥物", "二号主角", "配角",
]
FILL_GENERIC_RELATION = [
    "搭档", "对手", "恋人", "兄妹", "姐弟", "师徒", "宿敌",
    "青梅竹马", "前辈后辈", "同学", "队友", "仇人",
]

FILL_PLOTS = [
    "最终决战", "身份揭露", "主角觉醒", "黑化", "牺牲", "时间穿越",
    "平行世界", "失忆", "背叛", "复活", "魔王讨伐",
]

FILL_EVENTS = [
    "泳装回", "温泉回", "文化祭", "修学旅行", "圣诞回", "新年参拜",
    "情人节", "花火大会", "学园祭", "合宿训练",
]

FILL_MECHANICS = [
    "抽卡", "战斗", "养成", "装备", "技能树", "好感度",
    "联机", "PVP", "公会战", "活动副本", "深渊",
]

FILL_ITEMS = [
    "限定角色", "专武", "圣遗物", "突破素材", "皮肤", "称号",
    "坐骑", "限定道具", "活动奖励", "登录奖励",
]

FILL_TERMS = [
    "頑張れ", "まじで", "やばい", "すごい", "かわいい", "ありがとう",
    "ごめんなさい", "お疲れ様", "いただきます", "行ってきます",
    "ただいま", "おかえり", "すみません", "大丈夫", "ちょっと待って",
]

FILL_PLACES = [
    "秋叶原", "涉谷", "京都", "神奈川", "北海道", "冲绳", "池袋",
    "大阪", "名古屋", "镰仓", "富士山", "东京塔", "天空树",
]

FILL_CULTURE = [
    "学园祭", "部活", "お正月", "お盆", "花見", "花火大会",
    "修学旅行", "成人式", "七五三", "節分", "ひな祭り",
]

GAMES = [
    "原神", "崩坏星穹铁道", "明日方舟", "赛马娘", "公主连结",
    "Fate/Grand Order", "碧蓝航线", "蔚蓝档案", "偶像大师",
    "最终幻想XIV", "女神异闻录5", "塞尔达传说王国之泪",
    "艾尔登法环", "只狼", "怪物猎人", "东方Project",
]

FILL_ANIME2 = [
    "鬼灭之刃", "咒术回战", "进击的巨人", "死神", "海贼王",
    "火影忍者", "龙珠", "JOJO的奇妙冒险", "银魂", "犬夜叉",
    "名侦探柯南", "哆啦A梦",
]

def _fill_template(template):
    """Fill {placeholders} in a template string. Uses only generic descriptors for characters."""
    def _replacer(m):
        key = m.group(1)
        if key == "anime": return random.choice(ANIME_LIST)
        if key == "anime2": return random.choice(FILL_ANIME2)
        if key == "character": return random.choice(FILL_GENERIC_CHARS)
        if key == "character2": return random.choice(FILL_GENERIC_RELATION)
        if key == "plot_point": return random.choice(FILL_PLOTS)
        if key == "event": return random.choice(FILL_EVENTS)
        if key == "mechanic": return random.choice(FILL_MECHANICS)
        if key == "item": return random.choice(FILL_ITEMS)
        if key == "japanese_term": return random.choice(FILL_TERMS)
        if key == "place": return random.choice(FILL_PLACES)
        if key == "culture_element": return random.choice(FILL_CULTURE)
        if key == "game": return random.choice(GAMES)
        return m.group(0)
    return re.sub(r'\{(\w+)\}', _replacer, template)


def build_questions(target):
    """Build a diverse pool of L1_search questions."""
    questions = set()

    total_weight = sum(c["weight"] for c in QUESTION_TEMPLATES.values())

    while len(questions) < target * 3:  # 3x oversample for dedup
        # Pick a category by weight
        r = random.random() * total_weight
        cumulative = 0
        chosen_cat = None
        for cat_name, cfg in QUESTION_TEMPLATES.items():
            cumulative += cfg["weight"]
            if r <= cumulative:
                chosen_cat = cat_name
                break
        if not chosen_cat:
            chosen_cat = list(QUESTION_TEMPLATES.keys())[0]

        template = random.choice(QUESTION_TEMPLATES[chosen_cat]["templates"])
        question = _fill_template(template)

        # Skip too short questions
        if len(question) < 8:
            continue

        questions.add(question)

    return list(questions)[:target]


# ═══════════════════════════════════════════════════════════════
#  API Infrastructure
# ═══════════════════════════════════════════════════════════════

def next_key():
    global key_index
    with key_lock:
        k = API_KEYS[key_index]
        key_index = (key_index + 1) % len(API_KEYS)
    return k


def _get_session(api_key):
    if not hasattr(_thread_local, 'sessions'):
        _thread_local.sessions = {}
    if api_key not in _thread_local.sessions:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1, pool_maxsize=2, max_retries=2, pool_block=True)
        s.mount('https://', adapter)
        _thread_local.sessions[api_key] = s
    return _thread_local.sessions[api_key]


def save_entry(entry):
    with write_lock:
        with open(OUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_checkpoint(idx):
    with checkpoint_lock:
        with open(CKPT_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_idx": idx, "target": TARGET}, f)


def load_checkpoint():
    if CKPT_FILE.exists():
        try:
            with open(CKPT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"last_idx": 0, "target": TARGET}


def completed_questions():
    if not OUT_FILE.exists():
        return set()
    seen = set()
    # Try reading with different encodings
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin-1"]:
        try:
            with open(OUT_FILE, encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        entry = json.loads(line)
                        seen.add(entry["messages"][1]["content"])
                    except: pass
            break
        except:
            continue
    return seen


def call_api(system_prompt, user_prompt, max_retries=4):
    api_key = next_key()

    for attempt in range(max_retries):
        now = time.time()
        last = _key_last_call.get(api_key, 0)
        wait = RATE_LIMIT - (now - last)
        if wait > 0:
            time.sleep(wait)
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
                "temperature": 0.85,
                "max_tokens": 2048,
                "top_p": 0.95,
            }, timeout=120)
            data = resp.json()
            if "choices" in data and len(data["choices"]) > 0:
                c = data["choices"][0]["message"]["content"]
                if c and len(c) > 100:
                    return c
                else:
                    print(f"  API_EMPTY[{attempt}]: {len(c or '')}c", flush=True)
            else:
                err = str(data.get("error", ""))[:100]
                print(f"  API_ERR[{attempt}]: {resp.status_code} {err}", flush=True)
                if "rate" in err.lower() or resp.status_code == 429:
                    time.sleep(10 * (attempt + 1))
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
        except Exception as e:
            print(f"  API_NET[{attempt}]: {type(e).__name__}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    return None


# ═══════════════════════════════════════════════════════════════
#  Generation Logic
# ═══════════════════════════════════════════════════════════════

def build_generation_prompt(question: str) -> str:
    """Build the user prompt asking DeepSeek to generate a full L1_search entry."""
    return f"""请为以下用户搜索请求生成一条完整的训练数据。

## 用户搜索请求
{question}

## 要求

### 第1部分：搜索参考资料
先写一段详细的"搜索结果"，模拟搜索引擎或百科返回的内容。
要求：
- 包含具体人名、声优名、数据、年份等可查证事实
- 有多个信息点（3-6条），方便后续逐条确认
- 必须以 [参考资料] 开头，以 [/参考资料] 结尾（必须写闭合标签！）
- 中间附上 [来源: XX百科 / 维基百科 / 官方资料]
- 绝对不要用「（未找到相关结果）」这种占位符，必须写真实信息

### 第2部分：猫娘回复
紧接着 [/参考资料] 之后写猫娘的完整回复。严格遵循：

1. 先 <tool_call> 搜索工具调用
2. 灯酱(にゃ)抢到手机 → 元气兴奋地总结搜索结果（2-3个关键点）
3. 提酱(喵)夺回手机 → 逐条引用原文确认 ✓，说"这次没错""没什么要纠正的"
4. 灯酱插嘴撩猫 → 提酱傲娇回应
5. 不用角色标记、不用emoji、喵/にゃ区分、调情40%+信息60%
6. 总回复300~600字

输出格式：
[参考资料]
（详细的搜索结果，包含事实信息）
[/参考资料]

<tool_call>{{"name": "search_web", "arguments": {{"query": "..."}}}}</tool_call>

灯酱...にゃ！... (ノ>▽<)ノ

提酱...喵。(。-`ω´-)
根据资料，「引用原文」——确认 ✓
...

灯酱插嘴撩 (≧▽≦)
提酱傲娇 (｀・ω・´)

直接输出以上两部分，不要任何前缀说明。"""


def generate_one(question: str) -> dict | None:
    """Generate one L1_search entry."""
    user_prompt = build_generation_prompt(question)

    # Use a minimal system prompt to guide the generation
    gen_system = """你是一个训练数据生成器。你的任务是根据用户搜索请求，生成高质量的日语动画/游戏相关搜索参考资料和猫娘回复。
搜索参考资料必须真实准确，包含具体人名、数据、时间等可查证信息。
猫娘回复必须遵循格式规则：无角色标记、喵/にゃ区分、准确确认无误。"""

    result = call_api(gen_system, user_prompt)
    if not result:
        return None

    # Parse: split into reference and assistant response
    # Pattern: [参考资料] ... [/参考资料] ... <tool_call>...
    ref_match = re.search(r'\[参考资料\](.*?)\[/参考资料\]', result, re.DOTALL)
    if ref_match:
        ref_text = "[参考资料]\n" + ref_match.group(1).strip() + "\n[/参考资料]"
    else:
        # Generate a fallback reference from the question
        ref_text = "[参考资料]\n（未找到相关结果）\n[/参考资料]"

    # Assistant is everything after [/参考资料]
    assistant_parts = re.split(r'\[/参考资料\]', result, maxsplit=1)
    if len(assistant_parts) > 1:
        assistant_text = assistant_parts[1].strip()
    else:
        assistant_text = result.strip()

    # Clean up assistant: remove any leading [参考资料] blocks that might remain
    assistant_text = re.sub(r'^\[参考资料\].*?\[/参考资料\]', '', assistant_text, flags=re.DOTALL).strip()

    # If the result has no tool_call, prepend one
    if "<tool_call>" not in assistant_text:
        search_query = question[:80]
        tool_call = f'<tool_call>{{"name": "search_web", "arguments": {{"query": "{search_query}"}}}}</tool_call>\n\n'
        assistant_text = tool_call + assistant_text

    # If assistant text starts with another [参考资料] block, extract it
    # and add to ref_text. Handle both with and without closing [/参考资料]
    extra_ref = re.match(r'^\s*\[参考资料\](.*?)(?:\[/参考资料\]|\[来源[:：]|<tool_call>|灯酱|提酱)', assistant_text, re.DOTALL)
    if extra_ref:
        extra_content = extra_ref.group(1).strip()
        # Only use if it has real content (not placeholder)
        if extra_content and "未找到相关结果" not in extra_content and len(extra_content) > 30:
            ref_text = f"[参考资料]\n{extra_content}\n[/参考资料]"
            print(f"  [fix] extracted {len(extra_content)}ch ref from assistant", flush=True)
        # Remove the extracted block from assistant
        # Find where the real assistant content starts
        end_markers = ["<tool_call>", "灯酱", "提酱"]
        start_pos = 0
        for marker in end_markers:
            pos = assistant_text.find(marker)
            if pos > 0:
                start_pos = pos
                break
        if start_pos > 0:
            assistant_text = assistant_text[start_pos:].strip()

    # Build the full user message (question + reference)
    user_content = f"帮我查一下：{question}\n\n{ref_text}"

    # Validate
    if "にゃ" not in assistant_text:
        return None  # Missing 灯酱 marker
    if "喵" not in assistant_text:
        return None  # Missing 提酱 marker
    if len(assistant_text) < 120:
        return None  # Too short

    # Emoji check
    emoji_pattern = re.compile(r'[\U0001F300-\U0001F9FF]')
    bad = [e for e in emoji_pattern.findall(assistant_text) if e not in "▽△○☆°"]
    if bad:
        return None  # Has emoji

    # Role markers check
    if "【提酱】" in assistant_text or "【灯酱】" in assistant_text:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_text},
        ],
        "_category": "L1_search",
    }


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

# Global counters for concurrent generation
_success_count = 0
_fail_count = 0
_success_lock = Lock()
_prog_lock = Lock()  # for print ordering


def generate_with_tracking(args_tuple):
    """Generate one entry with progress tracking. Thread-safe."""
    global _success_count, _fail_count
    question, total_done, target = args_tuple

    entry = generate_one(question)
    if entry:
        save_entry(entry)
        with _success_lock:
            _success_count += 1
            current = _success_count + total_done
        with _prog_lock:
            print(f"[{current}/{target + total_done}] {question[:60]}... OK ({len(entry['messages'][2]['content'])}ch)", flush=True)
        return ("ok", entry)
    else:
        with _success_lock:
            _fail_count += 1
            current = _success_count + total_done
        with _prog_lock:
            print(f"[{current}/{target + total_done}] {question[:60]}... FAIL", flush=True)
        return ("fail", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--parallel", type=int, default=None)
    args = parser.parse_args()

    target = args.limit or TARGET
    n_parallel = args.parallel or PARALLEL

    # Load checkpoint
    if args.from_scratch and OUT_FILE.exists():
        OUT_FILE.unlink()
    if args.from_scratch and CKPT_FILE.exists():
        CKPT_FILE.unlink()

    ckpt = load_checkpoint()
    start_idx = ckpt["last_idx"]

    # Get completed questions
    done = completed_questions()
    total_done = len(done)
    print(f"已完成: {total_done} 条")

    # Build question pool
    questions = build_questions(target + 500)  # oversample
    # Filter out already done
    questions = [q for q in questions if q not in done]
    # Limit
    questions = questions[:target * 2]  # 2x for fallback on validation failures

    print(f"问题池: {len(questions)} 个")
    print(f"目标: {target} 条")
    print(f"从 #{start_idx} 开始")
    print(f"API Keys: {len(API_KEYS)} 个 → 双密钥并发")
    print(f"并行线程: {n_parallel}")
    print("-" * 50)

    # Build task list
    tasks = [(q, total_done, target) for q in questions]

    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        # Submit all tasks at once — executor handles the queue
        futures = [pool.submit(generate_with_tracking, t) for t in tasks]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"Thread error: {e}", flush=True)
            # Check if we've reached target
            if _success_count + total_done >= target:
                pool.shutdown(wait=False, cancel_futures=True)
                break

    print(f"\n{'='*50}")
    print(f"完成! 成功: {_success_count}, 失败: {_fail_count}")
    print(f"输出: {OUT_FILE}")
    print(f"总条目: {_success_count + total_done}")


if __name__ == "__main__":
    main()
