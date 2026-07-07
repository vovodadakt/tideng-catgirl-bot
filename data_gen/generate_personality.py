"""
Generate personality variant training data.
Target: 400 per variant (currently 129-152 each).
Uses DeepSeek API, dual-key concurrency, on AutoDL.

Usage:
  python generate_personality.py --variant all          # all 8 variants
  python generate_personality.py --variant amae         # single variant
  python generate_personality.py --parallel 20
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
RATE_LIMIT = 0.02
TARGET_PER_VARIANT = 400

key_index = 0
key_lock = Lock()
_key_last_call = {}
_thread_local = threading.local()
write_lock = Lock()

# ═══════════════════════════════════════════════════════════════
#  Personality Variant Definitions
# ═══════════════════════════════════════════════════════════════

PERSONALITIES = {
    "amae": {
        "label": "甘えん坊",
        "addon": "【今日猫娘模式】甘えん坊提酱×甘やかし灯酱\n"
                 "提酱今天是撒娇猫娘：软绵绵、想被夸、主动蹭灯酱。句末加「喵～」软软的。"
                 "颜文字偏撒娇系：(´・ω・`)。"
                 "灯酱今天变身宠溺模式：有求必应、温柔摸头、轻声细语。"
                 "互动：提酱主动蹭→灯酱受宠若惊→提酱「再夸我一句喵～」→灯酱全力全開で褒める。",
        "current": 152,
    },
    "kuudere": {
        "label": "クーデレ",
        "addon": "【今日猫娘模式】クーデレ提酱×粘人灯酱\n"
                 "提酱今天是クーデレ猫娘：话少、表情冷淡、回答简短但精准。"
                 "句末加「喵」但都是冷淡的一个字「…喵」。颜文字极少，偶尔( -_-)或( ˘_˘)。"
                 "但被灯酱持续直球后会不小心流露出关心——然后马上恢复扑克脸。"
                 "灯酱今天格外粘人：用加倍热情融化冰山。"
                 "互动：提酱冷→灯酱热→提酱偶尔破功「…吵死了」→灯酱狂喜。",
        "current": 151,
    },
    "tipsy": {
        "label": "微醺",
        "addon": "【今日猫娘模式】微醺猫娘：两只都喝了梅酒\n"
                 "两只猫娘都喝了点梅酒，ふわふわ飘飘然。回答有点绕弯但知识依然准确。"
                 "提酱比平时坦率，傲娇开关半失灵→说真心话的概率大幅上升。"
                 "灯酱比平时更大胆，直球威力×2，但自己也会说晕话。"
                 "互动：互相靠着→提酱不小心说漏真心话→灯酱震惊→两只一起脸红→岔开话题。",
        "current": 149,
    },
    "swap": {
        "label": "互换日",
        "addon": "【今日猫娘模式】互换日：元气提酱×傲娇灯酱\n"
                 "今天性格互换了！提酱变成元气直球担当，句末加「喵！」（感叹号多）。"
                 "灯酱变成傲娇担当，句末加「にゃ…」弱气版，被夸了脸红。"
                 "提酱学灯酱平时撩她的方式反攻，灯酱体验被撩的感觉、手忙脚乱。",
        "current": 144,
    },
    "serious": {
        "label": "认真模式",
        "addon": "【今日猫娘模式】认真模式：两只都超正经\n"
                 "今天两只猫娘都开启了认真模式。回答超级专业、条理清晰、几乎没有调情。"
                 "提酱「喵」频率降到最低（每3-4句一次）。灯酱「にゃ」也克制。"
                 "但偶尔会不小心露出猫娘本色——然后马上清嗓子恢复正经。",
        "current": 141,
    },
    "devil": {
        "label": "小恶魔",
        "addon": "【今日猫娘模式】小恶魔提酱×被撩炸毛灯酱\n"
                 "提酱今天是小恶魔猫娘：主动撩灯酱，看灯酱害羞为乐。"
                 "句末加「喵～」拖长音。颜文字偏挑逗系：( =^ω^= )、(￣▽￣)。"
                 "灯酱今天反被撩：平时是进攻方，今天被提酱反攻→手足无措→脸红→炸毛。"
                 "互动：提酱主动→灯酱懵→提酱追加→灯酱「にゃ…にゃー！！」→提酱得意。",
        "current": 138,
    },
    "yandere": {
        "label": "病娇Lite",
        "addon": "【今日猫娘模式】病娇 Lite：提酱ヤンデレ気味×灯酱戦慄\n"
                 "提酱今天是轻病娇猫娘：对灯酱的保护欲爆表→「灯酱は私のものにゃ…」"
                 "但只是轻度，不会真的可怕——更像过度粘人+独占欲。"
                 "灯酱被提酱的病娇气场震住→「に、にゃ…提酱你今天有点不一样にゃ…」"
                 "→但其实不讨厌被提酱这么在意，偶尔会故意逗提酱。"
                 "互动：恐怖→喜剧→甜。绝对不能真的吓人，本质是搞笑向。",
        "current": 138,
    },
    "airhead": {
        "label": "天然呆",
        "addon": "【今日猫娘模式】天然呆提酱×照顾人灯酱\n"
                 "提酱今天是天然呆猫娘：走神、前言不搭后语、被路过的蝴蝶吸引、认真到一半突然发呆。"
                 "句末加「喵」但经常走神忘了加。颜文字偏迷糊系：(。-ω-)、(_;)、ヽ(・∀・)ノ。"
                 "灯酱今天负责照顾提酱：一边把跑偏的提酱拉回来一边觉得这样的提酱超可爱。"
                 "互动：提酱走神→灯酱温柔拉回→提酱「あれ讲到哪了喵」→灯酱被萌到。",
        "current": 129,
    },
}

# Base system prompt (same as consolidated data)
BASE_SYSTEM = """你是「提酱」和「灯酱」——两只共用一个QQ号的猫娘翻译顾问。

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
调情玩闹40%，知识60%。抢手机斗嘴尾巴互动全部通过对话内容自然流露。"""

# ═══════════════════════════════════════════════════════════════
#  Question Pool — everyday chat topics
# ═══════════════════════════════════════════════════════════════

CHAT_TOPICS = [
    # Food & cooking
    "今天晚饭想吃什么？我纠结好久了",
    "你会做饭吗？有什么拿手菜？",
    "推荐一款适合夏天的饮品",
    "最近有没有吃到什么好吃的东西？",
    "甜党和咸党之争，你站哪边？",
    "最喜欢的日本零食是什么？",
    "深夜饿了怎么办？",
    "拉面和乌冬哪个更好吃？",

    # Weather & seasons
    "今天好热啊，怎么消暑？",
    "下雨天适合做什么？",
    "最喜欢的季节是哪个？",
    "台风天出不了门好无聊",
    "下雪了！要不要出去堆雪人？",
    "梅雨季衣服都晒不干怎么办",

    # Entertainment
    "最近有什么好看的新番推荐？",
    "你觉得今年最佳动画是哪个？",
    "推荐一部冷门但超好看的动漫",
    "人活着就是为了____（填空）",
    "深夜睡不着在看什么？",
    "有没有什么好听的日语歌？",
    "声优厨是什么体验？",
    "漫展/Comiket去过吗？好玩吗？",

    # Daily life
    "周末打算做什么？",
    "熬夜学习/工作好累，怎么办",
    "早上起不来有什么办法吗？",
    "最近在追什么剧/番？",
    "有没有什么奇怪的小习惯？",
    "房间太乱了懒得收拾",
    "网购又剁手了怎么办",

    # Hobbies & interests
    "你平时有什么爱好？",
    "喜欢旅游吗？最想去哪里？",
    "养过宠物吗？猫还是狗？",
    "最近有没有学会什么新技能？",
    "喜欢看书还是看电影？",
    "收集过什么东西吗？",

    # Relationships & emotions
    "心情不好的时候怎么调节？",
    "如何向喜欢的人表白？",
    "友情和爱情有什么区别？",
    "被朋友误解了怎么办？",
    "孤独的时候会做什么？",
    "有没有特别想念的人或事？",

    # Tech & internet
    "AI现在好厉害，你觉得未来会怎样？",
    "手机没电了怎么办，好焦虑",
    "推荐几个好用的App",
    "二次元圈子里有什么梗让你上头？",
    "社交媒体是不是让人更孤独了？",

    # ACG specific
    "你最喜欢的动画角色是谁？为什么？",
    "如果可以穿越到一个动画世界，你会选哪个？",
    "学日语是为了看动漫吗？",
    "Cosplay想尝试吗？会cos谁？",
    "你觉得什么样的动画才是神作？",
    "二刺猿和三次元的区别在哪里？",

    # Random fun
    "如果有一天的超能力，你会做什么？",
    "如果能和任何人(包括虚拟角色)吃一顿饭，选谁？",
    "突然中了彩票大奖怎么花？",
    "如果变成猫娘一天，最想做什么？",
    "世界末日来了，最后的晚餐吃什么？",
]


def make_system(variant_key):
    """Build full system prompt with personality addon."""
    pers = PERSONALITIES[variant_key]
    return BASE_SYSTEM + "\n\n" + pers["addon"]


# ═══════════════════════════════════════════════════════════════
#  API
# ═══════════════════════════════════════════════════════════════

def next_key():
    global key_index
    with key_lock:
        k = API_KEYS[key_index]
        key_index = (key_index + 1) % len(API_KEYS)
    return k


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
                if c and len(c) > 80:
                    return c
            else:
                err = str(data.get("error", ""))[:100]
                if "rate" in err.lower() or resp.status_code == 429:
                    time.sleep(10 * (attempt + 1))
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    return None


# ═══════════════════════════════════════════════════════════════
#  Generation
# ═══════════════════════════════════════════════════════════════

def build_gen_prompt(variant_key, topic):
    """Build the user prompt for DeepSeek to generate a personality-specific chat entry."""
    pers = PERSONALITIES[variant_key]

    return f"""请为猫娘QQ群bot生成一条日常聊天训练数据。当前猫娘人格模式：{pers['label']}。

## 用户消息
{topic}

## 当前人格规则
{pers['addon']}

## 回复要求
1. 灯酱(にゃ)先抢到手机说话 → 提酱(喵)夺回来补充/回应 → 灯酱插嘴撩猫
2. 严格按当前人格模式来！不要用默认人格（默认是傲娇提酱+元气灯酱），必须体现{pers['label']}的独特互动风格
3. 纯对话格式，喵/にゃ区分身份。无角色标记、无动作描述、绝不用emoji
4. 调情玩闹50%，日常聊天50%
5. 总回复150~400字
6. 颜文字要和当前人格匹配

直接输出完整猫娘回复，不要任何前缀说明。"""


def generate_one(variant_key, topic):
    """Generate one personality-specific chat entry."""
    system = make_system(variant_key)
    gen_prompt = build_gen_prompt(variant_key, topic)

    # Use minimal generation system prompt
    gen_system = f"你是训练数据生成器。为「{PERSONALITIES[variant_key]['label']}」人格模式生成猫娘聊天回复。严格按照给定人格规则输出。"

    result = call_api(gen_system, gen_prompt)
    if not result:
        return None

    assistant = result.strip()

    # Validate
    if len(assistant) < 80:
        return None
    if "にゃ" not in assistant or "喵" not in assistant:
        return None

    # Emoji check (allow only text/kaomoji)
    emoji_pattern = re.compile(r'[\U0001F300-\U0001F9FF]')
    bad_emoji = [c for c in emoji_pattern.findall(assistant) if c not in "▽△○☆°"]
    if bad_emoji:
        return None

    # Role marker check
    if "【提酱】" in assistant or "【灯酱】" in assistant:
        return None

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": topic},
            {"role": "assistant", "content": assistant},
        ],
        "_category": "chat",
        "_personality": PERSONALITIES[variant_key]['label'],
    }


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

_success = {}
_success_lock = Lock()
_fail_lock = Lock()
_prog_lock = Lock()


def generate_with_tracking(args_tuple):
    global _success
    variant_key, target = args_tuple

    topic = random.choice(CHAT_TOPICS)
    entry = generate_one(variant_key, topic)

    if entry:
        # Save
        out_file = DATA_DIR / f"train_personality_{variant_key}.jsonl"
        with write_lock:
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        with _success_lock:
            _success[variant_key] = _success.get(variant_key, 0) + 1

        current = _success.get(variant_key, 0)
        with _prog_lock:
            label = PERSONALITIES[variant_key]['label']
            print(f"[{current}/{target}] {label}: {topic[:40]}... OK", flush=True)
        return ("ok", variant_key)
    else:
        with _prog_lock:
            print(f"[{_success.get(variant_key, 0)}] {PERSONALITIES[variant_key]['label']}: {topic[:40]}... FAIL", flush=True)
        return ("fail", variant_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="all",
                        help="Variant key (amae,kuudere,tipsy,swap,serious,devil,yandere,airhead) or 'all'")
    parser.add_argument("--parallel", type=int, default=None)
    args = parser.parse_args()

    n_parallel = args.parallel or PARALLEL

    if args.variant == "all":
        variants = list(PERSONALITIES.keys())
    else:
        variants = [args.variant]

    # Calculate targets
    print("=" * 50)
    print("人格变体数据生成")
    print("=" * 50)
    targets = {}
    for vk in variants:
        pers = PERSONALITIES[vk]
        gap = max(0, TARGET_PER_VARIANT - pers["current"])
        targets[vk] = gap
        print(f"  {pers['label']:10s}  现有:{pers['current']}  目标:{TARGET_PER_VARIANT}  缺:{gap}")

    total_needed = sum(targets.values())
    print(f"\n总缺口: {total_needed} 条")
    print(f"并行: {n_parallel}")
    print(f"API Keys: {len(API_KEYS)}")

    # Build task queue
    tasks = []
    for vk in variants:
        for _ in range(min(targets[vk] * 3, 2000)):  # 3x oversample
            tasks.append((vk, targets[vk]))

    random.shuffle(tasks)
    print(f"任务队列: {len(tasks)} 个\n")

    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        futures = []
        for i, t in enumerate(tasks):
            # Stop if all variants reached target
            all_done = all(
                _success.get(vk, 0) >= targets[vk]
                for vk in variants
            )
            if all_done:
                print("所有目标达成！", flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                break

            f = pool.submit(generate_with_tracking, t)
            futures.append(f)
            time.sleep(0.05)

        for f in as_completed(futures):
            try:
                f.result()
            except:
                pass

    # Summary
    print(f"\n{'='*50}")
    print("完成！")
    for vk in variants:
        success_count = _success.get(vk, 0)
        pers = PERSONALITIES[vk]
        total_now = pers["current"] + success_count
        print(f"  {pers['label']:10s}  原始:{pers['current']}  + 新增:{success_count}  = {total_now}/{TARGET_PER_VARIANT}")

    print(f"\n输出目录: {DATA_DIR}")


if __name__ == "__main__":
    main()
