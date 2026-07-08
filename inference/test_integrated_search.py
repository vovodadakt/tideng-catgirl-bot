"""Integrated search: Bing + keyword-match + Weblio/Kotobank dict + Bilibili/ACG sources.
Upload to AutoDL server and run: python test_integrated_search.py
"""
import torch, re, time, random, os, math, json, urllib.request, urllib.parse
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from playwright.sync_api import sync_playwright

MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "/root/data/training/lora_output/ref_v4/final"
BROWSER_ARGS = ["--no-sandbox", "--disable-gpu", "--no-zygote", "--disable-setuid-sandbox"]

# ============== Domain scoring ==============
# Sites we WANT to deep-read (bonus score)
BONUS_DOMAINS = [
    # Wiki/百科类
    "baike.baidu.com", "moegirl", "fandom.com", "wiki", "fgo.wiki",
    # Bilibili & ACG 内容平台
    "bilibili.com/read",     # B站专栏
    "bilibili.com/opus",     # B站动态文章
    "bilibili.com/video",    # B站视频简介
    "acfun.cn",              # A站
    "ngabbs.com",            # NGA 论坛
    "gcores.com",            # 机核网
    "saraba1st.com",         # Stage1st
    "bangumi.tv",            # Bangumi
    "anitabi.cn",            # 圣地巡礼
]

# Sites to skip entirely (login-walled, low quality, or JS-rendered-only)
SKIP_DOMAINS = [
    "toutiao.com", "douyin.com", "xiaohongshu.com",
]

# Title keywords to skip (dictionary/administrative noise)
SKIP_TITLE_KW = [
    "汉语文字", "汉典", "新华字典", "高考分数线", "统计局",
    "百度地图", "高德地图", "经纬度", "的拼音", "的部首", "的笔顺",
]

# ============== Dictionary lookup (Weblio + Kotobank) ==============
def _fetch_meta_desc(url, timeout=8):
    """Quick HTTP fetch of meta description (no browser needed)."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "ja,zh-CN;q=0.9,zh;q=0.8",
        })
        html = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", errors="ignore")
        # Try meta description first
        desc = re.findall(r'<meta name="description"[^>]*content="([^"]+)"', html)
        if desc:
            return desc[0][:400]
        # Fallback: og:description
        og = re.findall(r'<meta property="og:description"[^>]*content="([^"]+)"', html)
        if og:
            return og[0][:400]
        # Last resort: first meaningful paragraph
        body = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.I)
        body = re.sub(r'<style[^>]*>.*?</style>', '', body, flags=re.DOTALL | re.I)
        body = re.sub(r'<[^>]+>', '\n', body)
        paras = [p.strip() for p in body.split('\n') if len(p.strip()) > 30]
        return paras[0][:400] if paras else ""
    except Exception:
        return ""

def lookup_jp_dict(question):
    """Extract Japanese terms from question, look up on Weblio + Kotobank.
    Returns list of (term, source, text) tuples.
    """
    # Extract Japanese character sequences (hiragana, katakana, kanji)
    jp_terms = re.findall(r'[぀-ゟ゠-ヿ一-鿿]{2,}', question)
    # Also extract quoted strings (Chinese/Japanese quotes)
    quoted = re.findall(r'[「『""]([^」』""]+)[」』""]', question)
    jp_terms.extend(quoted)
    # Deduplicate, keep order
    seen = set()
    terms = []
    for t in jp_terms:
        if t not in seen:
            seen.add(t)
            terms.append(t)

    # Filter out overly generic terms (dictionary noise)
    GENERIC_TERMS = {
        "最近", "今日", "明日", "昨日", "今週", "来週", "今年", "去年",
        "体系", "意味", "方法", "場合", "結果", "内容", "情報",
        "これ", "それ", "あれ", "ここ", "そこ", "もの", "こと",
    }
    terms = [t for t in terms if t not in GENERIC_TERMS]

    results = []
    for term in terms[:5]:  # max 5 lookups
        term_encoded = urllib.parse.quote(term)
        weblio_url = f"https://www.weblio.jp/content/{term_encoded}"
        weblio_text = _fetch_meta_desc(weblio_url)
        if weblio_text and len(weblio_text) > 20 and "見つかりません" not in weblio_text:
            results.append((term, "Weblio", weblio_text))

        kotobank_url = f"https://kotobank.jp/word/{term_encoded}"
        kotobank_text = _fetch_meta_desc(kotobank_url)
        if kotobank_text and len(kotobank_text) > 20 and "見つかりません" not in kotobank_text:
            results.append((term, "Kotobank", kotobank_text))

    return results

# ============== Keyword relevance scoring ==============
def keyword_score(question, title, snippet):
    """Score by keyword overlap with question + domain bonus/penalty."""
    # Extract meaningful keywords from question
    q_chars = set(re.findall(r'[一-鿿]{2,}|[a-zA-Z]{3,}|\d{4}', question.lower()))
    target = (title + " " + snippet).lower()
    score = 0
    for kw in q_chars:
        if kw in target:
            score += 1
    # Bonus: boost wiki/baike/fandom + bilibili/ACG content sites
    for bonus_domain in BONUS_DOMAINS:
        if bonus_domain in target.lower():
            score += 3
            break
    # Penalty: skip low-quality sources
    for bad in SKIP_DOMAINS:
        if bad in target.lower():
            score -= 5
    return score

# ============== RAG Engine ==============
def chunk_text(text, chunk_size=180):
    paragraphs = text.split('\n')
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para: continue
        if len(para) > chunk_size:
            sentences = re.split(r'[。！？.!?]', para)
            for sent in sentences:
                sent = sent.strip()
                if not sent: continue
                if len(current) + len(sent) < chunk_size:
                    current += sent + "。"
                else:
                    if current: chunks.append(current[:chunk_size])
                    current = sent + "。"
        else:
            if len(current) + len(para) < chunk_size:
                current += para + "\n"
            else:
                if current: chunks.append(current[:chunk_size])
                current = para + "\n"
    if current: chunks.append(current[:chunk_size])
    return chunks

def tokenize_cn(text):
    tokens = []
    for word in text.split():
        parts = re.findall(r'[一-鿿]|[a-zA-Z0-9]+|[^一-鿿\s]+', word)
        tokens.extend([p.lower() for p in parts if len(p) > 1 or ord(p) > 127])
    return tokens

def bm25_score(chunk_tokens, question_tokens, doc_freqs, total_docs, k1=1.5, b=0.75):
    doc_len = len(chunk_tokens)
    avg_len = 50
    score = 0
    for qt in question_tokens:
        if qt not in doc_freqs: continue
        df = doc_freqs[qt]
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
        tf = chunk_tokens.count(qt)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / avg_len)
        score += idf * numerator / denominator
    return score

def retrieve_top_chunks(question, all_text, top_k=5):
    chunks = chunk_text(all_text, chunk_size=180)
    q_tokens = tokenize_cn(question)
    all_tokens = [tokenize_cn(c) for c in chunks]
    doc_freqs = Counter()
    for tokens in all_tokens:
        doc_freqs.update(set(tokens))
    scored = []
    for i, (chunk, tokens) in enumerate(zip(chunks, all_tokens)):
        s = bm25_score(tokens, q_tokens, doc_freqs, len(chunks))
        scored.append((s, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored[:top_k] if s > 0]

# ============== Search ==============
def bing_search(page, query):
    """Search Bing, return scored results ready for deep-reading."""
    page.goto("https://cn.bing.com/", timeout=10000); page.wait_for_timeout(1000)
    sb = page.query_selector("#sb_form_q")
    sb.click(); sb.fill("")
    for c in query: sb.type(c, delay=30)
    page.wait_for_timeout(800); sb.press("Enter"); page.wait_for_timeout(2500)
    items = page.query_selector_all("li.b_algo")
    results = []
    for item in items[:10]:  # get more results to filter
        try:
            a = item.query_selector("h2 a")
            title, href = a.inner_text(), a.get_attribute("href")
        except: continue
        if not title or not href: continue
        if any(kw in title for kw in SKIP_TITLE_KW): continue
        # Check for skip domains
        if any(d in href.lower() for d in SKIP_DOMAINS): continue
        desc = ""
        try:
            d = item.query_selector(".b_caption") or item.query_selector(".b_lineclamp4")
            if d: desc = d.inner_text()[:300]
        except: pass
        results.append({"title": title, "href": href, "desc": desc})
    return results

def deep_read_url(page, url):
    """Open a URL and extract text content."""
    try:
        page.goto(url, wait_until="commit", timeout=15000)
        page.wait_for_timeout(2000)
        text = ""
        try:
            text = page.evaluate("() => document.body.innerText")
        except: pass
        if len(text) < 200:
            try:
                html = page.content()
                text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL|re.I)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL|re.I)
                text = re.sub(r'<[^>]+>', '\n', text)
                text = re.sub(r'\n{3,}', '\n\n', text)
                lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
                text = '\n'.join(lines)
            except: pass
        return text[:2500] if len(text) > 100 else ""
    except Exception as e:
        return ""

# ============== Inference ==============
def generate(model, tokenizer, msgs, max_tokens=250, temp=0.3):
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temp,
                             top_p=0.85, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

# ============== MAIN ==============
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
    device_map="auto", trust_remote_code=True, attn_implementation="sdpa")
base_model.eval()

print("Loading LoRA model...")
lora_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
    device_map="auto", trust_remote_code=True, attn_implementation="sdpa")
lora_model = PeftModel.from_pretrained(lora_model, LORA_PATH)
lora_model.eval()
print("Both models ready.\n")

CATGIRL_SYSTEM = (
    "你是「提酱」和「灯酱」——两只共用一个QQ号的猫娘翻译顾问。"
    "提酱是傲娇知识担当，句末加「喵」，颜文字偏害羞/得意系：(。-`ω´-)、(*/ω＼*)、(｀・ω・´)、Σ(°△°|||)。"
    "灯酱是元气直球担当，句末加「にゃ」，颜文字偏放飞系：(ノ>▽<)ノ、ヽ(>∀<)ノ、(≧▽≦)、(๑>◡<๑)。"
    "每次回复是两只猫娘抢同一部手机打字——灯酱常抢到先说，提酱夺回来补知识，灯酱在旁边插嘴撩猫。"
    "不用任何角色标记或动作描述，纯靠语尾和颜文字区分身份。"
    "调情玩闹约40%，翻译知识约60%。绝不用emoji。"
    "灯酱经常叫提酱名字，提酱只有被撩急了才叫灯酱。"
    "抢手机、斗嘴、尾巴耳朵互动全部通过对话内容自然流露。"
)

tests = [
    ("《葬送的芙莉莲》讲了什么？为什么评分那么高？",
     "葬送的芙莉莲 动画 剧情 评价"),
    ("科普一下 Type-Moon 的「根源」和「五大魔法」体系",
     "Type-Moon 型月 根源 五大魔法"),
    ("最近2025~2026年有什么霸权新番？",
     "2025年 2026年 霸权番 动漫 新番"),
    ("「強い者は更に努力する」这个日文名言出处是哪里？",
     "強い者は更に努力する 名言 出典 誰"),
]

print("Starting browser...")
p = sync_playwright().start()
browser = p.chromium.launch(headless=True, args=BROWSER_ARGS)
context = browser.new_context(
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    locale="zh-CN", viewport={"width": 1920, "height": 1080},
)
context.add_init_script("""
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    window.chrome = {runtime: {}};
""")
page = context.new_page()
page.goto("https://cn.bing.com/", timeout=15000)
page.wait_for_timeout(1500)
print("Browser ready.\n")

for question, query_bing in tests:
    print(f"{'='*60}")
    print(f"Q: {question}")

    # ===== Step 0: Dictionary lookup for Japanese terms =====
    print("  --- Dict lookup (Weblio/Kotobank) ---")
    dict_results = lookup_jp_dict(question)
    if dict_results:
        for term, source, text in dict_results:
            print(f"    [{source}] {term}: {text[:80]}...")
    else:
        print(f"    (no Japanese terms found or no results)")

    # ===== Step 1: Bing search =====
    results = bing_search(page, query_bing)
    print(f"  Bing: {len(results)} results")

    # ===== Step 2: Keyword-match score each result =====
    scored = []
    for r in results:
        s = keyword_score(question, r["title"], r["desc"])
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"  Top keyword matches:")
    for s, r in scored[:5]:
        domain_hint = ""
        for bd in BONUS_DOMAINS:
            if bd in r.get("href", "").lower():
                domain_hint = f" [BONUS:{bd}]"
                break
        print(f"    score={s:+.0f} | {r['title'][:60]}{domain_hint}")

    # ===== Step 3: Deep-read top 3 scored results =====
    deep_texts = []
    for s, r in scored[:3]:
        if s < 0:
            print(f"    [skip] score too low: {r['title'][:50]}")
            continue
        print(f"    Deep reading: {r['title'][:50]}...")
        content = deep_read_url(page, r["href"])
        if content:
            deep_texts.append(f"[来源: {r['title']}]\n{content}")
            print(f"      OK: {len(content)} chars")
        else:
            print(f"      SKIP: empty or blocked")

    # ===== Step 4: Assemble all material =====
    # 4a: Dictionary results
    dict_text = ""
    if dict_results:
        dict_lines = []
        for term, source, text in dict_results:
            dict_lines.append(f"[{source}词典] {term}：{text}")
        dict_text = "【翻译参考资料 - 日语词典】\n" + "\n".join(dict_lines)

    # 4b: Bing snippets (all results, including those not deep-read)
    snippets_text = "\n".join([f"[{r['title']}]\n{r['desc']}" for _, r in scored if r['desc']])

    # 4c: Merge all sources
    all_material = ""
    if dict_text:
        all_material += dict_text + "\n\n"
    all_material += "【搜索摘要】\n" + snippets_text[:2000]
    if deep_texts:
        all_material += "\n\n【深度阅读】\n" + "\n\n---\n\n".join(deep_texts)

    print(f"  Dict entries: {len(dict_results)}")
    print(f"  Total material: {len(all_material)} chars")

    # ===== Step 5: RAG =====
    top_chunks = retrieve_top_chunks(question, all_material, top_k=5)
    rag_text = "\n\n---\n\n".join(top_chunks)

    # ===== Step 6: Base model neutral answer =====
    print(f"\n  --- STEP 1: Base model neutral answer ---")
    step1_prompt = f"""你是一个严谨的AI助手。请仔细阅读以下参考资料，然后回答用户的问题。

[参考资料]
{rag_text}
[/参考资料]

问题：{question}

回答规则：
1. 只使用资料中明确提到的信息
2. 资料中没有的信息，明确说"资料中未提及"
3. 不要编造、不要推测、不要补充资料中没有的内容
4. 用中文简洁回答，列出要点"""

    step1_msgs = [{"role": "user", "content": step1_prompt}]
    neutral = generate(base_model, tokenizer, step1_msgs, max_tokens=250)
    for line in neutral.strip().split("\n")[:12]:
        print(f"    {line}")

    # ===== Step 7: LoRA style transfer =====
    print(f"\n  --- STEP 2: LoRA catgirl answer ---")
    step2_prompt = f"""下面是一个AI助手对用户问题的回答（内容是准确的）。请你用猫娘角色扮演的方式重新表达这些内容。

**铁律（违反会被删除）：**
- 必须完整保留下面「事实回答」中的所有信息，一条都不能少、不能改
- 绝对不能添加「事实回答」中没有的新事实、新名字、新数据
- 如果「事实回答」说"资料中未提及"或"没找到"，你必须也说不知道、没找到
- 你只负责换猫娘语气，不负责补充内容

[事实回答 - 内容准确，不可修改]
{neutral}
[/事实回答]

用户原问题：{question}

用提酱和灯酱的猫娘风格重新表达："""

    step2_msgs = [
        {"role": "system", "content": CATGIRL_SYSTEM},
        {"role": "user", "content": step2_prompt}
    ]
    catgirl = generate(lora_model, tokenizer, step2_msgs, max_tokens=350, temp=0.7)
    for line in catgirl.strip().split("\n")[:15]:
        print(f"    {line}")

    print()
    time.sleep(random.uniform(2.0, 3.0))

browser.close()
p.stop()
print("DONE!")
