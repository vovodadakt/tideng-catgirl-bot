"""Direct encyclopedia API search — no Bing site: garbage.
R1: Moegirl API + Wikipedia API + Baidu Baike search-box → deep read
R2: Extract entities from R1 → Bing general search supplements
Base model summarizes → output
"""
import torch, re, time, json, urllib.request, urllib.parse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from playwright.sync_api import sync_playwright

MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2.5-3B-Instruct"
BROWSER_ARGS = ["--no-sandbox", "--disable-gpu", "--no-zygote", "--disable-setuid-sandbox"]

# ══════════════════ Encyclopedia APIs ══════════════════

def search_moegirl(query, max_results=5):
    """Moegirl MediaWiki API search."""
    params = urllib.parse.urlencode({
        "action": "query", "list": "search",
        "srsearch": query, "srlimit": max_results,
        "format": "json"
    })
    url = "https://zh.moegirl.org.cn/api.php?" + params
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "CatgirlBot/1.0 (ACG research assistant)"
        })
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        results = []
        for r in data.get("query", {}).get("search", []):
            page_url = "https://zh.moegirl.org.cn/" + urllib.parse.quote(r["title"])
            results.append({
                "title": r["title"], "href": page_url,
                "desc": re.sub(r'<[^>]+>', '', r.get("snippet", ""))[:300],
                "source": "moegirl"
            })
        return results
    except Exception as e:
        print("    [moegirl API] " + str(e))
        return []


def search_wikipedia(query, max_results=5):
    """Wikipedia ZH MediaWiki API search."""
    params = urllib.parse.urlencode({
        "action": "query", "list": "search",
        "srsearch": query, "srlimit": max_results,
        "format": "json"
    })
    url = "https://zh.wikipedia.org/w/api.php?" + params
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "CatgirlBot/1.0 (ACG research assistant)"
        })
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        results = []
        for r in data.get("query", {}).get("search", []):
            page_url = "https://zh.wikipedia.org/wiki/" + urllib.parse.quote(r["title"])
            results.append({
                "title": r["title"], "href": page_url,
                "desc": re.sub(r'<[^>]+>', '', r.get("snippet", ""))[:300],
                "source": "wikipedia"
            })
        return results
    except Exception as e:
        print("    [wiki API] " + str(e))
        return []


def search_baidu_baike(page, query, max_results=5):
    """Baidu Baike search via simulated input."""
    try:
        q_enc = urllib.parse.quote(query)
        page.goto("https://baike.baidu.com/search?word=" + q_enc, timeout=12000)
        page.wait_for_timeout(2000)

        # Try result list
        items = page.query_selector_all(".result-item, .search-list dd, .search-list a")
        if not items:
            items = page.query_selector_all("a[href*='/item/']")

        results = []
        seen = set()
        for item in items:
            try:
                a = item.query_selector("a") or item
                href = a.get_attribute("href")
                if not href or "/item/" not in href:
                    continue
                if href.startswith("/"):
                    href = "https://baike.baidu.com" + href
                title = a.inner_text().strip()
                if not title or title in seen or len(title) < 2:
                    continue
                seen.add(title)
                desc = ""
                try:
                    parent = item.query_selector("..") or item
                    desc_el = parent.query_selector(".result-summary, .search-desc, dl dd, p")
                    if desc_el:
                        desc = desc_el.inner_text()[:300]
                except:
                    pass
                results.append({
                    "title": title, "href": href, "desc": desc, "source": "baike"
                })
            except:
                continue
            if len(results) >= max_results:
                break

        return results
    except Exception as e:
        print("    [baike] " + str(e))
        return []


# ══════════════════ Generic helpers ══════════════════

def bing_search_general(page, query, max_results=8):
    """General Bing search for R2 supplements."""
    try:
        page.goto("https://cn.bing.com/", timeout=8000)
        page.wait_for_timeout(600)
        sb = page.query_selector("#sb_form_q")
        sb.click(); sb.fill("")
        for c in query: sb.type(c, delay=15)
        page.wait_for_timeout(400); sb.press("Enter")
        page.wait_for_timeout(2500)
        items = page.query_selector_all("li.b_algo")
        results = []
        for item in items[:max_results]:
            try:
                a = item.query_selector("h2 a")
                title, href = a.inner_text(), a.get_attribute("href")
                desc = ""
                try:
                    d = item.query_selector(".b_caption") or item.query_selector(".b_lineclamp4")
                    if d: desc = d.inner_text()[:300]
                except: pass
                if title and href: results.append({"title": title, "href": href, "desc": desc})
            except: continue
        return results
    except: return []


def deep_read_url(page, url):
    """Extract text from a URL."""
    try:
        page.goto(url, wait_until="commit", timeout=12000)
        page.wait_for_timeout(2000)
        text = page.evaluate("() => document.body.innerText")
        if len(text) < 200:
            html = page.content()
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.I)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.I)
            text = re.sub(r'<[^>]+>', '\n', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
            text = '\n'.join(lines)
        return text[:3000] if len(text) > 100 else ""
    except: return ""


def extract_key_entities(text, question):
    """Extract key terms from clean encyclopedia text for R2 expansion."""
    entities = set()

    # Japanese terms
    jp = re.findall(r'[ァ-ヴー]{2,}|[ぁ-ゟ]{2,}', text)
    entities.update(jp[:5])

    # Quoted / bracketed terms
    quoted = re.findall(r'[「『""《]([^」』""》]{2,20})[」』""》]', text)
    entities.update(quoted)

    # Proper nouns: Chinese character sequences 2-6 chars
    cn = re.findall(r'[一-鿿]{2,6}', text)
    # Filter to meaningful nouns by frequency + position
    from collections import Counter
    freq = Counter(cn)
    # Keep terms that appear with other context words
    meaningful = {k for k, v in freq.items() if v >= 2 and len(k) >= 2}
    entities.update(list(meaningful)[:8])

    # Also from question
    q_terms = re.findall(r'[一-鿿]{2,6}|[ァ-ヴー]{2,}', question)
    entities.update(q_terms[:5])

    noise = {"一个","可以","没有","不是","这个","那个","我们","他们",
             "进行","使用","通过","其中","以及","因此","所以","但是",
             "目前","已经","还有","其他","关于","因为","如果","虽然",
             "什么","怎么","为什么","怎么样","有什么","是不是","能不能",
             "哪些","还是","就是","每个","所有","一种","不过","同时",
             "这样","那样","这种","那种","一般","比较","非常","特别",
             "基本","主要","可能","应该","一定","需要","不过","甚至",
             "并且","内容","资料","来源","参考","链接","相关","搜索"}
    entities = {e for e in entities if e not in noise and len(e) >= 2}

    return list(entities)[:6]


def generate(model, tokenizer, msgs, max_tokens=500, temp=0.3):
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temp,
                             top_p=0.85, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)


# ══════════════════ MAIN ══════════════════

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
    device_map="auto", trust_remote_code=True, attn_implementation="sdpa")
base_model.eval()
print("Ready.\n")

tests = [
    ("Q1-鬼灭之刃", "鬼灭之刃里水之呼吸的使用者有哪些？水之呼吸有什么招式？"),
    ("Q2-日语敬语", "お疲れ様です和ご苦労様です有什么区别？翻译时怎么处理？"),
    ("Q3-咒术回战", "咒术回战里五条悟的领域展开无量空处具体效果是什么？"),
    ("Q4-间谍过家家", "间谍过家家动画有几季？剧场版讲了什么？"),
    ("Q5-网络用语", "尊い在ACG圈是什么意思？怎么翻译成中文？"),
    ("Q6-ACG设定", "葬送的芙莉莲主角芙莉莲是什么人？故事讲了什么？"),
    ("Q7-冷门角色", "孤独摇滚里PA小姐的本名是什么？"),
    ("Q8-翻译", "ツンデレ怎么翻译成中文？"),
]

print("Starting browser...")
p = sync_playwright().start()
browser = p.chromium.launch(headless=True, args=BROWSER_ARGS)
context = browser.new_context(
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    locale="zh-CN", viewport={"width": 1920, "height": 1080},
)
context.add_init_script("""
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    window.chrome = {runtime: {}};
""")
page = context.new_page()
print("Browser ready.\n")

for label, question in tests:
    print("=" * 55)
    print(label + ": " + question[:65])
    print("=" * 55)

    t0 = time.time()

    # ═══ ROUND 1: Direct encyclopedia APIs ═══
    print("  [R1] API calls: moegirl + wiki + baike...")

    # All three in parallel (sequential HTTP calls but no dependency)
    moegirl_results = search_moegirl(question)
    wiki_results = search_wikipedia(question)
    baike_results = search_baidu_baike(page, question)

    print("    moegirl=%d | wiki=%d | baike=%d" % (
        len(moegirl_results), len(wiki_results), len(baike_results)))

    # Merge: baike first (best quality for Chinese ACG facts), then moegirl, then wiki
    r1_all = baike_results + moegirl_results + wiki_results

    # Deep read top 6
    r1_deep = []
    for r in r1_all[:6]:
        content = deep_read_url(page, r["href"])
        if content:
            label_src = "[" + r["source"] + "] " + r["title"]
            r1_deep.append(label_src + "\n" + content)
            print("    + %s: %s... (%dc)" % (r["source"], r["title"][:40], len(content)))
        else:
            print("    - %s: %s... (empty)" % (r["source"], r["title"][:40]))

    t1 = time.time()

    # ═══ ROUND 2: Extract entities + Bing supplement ═══
    r1_text = "\n\n---\n\n".join(r1_deep) if r1_deep else ""
    entities = extract_key_entities(r1_text, question)
    print("  [R2] Entities: " + str(entities[:6]))

    r2_deep = []
    if entities:
        seen_urls = {r["href"] for r in r1_all}
        for ent in entities[:4]:
            bing_results = bing_search_general(page, ent, max_results=3)
            for br in bing_results:
                if br["href"] in seen_urls:
                    continue
                if len(r2_deep) >= 3:
                    break
                content = deep_read_url(page, br["href"])
                if content:
                    r2_deep.append("[R2] " + br["title"] + "\n" + content)
                    seen_urls.add(br["href"])
                    print("    + R2: %s... (%dc)" % (br["title"][:40], len(content)))
            if len(r2_deep) >= 3:
                break

    t2 = time.time()

    # ═══ Base model summary ═══
    all_material = "=== Round 1 (encyclopedias) ===\n" + r1_text
    if r2_deep:
        all_material += "\n\n=== Round 2 (supplements) ===\n" + "\n\n---\n\n".join(r2_deep)

    total_chars = len(all_material)
    trimmed = all_material[:8000]

    SUMMARIZE_PROMPT = (
        "你是一个严谨的AI助手。请根据以下资料回答用户问题。\n\n"
        "[资料]\n" + trimmed + "\n[/资料]\n\n"
        "用户问题：" + question + "\n\n"
        "回答规则：\n"
        "1. 只使用资料中明确提到的信息，逐条列出\n"
        "2. 资料中没有的信息，明确写未知\n"
        "3. 不要编造、不要推测、不要补充\n"
        "4. 用简洁中文列出要点"
    )

    summary_msgs = [{"role": "user", "content": SUMMARIZE_PROMPT}]
    summary = generate(base_model, tokenizer, summary_msgs, max_tokens=500, temp=0.3)
    t3 = time.time()

    # ═══ Output ═══
    print("")
    print("  R1: %.1fs | R2: %.1fs | Summary: %.1fs | TOTAL: %.1fs" % (t1-t0, t2-t1, t3-t2, t3-t0))
    print("  Material: %d chars (%d R1 deep reads, %d R2)" % (total_chars, len(r1_deep), len(r2_deep)))
    print("  --- Summary ---")
    for line in summary.strip().split("\n")[:20]:
        print("  " + line)
    print("")

browser.close()
p.stop()
print("DONE!")
