"""Two-round search + Base model summary test.
ROUND 1: Search Baidu Baike, Moegirl, Wikipedia via site: queries
ROUND 2: Extract key entities, search Bing for supplements
Then: Base model summarizes into structured facts.
"""
import torch, re, time, random, os, urllib.request, urllib.parse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from playwright.sync_api import sync_playwright

MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2.5-3B-Instruct"
BROWSER_ARGS = ["--no-sandbox", "--disable-gpu", "--no-zygote", "--disable-setuid-sandbox"]

# ── Search helpers ──
def bing_search_site(page, query, site, timeout=3000):
    full_query = "site:" + site + " " + query
    try:
        page.goto("https://cn.bing.com/", timeout=8000)
        page.wait_for_timeout(600)
        sb = page.query_selector("#sb_form_q")
        sb.click()
        sb.fill("")
        for c in full_query:
            sb.type(c, delay=15)
        page.wait_for_timeout(400)
        sb.press("Enter")
        page.wait_for_timeout(timeout)
        items = page.query_selector_all("li.b_algo")
        results = []
        for item in items[:5]:
            try:
                a = item.query_selector("h2 a")
                title = a.inner_text()
                href = a.get_attribute("href")
                desc = ""
                try:
                    d = item.query_selector(".b_caption") or item.query_selector(".b_lineclamp4")
                    if d:
                        desc = d.inner_text()[:300]
                except:
                    pass
                if title and href:
                    results.append({"title": title, "href": href, "desc": desc, "site": site})
            except:
                continue
        return results
    except Exception as e:
        print("    site:" + site + " error: " + str(e))
        return []


def bing_search_general(page, query):
    try:
        page.goto("https://cn.bing.com/", timeout=8000)
        page.wait_for_timeout(600)
        sb = page.query_selector("#sb_form_q")
        sb.click()
        sb.fill("")
        for c in query:
            sb.type(c, delay=15)
        page.wait_for_timeout(400)
        sb.press("Enter")
        page.wait_for_timeout(2500)
        items = page.query_selector_all("li.b_algo")
        results = []
        for item in items[:8]:
            try:
                a = item.query_selector("h2 a")
                title = a.inner_text()
                href = a.get_attribute("href")
                desc = ""
                try:
                    d = item.query_selector(".b_caption") or item.query_selector(".b_lineclamp4")
                    if d:
                        desc = d.inner_text()[:300]
                except:
                    pass
                if title and href:
                    results.append({"title": title, "href": href, "desc": desc})
            except:
                continue
        return results
    except:
        return []


def deep_read_url(page, url):
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
    except:
        return ""


def extract_key_entities(text, question):
    entities = set()

    quoted = re.findall(r'[「『""《]([^」』""》]{2,20})[」』""》]', text)
    entities.update(quoted)

    jp = re.findall(r'[ァ-ヴー]{2,}|[ぁ-ゟ]{2,}', text)
    entities.update(jp[:5])

    cn_names = re.findall(r'(?:使用者|角色|人物|作者|声优)[：:]\s*([^\s,，。、\n]{2,8})', text)
    entities.update(cn_names)

    titles = re.findall(r'《([^》]{2,20})》', text)
    entities.update(titles)

    noise = {"一个", "可以", "没有", "不是", "这个", "那个", "我们", "他们",
             "进行", "使用", "通过", "其中", "以及", "因此", "所以", "但是",
             "目前", "已经", "还有", "其他", "关于", "因为", "如果", "虽然",
             "漫画", "动画", "作品", "故事", "剧情", "角色"}
    entities = {e for e in entities if e not in noise and len(e) >= 2}

    q_quoted = re.findall(r'[「『""]([^」』""]{2,20})[」』""]', question)
    entities.update(q_quoted)

    return list(entities)[:8]


def generate(model, tokenizer, msgs, max_tokens=500, temp=0.3):
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temp,
                             top_p=0.85, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)


# ── MAIN ──
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
    device_map="auto", trust_remote_code=True, attn_implementation="sdpa")
base_model.eval()
print("Ready.\n")

tests = [
    ("Q1-ACG事实", "鬼灭之刃里水之呼吸的使用者有哪些？水之呼吸有什么招式？",
     "鬼灭之刃 水之呼吸 使用者 招式"),
    ("Q2-翻译", "お疲れ様です和ご苦労様です有什么区别？翻译时怎么处理？",
     "お疲れ様です ご苦労様です 違い"),
    ("Q3-ACG设定", "咒术回战里五条悟的领域展开无量空处具体效果是什么？",
     "五条悟 无量空处 领域展开 效果"),
    ("Q4-动漫信息", "间谍过家家动画有几季？剧场版讲了什么？",
     "间谍过家家 动画 季 剧场版 内容"),
    ("Q5-网络用语", "尊い在ACG圈是什么意思？怎么翻译成中文？",
     "尊い ACG 意味 翻訳"),
]

print("Starting browser...")
p = sync_playwright().start()
browser = p.chromium.launch(headless=True, args=BROWSER_ARGS)
context = browser.new_context(
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    locale="zh-CN",
    viewport={"width": 1920, "height": 1080},
)
context.add_init_script("""
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    window.chrome = {runtime: {}};
""")
page = context.new_page()
print("Browser ready.\n")

for label, question, query in tests:
    print("=" * 55)
    print(label + ": " + question[:60])
    print("=" * 55)

    t0 = time.time()

    # ═══ ROUND 1: Encyclopedia search ═══
    print("  [R1] Encyclopedia search...")
    r1_all = []

    r1_baidu = bing_search_site(page, query, "baike.baidu.com")
    print("    baike: " + str(len(r1_baidu)) + " results")
    r1_all.extend(r1_baidu)

    r1_moegirl = bing_search_site(page, query, "moegirl.org.cn")
    print("    moegirl: " + str(len(r1_moegirl)) + " results")
    r1_all.extend(r1_moegirl)

    r1_wiki = bing_search_site(page, query, "zh.wikipedia.org")
    print("    wiki: " + str(len(r1_wiki)) + " results")
    r1_all.extend(r1_wiki)

    # Deep-read R1
    r1_all.sort(key=lambda x: 0 if "wiki" in x.get("site","") or "baike" in x.get("site","") else 1)
    r1_deep = []
    for r in r1_all[:5]:
        content = deep_read_url(page, r["href"])
        if content:
            r1_deep.append("[" + r["site"] + "] " + r["title"] + "\n" + content)
            print("    Read: " + r["title"][:45] + "... (" + str(len(content)) + "c)")
        else:
            print("    Skip: " + r["title"][:45] + "... (empty)")

    t1 = time.time()

    # ═══ ROUND 2: Entity-based supplementary ═══
    print("  [R2] Entity extraction + search...")
    r1_text = "\n\n---\n\n".join(r1_deep) if r1_deep else ""
    entities = extract_key_entities(r1_text, question)
    print("    Entities: " + str(entities[:6]))

    r2_all = []
    for ent in entities[:4]:
        ent_results = bing_search_general(page, ent + " " + question[:20])
        r2_all.extend(ent_results[:2])

    seen_urls = {r["href"] for r in r1_all}
    r2_new = [r for r in r2_all if r["href"] not in seen_urls][:4]

    r2_deep = []
    for r in r2_new[:3]:
        content = deep_read_url(page, r["href"])
        if content:
            r2_deep.append("[R2] " + r["title"] + "\n" + content)
            print("    Read: " + r["title"][:45] + "... (" + str(len(content)) + "c)")
        else:
            print("    Skip: " + r["title"][:45] + "... (empty)")

    t2 = time.time()

    # ═══ Base model summary ═══
    r2_text = "\n\n---\n\n".join(r2_deep) if r2_deep else "(R2: empty)"
    all_material = "=== Round 1 ===\n" + r1_text + "\n\n=== Round 2 ===\n" + r2_text
    total_chars = len(all_material)
    trimmed = all_material[:8000]

    SUMMARIZE_PROMPT = """你是一个严谨的AI助手。请根据以下资料回答用户问题。

[资料]
%s
[/资料]

用户问题：%s

回答规则：
1. 只使用资料中明确提到的信息，逐条列出
2. 资料中没有的信息，明确写未知
3. 不要编造、不要推测、不要补充
4. 用简洁中文列出要点""" % (trimmed, question)

    summary_msgs = [{"role": "user", "content": SUMMARIZE_PROMPT}]
    summary = generate(base_model, tokenizer, summary_msgs, max_tokens=500, temp=0.3)
    t3 = time.time()

    # ═══ Output ═══
    print("")
    print("  --- Timing ---")
    print("  R1: %.1fs | R2: %.1fs | Summary: %.1fs | Total: %.1fs" % (t1-t0, t2-t1, t3-t2, t3-t0))
    print("  Material: " + str(total_chars) + " chars")
    print("  --- Base Summary ---")
    for line in summary.strip().split("\n")[:20]:
        print("  " + line)
    print("")

browser.close()
p.stop()
print("DONE!")
