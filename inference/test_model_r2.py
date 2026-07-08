"""Model-driven two-round search.
R1: Browser-search baike + moegirl + weblio → deep read
R2: Base model reads R1 → decides what's missing → generates queries → Bing
Final: Base model summarizes all material
"""
import torch, re, time, json, urllib.request, urllib.parse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from playwright.sync_api import sync_playwright

MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2.5-3B-Instruct"
BROWSER_ARGS = ["--no-sandbox", "--disable-gpu", "--no-zygote", "--disable-setuid-sandbox"]

# ══════════════════ Browser-based searches ══════════════════

def search_baidu_baike(page, query, max_results=5):
    """Baidu Baike search via simulated input."""
    try:
        q_enc = urllib.parse.quote(query)
        page.goto("https://baike.baidu.com/search?word=" + q_enc, timeout=12000)
        page.wait_for_timeout(2000)
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
                results.append({"title": title, "href": href, "source": "baike"})
            except:
                continue
            if len(results) >= max_results:
                break
        return results
    except Exception as e:
        print("    [baike] " + str(e)[:80])
        return []


def search_moegirl_browser(page, query, max_results=5):
    """Moegirl百科 browser search — simulate typing in search box.
    Moegirl uses MediaWiki. Search URL: /index.php?search=..."""
    try:
        q_enc = urllib.parse.quote(query)
        search_url = "https://zh.moegirl.org.cn/index.php?search=" + q_enc
        page.goto(search_url, timeout=15000)
        page.wait_for_timeout(2500)

        # MediaWiki search results page has .mw-search-results or .mw-search-result
        items = page.query_selector_all(".mw-search-result, .mw-search-results li, .searchResult")
        if not items:
            # Fallback: any link to a wiki page
            items = page.query_selector_all("a[href*='/']")

        results = []
        seen = set()
        for item in items:
            try:
                a = item.query_selector("a") or item
                href = a.get_attribute("href")
                if not href:
                    continue
                # Keep only moegirl wiki page links
                if "/index.php" in href or "/wiki/" in href or href.startswith("/"):
                    pass
                else:
                    continue
                # Filter out non-content links
                if any(x in href for x in ["Special:", "User:", "Talk:", "File:", "Template:", "Help:", "Category:", "action="]):
                    continue
                title = a.inner_text().strip()
                if not title or title in seen or len(title) < 2:
                    continue
                if title in ["登录", "注册", "编辑", "讨论", "历史", "查看", "更多", "搜索", "帮助", "关于"]:
                    continue
                seen.add(title)
                # Build full URL
                if href.startswith("/"):
                    full_url = "https://zh.moegirl.org.cn" + href
                elif href.startswith("http"):
                    full_url = href
                else:
                    full_url = "https://zh.moegirl.org.cn/" + href
                results.append({"title": title, "href": full_url, "source": "moegirl"})
            except:
                continue
            if len(results) >= max_results:
                break

        # If still empty, try the main page search box approach
        if not results:
            page.goto("https://zh.moegirl.org.cn/", timeout=10000)
            page.wait_for_timeout(2000)
            try:
                search_box = page.query_selector("input[name='search'], #searchInput, input[type='search']")
                if search_box:
                    search_box.click()
                    search_box.fill("")
                    for c in query[:30]:
                        search_box.type(c, delay=20)
                    page.wait_for_timeout(300)
                    search_box.press("Enter")
                    page.wait_for_timeout(3000)
                    items = page.query_selector_all(".mw-search-result a, .mw-search-results a")
                    for item in items[:max_results]:
                        try:
                            title = item.inner_text().strip()
                            href = item.get_attribute("href")
                            if title and href and len(title) >= 2:
                                if href.startswith("/"):
                                    href = "https://zh.moegirl.org.cn" + href
                                if title not in seen:
                                    seen.add(title)
                                    results.append({"title": title, "href": href, "source": "moegirl"})
                        except:
                            continue
            except:
                pass

        return results
    except Exception as e:
        print("    [moegirl] " + str(e)[:80])
        return []


def search_weblio(page, query, max_results=4):
    """Search Weblio.jp dictionary for Japanese terms."""
    try:
        q_enc = urllib.parse.quote(query)
        # Weblio search page
        page.goto("https://www.weblio.jp/content/" + q_enc, timeout=12000)
        page.wait_for_timeout(2000)

        results = []

        # Check if we landed on a definition page directly
        title_el = page.query_selector("h1, .title, .kijiWTitle")
        if title_el:
            title = title_el.inner_text().strip()
            if title and len(title) > 1:
                content_el = page.query_selector("#main, .kiji, article, .description, .meaning")
                desc = ""
                if content_el:
                    desc = content_el.inner_text()[:500]
                results.append({
                    "title": "[Weblio] " + title,
                    "href": page.url,
                    "desc": desc,
                    "source": "weblio"
                })

        # Also get related/similar entries from sidebar or search results
        related = page.query_selector_all(".crosslink a, .relatedWord a, .synonyms a, .searchResult a")
        seen = set()
        for a in related[:max_results]:
            try:
                title = a.inner_text().strip()
                href = a.get_attribute("href")
                if title and href and title not in seen and len(title) >= 2:
                    seen.add(title)
                    if href.startswith("/"):
                        href = "https://www.weblio.jp" + href
                    results.append({"title": "[Weblio] " + title, "href": href, "source": "weblio"})
            except:
                continue

        return results
    except Exception as e:
        print("    [weblio] " + str(e)[:80])
        return []


def bing_search(page, query, max_results=5):
    """General Bing search."""
    try:
        page.goto("https://cn.bing.com/", timeout=8000)
        page.wait_for_timeout(600)
        sb = page.query_selector("#sb_form_q")
        if sb:
            sb.click()
            sb.fill("")
            for c in query:
                sb.type(c, delay=10)
            page.wait_for_timeout(300)
            sb.press("Enter")
        else:
            q_enc = urllib.parse.quote(query)
            page.goto("https://cn.bing.com/search?q=" + q_enc, timeout=8000)
        page.wait_for_timeout(2500)
        items = page.query_selector_all("li.b_algo")
        results = []
        for item in items[:max_results]:
            try:
                a = item.query_selector("h2 a")
                if not a:
                    continue
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
                    results.append({"title": title, "href": href, "desc": desc, "source": "bing"})
            except:
                continue
        return results
    except Exception as e:
        print("    [bing] " + str(e)[:80])
        return []


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
        # Clean up common noise
        lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
        return '\n'.join(lines)[:4000] if lines else ""
    except Exception as e:
        return ""


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
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
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

    # ═══ R1: Browser-search baike + moegirl + weblio ═══
    print("  [R1] Browser search: baike + moegirl + weblio...")

    # Extract Japanese terms from question for Weblio
    jp_terms = re.findall(r'[ぁ-ゟァ-ヴー]{2,}', question)

    baike_results = search_baidu_baike(page, question, max_results=4)
    moegirl_results = search_moegirl_browser(page, question, max_results=4)

    # Weblio: search both individual Japanese terms and full question
    weblio_results = []
    for term in jp_terms[:2]:
        weblio_results.extend(search_weblio(page, term))
    if jp_terms:
        weblio_results.extend(search_weblio(page, question))

    print("    baike=%d | moegirl=%d | weblio=%d" % (
        len(baike_results), len(moegirl_results), len(weblio_results)))

    # Merge: moegirl first (best for ACG), then baike, then weblio
    r1_all = moegirl_results + baike_results + weblio_results

    # Deep read top results
    r1_deep = []
    r1_labels = []
    for r in r1_all[:8]:
        content = deep_read_url(page, r["href"])
        if content:
            label = "[%s] %s" % (r["source"], r["title"])
            r1_labels.append(label)
            r1_deep.append(label + "\n" + content)
            print("    + %s: %s... (%dc)" % (r["source"], r["title"][:40], len(content)))
        else:
            print("    - %s: %s... (empty)" % (r["source"], r["title"][:40]))

    t1 = time.time()
    r1_text = "\n\n---\n\n".join(r1_deep) if r1_deep else "(no results)"

    # ═══ R2: Model reads R1 → generates queries → Bing ═══
    print("  [R2] Model reading R1 results...")

    # Truncate R1 text for model input
    r1_text_for_model = r1_text[:5000] if len(r1_text) > 5000 else r1_text

    R2_ROUTER_PROMPT = (
        "你是搜索助手。阅读第一轮搜索结果，判断是否需要补充搜索。\n\n"
        "用户问题：" + question + "\n\n"
        "第一轮搜索结果：\n" + r1_text_for_model + "\n\n"
        "请回答：\n"
        "1. 现有信息是否足以回答问题？[够/不够]\n"
        "2. 如果不够，缺什么？[一句话]\n"
        "3. 需要补充搜索的话，给出2-3个搜索查询词（每行一个，中文或日文）\n"
        "如果信息已经足够，查询词写\"无\"。"
    )

    r2_decision = generate(base_model, tokenizer,
        [{"role": "user", "content": R2_ROUTER_PROMPT}], max_tokens=200, temp=0.3)

    print("    Model decision:")
    for line in r2_decision.strip().split('\n')[:8]:
        print("      " + line)

    # Parse queries from model response
    queries = []
    for line in r2_decision.split('\n'):
        line = line.strip()
        # Skip metadata lines
        if any(line.startswith(p) for p in ["1.", "2.", "3.", "足够", "缺失", "查询", "够/", "如果"]):
            continue
        if line == "无" or line == "不需要":
            continue
        if len(line) >= 3 and len(line) <= 60:
            queries.append(line)

    t2 = time.time()

    # Run R2 searches
    r2_deep = []
    if queries:
        print("  [R2] Executing queries: " + str(queries[:3]))
        seen_urls = {r["href"] for r in r1_all}
        for q in queries[:3]:
            bing_results = bing_search(page, q, max_results=3)
            for br in bing_results:
                if br["href"] in seen_urls:
                    continue
                if len(r2_deep) >= 4:
                    break
                content = deep_read_url(page, br["href"])
                if content:
                    r2_deep.append("[R2-bing] " + br["title"] + "\n" + content)
                    seen_urls.add(br["href"])
                    print("    + R2: %s... (%dc)" % (br["title"][:40], len(content)))
                else:
                    print("    - R2: %s... (empty)" % br["title"][:40])
            if len(r2_deep) >= 4:
                break
    else:
        print("  [R2] Skipped — model says R1 is enough.")

    t3 = time.time()

    # ═══ Final summary ═══
    all_material = "=== Round 1 (encyclopedias + dictionary) ===\n" + r1_text[:6000]
    if r2_deep:
        all_material += "\n\n=== Round 2 (targeted supplements) ===\n" + "\n\n---\n\n".join(r2_deep)

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
    t4 = time.time()

    # ═══ Output ═══
    print("")
    print("  R1: %.1fs | R2-decision: %.1fs | R2-exec: %.1fs | Summary: %.1fs | TOTAL: %.1fs" % (
        t1-t0, t2-t1, t3-t2, t4-t3, t4-t0))
    print("  Material: %d chars (%d R1 deep reads, %d R2 deep reads)" % (
        total_chars, len(r1_deep), len(r2_deep)))
    print("  --- Summary ---")
    for line in summary.strip().split("\n")[:25]:
        print("  " + line)
    print("")

browser.close()
p.stop()
print("DONE!")
