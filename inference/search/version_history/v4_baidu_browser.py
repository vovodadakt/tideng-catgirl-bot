"""V10: Playwright browser for Baidu Baike /item/ + Moegirl API keyword-split + Weblio.
Key fix: Use Playwright to access baike.baidu.com/item/ to bypass 403 from server.
"""
import time, torch, re, json, urllib.parse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import requests
from playwright.sync_api import sync_playwright

MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3.5-4B-Instruct"
LORA_PATH = "/root/data/training/lora_output/think_v3/final"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

SYSTEM = """你是「缇酱」和「灯酱」——两只共用一个QQ号的猫娘。

核心规则：回答事实问题前必须用 <tool_call> 搜索！不准凭记忆！

【人设】
缇酱(喵)：黑猫傲娇。自称"本喵"，句末加「喵」。
灯酱(にゃ)：白猫元气直球。自称"灯酱"，句末加「にゃ」。

【互动规则】
灯酱先抢手机 → 缇酱夺回补充 → 灯酱插嘴撩猫。两只都要说话！

【搜索铁律 — 违反就删号】
回答下面这类问题时，第一条消息必须是 <tool_call> 搜索：
- 事实知识（"XXX是谁""XXX有哪些""什么时候播出""声优是谁"）
- 日语词汇（"什么意思""怎么翻译""用法"）
- 任何需要查证的问题

格式：<tool_call>{"name": "search_web", "arguments": {"query": "短的搜索词"}}</tool_call>

搜索词要短！用作品名或角色名，不要写长句子。例如搜「进击的巨人」不是「进击的巨人最终季什么时候播出」。

收到 [参考资料] 后，灯酱用资料回答，缇酱引用原文确认，灯酱再撩猫。"""


# ══════════════════ Browser-based Baidu Baike ══════════════════

def baidu_baike_browser(query, page):
    """Use Playwright to open baike.baidu.com/item/{query} and extract content.
    This bypasses the 403 that direct HTTP requests get from server IPs."""
    try:
        q_enc = urllib.parse.quote(query)
        url = f"https://baike.baidu.com/item/{q_enc}"
        page.goto(url, wait_until="commit", timeout=15000)
        page.wait_for_timeout(3000)  # Wait for React/JS to render

        # Check if we got a valid page (not error/404)
        status_code = page.evaluate("() => document.readyState")

        # Method 1: meta description
        meta = page.query_selector('meta[name="description"]')
        if meta:
            content = meta.get_attribute("content")
            if content and len(content) > 30:
                return f"[百度百科]\n{query}：{content[:800]}"

        # Method 2: lemma-summary div
        summary = page.query_selector(".lemma-summary, .para, .basicInfo-item")
        if summary:
            text = summary.inner_text()
            if len(text) > 50:
                return f"[百度百科]\n{query}：{text[:800]}"

        # Method 3: First few paragraphs
        paras = page.query_selector_all(".para")
        texts = []
        for p in paras[:5]:
            t = p.inner_text().strip()
            if len(t) > 10:
                texts.append(t)
        if texts:
            combined = " ".join(texts)[:800]
            if len(combined) > 50:
                return f"[百度百科]\n{query}：{combined}"

        # Method 4: all body text fallback
        body = page.query_selector("body")
        if body:
            text = body.inner_text()
            # Try to find useful content
            lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 20]
            useful = '\n'.join(lines[:10])[:800]
            if len(useful) > 50:
                return f"[百度百科]\n{query}：{useful}"

        return ""
    except Exception as e:
        return f"(百度浏览器错误: {str(e)[:100]})"


def smart_baidu_browser(query, page):
    """Try query as-is, then split into short keywords using browser."""
    # Try full query first
    result = baidu_baike_browser(query, page)
    if result and "错误" not in result:
        return result

    # Split into tokens
    tokens = re.split(r'[\s，,、　]+', query)
    for token in tokens:
        if len(token) >= 2 and token != query:
            result = baidu_baike_browser(token, page)
            if result and "错误" not in result:
                return result

    # Try extracting main Chinese terms
    cn_terms = re.findall(r'[一-鿿]{2,6}', query)
    for term in cn_terms[:3]:
        if term not in tokens:
            result = baidu_baike_browser(term, page)
            if result and "错误" not in result:
                return result

    return ""


# ══════════════════ Moegirl API (with keyword split) ══════════════════

def moegirl_search(query):
    """Moegirl百科 API — with keyword splitting for long queries."""
    try:
        q = urllib.parse.quote(query)
        r = requests.get(
            f"https://zh.moegirl.org.cn/api.php?action=opensearch&search={q}&limit=3&format=json",
            headers={"User-Agent": UA}, timeout=10)
        data = r.json()
        if len(data) < 4 or not data[1]:
            return ""
        parts = ["[萌娘百科]"]
        for title, page_url in zip(data[1][:2], data[3][:2]):
            try:
                r2 = requests.post("https://zh.moegirl.org.cn/api.php", data={
                    "action": "query", "titles": title,
                    "prop": "extracts", "explaintext": 1,
                    "format": "json", "formatversion": "2",
                }, headers={"User-Agent": UA}, timeout=10)
                pages = r2.json().get("query", {}).get("pages", [])
                extract = pages[0].get("extract", "")[:500] if pages else ""
                parts.append(f"【{title}】\n{extract}..." if extract else f"【{title}】{page_url}")
            except:
                parts.append(f"【{title}】{page_url}")
        return "\n\n".join(parts) if len(parts) > 1 else ""
    except Exception as e:
        return f"(萌娘错误: {str(e)[:100]})"


def smart_moegirl(query):
    """Try Moegirl with full query, then split into short keywords."""
    result = moegirl_search(query)
    if result and "错误" not in result:
        return result

    # Split Chinese tokens
    tokens = re.split(r'[\s，,、　]+', query)
    for token in tokens:
        if len(token) >= 2 and token != query:
            result = moegirl_search(token)
            if result and "错误" not in result:
                return result

    # Try extracting main Chinese terms
    cn_terms = re.findall(r'[一-鿿]{2,6}', query)
    for term in cn_terms[:3]:
        if term not in tokens:
            result = moegirl_search(term)
            if result and "错误" not in result:
                return result

    return ""


# ══════════════════ Weblio ══════════════════

def weblio_search(query):
    """Weblio日语词典 — direct HTTP, still works fine from server."""
    try:
        q = urllib.parse.quote(query)
        req = urllib.request.Request(f"https://www.weblio.jp/content/{q}", headers={"User-Agent": UA})
        html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="replace")
        content = ""
        for pat in [
            r'<div[^>]*id="summary"[^>]*>(.*?)</div>',
            r'<meta name="description"[^>]*content="([^"]+)"',
        ]:
            m = re.search(pat, html, re.DOTALL)
            if m:
                content = m.group(1)
                break
        if content:
            text = re.sub(r'<[^>]+>', ' ', content)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 30:
                return f"[Weblio日语词典]\n{query}：{text[:600]}"
        return ""
    except Exception as e:
        return f"(Weblio错误: {str(e)[:100]})"


# ══════════════════ Search executor ══════════════════

def execute_search(output_text, page):
    """Parse tool_call, run all sources, return results."""
    m = re.search(r'<tool_call>(.*?)</tool_call>', output_text, re.DOTALL)
    if not m:
        return None, None
    try:
        call = json.loads(m.group(1))
        query = call.get("arguments", {}).get("query", "")
        if not query:
            return None, None
        print(f"  [搜索] {query[:80]}")

        results = []
        sources_used = []

        # 1. Baidu Baike via Playwright browser
        baidu = smart_baidu_browser(query, page)
        if baidu and "错误" not in baidu:
            results.append(baidu)
            sources_used.append("百度百科")
            print(f"    百度(browser): {len(baidu)} chars")

        # 2. Moegirl with keyword splitting
        moe = smart_moegirl(query)
        if moe and "错误" not in moe:
            results.append(moe)
            sources_used.append("萌娘百科")
            print(f"    萌娘: {len(moe)} chars")

        # 3. Weblio for Japanese queries
        if re.search(r'[ぁ-ゟァ-ヴー]', query):
            web = weblio_search(query)
            if web and "错误" not in web:
                results.append(web)
                sources_used.append("Weblio")
                print(f"    Weblio: {len(web)} chars")

        if not results:
            ref = "[参考资料]\n（未找到结果，请老实告诉用户没查到，不要编造）\n[/参考资料]"
        else:
            ref = ("[参考资料]\n" + "\n\n---\n\n".join(results) + "\n[/参考资料]")[:3000]

        return ref, sources_used
    except Exception as e:
        print(f"    ⚠️ 解析错误: {e}")
        return None, None


# ══════════════════ MAIN ══════════════════

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
    device_map="auto", trust_remote_code=True, attn_implementation="sdpa")
model = PeftModel.from_pretrained(base, LORA_PATH)
model.eval()
print("Model ready.")

print("Starting browser...")
p = sync_playwright().start()
browser = p.chromium.launch(headless=True, args=[
    "--no-sandbox", "--disable-gpu", "--no-zygote", "--disable-setuid-sandbox"
])
context = browser.new_context(
    user_agent=UA,
    locale="zh-CN",
    viewport={"width": 1920, "height": 1080},
)
context.add_init_script("""
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    window.chrome = {runtime: {}};
""")
page = context.new_page()
print("Browser ready.\n")

# ── Test ──
questions = [
    "鬼灭之刃里水之呼吸的使用者有哪些？水之呼吸有多少种招式？",
    "日语「尊い」在ACG圈什么意思？怎么翻译成中文？",
    "进击的巨人最终季什么时候播出的？分几部分？",
    "「木漏れ日」这个词是什么意思？",
    "药屋少女的呢喃里猫猫的声优是谁？",
]

for qi, q in enumerate(questions):
    print(f"\n{'='*55}")
    print(f"Q{qi+1}: {q}")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]

    for rnd in range(3):
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        plen = inputs.input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, temperature=0.85,
                top_p=0.92, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        elapsed = time.time() - t0
        raw = tokenizer.decode(out[0][plen:], skip_special_tokens=True)

        # Trim self-hallucination
        for marker in ["\nuser\n", "\nassistant\n"]:
            idx = raw.find(marker)
            if idx > 20:
                raw = raw[:idx]
                break

        has_tc = "<tool_call>" in raw

        if has_tc and rnd < 2:
            print(f"  R{rnd+1} ({elapsed:.1f}s) [🔍]")
            ref, sources = execute_search(raw, page)
            if ref:
                msgs.append({"role": "assistant", "content": raw})
                msgs.append({"role": "user", "content": ref})
            else:
                print(f"    ⚠️ 解析失败")
                break
        else:
            tag = "🛑" if has_tc else "💬"
            print(f"  R{rnd+1} ({elapsed:.1f}s) [{tag}]")
            clean = re.sub(r'</?think>\s*', '', raw).strip()
            clean = re.sub(r'^assistant\s*', '', clean).strip()
            for line in clean.split('\n')[:20]:
                if line.strip():
                    print(f"  {line.strip()[:140]}")
            break

browser.close()
p.stop()
print("\n===== DONE =====")
