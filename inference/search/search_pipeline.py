"""V22: Multi-keyword × (Baidu+Moegirl) → Bing only if R1 < 500c.
Clean, simple, no model-based relevance/cleaning.
"""
import time, torch, re, json, urllib.parse, urllib.request
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import requests
from playwright.sync_api import sync_playwright

MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3.5-4B-Instruct"
LORA_PATH = "/root/data/training/lora_output/think_v3/final"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

KW_EXAMPLES = {
    "鬼灭之刃里水之呼吸的使用者有哪些？水之呼吸有多少种招式？": "鬼灭之刃、水之呼吸",
    "进击的巨人最终季什么时候播出的？分几部分？": "进击的巨人、最终季",
    "药屋少女的呢喃里猫猫的声优是谁？": "药屋少女的呢喃、猫猫、声优",
    "「木漏れ日」这个词是什么意思？": "木漏れ日",
}

SYSTEMS = {
    "keywords": "提取搜索关键词。只输出用顿号分隔的关键词，不要其他内容。",
    "answer": """你是「提酱」和「灯酱」——两只共用一个QQ号的猫娘翻译顾问。

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

【当前模式：搜索准确回答】
灯酱先总结搜索结果 → 提酱引用来源逐条确认。
材料里有的引用，材料里没有的诚实说不知道。不准编造数字、日期、人名。""",
}

KW_GARBAGE = ['user', '问题', 'assistant', 'system', 'User', 'Question', '参考', '搜索']


def base_generate(kw_model, tokenizer, system_key, prompt, max_tokens=100, temp=0.1):
    msgs = [{"role": "system", "content": SYSTEMS[system_key]},
            {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(kw_model.device)
    with torch.no_grad():
        out = kw_model.generate(**inputs, max_new_tokens=max_tokens, temperature=temp,
            do_sample=(temp > 0.1), top_p=0.5, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()


def extract_keywords(question, kw_model, tokenizer):
    examples = "\n".join([f"问题：{q}\n关键词：{kw}" for q, kw in KW_EXAMPLES.items()])
    prompt = f"{examples}\n问题：{question}\n关键词："
    raw = base_generate(kw_model, tokenizer, "keywords", prompt, max_tokens=30, temp=0.1)
    raw = raw.split('\n')[0].strip()
    raw = re.sub(r'^关键词[：:]\s*', '', raw)
    raw = re.sub(r'^Keywords[：:]\s*', '', raw)
    raw = re.sub(r'\s*user.*$', '', raw)
    raw = re.sub(r'\s*问题.*$', '', raw)
    raw = re.sub(r'\s*assistant.*$', '', raw)
    keywords = re.split(r'[,，、]+', raw)
    clean_kws = []
    for k in keywords:
        k = k.strip().strip("\"'「」")
        k = k.split('\n')[0].strip()
        if len(k) < 2 or len(k) > 30:
            continue
        if any(g in k.lower() for g in KW_GARBAGE):
            continue
        clean_kws.append(k)
    if not clean_kws:
        short = question.strip().strip('「」').strip('""').strip("''")
        short = re.sub(r'^(请问|问一下|你知道|告诉我|什么是|什么是「|「|"|\')\s*', '', short)
        short = short[:15].rstrip('，。？?！!的')
        if len(short) >= 2:
            clean_kws = [short]
    return clean_kws[:3]


# ═══════════ Baidu Baike (Browser) ═══════════

def baidu_baike_browser(query, page):
    try:
        q_enc = urllib.parse.quote(query)
        url = f"https://baike.baidu.com/item/{q_enc}"
        page.goto(url, wait_until="commit", timeout=15000)
        page.wait_for_timeout(3000)

        meta = page.query_selector('meta[name="description"]')
        if meta:
            content = meta.get_attribute("content")
            if content and len(content) > 30:
                return f"[百度百科]\n{query}：{content[:800]}"

        summary = page.query_selector(".lemma-summary, .basicInfo-item")
        if summary:
            text = summary.inner_text()
            if len(text) > 50:
                return f"[百度百科]\n{query}：{text[:800]}"

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

        body = page.query_selector("body")
        if body:
            text = body.inner_text()
            lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 20]
            useful = '\n'.join(lines[:10])[:800]
            if len(useful) > 50:
                return f"[百度百科]\n{query}：{useful}"
        return ""
    except:
        return ""


# ═══════════ Moegirl API ═══════════

def moegirl_search(query):
    try:
        q = urllib.parse.quote(query)
        r = requests.get(
            f"https://zh.moegirl.org.cn/api.php?action=opensearch&search={q}&limit=2&format=json",
            headers={"User-Agent": UA}, timeout=8)
        data = r.json()
        if len(data) < 4 or not data[1]:
            return ""
        parts = ["[萌娘百科]"]
        for title, page_url in zip(data[1][:2], data[3][:2]):
            try:
                r2 = requests.post("https://zh.moegirl.org.cn/api.php", data={
                    "action": "query", "titles": title, "prop": "extracts",
                    "explaintext": 1, "format": "json", "formatversion": "2",
                }, headers={"User-Agent": UA}, timeout=10)
                pages = r2.json().get("query", {}).get("pages", [])
                extract = pages[0].get("extract", "") if pages else ""
                if len(extract) > 600:
                    paras = [p.strip() for p in extract.split('\n') if len(p.strip()) > 10]
                    relevant = f"【{title}】\n" + "\n".join(paras[:8])[:1500]
                else:
                    relevant = f"【{title}】\n{extract}"
                if relevant:
                    parts.append(relevant)
                elif extract:
                    parts.append(f"【{title}】\n{extract[:600]}")
            except:
                parts.append(f"【{title}】{page_url}")
        return "\n\n".join(parts) if len(parts) > 1 else ""
    except:
        return ""


# ═══════════ Bing deep search ═══════════

def bing_deep_search(keywords, page, max_results=3):
    query = " ".join(keywords[:2]) if keywords else ""
    if not query:
        return ""
    try:
        q_enc = urllib.parse.quote(query)
        page.goto(f"https://cn.bing.com/search?q={q_enc}", timeout=12000)
        page.wait_for_timeout(2000)

        items = page.query_selector_all("li.b_algo h2 a")
        urls, titles = [], []
        for item in items[:max_results]:
            href = item.get_attribute("href")
            title = item.inner_text().strip()
            if href and title and len(title) > 3:
                urls.append(href)
                titles.append(title)

        if not urls:
            return ""

        results = []
        for url, title in zip(urls, titles):
            content = deep_read(page, url)
            if content:
                results.append(f"[Bing] {title}\n{url}\n{content}")
                print(f"      Bing: {title[:50]}... ({len(content)}c)")
            else:
                print(f"      Bing skip: {title[:50]}")

        return "\n\n---\n\n".join(results)[:3000] if results else ""
    except Exception as e:
        print(f"      Bing error: {str(e)[:80]}")
        return ""


def deep_read(page, url, timeout=12000):
    try:
        page.goto(url, wait_until="commit", timeout=timeout)
        page.wait_for_timeout(2000)
        text = page.evaluate("() => document.body.innerText")
        lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 15]
        return '\n'.join(lines)[:2000] if lines else ""
    except:
        return ""


# ═══════════ Main Pipeline ═══════════

def search_pipeline(question, kw_model, tokenizer, ans_model, page):
    t0 = time.time()
    keywords = extract_keywords(question, kw_model, tokenizer)
    print(f"  [关键词] {keywords}")

    # ── R1: Baidu + Moegirl for each keyword ──
    r1_parts = []
    total_c = 0
    for kw in keywords:
        baidu = baidu_baike_browser(kw, page)
        if baidu:
            r1_parts.append(baidu)
            total_c += len(baidu)
            print(f"    百度({kw}): {len(baidu)}c")

        moe = moegirl_search(kw)
        if moe:
            r1_parts.append(moe)
            total_c += len(moe)
            print(f"    萌娘({kw}): {len(moe)}c")

    print(f"  [R1] total={total_c}c, parts={len(r1_parts)}")
    r1_text = "\n\n---\n\n".join(r1_parts)[:5000] if r1_parts else ""
    all_refs = [r1_text] if r1_text else []
    stage = "R1"

    # ── R2: Bing ONLY if R1 < 500c ──
    if total_c < 500:
        print(f"  [R2] Bing (R1不足)...")
        bing = bing_deep_search(keywords, page)
        if bing:
            all_refs.append(bing)
            stage = "R1+Bing"

    # ── Merge ──
    ref = "\n\n---\n\n".join(all_refs)[:3000].strip()

    if len(ref) < 200:
        instruction = "搜索未能找到答案。请诚实告诉用户：没找到相关信息，建议换个关键词。不准编造。"
        stage += "(不足)"
    else:
        instruction = "请严格只使用参考资料中的信息回答。材料里有就引用，材料里没有就说不知道。不准编造数字、日期、人名、招式名。"

    user_msg = f"{question}\n\n[参考资料]\n{ref}\n[/参考资料]\n\n{instruction}"
    msgs = [{"role": "system", "content": SYSTEMS["answer"]}, {"role": "user", "content": user_msg}]

    text = tokenizer.apply_chat_template(msgs, tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(text, return_tensors="pt").to(ans_model.device)
    plen = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = ans_model.generate(**inputs, max_new_tokens=500, temperature=0.85,
            top_p=0.92, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    elapsed = time.time() - t0

    raw = tokenizer.decode(out[0][plen:], skip_special_tokens=True)
    for marker in ["\nuser", "\nassistant", "\n<|user|>", "\n<|assistant|>"]:
        idx = raw.find(marker)
        if idx > 20:
            raw = raw[:idx]
            break
    clean = re.sub(r'<tool_call>.*?</tool_call>\s*', '', raw, flags=re.DOTALL)
    clean = re.sub(r'</?think>\s*', '', clean).strip()
    clean = re.sub(r'^assistant\s*', '', clean).strip()

    print(f"  [ref {len(ref)}c] ({elapsed:.1f}s) [{stage}]")
    for line in clean.split('\n')[:30]:
        if line.strip():
            print(f"  {line.strip()[:150]}")

    return clean


# ═══════════ MAIN ═══════════

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

kw_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
    device_map="cuda:0", trust_remote_code=True, attn_implementation="sdpa")
kw_model.eval()
print("Base model ready.")

print("Loading LoRA...")
base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
    device_map="cuda:0", trust_remote_code=True, attn_implementation="sdpa")
ans_model = PeftModel.from_pretrained(base, LORA_PATH)
ans_model.eval()
print("LoRA ready.")

print("Starting browser...")
p = sync_playwright().start()
browser = p.chromium.launch(headless=True, args=[
    "--no-sandbox", "--disable-gpu", "--no-zygote", "--disable-setuid-sandbox"
])
context = browser.new_context(user_agent=UA, locale="zh-CN", viewport={"width": 1920, "height": 1080})
context.add_init_script("""
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    window.chrome = {runtime: {}};
""")
page = context.new_page()
print("Browser ready.\n")


questions = [
    "鬼灭之刃里水之呼吸的使用者有哪些？水之呼吸有多少种招式？",
    "进击的巨人最终季什么时候播出的？分几部分？",
    "「木漏れ日」这个词是什么意思？",
    "药屋少女的呢喃里猫猫的声优是谁？",
]

for qi, q in enumerate(questions):
    print(f"\n{'='*55}")
    print(f"Q{qi+1}: {q}")
    search_pipeline(q, kw_model, tokenizer, ans_model, page)

browser.close()
p.stop()
print("\n===== DONE =====")
