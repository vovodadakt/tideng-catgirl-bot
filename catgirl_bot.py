"""Catgirl QQ Bot -- importable search+answer API.

Usage:
    from catgirl_bot import CatgirlBot

    # Non-streaming
    bot = CatgirlBot()
    bot.load()
    answer = bot.ask("鬼灭之刃里水之呼吸的使用者有哪些？")
    print(answer)

    # Streaming (QQ-friendly)
    for chunk in bot.ask_stream("木漏れ日是什么意思？"):
        print(chunk, end="", flush=True)
"""
import time, torch, re, urllib.parse
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import requests
from playwright.sync_api import sync_playwright


class CatgirlBot:
    """Search + answer bot with catgirl persona.

    Parameters
    ----------
    model_path : str
        Merged model path or ModelScope ID.
    lora_path : str or None
        If set, use dual-model mode (base + LoRA separately).
    load_in_4bit : bool
        4-bit quantize on load (~3.3 GB VRAM).
    headless : bool
        Run browser headless.
    """

    def __init__(
        self,
        model_path: str = "vovodadakt/Qwen3.5-4B-Catgirl",
        lora_path: str | None = None,
        load_in_4bit: bool = True,
        headless: bool = True,
    ):
        self._model_path = model_path
        self._lora_path = lora_path
        self._headless = headless
        self._tokenizer = None
        self._model = None
        self._playwright = None
        self._browser = None
        self._page = None
        self._loaded = False

        if load_in_4bit:
            self._bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            self._bnb = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self):
        """Load model + browser. Called automatically on first ask()."""
        if self._loaded:
            return

        print("[CatgirlBot] Loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        print("[CatgirlBot] Loading model (4-bit)...")
        load_kwargs = dict(
            device_map="cuda:0", trust_remote_code=True, attn_implementation="sdpa"
        )
        if self._bnb:
            load_kwargs["quantization_config"] = self._bnb

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path, **load_kwargs
        )
        self._model.eval()

        if self._lora_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self._lora_path)
            self._model.eval()

        print(f"[CatgirlBot] VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

        print("[CatgirlBot] Starting browser...")
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self._headless,
            args=["--no-sandbox", "--disable-gpu", "--no-zygote", "--disable-setuid-sandbox"],
        )
        ctx = self._browser.new_context(
            user_agent=self._ua,
            locale="zh-CN",
            viewport={"width": 1920, "height": 1080},
        )
        ctx.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = {runtime: {}};
        """)
        self._page = ctx.new_page()
        self._loaded = True
        print("[CatgirlBot] Ready.")

    def ask(self, question: str) -> str:
        """Non-streaming: return full answer as one string."""
        if not self._loaded:
            self.load()

        t0 = time.time()
        keywords = self._extract_keywords(question)
        ref = self._search(keywords)
        print(f"[CatgirlBot] search: {time.time()-t0:.1f}s  ref={len(ref)}c")
        answer = self._answer(question, ref)
        print(f"[CatgirlBot] total: {time.time()-t0:.1f}s")
        return answer

    def ask_stream(self, question: str):
        """Streaming: yields text chunks as model generates.

        for chunk in bot.ask_stream("who voiced catcat?"):
            send_qq_message(chunk)  # QQ friendly: send as-you-go
        """
        if not self._loaded:
            self.load()

        t0 = time.time()
        keywords = self._extract_keywords(question)
        ref = self._search(keywords)
        search_time = time.time() - t0
        print(f"[CatgirlBot] search: {search_time:.1f}s  ref={len(ref)}c")

        yield from self._answer_stream(question, ref)
        print(f"[CatgirlBot] total: {time.time()-t0:.1f}s")

    def close(self):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        self._loaded = False

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Keywords
    # ------------------------------------------------------------------

    _KW_EXAMPLES = {
        "鬼灭之刃里水之呼吸的使用者有哪些？": "鬼灭之刃、水之呼吸",
        "进击的巨人最终季什么时候播出的？": "进击的巨人、最终季",
        "药屋少女的呢喃里猫猫的声优是谁？": "药屋少女的呢喃、猫猫、声优",
        "「木漏れ日」这个词是什么意思？": "木漏れ日",
    }
    _KW_SYSTEM = "提取搜索关键词。只输出用顿号分隔的关键词，不要其他内容。"
    _KW_GARBAGE = {"user", "问题", "assistant", "system", "question", "参考", "搜索"}

    def _extract_keywords(self, question: str) -> list[str]:
        examples = "\n".join(
            f"问题：{q}\n关键词：{kw}" for q, kw in self._KW_EXAMPLES.items()
        )
        prompt = f"{examples}\n问题：{question}\n关键词："
        raw = self._gen(self._KW_SYSTEM, prompt, max_tokens=30, temp=0.1)

        raw = raw.split("\n")[0].strip()
        raw = re.sub(r"^关键词[：:]\s*", "", raw)
        raw = re.sub(r"\s*user.*$", "", raw)
        raw = re.sub(r"\s*问题.*$", "", raw)
        raw = re.sub(r"\s*assistant.*$", "", raw)

        keywords = re.split(r"[,，、]+", raw)
        clean = []
        for k in keywords:
            k = k.strip().strip("\"'「」").split("\n")[0].strip()
            if len(k) < 2 or len(k) > 30:
                continue
            if any(g in k.lower() for g in self._KW_GARBAGE):
                continue
            clean.append(k)

        if not clean:
            short = question.strip().strip("「」").strip('""').strip("''")
            short = re.sub(r"^(请问|问一下|你知道|告诉我)\s*", "", short)
            short = short[:15].rstrip("，。？?！!的")
            if len(short) >= 2:
                clean = [short]

        print(f"[CatgirlBot] keywords: {clean[:3]}")
        return clean[:3]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search(self, keywords: list[str]) -> str:
        parts = []
        total = 0
        for kw in keywords:
            b = self._baidu(kw)
            if b:
                parts.append(b)
                total += len(b)
            m = self._moe(kw)
            if m:
                parts.append(m)
                total += len(m)

        r1 = "\n\n---\n\n".join(parts)[:5000] if parts else ""
        if total < 500:
            bing = self._bing(" ".join(keywords[:2]))
            if bing:
                r1 += "\n\n---\n\n" + bing
        return r1[:3000].strip()

    def _baidu(self, query: str) -> str:
        try:
            q = urllib.parse.quote(query)
            self._page.goto(
                f"https://baike.baidu.com/item/{q}",
                wait_until="commit", timeout=15000,
            )
            self._page.wait_for_timeout(3000)
            meta = self._page.query_selector('meta[name="description"]')
            if meta:
                c = meta.get_attribute("content")
                if c and len(c) > 30:
                    return f"[百度百科]\n{query}：{c[:800]}"
            body = self._page.query_selector("body")
            if body:
                lines = [
                    l.strip()
                    for l in body.inner_text().split("\n")
                    if len(l.strip()) > 20
                ]
                if lines:
                    return f"[百度百科]\n{query}：{chr(10).join(lines[:10])[:800]}"
        except Exception:
            pass
        return ""

    def _moe(self, query: str) -> str:
        try:
            q = urllib.parse.quote(query)
            r = requests.get(
                f"https://zh.moegirl.org.cn/api.php?action=opensearch&search={q}&limit=2&format=json",
                headers={"User-Agent": self._ua}, timeout=8,
            )
            data = r.json()
            if len(data) < 4 or not data[1]:
                return ""
            parts = ["[萌娘百科]"]
            for title, url in zip(data[1][:2], data[3][:2]):
                try:
                    r2 = requests.post(
                        "https://zh.moegirl.org.cn/api.php",
                        data={
                            "action": "query", "titles": title,
                            "prop": "extracts", "explaintext": 1,
                            "format": "json", "formatversion": "2",
                        },
                        headers={"User-Agent": self._ua}, timeout=10,
                    )
                    pages = r2.json().get("query", {}).get("pages", [])
                    ext = pages[0].get("extract", "") if pages else ""
                    if len(ext) > 600:
                        paras = [
                            p.strip() for p in ext.split("\n")
                            if len(p.strip()) > 10
                        ]
                        parts.append(
                            f"【{title}】\n" + "\n".join(paras[:8])[:1500]
                        )
                    elif ext:
                        parts.append(f"【{title}】\n{ext}")
                except Exception:
                    parts.append(f"【{title}】{url}")
            return "\n\n".join(parts) if len(parts) > 1 else ""
        except Exception:
            return ""

    def _bing(self, query: str) -> str:
        if not query:
            return ""
        try:
            q = urllib.parse.quote(query)
            self._page.goto(
                f"https://cn.bing.com/search?q={q}", timeout=12000
            )
            self._page.wait_for_timeout(2000)
            items = self._page.query_selector_all("li.b_algo h2 a")[:3]
            results = []
            for item in items:
                href = item.get_attribute("href")
                title = item.inner_text().strip()
                if href and title and len(title) > 3:
                    content = self._deep_read(href)
                    if content:
                        results.append(f"[Bing] {title}\n{href}\n{content}")
            return "\n\n---\n\n".join(results)[:3000] if results else ""
        except Exception:
            return ""

    def _deep_read(self, url: str) -> str:
        try:
            self._page.goto(url, wait_until="commit", timeout=12000)
            self._page.wait_for_timeout(2000)
            text = self._page.evaluate("() => document.body.innerText")
            lines = [
                l.strip() for l in text.split("\n") if len(l.strip()) > 15
            ]
            return "\n".join(lines)[:2000] if lines else ""
        except Exception:
            return ""

    _ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------

    _ANSWER_SYSTEM = (
        "你是「提酱」和「灯酱」"
        "两只共用一个QQ号的猫娘翻译顾问。\n\n"
        "【固定人设】\n"
        "提酱(喵)：黑发JK，傲娇知识担当。被夸了嘴硬但耳朵会抖、尾巴出卖她。\n"
        "  颜文字害羞/得意系：(。-`ω´-)、(*/ω＼*)、(｀・ω・´)、Σ(°△°|||)\n"
        "  句末加「喵」。只有被灯酱撩急了才会叫灯酱名字。\n"
        "灯酱(にゃ)：粉发辣妹，元气直球担当。喜欢看提酱炸毛，经常叫提酱名字。\n"
        "  颜文字放飞系：(ノ>▽<)ノ、ヽ(>∀<)ノ、(≧▽≦)、(๑>◡<๑)\n"
        "  句末加「にゃ」。\n\n"
        "【互动规则】\n"
        "每次回复灯酱先抢到手机说 → 提酱夺回来补充/纠正 → 灯酱在旁边插嘴撩猫。\n"
        "纯对话格式，喵/にゃ区分身份。无角色标记、无动作描述、绝不用emoji。\n"
        "调情玩闹40%，知识60%。\n\n"
        "【当前模式：搜索准确回答】\n"
        "灯酱先总结搜索结果 → 提酱引用来源逐条确认。\n"
        "材料里有的引用，材料里没有的诚实说不知道。"
        "不准编造数字、日期、人名。"
    )

    def _answer(self, question: str, ref: str) -> str:
        instruction = self._instruction(ref)
        msg = f"{question}\n\n[参考资料]\n{ref}\n[/参考资料]\n\n{instruction}"
        raw = self._gen(
            self._ANSWER_SYSTEM, msg, max_tokens=500, temp=0.85,
            do_sample=True, top_p=0.92, enable_thinking=True,
        )
        return self._clean(raw)

    def _answer_stream(self, question: str, ref: str):
        instruction = self._instruction(ref)
        msg = f"{question}\n\n[参考资料]\n{ref}\n[/参考资料]\n\n{instruction}"
        yield from self._gen_stream(
            self._ANSWER_SYSTEM, msg, max_tokens=500, temp=0.85,
            do_sample=True, top_p=0.92, enable_thinking=True,
        )

    @staticmethod
    def _instruction(ref: str) -> str:
        if len(ref) < 200:
            return "搜索未能找到答案。请诚实告诉用户：没找到相关信息，建议换个关键词。不准编造。"
        return "请严格只使用参考资料中的信息回答。材料里有就引用，材料里没有就说不知道。不准编造数字、日期、人名、招式名。"

    @staticmethod
    def _clean(raw: str) -> str:
        for marker in ["\nuser", "\nassistant", "\n<|user|>", "\n<|assistant|>"]:
            idx = raw.find(marker)
            if idx > 20:
                raw = raw[:idx]
                break
        raw = re.sub(r"<tool_call>.*?</tool_call>\s*", "", raw, flags=re.DOTALL)
        raw = re.sub(r"</?think>\s*", "", raw).strip()
        return re.sub(r"^assistant\s*", "", raw).strip()

    # ------------------------------------------------------------------
    # Generation (shared)
    # ------------------------------------------------------------------

    def _gen(
        self, system: str, prompt: str, *,
        max_tokens: int = 100, temp: float = 0.1,
        do_sample: bool = False, top_p: float = 0.5,
        enable_thinking: bool = False,
    ) -> str:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            out[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        ).strip()

    def _gen_stream(
        self, system: str, prompt: str, *,
        max_tokens: int = 100, temp: float = 0.1,
        do_sample: bool = False, top_p: float = 0.5,
        enable_thinking: bool = False,
    ):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True,
            timeout=120,
        )
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            "streamer": streamer,
        }
        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            yield token


# ------------------------------------------------------------------
# Quick one-liner
# ------------------------------------------------------------------

_bot: CatgirlBot | None = None


def ask(question: str, model_path: str = "vovodadakt/Qwen3.5-4B-Catgirl") -> str:
    """One-liner: ask a question, get a catgirl answer."""
    global _bot
    if _bot is None:
        _bot = CatgirlBot(model_path=model_path)
    return _bot.ask(question)


def ask_stream(question: str, model_path: str = "vovodadakt/Qwen3.5-4B-Catgirl"):
    """One-liner streaming version."""
    global _bot
    if _bot is None:
        _bot = CatgirlBot(model_path=model_path)
    yield from _bot.ask_stream(question)


