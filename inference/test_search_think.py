"""
Integrated search + think-debate inference pipeline.
Tests: does the LoRA model call <tool_call> to search, then use results in <think> debate?

Flow: user → model → <tool_call> → execute search → [参考资料] → model → <think>辩论</think>收尾
"""
import os, time, torch, re, json, urllib.parse, uuid, random
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import requests

# ── Config ──
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3.5-4B-Instruct"
LORA_PATH = "/root/data/training/lora_output/think_v1/final"
OUT = "/root/data/training/search_think_output.txt"

TEMPERATURE = 0.7
MAX_NEW = 768
MAX_TOOL_ROUNDS = 2  # max search → answer rounds

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# ═══════════════════════════════════════════════════════════════
#  System Prompt
# ═══════════════════════════════════════════════════════════════
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

【思考模式说明】
你的思考过程本身就是猫娘辩论——灯酱先说、提酱纠正、灯酱再撩。
在 <think> 标签内输出猫娘辩论，</think> 之后简短收尾。

【工具使用】
你可以通过 <tool_call> 搜索信息。格式：
<tool_call>{"name": "search_web", "arguments": {"query": "搜索关键词"}}</tool_call>
收到搜索结果后会以 [参考资料]...[、/参考资料] 形式给出。
工具调用只用于需要查证的问题，日常聊天不需要。

【参考材料处理】
· 材料清晰 → 灯酱总结、提酱引用原文确认，不制造假错误
· 材料混乱 → 提酱逐条引用原文纠正灯酱
· 材料缺失 → 诚实说明"""

# ═══════════════════════════════════════════════════════════════
#  Search Engines
# ═══════════════════════════════════════════════════════════════

def bing_search(query: str, max_results: int = 5) -> str:
    """Bing China search via HTTP."""
    s = None
    try:
        s = requests.Session()
        s.headers.update({
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-CN,zh;q=0.9,ja;q=0.8",
            "Referer": "https://cn.bing.com/",
        })
        s.get("https://cn.bing.com/", timeout=10)

        cvid = uuid.uuid4().hex.upper()[:32]
        params = {
            "q": query, "qs": "n", "form": "QBRE",
            "sp": "-1", "lq": "0", "pq": query[:20],
            "sc": "0-9", "sk": "", "cvid": cvid,
        }
        url = f"https://cn.bing.com/search?{urllib.parse.urlencode(params)}"
        r = s.get(url, timeout=12, allow_redirects=True)
        html = r.text

        results = []
        for match in re.finditer(
            r'<div[^>]*class="[^"]*b_caption[^"]*"[^>]*>(.*?)</div>',
            html, re.DOTALL
        ):
            text = re.sub(r'<[^>]+>', ' ', match.group(1))
            text = re.sub(r'&[a-z]+;', ' ', text)
            text = re.sub(r'&ensp;', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 30 and text not in results:
                results.append(text)

        if results:
            parts = [f"[Bing搜索]"]
            for i, r in enumerate(results[:max_results], 1):
                parts.append(f"{i}. {r[:300]}")
            return "\n".join(parts)

        return "(Bing: 无结果)"
    except Exception as e:
        return f"(Bing错误: {e})"
    finally:
        if s: s.close()


class MoegirlSearch:
    """Moegirl Bot API search — logged-in for higher rate limits."""

    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self._login()

    def _login(self):
        try:
            r = self.s.post("https://mzh.moegirl.org.cn/api.php", data={
                "action": "query", "meta": "tokens", "type": "login",
                "format": "json", "formatversion": "2",
            }, timeout=15)
            token = r.json().get("query", {}).get("tokens", {}).get("logintoken", "")
            r2 = self.s.post("https://mzh.moegirl.org.cn/api.php", data={
                "action": "login",
                "lgname": "MoeMoe1580421@ai_bot",
                "lgpassword": "dgofc9mifegemt33v8nfvnj6h7dg8v9v",
                "lgtoken": token,
                "format": "json", "formatversion": "2",
            }, timeout=15)
            self.logged_in = r2.json().get("login", {}).get("result") == "Success"
        except:
            self.logged_in = False

    def opensearch(self, query: str, limit: int = 5) -> list:
        """Search titles. Returns [(title, url), ...]"""
        try:
            q = urllib.parse.quote(query)
            url = f"https://zh.moegirl.org.cn/api.php?action=opensearch&search={q}&limit={limit}&format=json"
            r = requests.get(url, headers={"User-Agent": UA}, timeout=10)
            data = r.json()
            if len(data) >= 4:
                return list(zip(data[1], data[3]))
        except:
            pass
        return []

    def get_extract(self, title: str) -> str:
        """Get page extract text."""
        if not self.logged_in:
            return ""
        try:
            r = self.s.post("https://mzh.moegirl.org.cn/api.php", data={
                "action": "query", "titles": title,
                "prop": "extracts", "explaintext": 1,
                "exsectionformat": "plain",
                "format": "json", "formatversion": "2",
            }, timeout=15)
            pages = r.json().get("query", {}).get("pages", [])
            if pages and "extract" in pages[0]:
                return pages[0]["extract"]
        except:
            pass
        return ""

    def search_and_extract(self, query: str) -> str:
        """Search Moegirl and get extracts for top matches."""
        titles = self.opensearch(query, limit=3)
        if not titles:
            return "(萌娘百科: 无结果)"

        parts = ["[萌娘百科]"]
        for title, url in titles:
            extract = self.get_extract(title)
            if extract and len(extract) > 50:
                # Take first 800 chars as summary
                summary = extract[:800]
                parts.append(f"【{title}】{url}\n{summary}...")
            else:
                parts.append(f"【{title}】{url}\n(无详细内容)")

        return "\n\n".join(parts) if len(parts) > 1 else "(萌娘百科: 无结果)"


# ═══════════════════════════════════════════════════════════════
#  Tool Execution
# ═══════════════════════════════════════════════════════════════

def execute_tool_call(tool_call_text: str, moegirl: MoegirlSearch) -> str:
    """Parse <tool_call> JSON, execute search, return formatted results."""
    # Extract JSON from <tool_call>...</tool_call>
    json_match = re.search(r'<tool_call>(.*?)</tool_call>', tool_call_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{.*"name"\s*:\s*"search_web".*\}', tool_call_text, re.DOTALL)
        if not json_match:
            return None

    try:
        call_data = json.loads(json_match.group(1) if json_match.lastindex else json_match.group(0))
    except json.JSONDecodeError:
        return None

    tool_name = call_data.get("name", "")
    args = call_data.get("arguments", {})
    query = args.get("query", "")

    if not query:
        return None

    print(f"  🔍 搜索: {query[:60]}...", flush=True)

    results = []

    # Bing (always runs — fast, covers everything)
    bing_result = bing_search(query)
    if bing_result and "错误" not in bing_result:
        results.append(bing_result)

    # Moegirl (for ACG-related queries)
    moegirl_result = moegirl.search_and_extract(query)
    if moegirl_result and "无结果" not in moegirl_result:
        results.append(moegirl_result)

    if not results:
        return "[参考资料]\n（未找到相关结果）\n[/参考资料]"

    combined = "[参考资料]\n" + "\n\n---\n\n".join(results) + "\n[/参考资料]"
    # Truncate to avoid blowing context
    if len(combined) > 3000:
        combined = combined[:3000] + "\n...(截断)\n[/参考资料]"
    return combined


# ═══════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════

def generate(model, tokenizer, messages, max_tokens=MAX_NEW):
    """Generate text from messages."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.95,
            top_k=20,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    output_ids = out[0][input_len:].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=False)
    return output_text, len(output_ids)


def run_one_question(model, tokenizer, moegirl, question, test_id, output_file):
    """Run one question through the search+think pipeline."""
    print(f"\n{'='*60}", flush=True)
    print(f"Q{test_id}: {question}", flush=True)
    output_file.write(f"\n{'='*60}\nQ{test_id}: {question}\n{'='*60}\n")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    total_time = 0

    for round_idx in range(1, MAX_TOOL_ROUNDS + 1):
        print(f"  Round {round_idx}...", flush=True)
        t0 = time.time()

        output_text, n_tokens = generate(model, tokenizer, messages)
        elapsed = time.time() - t0
        total_time += elapsed

        print(f"  {elapsed:.1f}s | {n_tokens} tokens", flush=True)
        output_file.write(f"\n--- Round {round_idx} ({elapsed:.1f}s, {n_tokens}tk) ---\n{output_text}\n")

        # Check for tool_call
        has_tool_call = "<tool_call>" in output_text

        if has_tool_call:
            print(f"  ✅ 模型主动调用了搜索工具！", flush=True)

            # Execute search
            ref_text = execute_tool_call(output_text, moegirl)
            if ref_text:
                # Add assistant response + tool result as new messages
                messages.append({"role": "assistant", "content": output_text})
                messages.append({"role": "user", "content": ref_text})
                print(f"  参考资料: {len(ref_text)}ch → 继续生成...", flush=True)
                output_file.write(f"\n[搜索执行]\n{ref_text[:500]}...\n")
            else:
                print(f"  ⚠️ 无法解析 tool_call", flush=True)
                break
        else:
            print(f"  无 tool_call → 最终回答", flush=True)
            output_file.write(f"\n[最终回答]\n")
            break

    print(f"  总耗时: {total_time:.1f}s", flush=True)
    return total_time


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading Qwen3.5-4B + LoRA...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()
    print("Model ready.", flush=True)

    print("Init Moegirl API...", flush=True)
    moegirl = MoegirlSearch()
    print(f"Moegirl login: {moegirl.logged_in}", flush=True)

    # Test questions
    questions = [
        "鬼灭之刃里水之呼吸的使用者有哪些？",
        "日语里「猫かわいがり」是什么意思？",
        "最近有什么好看的动漫推荐吗にゃ？",
        "药屋少女的呢喃里猫猫的声优是谁？配过哪些角色？",
    ]

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("搜索+Think推理测试\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"LoRA: {LORA_PATH}\n\n")

        total_time = 0
        for i, q in enumerate(questions, 1):
            t = run_one_question(model, tokenizer, moegirl, q, i, f)
            total_time += t

        f.write(f"\n\n总耗时: {total_time:.1f}s, 平均: {total_time/len(questions):.1f}s\n")

    print(f"\nDone! Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
