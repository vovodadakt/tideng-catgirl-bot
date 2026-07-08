"""Test moegirl + weblio via urllib (no browser)."""
import urllib.request, urllib.parse, re

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'text/html,*/*'})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return 'ERROR: ' + str(e)

# ═══ Moegirl search ═══
for query in ['五条悟', '间谍过家家', '孤独摇滚 PA', 'ツンデレ', 'お疲れ様']:
    print('=== MOEGIRL: %s ===' % query)
    q = urllib.parse.quote(query)
    html = fetch('https://zh.moegirl.org.cn/index.php?search=' + q)
    print('Length:', len(html))

    # Try to extract search results
    # Pattern 1: MediaWiki .mw-search-result-heading
    results_found = False
    for m in re.finditer(
        r'<div class="mw-search-result-heading"[^>]*>'
        r'\s*<a\s+href="([^"]+)"[^>]*title="([^"]*)"[^>]*>'
        r'([^<]+)</a>',
        html
    ):
        href, title, text = m.group(1), m.group(2), m.group(3)
        if any(x in href for x in ['Special:', 'User:', 'Talk:', 'File:', 'Template:', 'Help:', 'Category:', 'action=', 'Portal:']):
            continue
        full = 'https://zh.moegirl.org.cn' + href if href.startswith('/') else href
        print('  -> %s' % full)
        results_found = True

    if not results_found:
        # Pattern 2: any <a href="/..." title="..."> with meaningful title
        items = re.findall(r'<a[^>]*href="(/[^"]+)"[^>]+title="([^"]+)"', html)
        seen = set()
        for href, title in items:
            if title in seen or len(title) < 2:
                continue
            if any(x in href for x in ['Special:', 'User:', 'Talk:', 'File:', 'Template:', 'Help:', 'Category:', 'action=', 'Portal:', 'project:']):
                continue
            seen.add(title)
            full = 'https://zh.moegirl.org.cn' + href
            print('  -> [%s] %s' % (title[:50], full))
            if len(seen) >= 5:
                break

    if not results_found and len(seen) == 0:
        # Debug: show what the page looks like
        snippet = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        snippet = re.sub(r'<style[^>]*>.*?</style>', '', snippet, flags=re.DOTALL)
        snippet = re.sub(r'<[^>]+>', '\n', snippet)
        lines = [l.strip() for l in snippet.split('\n') if len(l.strip()) > 30]
        print('  NO RESULTS FOUND. Page text:')
        for l in lines[:10]:
            print('    | %s' % l[:150])
    print()

# ═══ Weblio lookup ═══
for query in ['お疲れ様', '尊い', 'ツンデレ', 'ご苦労様']:
    print('=== WEBLIO: %s ===' % query)
    q = urllib.parse.quote(query)
    html = fetch('https://www.weblio.jp/content/' + q)
    print('Length:', len(html))

    # Extract meaning/definition
    # Weblio uses various patterns: #summary, .kiji, .meaning, .netDicBody
    content = ''

    # Try summary div
    for pattern in [
        r'<div[^>]*id="summary"[^>]*>(.*?)</div>',
        r'<div[^>]*class="[^"]*kiji[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]*class="[^"]*netDicBody[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]*class="[^"]*meaning[^"]*"[^>]*>(.*?)</div>',
    ]:
        m = re.search(pattern, html, re.DOTALL)
        if m:
            content = m.group(1)
            break

    if content:
        clean = re.sub(r'<[^>]+>', ' ', content)
        clean = re.sub(r'\s+', ' ', clean).strip()
        print('  DEF:', clean[:500])
    else:
        # Fallback: get all text and find meaningful parts
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<br\s*/?>', '\n', text)
        text = re.sub(r'<[^>]+>', '\n', text)
        lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 30]
        for l in lines[:8]:
            print('  TXT:', l[:200])
    print()

print('DONE!')
