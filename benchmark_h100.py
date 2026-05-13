#!/usr/bin/env python3
"""
H100 VLM Benchmark — compare large vision-language models against V7 PaddleOCR
on 10 Macedonian Cyrillic book pages.

Models (Ollama):
    qwen2.5vl:72b  Qwen2.5-VL 72B  ~45 GB
    gemma4:31b     Gemma4 31B       ~19 GB
    phi4-vision    Phi-4 Vision      ~5 GB
    deepseek-ocr   DeepSeek-OCR      ~4 GB

Setup on H100:
    git clone https://github.com/mjurukov/macedonian-ocr
    cd macedonian-ocr
    bash setup_faculty.sh                  # installs Ollama, downloads V7 model
    python3 benchmark_h100.py --pull       # downloads ~73 GB of Ollama models
    python3 benchmark_h100.py

Resume after interruption:
    python3 benchmark_h100.py --resume results_h100.json

Run only specific models:
    python3 benchmark_h100.py --models qwen2.5vl:72b,deepseek-ocr
"""

import argparse, base64, json, os, re, subprocess, sys, time
import urllib.request, urllib.error

# ── Local model references  (benchmark_final.py run, corrected GT for 0010) ───

V7_REF = {
    '0001': 0.0018,
    '0002': 0.0022,
    '0003': 0.0072,
    '0004': 0.0031,
    '0005': 0.0036,
    '0006': 0.0067,
    '0007': 0.0024,
    '0008': 0.0012,
    '0009': 0.0005,
    '0010': 0.0023,  # corrected GT (0010.txt was updated)
}

V8_REF = {
    '0001': 0.0028,
    '0002': 0.0018,
    '0003': 0.0031,
    '0004': 0.0031,
    '0005': 0.0027,
    '0006': 0.0058,
    '0007': 0.0009,
    '0008': 0.0019,
    '0009': 0.0005,
    '0010': 0.0032,
}

V7_AVG_CER = sum(V7_REF.values()) / len(V7_REF)
V8_AVG_CER = sum(V8_REF.values()) / len(V8_REF)

# ── Model list ─────────────────────────────────────────────────────────────────
#
#  (ollama_tag, short_key, display_label, prompt_key)
#   short_key    — JSON key for save/resume, keep it stable
#   prompt_key   — 'vlm' for general-purpose models, 'ocr' for OCR-specialized
#

DEFAULT_MODELS = [
    ('qwen2.5vl:72b', 'qwen72b',   'Qwen2.5-VL 72B', 'vlm'),
    ('gemma4:31b',    'gemma4_31', 'Gemma4 31B',      'vlm'),
    ('phi4-vision',   'phi4v',     'Phi-4 Vision',    'vlm'),
    ('deepseek-ocr',  'deepseek',  'DeepSeek-OCR',    'ocr'),
]

# Approximate download sizes for the pull progress header
_DL_SIZES = {
    'qwen2.5vl:72b': '~45 GB',
    'gemma4:31b':    '~19 GB',
    'phi4-vision':   '~5 GB',
    'deepseek-ocr':  '~4 GB',
}

TEST_DIR   = 'data/test_data/test_books'
OLLAMA_URL = 'http://localhost:11434'  # overridden by --ollama-url
BASE       = os.path.dirname(os.path.abspath(__file__))

# ── Prompts ────────────────────────────────────────────────────────────────────

# For OCR-specialized models (deepseek-ocr): they already know to output clean
# text, so a short prompt is enough.
PROMPT_OCR = (
    "Transcribe every word of text visible in this image exactly as written. "
    "Preserve all Cyrillic characters including ѓ Ѓ ќ Ќ ѕ Ѕ љ Љ њ Њ џ Џ ј Ј ѐ Ѐ ѝ Ѝ. "
    "Output only the transcribed text."
)

# For general-purpose VLMs (Qwen, Gemma, Phi): these models tend to add
# preamble ("Sure, here is...") or commentary unless told very explicitly.
PROMPT_VLM = (
    "You are an OCR system. Your only job is to output the text from the image.\n"
    "STRICT RULES — follow exactly:\n"
    "1. Output ONLY the text that appears in the image. Nothing else.\n"
    "2. Do NOT write any introduction, greeting, or preamble of any kind.\n"
    "   Do not start with 'Sure', 'Here is', 'The text reads', or anything similar.\n"
    "3. Do NOT add any explanation, commentary, or closing remark after the text.\n"
    "4. Start your response with the very first word visible in the image.\n"
    "5. End your response with the very last word visible in the image.\n"
    "6. Preserve all Cyrillic characters exactly — especially "
    "ѓ Ѓ ќ Ќ ѕ Ѕ љ Љ њ Њ џ Џ ј Ј ѐ Ѐ ѝ Ѝ — do not substitute them with "
    "similar-looking characters from Russian or Bulgarian.\n"
    "7. Do not correct spelling or change any word."
)

PROMPTS = {'ocr': PROMPT_OCR, 'vlm': PROMPT_VLM}

# ── Metrics ────────────────────────────────────────────────────────────────────

def _edit(s1, s2):
    if len(s1) < len(s2):
        return _edit(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c in s1:
        curr = [prev[0] + 1]
        for j, d in enumerate(s2):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (0 if c == d else 1)))
        prev = curr
    return prev[-1]

def cer(pred, gt):
    p, g = ' '.join(pred.split()), ' '.join(gt.split())
    if not g:
        return 0.0 if not p else 1.0
    return _edit(p, g) / len(g)

def wer(pred, gt):
    pw, gw = pred.split(), gt.split()
    if not gw:
        return 0.0 if not pw else 1.0
    return _edit(pw, gw) / len(gw)

# ── Ollama helpers ─────────────────────────────────────────────────────────────

def _get(path, timeout=5):
    with urllib.request.urlopen(f'{OLLAMA_URL}{path}', timeout=timeout) as r:
        return json.loads(r.read())

def ollama_available():
    try:
        _get('/api/tags')
        return True
    except Exception:
        return False

def model_pulled(tag):
    try:
        names = [m['name'] for m in _get('/api/tags').get('models', [])]
        return any(n == tag or n.startswith(tag + ':') or n.startswith(tag.split(':')[0] + ':') for n in names)
    except Exception:
        return False

def pull_model(tag):
    size = _DL_SIZES.get(tag, '')
    print(f'  Pulling {tag} {size} ...')
    r = subprocess.run(['ollama', 'pull', tag])
    return r.returncode == 0

def unload_model(tag):
    """Evict model from GPU memory so the next model gets a clean slate."""
    payload = json.dumps({'model': tag, 'keep_alive': 0}).encode()
    req = urllib.request.Request(
        f'{OLLAMA_URL}/api/generate',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            pass
    except Exception:
        pass

def run_model(tag, img_path, prompt, timeout=300):
    with open(img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    payload = json.dumps({
        'model': tag,
        'prompt': prompt,
        'images': [img_b64],
        'stream': False,
        'options': {'temperature': 0},
    }).encode()
    req = urllib.request.Request(
        f'{OLLAMA_URL}/api/generate',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    ms = (time.perf_counter() - t0) * 1000
    text = data.get('response', '').strip()
    return ' '.join(text.split()), ms

# ── Test data ──────────────────────────────────────────────────────────────────

def load_pairs(test_dir):
    pairs = []
    for i in range(1, 100):
        img = os.path.join(test_dir, f'{i:04d}.jpg')
        txt = os.path.join(test_dir, f'{i:04d}.txt')
        if os.path.exists(img) and os.path.exists(txt):
            gt = ' '.join(open(txt, encoding='utf-8').read().strip().split())
            pairs.append((f'{i:04d}', img, gt))
    return pairs

# ── Save / resume ──────────────────────────────────────────────────────────────

def save_results(results, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def load_results(path):
    if path and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}

# ── Summary printer ────────────────────────────────────────────────────────────

def print_summary(pairs, results, model_list):
    keys   = [key for _, key, *_ in model_list if key in results]
    labels = {key: results[key]['label'] for key in keys}

    def avg(key, metric):
        vals = [r[metric] for r in results[key]['pages']]
        return sum(vals) / len(vals) if vals else 1.0

    wins = {k: 0 for k in keys}
    for i in range(len(pairs)):
        row = {k: results[k]['pages'][i]['cer']
               for k in keys if i < len(results[k]['pages'])}
        if row:
            wins[min(row, key=row.get)] += 1

    ranked = sorted(keys, key=lambda k: avg(k, 'cer'))

    W = 72
    print(f'\n{"="*W}')
    print(f'  SUMMARY — {len(pairs)} pages')
    print(f'{"="*W}')
    print(f'  {"#":<3} {"Engine":<24} {"Avg CER":>8} {"Avg WER":>8} {"Wins":>6}  {"vs V8":>8}')
    print(f'  {"─"*3} {"─"*24} {"─"*8} {"─"*8} {"─"*6}  {"─"*8}')
    # Both local models shown as reference rows
    print(f'  {"ref":<3} {"V8 PaddleOCR (ours)":<24} {V8_AVG_CER*100:>7.2f}%  {"──":>7}  {"──":>6}  {"baseline":>8}')
    print(f'  {"ref":<3} {"V7 PaddleOCR":<24} {V7_AVG_CER*100:>7.2f}%  {"──":>7}  {"──":>6}  {(V7_AVG_CER-V8_AVG_CER)*100:>+7.2f}%')
    for rank, k in enumerate(ranked, 1):
        ac = avg(k, 'cer')
        aw = avg(k, 'wer')
        delta = (ac - V8_AVG_CER) * 100
        sign  = '+' if delta >= 0 else ''
        print(f'  {rank:<3} {labels[k]:<24} {ac*100:>7.2f}%  {aw*100:>6.1f}%  {wins[k]:>3}/{len(pairs)}'
              f'  {sign}{delta:>+.2f}%')

    print(f'\n  Character Accuracy:')
    for label, avg_cer in [('V8 PaddleOCR (ours)', V8_AVG_CER), ('V7 PaddleOCR', V7_AVG_CER)]:
        bar = '#' * int((1 - avg_cer) * 50)
        print(f'    {label:<24} {(1-avg_cer)*100:>6.2f}%  {bar}')
    for k in ranked:
        ac  = avg(k, 'cer')
        bar = '#' * int((1 - ac) * 50)
        print(f'    {labels[k]:<24} {(1-ac)*100:>6.2f}%  {bar}')

    print(f'\n  Per-page CER:')
    col_keys = ['v8', 'v7'] + ranked
    col_hdrs = ['V8(ours)', 'V7(ref)'] + [labels[k][:10] for k in ranked]
    hdr_line = f'  {"Page":<8}' + ''.join(f' {h:>12}' for h in col_hdrs)
    print(hdr_line)
    print(f'  {"─"*8}' + ''.join(f' {"─"*12}' for _ in col_keys))

    page_cer = {'v8': V8_REF, 'v7': V7_REF}
    for k in keys:
        page_cer[k] = {r['page']: r['cer'] for r in results[k]['pages']}

    for name, _, _ in pairs:
        cers = {k: page_cer.get(k, {}).get(name) for k in col_keys}
        valid = {k: v for k, v in cers.items() if v is not None}
        best  = min(valid.values()) if valid else None
        row   = f'  {name:<8}'
        for k in col_keys:
            c = cers.get(k)
            if c is None:
                row += f' {"──":>12}'
            else:
                marker = '*' if (best is not None and abs(c - best) < 1e-9) else ' '
                row += f' {c*100:>9.2f}%{marker}'
        print(row)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global OLLAMA_URL
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test-dir', default=os.path.join(BASE, TEST_DIR),
                    help='Directory with 0001.jpg / 0001.txt pairs')
    ap.add_argument('--pull', action='store_true',
                    help='Pull any missing Ollama models before running')
    ap.add_argument('--pull-only', action='store_true',
                    help='Pull models and exit without running benchmark')
    ap.add_argument('--resume', metavar='FILE',
                    help='Load saved JSON and skip already-completed models')
    ap.add_argument('--save', metavar='FILE', default='results_h100.json',
                    help='Where to save results (default: results_h100.json)')
    ap.add_argument('--models', metavar='TAG[,TAG...]',
                    help='Override model list with comma-separated Ollama tags')
    ap.add_argument('--ollama-url', default=OLLAMA_URL,
                    help=f'Ollama base URL (default: {OLLAMA_URL})')
    args = ap.parse_args()
    OLLAMA_URL = args.ollama_url.rstrip('/')

    # Build model list
    if args.models:
        model_list = []
        for tag in args.models.split(','):
            tag = tag.strip()
            key = re.sub(r'[^a-z0-9_]', '_', tag.lower())[:16]
            model_list.append((tag, key, tag, 'vlm'))  # unknown models default to vlm prompt
    else:
        model_list = list(DEFAULT_MODELS)

    # ── Pull phase ─────────────────────────────────────────────────────────────
    if args.pull or args.pull_only:
        if not ollama_available():
            print('ERROR: Ollama not reachable. Start it with: ollama serve')
            sys.exit(1)
        total = sum(
            float(re.search(r'[\d.]+', _DL_SIZES.get(tag, '0 GB')).group())
            for tag, *_ in model_list
            if not model_pulled(tag)
        )
        print(f'\nPulling missing models  (up to ~{total:.0f} GB):')
        for tag, _, label, *_ in model_list:
            if model_pulled(tag):
                print(f'  {label:<26}  already present')
            else:
                if not pull_model(tag):
                    print(f'  WARNING: failed to pull {tag}')
        if args.pull_only:
            print('\nDone.')
            return

    # ── Sanity checks ──────────────────────────────────────────────────────────
    if not ollama_available():
        print('ERROR: Ollama not reachable. Start it with: ollama serve')
        sys.exit(1)

    pairs = load_pairs(args.test_dir)
    if not pairs:
        print(f'ERROR: No .jpg/.txt pairs found in {args.test_dir}')
        sys.exit(1)

    # ── Resume ────────────────────────────────────────────────────────────────
    results = load_results(args.resume)
    if results:
        done = [results[k]['label'] for k in results]
        print(f'Resuming — skipping completed: {", ".join(done)}')

    pending = [(tag, key, label, pk) for tag, key, label, pk in model_list
               if key not in results]

    # ── Header ────────────────────────────────────────────────────────────────
    print(f'\n{"="*68}')
    print(f'  H100 VLM Benchmark — {len(pairs)} pages')
    print(f'  V8 PaddleOCR (ours): avg CER {V8_AVG_CER*100:.2f}%  ← beat this')
    print(f'  V7 PaddleOCR:        avg CER {V7_AVG_CER*100:.2f}%')
    print(f'{"="*68}')
    for tag, key, label, pk in model_list:
        size  = _DL_SIZES.get(tag, '')
        check = '  [done]' if key in results else ''
        ptype = '  [ocr prompt]' if pk == 'ocr' else '  [vlm prompt]'
        print(f'  {label:<26}  {size:<9}{ptype}{check}')
    print(f'{"="*68}\n')

    # ── Run each model sequentially ────────────────────────────────────────────
    for tag, key, label, prompt_key in pending:
        if not model_pulled(tag):
            print(f'  [SKIP] {label} — not pulled  (run with --pull first)')
            continue

        prompt = PROMPTS[prompt_key]
        print(f'  ── {label} {"─"*(50-len(label))}')
        per_page = []

        for i, (name, img_path, gt) in enumerate(pairs):
            # First page gets extra time for model load into GPU memory
            timeout = 900 if i == 0 else 300
            try:
                pred, ms = run_model(tag, img_path, prompt, timeout=timeout)
                c = cer(pred, gt)
                w = wer(pred, gt)
                v8_c  = V8_REF.get(name)
                delta = f'  Δ={((c - v8_c)*100):+.2f}% vs V8' if v8_c is not None else ''
                print(f'  {name}.jpg  CER={c*100:.2f}%  WER={w*100:.1f}%  ({ms/1000:.0f}s){delta}')
                per_page.append({
                    'page': name, 'cer': round(c, 6), 'wer': round(w, 6),
                    'ms': round(ms), 'pred': pred,
                })
            except Exception as e:
                print(f'  {name}.jpg  FAIL: {e}')
                per_page.append({
                    'page': name, 'cer': 1.0, 'wer': 1.0,
                    'ms': 0, 'pred': f'FAIL: {e}',
                })

        n = len(per_page)
        avg_c = sum(r['cer'] for r in per_page) / n
        avg_w = sum(r['wer'] for r in per_page) / n
        delta_v8 = (avg_c - V8_AVG_CER) * 100
        sign = '+' if delta_v8 >= 0 else ''
        print(f'  → avg CER={avg_c*100:.2f}%  avg WER={avg_w*100:.1f}%'
              f'  ({sign}{delta_v8:.2f}% vs V8)')
        print(f'  Unloading {tag} from GPU memory...')
        unload_model(tag)
        time.sleep(2)
        print()

        results[key] = {'tag': tag, 'label': label, 'pages': per_page}
        save_results(results, args.save)
        print(f'  [saved → {args.save}]\n')

    if not results:
        print('No results collected.'); return

    print_summary(pairs, results, model_list)
    print(f'\n  Full predictions saved to: {args.save}')


if __name__ == '__main__':
    main()
