#!/usr/bin/env python3
"""
Batch Book Aligner — process 20-30 book folders at once.

Expects this structure:
    books-root/
      book_001/
        page0001.jpg  (or 0001.jpg, page_0001.jpg — auto-detected)
        page0002.jpg
        book_001.txt  (one .txt file = full book ground truth)
      book_002/
        ...

Usage:
    # Dry run first — see alignment quality per book
    python3 align_all_books.py --books-root /path/to/books --out aligned/ --dry-run

    # Full run, specify which books to hold out for validation
    python3 align_all_books.py \
        --books-root /path/to/books \
        --out aligned/ \
        --val-books book_003,book_007,book_015 \
        --rec output/mk_rec_v7_infer

    # Auto-pick val books (last N books alphabetically)
    python3 align_all_books.py --books-root /path/to/books --out aligned/ --auto-val 3

Output:
    aligned/crops/           all line crop JPGs
    aligned/train.txt        PaddleOCR format labels (training books)
    aligned/val.txt          PaddleOCR format labels (val books)
    aligned/val.jsonl        JSONL with metadata per crop (val books)
    aligned/stats.json       per-book alignment statistics
"""

import os, sys, re, cv2, json, glob, argparse, difflib, unicodedata, tempfile
import numpy as np
from pathlib import Path

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# ── Text helpers ──────────────────────────────────────────────────────────────

_DASH_RE  = re.compile(r'[\-\u2010\u2011\u2012\u2013\u2014\u2015\u00AD]')
_SPACE_RE = re.compile(r'\s+')
_QUOTE_RE = re.compile(r'[„""«»\u2018\u2019]')

def normalise(text):
    text = _QUOTE_RE.sub('"', text)
    text = _DASH_RE.sub('-', text)
    text = unicodedata.normalize('NFC', text)
    return _SPACE_RE.sub(' ', text).strip().lower()

def dehyphenate(lines):
    result, i = [], 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line and _DASH_RE.search(line[-1:]) and i + 1 < len(lines):
            m = re.search(r'(\S+)' + _DASH_RE.pattern + r'$', line)
            nxt = lines[i + 1].lstrip()
            if m and nxt and nxt[0].islower():
                nm = re.match(r'(\S+)(.*)', nxt)
                if nm:
                    line = line[:m.start()] + m.group(1) + nm.group(1)
                    rest = nm.group(2).lstrip()
                    lines[i + 1] = rest if rest else ''
                    if not rest: i += 1
        result.append(line)
        i += 1
    return result

def cer(pred, gt):
    if not gt: return 0.0 if not pred else 1.0
    p, g = normalise(pred), normalise(gt)
    if p == g: return 0.0
    prev = list(range(len(g) + 1))
    for cp in p:
        curr = [prev[0] + 1]
        for j, cg in enumerate(g):
            curr.append(min(curr[j]+1, prev[j+1]+1, prev[j]+(0 if cp==cg else 1)))
        prev = curr
    return prev[-1] / len(g)

# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# ── OCR engine ────────────────────────────────────────────────────────────────

_engine = None

def get_engine(rec_dir):
    global _engine
    if _engine is None:
        from paddleocr import PaddleOCR
        kw = dict(use_doc_orientation_classify=False,
                  use_doc_unwarping=False,
                  use_textline_orientation=False)
        if rec_dir:
            kw['text_recognition_model_dir'] = rec_dir
        print(f"  Loading OCR (rec: {rec_dir or 'stock'})...")
        _engine = PaddleOCR(**kw)
    return _engine

def ocr_page(img_bgr, rec_dir, min_score=0.3):
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    cv2.imwrite(tmp.name, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    result = get_engine(rec_dir).predict(tmp.name)
    os.unlink(tmp.name)
    dets = []
    for page in result:
        boxes  = page.get('rec_boxes',  [])
        texts  = page.get('rec_texts',  [])
        scores = page.get('rec_scores', [])
        for box, text, score in zip(boxes, texts, scores):
            if score >= min_score and text.strip():
                dets.append((text.strip(), float(score), box))
    dets.sort(key=lambda d: (min(p[1] for p in d[2]) if d[2] is not None else 0))
    return dets

# ── Alignment ─────────────────────────────────────────────────────────────────

def align_page(ocr_text, gt_norm, cursor, window_factor=2.5):
    ocr_norm = normalise(ocr_text)
    if not ocr_norm:
        return cursor, cursor, 0.0
    search_end = min(len(gt_norm), cursor + int(len(ocr_norm) * window_factor))
    window = gt_norm[cursor:search_end]
    if not window:
        return cursor, cursor, 0.0
    best_score, best_pos = 0.0, 0
    step = max(1, len(ocr_norm) // 20)
    positions = list(range(0, max(1, len(window) - len(ocr_norm) + 1), step))
    if positions and positions[-1] != max(0, len(window) - len(ocr_norm)):
        positions.append(max(0, len(window) - len(ocr_norm)))
    for pos in positions:
        score = difflib.SequenceMatcher(
            None, ocr_norm, window[pos:pos + len(ocr_norm)], autojunk=False).ratio()
        if score > best_score:
            best_score, best_pos = score, pos
    gt_start = cursor + best_pos
    gt_end   = min(len(gt_norm), gt_start + len(ocr_norm))
    return gt_start, gt_end, best_score

def _extract_original(original, normalised, start, end):
    norm_pos, chars = 0, []
    for ch in original:
        n = normalise(ch)
        if n and norm_pos >= start:
            chars.append(ch)
        if n:
            norm_pos += len(n)
        if norm_pos >= end:
            break
    return ''.join(chars).strip() or original[start:end]

def align_lines(detections, gt_segment, max_cer=0.25):
    gt_norm, gt_cursor, results = normalise(gt_segment), 0, []
    for ocr_text, ocr_score, box in detections:
        ocr_norm = normalise(ocr_text)
        if not ocr_norm or len(ocr_norm) < 3:
            continue
        search_len = min(len(gt_norm) - gt_cursor, int(len(ocr_norm) * 2.5))
        if search_len <= 0:
            break
        window = gt_norm[gt_cursor:gt_cursor + search_len]
        best_score, best_pos = 0.0, 0
        step = max(1, len(ocr_norm) // 10)
        for pos in range(0, max(1, len(window) - len(ocr_norm) + 1), step):
            score = difflib.SequenceMatcher(
                None, ocr_norm, window[pos:pos + len(ocr_norm)], autojunk=False).ratio()
            if score > best_score:
                best_score, best_pos = score, pos
        line_cer = 1.0 - best_score
        if line_cer <= max_cer:
            gs = gt_cursor + best_pos
            ge = min(len(gt_norm), gs + len(ocr_norm))
            gt_line = _extract_original(gt_segment, gt_norm, gs, ge)
            results.append((ocr_text, gt_line, line_cer, box))
            gt_cursor = gs + max(1, len(ocr_norm) // 2)
    return results

def extract_crop(img_bgr, box, pad=4):
    if box is None or len(box) == 0:
        return None
    pts = np.array(box, dtype=np.float32)
    x1 = int(max(0, pts[:, 0].min() - pad))
    y1 = int(max(0, pts[:, 1].min() - pad))
    x2 = int(min(img_bgr.shape[1], pts[:, 0].max() + pad))
    y2 = int(min(img_bgr.shape[0], pts[:, 1].max() + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]

# ── Book detection ────────────────────────────────────────────────────────────

def find_books(books_root):
    """Find all book folders: must contain at least one image and exactly one .txt."""
    books_root = Path(books_root)
    books = []
    for folder in sorted(books_root.iterdir()):
        if not folder.is_dir():
            continue
        txts = list(folder.glob('*.txt'))
        imgs = []
        for ext in ['*.jpg','*.jpeg','*.png','*.JPG','*.JPEG','*.PNG']:
            imgs.extend(folder.glob(ext))
        if len(txts) == 1 and len(imgs) > 0:
            books.append((folder.name, folder, sorted(set(imgs)), txts[0]))
        elif len(txts) > 1:
            print(f"  [SKIP] {folder.name}: multiple .txt files found")
        elif len(imgs) == 0:
            pass  # empty or non-book folder, silently skip
    return books

# ── Per-book processing ───────────────────────────────────────────────────────

def process_book(book_name, book_dir, images, gt_file,
                 crops_dir, rec_dir, min_score, max_cer, dry_run, is_val):

    gt_text = gt_file.read_text(encoding='utf-8')
    gt_text = re.sub(r'\r\n|\r', '\n', gt_text)
    gt_text = re.sub(r'\n{3,}', '\n\n', gt_text).strip()
    gt_norm = normalise(gt_text)

    gt_cursor   = 0
    train_labels = []
    val_records  = []
    page_stats   = []

    for page_idx, img_path in enumerate(images):
        page_name = img_path.stem
        try:
            img_bgr = preprocess(img_path)
            dets    = ocr_page(img_bgr, rec_dir)
        except Exception as e:
            page_stats.append({'page': page_name, 'score': 0, 'det': 0, 'kept': 0, 'error': str(e)})
            continue

        if not dets:
            page_stats.append({'page': page_name, 'score': 0, 'det': 0, 'kept': 0})
            continue

        ocr_lines = dehyphenate([d[0] for d in dets])
        ocr_text  = ' '.join(ocr_lines)

        gt_start, gt_end, page_score = align_page(ocr_text, gt_norm, gt_cursor)
        if page_score < min_score:
            page_stats.append({'page': page_name, 'score': round(page_score, 3),
                                'det': len(dets), 'kept': 0, 'skipped': True})
            continue

        gt_segment = _extract_original(gt_text, gt_norm, gt_start, gt_end)
        gt_cursor  = gt_end

        aligned = align_lines(dets, gt_segment, max_cer=max_cer)
        kept = 0

        if not dry_run:
            for line_idx, (ocr_text_line, gt_line, line_cer, box) in enumerate(aligned):
                if not gt_line.strip():
                    continue
                crop = extract_crop(img_bgr, box)
                if crop is None or crop.size == 0:
                    continue

                crop_name = f"{book_name}__{page_name}_l{line_idx:03d}.jpg"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                abs_path = str(crop_path.absolute())

                if is_val:
                    val_records.append({
                        'image':       abs_path,
                        'label':       gt_line,
                        'book':        book_name,
                        'page':        page_name,
                        'line':        line_idx,
                        'align_score': round(page_score, 4),
                        'ocr_cer':     round(line_cer, 4),
                    })
                else:
                    train_labels.append(f"{abs_path}\t{gt_line}")
                kept += 1

        page_stats.append({'page': page_name, 'score': round(page_score, 3),
                            'det': len(dets), 'kept': kept if not dry_run else len(aligned)})

    total_det  = sum(s.get('det', 0) for s in page_stats)
    total_kept = sum(s.get('kept', 0) for s in page_stats)
    good_pages = sum(1 for s in page_stats if s.get('score', 0) >= min_score)
    avg_score  = sum(s.get('score', 0) for s in page_stats) / max(1, len(page_stats))

    return {
        'train_labels': train_labels,
        'val_records':  val_records,
        'stats': {
            'book':       book_name,
            'is_val':     is_val,
            'pages':      len(images),
            'aligned':    good_pages,
            'det_total':  total_det,
            'kept_total': total_kept,
            'avg_score':  round(avg_score, 3),
            'page_detail': page_stats,
        }
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Batch book aligner for 20-30 book folders')
    ap.add_argument('--books-root', required=True, help='Root folder containing book subfolders')
    ap.add_argument('--out',        required=True, help='Output directory')
    ap.add_argument('--rec',        default='output/mk_rec_v5_cyrillic_infer',
                    help='Recognition model dir (use V7 when ready)')
    ap.add_argument('--val-books',  default='',
                    help='Comma-separated book folder names to use as validation')
    ap.add_argument('--auto-val',   type=int, default=0,
                    help='Auto-pick last N books (alphabetically) as validation')
    ap.add_argument('--min-score',  type=float, default=0.5,
                    help='Min page alignment score to keep (default: 0.5)')
    ap.add_argument('--max-cer',    type=float, default=0.25,
                    help='Max line CER to keep a crop (default: 0.25)')
    ap.add_argument('--dry-run',    action='store_true',
                    help='Preview alignment quality without saving files')
    args = ap.parse_args()

    out_dir   = Path(args.out)
    crops_dir = out_dir / 'crops'
    if not args.dry_run:
        crops_dir.mkdir(parents=True, exist_ok=True)

    # Discover books
    books = find_books(args.books_root)
    if not books:
        print(f"ERROR: No book folders found in {args.books_root}")
        sys.exit(1)

    # Determine val set
    val_set = set()
    if args.val_books:
        val_set = set(v.strip() for v in args.val_books.split(',') if v.strip())
    elif args.auto_val > 0:
        val_set = set(name for name, *_ in books[-args.auto_val:])

    print(f"\n  Batch Book Aligner")
    print(f"  {'─'*55}")
    print(f"  Books root : {args.books_root}")
    print(f"  Output     : {args.out}")
    print(f"  Model      : {args.rec}")
    print(f"  Books found: {len(books)}")
    print(f"  Val books  : {val_set if val_set else '(none — all go to train)'}")
    print(f"  min_score  : {args.min_score}  max_cer: {args.max_cer}")
    if args.dry_run:
        print(f"  DRY RUN — no files will be written")
    print(f"  {'─'*55}\n")

    all_train_labels = []
    all_val_records  = []
    all_stats        = []

    for book_name, book_dir, images, gt_file in books:
        is_val = book_name in val_set
        tag    = '[VAL]' if is_val else '[TRN]'
        print(f"  {tag} {book_name}  ({len(images)} pages)", flush=True)

        result = process_book(
            book_name, book_dir, images, gt_file,
            crops_dir, args.rec, args.min_score, args.max_cer,
            args.dry_run, is_val)

        all_train_labels.extend(result['train_labels'])
        all_val_records.extend(result['val_records'])
        all_stats.append(result['stats'])

        s = result['stats']
        print(f"         pages={s['pages']}  aligned={s['aligned']}  "
              f"det={s['det_total']}  kept={s['kept_total']}  avg_score={s['avg_score']}")

    # Write outputs
    if not args.dry_run:
        if all_train_labels:
            train_file = out_dir / 'train.txt'
            train_file.write_text('\n'.join(all_train_labels), encoding='utf-8')
            print(f"\n  train.txt  : {len(all_train_labels):,} lines → {train_file}")

        if all_val_records:
            val_txt  = out_dir / 'val.txt'
            val_jsonl = out_dir / 'val.jsonl'
            val_txt.write_text(
                '\n'.join(f"{r['image']}\t{r['label']}" for r in all_val_records),
                encoding='utf-8')
            val_jsonl.write_text(
                '\n'.join(json.dumps(r, ensure_ascii=False) for r in all_val_records),
                encoding='utf-8')
            print(f"  val.txt    : {len(all_val_records):,} lines → {val_txt}")
            print(f"  val.jsonl  : {len(all_val_records):,} records → {val_jsonl}")

        stats_file = out_dir / 'stats.json'
        stats_file.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"  stats.json : {stats_file}")

    # Summary table
    total_pages = sum(s['pages'] for s in all_stats)
    total_aligned = sum(s['aligned'] for s in all_stats)
    total_kept = sum(s['kept_total'] for s in all_stats)
    total_det  = sum(s['det_total'] for s in all_stats)

    print(f"\n  {'='*55}")
    print(f"  SUMMARY")
    print(f"  {'='*55}")
    print(f"  {'Book':<25} {'Pages':>6} {'Aligned':>8} {'Kept':>7} {'Score':>7} {'Set':>5}")
    print(f"  {'─'*25} {'─'*6} {'─'*8} {'─'*7} {'─'*7} {'─'*5}")
    for s in all_stats:
        tag = 'VAL' if s['is_val'] else 'TRN'
        print(f"  {s['book']:<25} {s['pages']:>6} {s['aligned']:>8} "
              f"{s['kept_total']:>7} {s['avg_score']:>7.3f} {tag:>5}")
    print(f"  {'─'*25} {'─'*6} {'─'*8} {'─'*7} {'─'*7} {'─'*5}")
    print(f"  {'TOTAL':<25} {total_pages:>6} {total_aligned:>8} {total_kept:>7}")
    print(f"\n  Lines detected : {total_det:,}")
    print(f"  Lines kept     : {total_kept:,}  "
          f"({100*total_kept/max(1,total_det):.1f}% of detected)")
    if val_set:
        train_kept = sum(s['kept_total'] for s in all_stats if not s['is_val'])
        val_kept   = sum(s['kept_total'] for s in all_stats if s['is_val'])
        print(f"  Train crops    : {train_kept:,}")
        print(f"  Val crops      : {val_kept:,}")
    print()

    # Warn about poorly aligned books
    bad = [(s['book'], s['avg_score']) for s in all_stats if s['avg_score'] < args.min_score]
    if bad:
        print(f"  Books with low average alignment (check GT file or photo order):")
        for name, score in bad:
            print(f"    {name}: avg_score={score:.3f}")
        print()

if __name__ == '__main__':
    main()
