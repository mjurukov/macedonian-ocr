#!/usr/bin/env python3
"""
Book Aligner — align photographed book pages to a ground truth .txt file.

Given a folder of page photos and one big GT text file, this script:
  1. Runs OCR (stock detection + your rec model) on every page
  2. Aligns each page's OCR output to the correct position in the GT text
  3. Saves per-page GT segments + individual line crops with GT labels
  4. Outputs a PaddleOCR-format label file ready for V8 training

Usage:
    python3 align_book.py --book-dir /path/to/photos --gt /path/to/book.txt --out /path/to/output
    python3 align_book.py --book-dir photos/ --gt book.txt --out aligned/ --rec output/mk_rec_v7_infer
    python3 align_book.py --book-dir photos/ --gt book.txt --out aligned/ --dry-run

Options:
    --book-dir   Folder of page photos (jpg/png, sorted by filename)
    --gt         Ground truth .txt file (full book text)
    --out        Output directory for crops + label file
    --rec        Recognition model dir (default: V5; swap to V7 when ready)
    --min-score  Min alignment score 0-1 to keep a page (default: 0.5)
    --max-cer    Max CER to keep an individual line crop (default: 0.25)
    --dry-run    Run OCR and alignment but don't save crops (preview only)
"""

import os, sys, re, cv2, glob, argparse, difflib, unicodedata
import numpy as np
from pathlib import Path

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# ── Text normalisation ────────────────────────────────────────────────────────

_DASH_RE   = re.compile(r'[\-\u2010\u2011\u2012\u2013\u2014\u2015\u00AD]')
_SPACE_RE  = re.compile(r'\s+')
_QUOTE_RE  = re.compile(r'[„""«»\u2018\u2019]')

def normalise(text):
    """Lowercase, collapse whitespace, normalise quotes and dashes for comparison."""
    text = _QUOTE_RE.sub('"', text)
    text = _DASH_RE.sub('-', text)
    text = unicodedata.normalize('NFC', text)
    text = _SPACE_RE.sub(' ', text).strip().lower()
    return text

def dehyphenate_lines(lines):
    """Merge hyphenated line endings across OCR line list."""
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

# ── Edit distance ─────────────────────────────────────────────────────────────

def cer(pred, gt):
    """Character error rate between two strings."""
    if not gt: return 0.0 if not pred else 1.0
    p, g = normalise(pred), normalise(gt)
    if p == g: return 0.0
    # Standard Levenshtein
    prev = list(range(len(g) + 1))
    for cp in p:
        curr = [prev[0] + 1]
        for j, cg in enumerate(g):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (0 if cp == cg else 1)))
        prev = curr
    return prev[-1] / len(g)

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(img_path):
    """CLAHE + unsharp mask on grayscale page photo."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# ── OCR ───────────────────────────────────────────────────────────────────────

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
        print(f"  Loading OCR engine (rec: {rec_dir or 'stock'})...")
        _engine = PaddleOCR(**kw)
    return _engine

def ocr_page(img_bgr, rec_dir, min_score=0.3):
    """Run OCR on a preprocessed BGR image. Returns list of (text, score, bbox)."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    cv2.imwrite(tmp.name, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    engine = get_engine(rec_dir)
    result = engine.predict(tmp.name)
    os.unlink(tmp.name)

    detections = []
    for page in result:
        boxes  = page.get('rec_boxes',  [])
        texts  = page.get('rec_texts',  [])
        scores = page.get('rec_scores', [])
        for box, text, score in zip(boxes, texts, scores):
            if score >= min_score and text.strip():
                detections.append((text.strip(), float(score), box))
    # Sort top-to-bottom by the top-left y coordinate of each box
    detections.sort(key=lambda d: (min(p[1] for p in d[2]) if d[2] is not None else 0))
    return detections

# ── Page-level alignment ──────────────────────────────────────────────────────

def align_page_to_gt(ocr_text, gt_chars, cursor, window_factor=2.5):
    """
    Find the best matching position in gt_chars for ocr_text, starting from cursor.
    Returns (start, end, score) in gt_chars index space.
    score is SequenceMatcher ratio (0-1, higher=better).
    """
    ocr_norm  = normalise(ocr_text)
    gt_norm   = normalise(gt_chars)
    ocr_len   = len(ocr_norm)
    if ocr_len == 0:
        return cursor, cursor, 0.0

    # Search window: from cursor to cursor + ocr_len * window_factor
    search_start = cursor
    search_end   = min(len(gt_norm), cursor + int(ocr_len * window_factor))
    window        = gt_norm[search_start:search_end]

    if not window:
        return cursor, cursor, 0.0

    best_score, best_pos = 0.0, 0
    # Slide a window of ocr_len characters through the GT search window
    step = max(1, ocr_len // 20)
    positions = list(range(0, max(1, len(window) - ocr_len + 1), step))
    if positions and positions[-1] != len(window) - ocr_len:
        positions.append(max(0, len(window) - ocr_len))

    for pos in positions:
        candidate = window[pos:pos + ocr_len]
        score = difflib.SequenceMatcher(None, ocr_norm, candidate, autojunk=False).ratio()
        if score > best_score:
            best_score = score
            best_pos   = pos

    gt_start = search_start + best_pos
    gt_end   = min(len(gt_norm), gt_start + ocr_len)
    return gt_start, gt_end, best_score

# ── Line-level alignment ──────────────────────────────────────────────────────

def align_lines_to_gt_segment(detections, gt_segment, max_cer=0.25):
    """
    Match each detected OCR line to a position in the GT segment.
    Returns list of (text, gt_text, score, box) for lines that pass max_cer.
    """
    gt_words  = gt_segment.split()
    gt_norm   = normalise(gt_segment)
    results   = []
    gt_cursor = 0

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
            cand  = window[pos:pos + len(ocr_norm)]
            score = difflib.SequenceMatcher(None, ocr_norm, cand, autojunk=False).ratio()
            if score > best_score:
                best_score, best_pos = score, pos

        line_cer = 1.0 - best_score
        if line_cer <= max_cer:
            gt_start = gt_cursor + best_pos
            gt_end   = min(len(gt_norm), gt_start + len(ocr_norm))
            # Extract GT text (original case) by mapping char positions
            gt_line  = _extract_original(gt_segment, gt_norm, gt_start, gt_end)
            results.append((ocr_text, gt_line, line_cer, box))
            gt_cursor = gt_start + max(1, len(ocr_norm) // 2)

    return results

def _extract_original(original, normalised, start, end):
    """Map normalised char positions back to original-case text."""
    # Build a char-position map from original → normalised
    norm_pos = 0
    orig_chars = []
    for ch in original:
        n = normalise(ch)
        if n and norm_pos >= start:
            orig_chars.append(ch)
        if n:
            norm_pos += len(n)
        if norm_pos >= end:
            break
    result = ''.join(orig_chars).strip()
    return result if result else original[start:end]

# ── Crop extraction ───────────────────────────────────────────────────────────

def extract_crop(img_bgr, box, pad=4):
    """Crop a line region from the image given a polygon box."""
    if box is None or len(box) == 0:
        return None
    pts = np.array(box, dtype=np.float32)
    x1, y1 = int(max(0, pts[:, 0].min() - pad)), int(max(0, pts[:, 1].min() - pad))
    x2, y2 = int(min(img_bgr.shape[1], pts[:, 0].max() + pad)), \
              int(min(img_bgr.shape[0], pts[:, 1].max() + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]

# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(args):
    book_dir = Path(args.book_dir)
    out_dir  = Path(args.out)
    crops_dir = out_dir / 'crops'

    if not args.dry_run:
        crops_dir.mkdir(parents=True, exist_ok=True)

    # Load GT text
    gt_text = Path(args.gt).read_text(encoding='utf-8')
    # Remove form feeds, normalise line endings, collapse blank lines
    gt_text = re.sub(r'\r\n|\r', '\n', gt_text)
    gt_text = re.sub(r'\n{3,}', '\n\n', gt_text).strip()
    gt_norm = normalise(gt_text)
    print(f"  GT text: {len(gt_text):,} chars, {len(gt_text.split()):,} words")

    # Find page images
    exts   = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    for ext in exts:
        images.extend(book_dir.glob(ext))
    images = sorted(set(images))
    if not images:
        print(f"ERROR: No images found in {book_dir}")
        sys.exit(1)
    print(f"  Pages: {len(images)}")

    gt_cursor  = 0
    labels     = []
    page_stats = []

    print(f"\n  Processing pages...\n")

    for page_idx, img_path in enumerate(images):
        page_name = img_path.stem
        print(f"  [{page_idx+1:3d}/{len(images)}] {img_path.name}", end='', flush=True)

        # OCR
        try:
            img_bgr    = preprocess_image(str(img_path))
            detections = ocr_page(img_bgr, args.rec, min_score=0.3)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if not detections:
            print(f"  — no detections, skipping")
            continue

        ocr_lines  = dehyphenate_lines([d[0] for d in detections])
        ocr_text   = ' '.join(ocr_lines)

        # Page-level alignment
        gt_start, gt_end, page_score = align_page_to_gt(
            ocr_text, gt_norm, gt_cursor, window_factor=2.5)

        if page_score < args.min_score:
            print(f"  — alignment score {page_score:.2f} too low (< {args.min_score}), skipping")
            page_stats.append((page_name, len(detections), 0, page_score, 0))
            continue

        # Extract GT segment for this page (original case)
        gt_segment = _extract_original(gt_text, gt_norm, gt_start, gt_end)
        gt_cursor  = gt_end

        # Line-level alignment
        aligned = align_lines_to_gt_segment(detections, gt_segment, max_cer=args.max_cer)
        kept = 0

        if not args.dry_run:
            for line_idx, (ocr_text_line, gt_line, line_cer, box) in enumerate(aligned):
                crop = extract_crop(img_bgr, box)
                if crop is None or crop.size == 0:
                    continue
                crop_name = f"{page_name}_l{line_idx:03d}.jpg"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                labels.append(f"{crop_path.absolute()}\t{gt_line}")
                kept += 1

        print(f"  — align={page_score:.2f}  det={len(detections)}  kept={kept if not args.dry_run else len(aligned)}/{len(detections)}")
        page_stats.append((page_name, len(detections), len(aligned), page_score, kept))

    # Save label file
    if not args.dry_run and labels:
        label_file = out_dir / 'train.txt'
        label_file.write_text('\n'.join(labels), encoding='utf-8')
        print(f"\n  Saved {len(labels)} line labels → {label_file}")

    # Summary
    total_det    = sum(s[1] for s in page_stats)
    total_kept   = sum(s[4] for s in page_stats)
    good_pages   = sum(1 for s in page_stats if s[3] >= args.min_score)
    avg_score    = sum(s[3] for s in page_stats) / len(page_stats) if page_stats else 0

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Pages processed : {len(page_stats)}")
    print(f"  Pages aligned   : {good_pages} / {len(page_stats)}  (score >= {args.min_score})")
    print(f"  Avg align score : {avg_score:.3f}")
    print(f"  Lines detected  : {total_det}")
    if not args.dry_run:
        print(f"  Lines kept      : {total_kept}  ({100*total_kept/max(1,total_det):.1f}% of detected)")
        print(f"  Output          : {out_dir}/")
        print(f"  Label file      : {out_dir}/train.txt")
    else:
        print(f"  Lines alignable : {sum(s[2] for s in page_stats)}")
        print(f"  (dry-run: no files written)")
    print()

    # Warn about low-alignment pages
    bad = [(s[0], s[3]) for s in page_stats if s[3] < args.min_score]
    if bad:
        print(f"  Pages skipped due to low alignment ({len(bad)}):")
        for name, score in bad[:10]:
            print(f"    {name}: score={score:.3f}")
        if len(bad) > 10:
            print(f"    ... and {len(bad)-10} more")

def main():
    ap = argparse.ArgumentParser(description='Align book photos to ground truth text')
    ap.add_argument('--book-dir', required=True, help='Folder of page photos')
    ap.add_argument('--gt',       required=True, help='Ground truth .txt file')
    ap.add_argument('--out',      required=True, help='Output directory')
    ap.add_argument('--rec',      default='output/mk_rec_v5_cyrillic_infer',
                    help='Recognition model dir (default: V5)')
    ap.add_argument('--min-score', type=float, default=0.5,
                    help='Min page alignment score to keep (default: 0.5)')
    ap.add_argument('--max-cer',   type=float, default=0.25,
                    help='Max line CER to keep a crop (default: 0.25)')
    ap.add_argument('--dry-run',   action='store_true',
                    help='Preview only, do not save crops')
    args = ap.parse_args()

    if not os.path.isdir(args.book_dir):
        print(f"ERROR: --book-dir not found: {args.book_dir}"); sys.exit(1)
    if not os.path.isfile(args.gt):
        print(f"ERROR: --gt not found: {args.gt}"); sys.exit(1)

    print(f"\n  Book Aligner")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Photos : {args.book_dir}")
    print(f"  GT     : {args.gt}")
    print(f"  Output : {args.out}")
    print(f"  Model  : {args.rec}")
    print(f"  Params : min_score={args.min_score}  max_cer={args.max_cer}")
    print(f"  ─────────────────────────────────────────────\n")

    run(args)

if __name__ == '__main__':
    main()
