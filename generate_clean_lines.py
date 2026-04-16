#!/usr/bin/env python3
"""
MK-OCR Clean Line Generator
============================
Generates clean individual text line images for recognition training.
Unlike page-crop approaches, lines are rendered directly → no warping artifacts.

Philosophy:
  - Clean simple backgrounds (white/cream/aged paper)
  - MILD augmentation only: slight blur, noise, brightness — NO rotation, NO heavy JPEG
  - Natural sentence/phrase lengths from real Macedonian books
  - All validated Cyrillic fonts, uniform distribution
  - Multiprocessing for speed

Usage:
    python3 generate_clean_lines.py \
        --books-dir ~/macedonian-llm/books_clean/books_txt \
        --font-dirs ~/macedonian-ocr/fonts /usr/share/fonts \
        --output-dir ~/macedonian-ocr/train_data/clean_lines_v1 \
        --num 300000 \
        --workers 0

Output:
    train_data/clean_lines_v1/
        images/        ← JPEG images
        train.txt      ← absolute-path labels for PaddleOCR
        val.txt
"""

import argparse
import glob
import json
import math
import os
import random
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance


# ── Constants ──────────────────────────────────────────────────────────────────

MK_TEST = "ЃѓЅѕЈјЉљЊњЌќЏџЀѐЍѝ\u201e\u201c"
# Macedonian Cyrillic + special chars allowed in labels
RE_MK_VALID = re.compile(r'^[\u0400-\u04FF\u0500-\u052F\u201c\u201e\u2014\u2013 ,.!?:;()\-–—\'\"0-9]+$')
# Filter corpus lines to those with Macedonian-specific chars (at least one)
RE_MK_REQUIRED = re.compile(r'[а-яА-ЯѓЃќЌљЉњЊџЏѕЅјЈЀѐЍѝ]')

# Reuse these across workers (set by _worker_init)
_FONTS = None
_LINES = None


# ── Font validation ─────────────────────────────────────────────────────────────

def validate_font(path, size=32):
    try:
        font = ImageFont.truetype(path, size)
        for ch in MK_TEST:
            img = Image.new("L", (size * 2, size * 2), 255)
            ImageDraw.Draw(img).text((4, 4), ch, font=font, fill=0)
            if np.array(img).min() > 240:
                return False
        return True
    except Exception:
        return False


def load_fonts(dirs, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = [p for p in json.load(f) if os.path.exists(p)]
        if cached:
            print(f"  Loaded {len(cached)} fonts from cache")
            return cached

    candidates = []
    for d in dirs:
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            continue
        for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
            candidates.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))
    candidates = sorted(set(candidates))

    print(f"  Validating {len(candidates)} fonts for Macedonian Cyrillic...")
    valid = [f for f in candidates if validate_font(f)]
    print(f"  Accepted: {len(valid)}/{len(candidates)}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(valid, f, indent=2)
    return valid


# ── Corpus loading ──────────────────────────────────────────────────────────────

def load_corpus(books_dir, min_len=12, max_len=85):
    """Load clean text lines from all books. Returns deduplicated list."""
    books_dir = os.path.expanduser(books_dir)
    seen = set()
    lines = []

    book_files = sorted(glob.glob(os.path.join(books_dir, "**", "*.txt"), recursive=True))
    if not book_files:
        # Try flat directory
        book_files = sorted(glob.glob(os.path.join(books_dir, "*.txt")))

    print(f"  Found {len(book_files)} book files")

    for fp in book_files:
        try:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    line = raw.strip()
                    # Length filter
                    if not (min_len <= len(line) <= max_len):
                        continue
                    # Must contain Macedonian characters
                    if not RE_MK_REQUIRED.search(line):
                        continue
                    # Skip lines that are mostly numbers/punctuation
                    alpha = sum(c.isalpha() for c in line)
                    if alpha < len(line) * 0.6:
                        continue
                    # Dedup
                    key = line.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append(line)
        except Exception:
            continue

    print(f"  Loaded {len(lines):,} unique lines")
    return lines


# ── Rendering ──────────────────────────────────────────────────────────────────

def _random_bg(w, h, rng):
    style = rng.choice(["white", "white", "cream", "aged"])  # weight toward white
    if style == "white":
        v = rng.randint(242, 255)
        return Image.new("RGB", (w, h), (v, v, v))
    elif style == "cream":
        r = rng.randint(232, 250)
        g = rng.randint(225, 242)
        b = rng.randint(200, 225)
        return Image.new("RGB", (w, h), (r, g, b))
    else:  # aged
        base = np.array([rng.randint(210, 238), rng.randint(200, 228), rng.randint(170, 208)])
        arr = np.full((h, w, 3), base, dtype=np.uint8)
        noise = np.random.normal(0, rng.uniform(1, 4), (h, w, 3)).astype(np.int16)
        return Image.fromarray(np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))


def _random_ink(rng):
    base = rng.randint(8, 45)
    tint = rng.choice(["none", "none", "warm", "cool"])  # mostly neutral
    if tint == "warm":
        return (base + rng.randint(3, 12), base, base)
    elif tint == "cool":
        return (base, base, base + rng.randint(3, 12))
    return (base, base, base)


def _mild_augment(img, rng):
    """Clean augmentation — no rotation, very mild effects only."""
    ops = []

    # Slight blur (50% chance, very mild)
    if rng.random() < 0.5:
        sigma = rng.uniform(0.1, 0.55)
        ops.append(lambda i, s=sigma: i.filter(ImageFilter.GaussianBlur(s)))

    # Light noise (40% chance)
    if rng.random() < 0.4:
        std = rng.uniform(1.0, 5.0)
        def add_noise(i, s=std):
            arr = np.array(i, dtype=np.float32)
            arr += np.random.normal(0, s, arr.shape)
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        ops.append(add_noise)

    # Brightness (50% chance, very mild)
    if rng.random() < 0.5:
        f = rng.uniform(0.90, 1.10)
        ops.append(lambda i, ff=f: ImageEnhance.Brightness(i).enhance(ff))

    # Contrast (30% chance)
    if rng.random() < 0.3:
        f = rng.uniform(0.90, 1.10)
        ops.append(lambda i, ff=f: ImageEnhance.Contrast(i).enhance(ff))

    # Very light JPEG (30% chance, high quality only)
    if rng.random() < 0.3:
        q = rng.randint(88, 97)
        def jpeg(i, qq=q):
            buf = BytesIO()
            i.save(buf, "JPEG", quality=qq)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        ops.append(jpeg)

    for fn in ops:
        try:
            img = fn(img)
        except Exception:
            pass
    return img


def render_line(text, font_path, font_size, rng):
    """Render a single text line. Returns PIL Image or None."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return None

    dummy = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    if tw < 8 or th < 6:
        return None
    # Skip if too wide for the recognition model (>1600px at 48px height)
    if tw > 1400:
        return None

    pad_x = rng.randint(4, 18)
    pad_y = rng.randint(3, 10)
    w, h = tw + 2 * pad_x, th + 2 * pad_y

    ink = _random_ink(rng)
    img = _random_bg(w, h, rng)
    ImageDraw.Draw(img).text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=ink)
    img = _mild_augment(img, rng)
    return img


# ── Worker (multiprocessing) ────────────────────────────────────────────────────

def _worker_init(fonts, lines, seed_base):
    global _FONTS, _LINES
    _FONTS = fonts
    _LINES = lines
    # Each worker gets a different seed
    pid = os.getpid()
    random.seed(seed_base + pid)
    np.random.seed((seed_base + pid) % (2**31))


def _worker_generate(task):
    """Generate one batch of images. Returns list of (abs_path, label)."""
    global _FONTS, _LINES

    idx_start, count, img_dir, seed = task
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed % (2**31))

    class _NpRng:
        def random(self): return np_rng.random()
        def uniform(self, a, b): return np_rng.uniform(a, b)
        def randint(self, a, b): return np_rng.randint(a, b + 1)
        def choice(self, seq): return seq[np_rng.randint(0, len(seq))]

    local_rng = _NpRng()
    results = []
    idx = idx_start
    attempts = 0
    max_attempts = count * 5

    while len(results) < count and attempts < max_attempts:
        attempts += 1

        text = rng.choice(_LINES)

        # Trim long lines to a natural phrase
        if len(text) > 75:
            words = text.split()
            if len(words) > 4:
                start = rng.randint(0, max(0, len(words) - 4))
                length = rng.randint(3, min(14, len(words) - start))
                text = " ".join(words[start:start + length])

        if len(text) < 10:
            continue

        font_path = rng.choice(_FONTS)
        font_size = rng.randint(22, 40)

        img = render_line(text, font_path, font_size, local_rng)
        if img is None:
            continue

        fname = f"cl_{idx:07d}.jpg"
        fpath = os.path.join(img_dir, fname)
        try:
            img.save(fpath, "JPEG", quality=rng.randint(90, 97))
        except Exception:
            continue

        results.append((fpath, text))
        idx += 1

    return results


# ── Combine labels with existing data ──────────────────────────────────────────

def build_combined_labels(new_train, new_val, existing_label_files, out_dir):
    """Merge new labels with existing clean training files."""
    def read_labels(path):
        lines = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "\t" in line:
                    img_path, label = line.split("\t", 1)
                    # Ensure absolute paths
                    if not os.path.isabs(img_path):
                        base = os.path.dirname(path)
                        img_path = os.path.abspath(os.path.join(base, img_path))
                    lines.append(f"{img_path}\t{label}")
        return lines

    combined_train = list(new_train)
    combined_val = list(new_val)

    for lf in existing_label_files:
        if not os.path.exists(lf):
            print(f"  [SKIP] {lf} not found")
            continue
        labels = read_labels(lf)
        # 95/5 split for existing data too
        n_val = max(1, len(labels) // 20)
        combined_val.extend(labels[:n_val])
        combined_train.extend(labels[n_val:])
        print(f"  Merged {len(labels):,} from {os.path.basename(lf)}")

    # Shuffle
    random.shuffle(combined_train)
    random.shuffle(combined_val)

    train_path = os.path.join(out_dir, "combined_train.txt")
    val_path = os.path.join(out_dir, "combined_val.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_train))
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_val))

    print(f"\n  Combined train: {len(combined_train):,}")
    print(f"  Combined val:   {len(combined_val):,}")
    return train_path, val_path


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MK-OCR Clean Line Generator")
    parser.add_argument("--books-dir", default="~/macedonian-llm/books_clean/books_txt",
                        help="Directory with .txt corpus files")
    parser.add_argument("--font-dirs", nargs="+",
                        default=["~/macedonian-ocr/fonts", "/usr/share/fonts"],
                        help="Directories to search for fonts")
    parser.add_argument("--output-dir", default="~/macedonian-ocr/train_data/clean_lines_v1",
                        help="Output directory")
    parser.add_argument("--num", type=int, default=300000,
                        help="Number of new images to generate")
    parser.add_argument("--workers", type=int, default=0,
                        help="Worker processes (0 = cpu_count - 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-existing", nargs="*",
                        default=[
                            "~/macedonian-ocr/train_data/combined_train_v6.txt",
                            "~/macedonian-ocr/train_data/v7/targeted_v2/train.txt",
                        ],
                        help="Existing label files to merge into combined output")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(os.path.expanduser(args.output_dir))
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)

    print("=" * 65)
    print("  MK-OCR Clean Line Generator")
    print("=" * 65)
    print(f"  Target images : {args.num:,}")
    print(f"  Workers       : {n_workers}")
    print(f"  Output        : {out}")
    print("=" * 65)

    # ── Step 1: Fonts ──
    print("\n[1/4] Loading and validating fonts...")
    font_dirs = [os.path.expanduser(d) for d in args.font_dirs]
    fonts = load_fonts(font_dirs, str(out / "font_cache.json"))
    if len(fonts) < 2:
        print("ERROR: Need at least 2 validated fonts. Check font dirs.")
        sys.exit(1)
    print(f"  Using {len(fonts)} fonts")

    # ── Step 2: Corpus ──
    print("\n[2/4] Loading corpus...")
    books_dir = os.path.expanduser(args.books_dir)
    # Try the specified dir, then fall back to the other known location
    if not os.path.isdir(books_dir):
        fallback = os.path.expanduser("~/macedonian-llm/books")
        if os.path.isdir(fallback):
            print(f"  {books_dir} not found, using {fallback}")
            books_dir = fallback
        else:
            print(f"ERROR: books dir not found: {books_dir}")
            sys.exit(1)

    lines = load_corpus(books_dir)
    if len(lines) < 1000:
        print(f"ERROR: Only {len(lines)} lines — corpus too small")
        sys.exit(1)

    # ── Step 3: Generate ──
    print(f"\n[3/4] Generating {args.num:,} images with {n_workers} workers...")

    batch_size = max(500, args.num // (n_workers * 10))
    tasks = []
    idx = 0
    while idx < args.num:
        cnt = min(batch_size, args.num - idx)
        task_seed = args.seed + idx
        tasks.append((idx, cnt, str(img_dir), task_seed))
        idx += cnt

    all_results = []

    if n_workers == 1:
        # Single-process fallback
        _worker_init(fonts, lines, args.seed)
        for i, task in enumerate(tasks):
            res = _worker_generate(task)
            all_results.extend(res)
            done = len(all_results)
            pct = done / args.num * 100
            print(f"\r  {done:>7,} / {args.num:,}  ({pct:.1f}%)", end="", flush=True)
    else:
        with Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(fonts, lines, args.seed),
        ) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker_generate, tasks)):
                all_results.extend(res)
                done = len(all_results)
                pct = done / args.num * 100
                print(f"\r  {done:>7,} / {args.num:,}  ({pct:.1f}%)", end="", flush=True)

    print()  # newline after progress

    # Split new data 95/5
    random.shuffle(all_results)
    n_val = max(1, min(500, int(len(all_results) * 0.05)))
    new_val_results = all_results[:n_val]
    new_train_results = all_results[n_val:]

    new_train = [f"{p}\t{l}" for p, l in new_train_results]
    new_val = [f"{p}\t{l}" for p, l in new_val_results]

    # Write standalone train/val for this dataset
    with open(out / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(new_train))
    with open(out / "val.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(new_val))

    print(f"  Generated: {len(all_results):,} images")
    print(f"  Train: {len(new_train):,} | Val: {len(new_val):,}")

    # ── Step 4: Merge with existing clean data ──
    print("\n[4/4] Merging with existing clean datasets...")
    existing = [os.path.expanduser(p) for p in (args.merge_existing or [])]
    combined_dir = str(out)
    train_path, val_path = build_combined_labels(new_train, new_val, existing, combined_dir)

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)
    print(f"  New images      : {out}/images/")
    print(f"  New train/val   : {out}/train.txt, val.txt")
    print(f"  Combined train  : {train_path}")
    print(f"  Combined val    : {val_path}")
    print()
    print("  Next steps:")
    print(f"    1. Update mk_rec_v7.yml to point to combined labels")
    print(f"    2. Run: bash train_v7.sh")
    print("=" * 65)


if __name__ == "__main__":
    main()
