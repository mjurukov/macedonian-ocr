# Macedonian OCR

Fine-tuned PaddleOCR recognition model for Macedonian Cyrillic text in scanned books.

## Results

| Engine | CER | WER | MK-Acc |
|--------|-----|-----|--------|
| **V7 (this model)** | **0.45%** | **~2%** | **99.6%** |
| Tesseract (`mkd`) | 2.15% | 6.3% | 96.3% |
| V5 (previous) | 2.21% | 3.9% | 99.2% |

Tested on 10 labeled pages from Macedonian books. MK-Acc measures accuracy on Macedonian-specific letters (ѓ, ќ, љ, њ, џ, ѕ, ј).

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/mjurukov/macedonian-ocr
cd macedonian-ocr
bash 01_setup.sh

# 2. Download model weights
# (releases page → mk_rec_v7_infer.zip → extract to output/)

# 3. Run OCR on an image
source venv/bin/activate
python3 visualize_detections.py your_page.jpg --out viz/
```

## Benchmark

```bash
source venv/bin/activate
python3 compare_prep.py
```

## How it works

- **Base model:** PaddleOCR PP-OCRv5 SVTR_LCNet with MultiHead (CTC + NRTR)
- **Dictionary:** 170 characters — full Macedonian Cyrillic, Latin, digits, punctuation
- **Training data:** 583k synthetic line images rendered from real Macedonian book text
- **Detection tuning:** `text_det_thresh=0.25`, `text_det_box_thresh=0.30`, `text_det_unclip_ratio=2.2` (found via `sweep_det.py` — 4.7× better than PaddleOCR defaults on book pages)

## Training your own model

### 1. Generate synthetic training data
```bash
python3 generate_clean_lines.py \
    --books-dir /path/to/macedonian/books_txt \
    --font-dirs fonts/ /usr/share/fonts \
    --output-dir train_data/clean_lines_v1 \
    --num 300000
```

### 2. Align real book pages (optional, for V8+)
```bash
python3 align_all_books.py \
    --books-root /path/to/books \
    --out aligned/ \
    --val-books book_003,book_007 \
    --rec output/mk_rec_v7_infer
```

### 3. Train
```bash
# Edit configs/mk_rec_v7.yml — update label_file_list paths
bash train_v7.sh
```

### 4. Export
```bash
cd PaddleOCR
python3 tools/export_model.py \
    -c /path/to/macedonian-ocr/configs/mk_rec_v7.yml \
    -o Global.pretrained_model=../output/mk_rec_v7/best_accuracy \
       Global.character_dict_path=../configs/mk_dict.txt \
       Global.save_inference_dir=../output/mk_rec_v7_infer

sed -i 's/model_name: cyrillic_PP-OCRv5_mobile_rec/model_name: PP-OCRv5_server_rec/' \
    ../output/mk_rec_v7_infer/inference.yml
```

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_clean_lines.py` | Render synthetic training line images from book text |
| `align_book.py` | Align one book's photos to its GT text, output labeled crops |
| `align_all_books.py` | Batch version for 20-30 books, train/val split by book |
| `compare_prep.py` | Benchmark V7 vs V5 vs Tesseract on test pages |
| `sweep_det.py` | Grid search over PaddleOCR detection thresholds |
| `visualize_detections.py` | Draw bounding boxes + recognized text side by side |
| `annotate_server.py` | Local web tool for annotating bounding boxes |
| `mk_postprocess.py` | Spell correction, dehyphenation, fragment merging |

## Requirements

- WSL2 or Linux, NVIDIA GPU (tested on RTX 2080 Ti)
- CUDA 11.8+, Python 3.10+
- See `01_setup.sh` for full dependency install
