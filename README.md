# Macedonian OCR

Fine-tuned PaddleOCR recognition model for Macedonian Cyrillic text in scanned books.

## Results

Tested on 10 labeled pages from Macedonian books. MK-Acc measures accuracy on the 14 Macedonian-specific Cyrillic letters (ѓ, ќ, љ, њ, џ, ѕ, ј and uppercase).

| Engine | CER | WER | MK-Acc |
|--------|-----|-----|--------|
| **V8 (this model)** | **0.26%** | — | — |
| V7 | 0.31% | ~2% | 99.6% |
| Tesseract (`mkd`) | 1.34% | — | — |
| DeepSeek-OCR | TODO | — | — |
| Qwen2.5-VL 72B | TODO | — | — |
| Gemma4 31B | TODO | — | — |

The TODO rows will be filled in after running `benchmark_h100.py` on H100 hardware.

## VLM Benchmark (Docker — no sudo required)

Compares Qwen2.5-VL 72B, Gemma4 31B, Phi-4 Vision and DeepSeek-OCR against the
PaddleOCR baseline on 10 labeled Macedonian book pages.
Requires Docker with NVIDIA Container Toolkit (`--gpus all`).

```bash
git clone https://github.com/mjurukov/macedonian-ocr
cd macedonian-ocr

# Build once (~5 min, ~600 MB image)
docker build -t macedonian-ocr-benchmark .

# Run — pulls Ollama models (~73 GB total) then runs benchmark
mkdir -p results
docker run --gpus all --rm \
    -v ollama_cache:/root/.ollama \
    -v "$(pwd)":/output \
    macedonian-ocr-benchmark \
    --pull --save /output/results_h100.json
```

Resume after an interruption:
```bash
docker run --gpus all --rm \
    -v ollama_cache:/root/.ollama \
    -v "$(pwd)":/output \
    macedonian-ocr-benchmark \
    --resume /output/results_h100.json --save /output/results_h100.json
```

The `ollama_cache` named volume persists downloaded models between runs.
Results are written to `results_h100.json` in the current directory.

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/mjurukov/macedonian-ocr
cd macedonian-ocr
bash 01_setup.sh
source venv/bin/activate

# 2. Download model weights from HuggingFace
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mjurukov/macedonian-ocr-v7', local_dir='output/mk_rec_v7_infer')
"

# 3. Run OCR
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    text_recognition_model_dir='output/mk_rec_v7_infer',
    text_det_thresh=0.25,
    text_det_box_thresh=0.30,
    text_det_unclip_ratio=2.2,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)
result = ocr.predict('your_page.jpg')
for page in result:
    for text, score in zip(page['rec_texts'], page['rec_scores']):
        print(f'{score:.2f}  {text}')
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
