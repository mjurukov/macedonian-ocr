# Macedonian OCR

Fine-tuned PaddleOCR PP-OCRv6 detection + PP-OCRv5 recognition for Macedonian Cyrillic text in scanned books.

## Models

| Model | HuggingFace | Params | CER |
|-------|------------|--------|-----|
| V8 (latest) | [mjurukov/macedonian-ocr-v8](https://huggingface.co/mjurukov/macedonian-ocr-v8) | 14M rec + 34.5M det | 0.26% |
| V7 (previous) | [mjurukov/macedonian-ocr-v7](https://huggingface.co/mjurukov/macedonian-ocr-v7) | 14M rec + 11M det | 1.02% |

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
snapshot_download('mjurukov/macedonian-ocr-v8', local_dir='output/mk_rec_v8_infer')
"

# 3. Run OCR
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    text_detection_model_name='PP-OCRv6_medium_det',
    text_recognition_model_name='PP-OCRv5_server_rec',
    text_recognition_model_dir='output/mk_rec_v8_infer',
    text_det_thresh=0.15,
    text_det_box_thresh=0.28,
    text_det_unclip_ratio=3.0,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
result = ocr.predict('your_page.jpg')
for page in result:
    for text, score in zip(page['rec_texts'], page['rec_scores']):
        print(f'{score:.2f}  {text}')
```

## How it works

- **Detection:** PP-OCRv6_medium_det — 36% faster than v5 with equivalent accuracy
- **Recognition:** Fine-tuned PP-OCRv5 SVTR_LCNet with MultiHead (CTC + NRTR)
- **Dictionary:** 170 characters — full Macedonian Cyrillic, Latin, digits, punctuation
- **Preprocessing:** grayscale → deskew → illumination flattening → unsharp masking (0.4 gain)
- **Training data:** 583k synthetic + 172k targeted + 42k IED-boost + 1,674 real annotated lines (~800k total)
- **Detection tuning:** `text_det_thresh=0.15`, `text_det_box_thresh=0.28`, `text_det_unclip_ratio=3.0` (found via `sweep_det.py` — substantially better than PaddleOCR defaults on book pages)
- **Post-processing:** `mk_postprocess.py` — dehyphenation, fragment merging, and spell correction with a 256k-word frequency dictionary, optionally arbitrated by a KenLM 3-gram language model (see below)

## Language model post-processing

Word corrections are picked by contextual KenLM score when `lm/mk_3gram.binary`
exists ("ge reche" → "go reche" because the trigram is more probable), falling
back to unigram frequency otherwise. To rebuild the LM from a Macedonian text
corpus:

```bash
pip install kenlm   # Python scoring module
# lmplz/build_binary come from a KenLM source build (needs cmake + boost):
#   git clone https://github.com/kpu/kenlm && cd kenlm && mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release && make -j lmplz build_binary

mkdir -p lm
python3 - <<'EOF'
import glob, re
WORD = re.compile(r'[а-яА-ЯѓЃќЌљЉњЊџЏѕЅјЈѐЀѝЍ]+')
with open('lm/corpus.txt', 'w', encoding='utf-8') as out:
    for fp in glob.glob('/path/to/books_txt/**/*.txt', recursive=True):
        for line in open(fp, encoding='utf-8', errors='ignore'):
            words = [w.lower() for w in WORD.findall(line)]
            if len(words) >= 2:
                out.write(' '.join(words) + '\n')
EOF
lmplz -o 3 --prune 0 1 1 -S 30% -T /tmp < lm/corpus.txt > lm/mk_3gram.arpa
build_binary lm/mk_3gram.arpa lm/mk_3gram.binary
```

The OCR server (`ocr_app/main.py`) and `benchmark_final.py` pick up
`lm/mk_3gram.binary` automatically.

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
# Edit configs/mk_rec_v8.yml — update label_file_list paths
bash train_v8.sh
```

### 4. Export
```bash
cd PaddleOCR
python3 tools/export_model.py \
    -c /path/to/macedonian-ocr/configs/mk_rec_v8.yml \
    -o Global.pretrained_model=../output/mk_rec_v8/best_accuracy \
       Global.character_dict_path=../configs/mk_dict.txt \
       Global.save_inference_dir=../output/mk_rec_v8_infer

sed -i 's/model_name: cyrillic_PP-OCRv5_mobile_rec/model_name: PP-OCRv5_server_rec/' \
    ../output/mk_rec_v8_infer/inference.yml
```

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_clean_lines.py` | Render synthetic training line images from book text |
| `generate_targeted_v8.py` | Generate 172k targeted samples (confusion pairs, rare chars) |
| `align_book.py` | Align one book's photos to its GT text, output labeled crops |
| `align_all_books.py` | Batch version for 20-30 books, train/val split by book |
| `mk_postprocess.py` | KenLM spell correction, dehyphenation, fragment merging |
| `analyze_errors.py` | Character confusion matrix, word-level diffs |
| `ocr_dump.py` | OCR all pages in a directory, output editable TSV for correction |
| `ocr_convert.py` | Convert corrected TSV to PaddleOCR training format |
| `ocr_app/main.py` | FastAPI OCR server with `/ocr` and `/ocr/full` endpoints |
| `sweep_det.py` | Grid search over PaddleOCR detection thresholds |
| `preprocess_utils.py` | Shared preprocessing: deskew, illumination flatten, unsharp |

## Requirements

- Linux, NVIDIA GPU (tested on RTX 2080 Ti, H100)
- CUDA 12.x, Python 3.12+
- PaddlePaddle 3.3.1+, PaddleOCR 3.7.0+
- See `01_setup.sh` for full dependency install
