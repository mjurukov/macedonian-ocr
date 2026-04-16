#!/bin/bash
# train_v7.sh — Bulletproof MK-OCR Recognition Training V7
#
# Fine-tunes from best V5 checkpoint for 150 epochs.
# ~800k samples: page crops + targeted (quote-heavy) + existing V6 data.
#
# Usage:
#   cd /home/kali/macedonian-ocr
#   source venv/bin/activate
#   bash train_v7.sh
#
# To run detached (recommended for basement PC):
#   nohup bash train_v7.sh > ~/train_v7.log 2>&1 &
#   echo "PID: $!"
#   tail -f ~/train_v7.log

set -euo pipefail

# ── Paths ──
PADDLE_DIR=/home/kali/macedonian-ocr/PaddleOCR
REC_CONFIG=/home/kali/macedonian-ocr/configs/mk_rec_v7.yml
V5_CHECKPOINT=/home/kali/macedonian-ocr/output/mk_rec_v5_cyrillic/best_accuracy
SAVE_DIR=/home/kali/macedonian-ocr/output/mk_rec_v7
TRAIN_LABELS=/home/kali/macedonian-ocr/train_data/clean_lines_v1/combined_train.txt
VAL_LABELS=/home/kali/macedonian-ocr/train_data/clean_lines_v1/combined_val.txt

# ── Pre-flight checks ──
echo "============================================================"
echo "  MK-OCR Recognition V7 — Pre-flight checks"
echo "============================================================"

fail=0

check() {
    if [ -e "$1" ]; then
        echo "  OK  $1"
    else
        echo "  MISSING: $1"
        fail=1
    fi
}

check "$PADDLE_DIR/tools/train.py"
check "$REC_CONFIG"
check "${V5_CHECKPOINT}.pdparams"
check "$TRAIN_LABELS"
check "$VAL_LABELS"

TRAIN_COUNT=$(wc -l < "$TRAIN_LABELS")
VAL_COUNT=$(wc -l < "$VAL_LABELS")
echo ""
echo "  Train samples : $TRAIN_COUNT"
echo "  Val samples   : $VAL_COUNT"

if [ "$TRAIN_COUNT" -lt 300000 ]; then
    echo "  ERROR: Too few training samples ($TRAIN_COUNT). Did generate_clean_lines.py complete?"
    fail=1
fi

# Spot-check 5 random image paths actually exist
echo ""
echo "  Spot-checking image paths..."
python3 - << 'PYEOF'
import os, random, sys
lines = open("/home/kali/macedonian-ocr/train_data/clean_lines_v1/combined_train.txt").readlines()
samples = random.sample(lines, min(20, len(lines)))
missing = []
for s in samples:
    path = s.split("\t")[0].strip()
    if not os.path.exists(path):
        missing.append(path)
if missing:
    print(f"  MISSING images:")
    for p in missing:
        print(f"    {p}")
    sys.exit(1)
else:
    print(f"  All spot-checked images found")
PYEOF

if [ "$fail" -eq 1 ]; then
    echo ""
    echo "  Pre-flight FAILED. Fix the above issues and retry."
    exit 1
fi

echo ""
echo "  All checks passed."
echo ""
echo "  Starting from : $V5_CHECKPOINT (acc ~0.49)"
echo "  Saving to     : $SAVE_DIR"
echo "  Epochs        : 50 (batch_size=96 for ~2.5 day runtime)"
echo "  LR            : 0.0002 (lower than V5 for fine-tuning)"
echo "  Train samples : $TRAIN_COUNT"
echo "============================================================"
echo ""

mkdir -p "$SAVE_DIR"

cd "$PADDLE_DIR"

python3 tools/train.py \
    -c "$REC_CONFIG" \
    -o \
    Global.pretrained_model="$V5_CHECKPOINT" \
    Global.save_model_dir="$SAVE_DIR"

echo ""
echo "============================================================"
echo "  Training complete. Best model at:"
echo "  $SAVE_DIR/best_accuracy.pdparams"
echo ""
echo "  Export for inference:"
echo "    cd $PADDLE_DIR"
echo "    source ../venv/bin/activate"
echo "    python3 tools/export_model.py \\"
echo "        -c ppocr/configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \\"
echo "        -o Global.pretrained_model=$SAVE_DIR/best_accuracy \\"
echo "           Global.character_dict_path=/home/kali/macedonian-ocr/configs/mk_dict.txt \\"
echo "           Global.save_inference_dir=/home/kali/macedonian-ocr/output/mk_rec_v7_infer"
echo "============================================================"
