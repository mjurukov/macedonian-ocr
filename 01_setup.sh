#!/bin/bash
# ============================================================
# Macedonian OCR - PaddleOCR Setup for WSL + RTX 2080 Ti
# ============================================================
# Run this on your WSL environment
# Tested with: Ubuntu 22.04/24.04 on WSL2, CUDA 11.8/12.x
#
# Prerequisites:
#   - WSL2 with NVIDIA GPU passthrough working
#   - nvidia-smi should show your RTX 2080 Ti
#   - Python 3.8-3.11 (3.10 recommended)
# ============================================================

set -e

echo "============================================"
echo "  Macedonian OCR - PaddleOCR Setup"
echo "============================================"

# --- Check GPU ---
echo ""
echo "[1/7] Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Make sure NVIDIA drivers are installed for WSL."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# --- Create project directory ---
echo "[2/7] Creating project structure..."
PROJECT_DIR="$HOME/macedonian-ocr"
mkdir -p "$PROJECT_DIR"/{data/{train_data,test_data,synthetic},models,configs,tools,output}
cd "$PROJECT_DIR"

# --- Python venv ---
echo "[3/7] Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# --- Install PaddlePaddle with CUDA ---
echo "[4/7] Installing PaddlePaddle (CUDA)..."
# For CUDA 11.8 (most common on WSL with 2080 Ti):
pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# If you have CUDA 12.x, use this instead:
# pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

# Verify PaddlePaddle GPU
python3 -c "
import paddle
print('PaddlePaddle version:', paddle.__version__)
print('GPU available:', paddle.device.is_compiled_with_cuda())
if paddle.device.is_compiled_with_cuda():
    print('GPU count:', paddle.device.cuda.device_count())
    print('GPU name:', paddle.device.cuda.get_device_name(0))
"

# --- Install PaddleOCR ---
echo "[5/7] Installing PaddleOCR..."
pip install paddleocr
pip install ppocrlabel  # Annotation tool

# Clone PaddleOCR repo for training scripts
if [ ! -d "PaddleOCR" ]; then
    git clone https://github.com/PaddlePaddle/PaddleOCR.git
    cd PaddleOCR
    pip install -r requirements.txt
    cd ..
fi

# --- Download pretrained models ---
echo "[6/7] Downloading pretrained PP-OCRv4 models..."
mkdir -p models/pretrained

# Detection model (language-agnostic, no need to retrain)
cd models/pretrained
if [ ! -d "en_PP-OCRv4_det_train" ]; then
    wget -q https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_train.tar
    tar xf en_PP-OCRv4_det_train.tar
    rm en_PP-OCRv4_det_train.tar
    echo "  -> Detection model downloaded"
fi

# Recognition model (this is what we'll fine-tune for Macedonian)
if [ ! -d "en_PP-OCRv4_rec_train" ]; then
    wget -q https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar
    tar xf en_PP-OCRv4_rec_train.tar
    rm en_PP-OCRv4_rec_train.tar
    echo "  -> Recognition model downloaded"
fi

# Also grab the Cyrillic/multilingual model as an alternative base
if [ ! -d "rec_svtr_tiny_none_ctc_en_train" ]; then
    wget -q https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_train.tar
    tar xf cyrillic_PP-OCRv3_rec_train.tar
    rm cyrillic_PP-OCRv3_rec_train.tar
    echo "  -> Cyrillic recognition model downloaded"
fi
cd "$PROJECT_DIR"

# --- Copy dictionary ---
echo "[7/7] Setting up Macedonian dictionary..."
# Copy the mk_dict.txt you already have, or the one from this repo
if [ -f "mk_dict.txt" ]; then
    cp mk_dict.txt configs/mk_dict.txt
else
    echo "WARNING: mk_dict.txt not found in project root."
    echo "Copy the provided mk_dict.txt to $PROJECT_DIR/configs/mk_dict.txt"
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Project directory: $PROJECT_DIR"
echo ""
echo "Next steps:"
echo "  1. Activate venv:  source $PROJECT_DIR/venv/bin/activate"
echo "  2. Test baseline:  python3 test_baseline.py"
echo "  3. Generate synthetic data:  python3 generate_synthetic.py"
echo "  4. Train:  see train_rec.sh"
echo ""
