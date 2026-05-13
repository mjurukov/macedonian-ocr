#!/bin/bash
# =============================================================================
#  Macedonian OCR — Faculty H100 Setup
# =============================================================================
#  Prepares a fresh Linux machine to run the VLM benchmark.
#  Tested on Ubuntu 22.04 / 24.04 with H100 (CUDA 12).
#
#  What this does:
#    1. Checks prerequisites (Python 3.10+, CUDA)
#    2. Installs Ollama
#    3. Creates a minimal Python venv and downloads the V7 model from HuggingFace
#    4. Prints the commands to pull Ollama models and run the benchmark
#
#  Usage:
#    git clone https://github.com/mjurukov/macedonian-ocr
#    cd macedonian-ocr
#    bash setup_faculty.sh
#    bash setup_faculty.sh --pull-models   # also pull all Ollama models (~73 GB)
# =============================================================================

set -e

PULL_MODELS=0
for arg in "$@"; do
    [[ "$arg" == "--pull-models" ]] && PULL_MODELS=1
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv_faculty"

echo ""
echo "============================================================"
echo "  Macedonian OCR — Faculty Setup"
echo "============================================================"
echo ""

# ── 1. Python ─────────────────────────────────────────────────────────────────

echo "[1/4] Checking Python..."
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c 'import sys; print(sys.version_info[:2])')
        if "$cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
            PYTHON="$cmd"
            echo "  Found $cmd ($VER) ✓"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python 3.10+ not found."
    echo "  Install it with:  sudo apt install python3.11 python3.11-venv"
    exit 1
fi

# ── 2. GPU ────────────────────────────────────────────────────────────────────

echo ""
echo "[2/4] Checking CUDA..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "  WARNING: nvidia-smi not found — Ollama will run on CPU (very slow)."
else
    nvidia-smi --query-gpu=name,memory.total,driver_version \
               --format=csv,noheader | while read line; do
        echo "  GPU: $line"
    done
fi

# ── 3. Ollama ─────────────────────────────────────────────────────────────────

echo ""
echo "[3/4] Installing Ollama..."
if command -v ollama &>/dev/null; then
    echo "  Already installed: $(ollama --version 2>&1 | head -1)"
else
    echo "  Downloading and installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  Ollama installed ✓"
fi

# Start ollama serve if not already running
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "  Starting ollama serve in background..."
    nohup ollama serve > "$PROJECT_DIR/ollama.log" 2>&1 &
    OLLAMA_PID=$!
    echo "  PID $OLLAMA_PID — logs: $PROJECT_DIR/ollama.log"
    sleep 3
    if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "  WARNING: Ollama did not start cleanly. Check ollama.log"
    else
        echo "  Ollama running ✓"
    fi
else
    echo "  Already running ✓"
fi

# ── 4. V7 model from HuggingFace ─────────────────────────────────────────────

echo ""
echo "[4/4] Downloading V7 model from HuggingFace..."

INFER_DIR="$PROJECT_DIR/output/mk_rec_v7_infer"
mkdir -p "$INFER_DIR"

# Check if already downloaded (inference.pdiparams is the large weight file)
if [[ -f "$INFER_DIR/inference.pdiparams" ]]; then
    echo "  Already present: $INFER_DIR ✓"
else
    # Set up a minimal venv just for the huggingface_hub download
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "  Creating minimal venv at $VENV_DIR ..."
        "$PYTHON" -m venv "$VENV_DIR"
    fi
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r "$PROJECT_DIR/requirements_faculty.txt"

    echo "  Downloading mjurukov/macedonian-ocr-v7 → $INFER_DIR"
    echo "  (this is ~70 MB)"
    export INFER_DIR
    "$VENV_DIR/bin/python3" - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

dest = os.environ['INFER_DIR']
snapshot_download(
    repo_id='mjurukov/macedonian-ocr-v7',
    local_dir=dest,
    ignore_patterns=['*.gitattributes', 'README.md', '.gitignore'],
)
print(f'  Downloaded to {dest} ✓')
PYEOF
fi

# ── Optional: pull Ollama models ──────────────────────────────────────────────

if [[ "$PULL_MODELS" -eq 1 ]]; then
    echo ""
    echo "[extra] Pulling Ollama models (~73 GB total) ..."
    echo "  This will take a long time on a first run."
    for model in qwen2.5vl:72b gemma4:31b phi4-vision deepseek-ocr; do
        echo "  → $model"
        ollama pull "$model"
    done
    echo "  All models pulled ✓"
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  V7 model:  $INFER_DIR"
echo ""
echo "  Next steps:"
echo ""
if [[ "$PULL_MODELS" -eq 0 ]]; then
echo "  1. Pull Ollama models (~73 GB — skip if already done):"
echo "     python3 benchmark_h100.py --pull-only"
echo ""
fi
echo "  $([ "$PULL_MODELS" -eq 1 ] && echo 1 || echo 2). Run the benchmark:"
echo "     python3 benchmark_h100.py"
echo ""
echo "  Resume if interrupted:"
echo "     python3 benchmark_h100.py --resume results_h100.json"
echo ""
echo "  Ollama logs: $PROJECT_DIR/ollama.log"
echo ""
