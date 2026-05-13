#!/bin/bash
# =============================================================================
#  Macedonian OCR — Faculty H100 Setup
# =============================================================================
#  Prepares a fresh Linux machine to run the VLM benchmark.
#  Tested on Ubuntu 22.04 / 24.04 with H100 (CUDA 12).
#
#  Full workflow:
#    git clone https://github.com/mjurukov/macedonian-ocr
#    cd macedonian-ocr
#    bash setup_faculty.sh
#    python3 benchmark_h100.py --pull    # downloads ~73 GB of models + runs benchmark
#
#  Results are saved to:  macedonian-ocr/results_h100.json
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv_faculty"

echo ""
echo "============================================================"
echo "  Macedonian OCR — Faculty Setup"
echo "  Project dir: $PROJECT_DIR"
echo "============================================================"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────

echo "[1/4] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    curl git python3 python3-venv python3-pip
echo "  Done ✓"

# ── 2. GPU ────────────────────────────────────────────────────────────────────

echo ""
echo "[2/4] Checking GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "  WARNING: nvidia-smi not found."
    echo "  Ollama will run on CPU — the benchmark will be very slow."
    echo "  Make sure NVIDIA drivers are installed before continuing."
else
    nvidia-smi --query-gpu=name,memory.total,driver_version \
               --format=csv,noheader | while read -r line; do
        echo "  GPU: $line"
    done
fi

# ── 3. Ollama ─────────────────────────────────────────────────────────────────

echo ""
echo "[3/4] Installing Ollama..."
if command -v ollama &>/dev/null; then
    echo "  Already installed: $(ollama --version 2>&1 | head -1)"
else
    curl -fsSL https://ollama.com/install.sh | sh
    # The install script sets up a systemd service and starts it automatically.
    echo "  Installed ✓"
fi

# Wait for the Ollama service to respond (systemd starts it after install)
echo "  Waiting for Ollama service..."
for i in $(seq 1 15); do
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "  Ollama running ✓"
        break
    fi
    if [ "$i" -eq 15 ]; then
        echo ""
        echo "  ERROR: Ollama is not responding after 30 seconds."
        echo "  Try starting it manually:  ollama serve"
        echo "  Then re-run:               bash setup_faculty.sh"
        exit 1
    fi
    sleep 2
done

# ── 4. Download model weights from HuggingFace ────────────────────────────────

echo ""
echo "[4/4] Downloading model from HuggingFace (mjurukov/macedonian-ocr-v7)..."

INFER_DIR="$PROJECT_DIR/output/mk_rec_v7_infer"
mkdir -p "$INFER_DIR"

if [[ -f "$INFER_DIR/inference.pdiparams" ]]; then
    echo "  Already present ✓"
else
    echo "  Creating Python venv for download..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r "$PROJECT_DIR/requirements_faculty.txt"

    echo "  Downloading (~70 MB)..."
    export INFER_DIR
    "$VENV_DIR/bin/python3" - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mjurukov/macedonian-ocr-v7',
    local_dir=os.environ['INFER_DIR'],
    ignore_patterns=['*.gitattributes', 'README.md', '.gitignore'],
)
PYEOF
    echo "  Downloaded to $INFER_DIR ✓"
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Next — pull Ollama models and run the benchmark:"
echo ""
echo "    cd $PROJECT_DIR"
echo "    python3 benchmark_h100.py --pull"
echo ""
echo "  This will download ~73 GB of models and then run automatically."
echo "  Results will be saved to:"
echo "    $PROJECT_DIR/results_h100.json"
echo ""
echo "  If the run is interrupted, resume with:"
echo "    python3 benchmark_h100.py --resume results_h100.json"
echo ""
