#!/usr/bin/env bash
# setup.sh — Install Official Kokoro TTS (hexgrad) on Ubuntu
# After running: source .venv/bin/activate && python kokoro-generate.py input.txt output.wav --voice bm_daniel
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Step 1: System dependencies ────────────────────────────────────
info "Installing system dependencies (espeak-ng, python3-venv)..."
sudo apt-get update -qq
sudo apt-get install -y espeak-ng python3 python3-venv python3-pip

# ── Step 2: Python virtual environment ─────────────────────────────
# kokoro requires Python >=3.10,<3.13 — do NOT use 3.13
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON_BIN="$(command -v "$candidate")"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    error "No compatible Python (3.10-3.12) found. kokoro requires Python <3.13."
    error "Install Python 3.12 with: sudo apt-get install python3.12 python3.12-venv"
    exit 1
fi

info "Using $PYTHON_BIN ($($PYTHON_BIN --version))"

if [ -d ".venv" ]; then
    info "Virtual environment .venv/ already exists, reusing it."
else
    info "Creating Python virtual environment at .venv/ ..."
    "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
info "Python: $(python --version) at $(which python)"

# ── Step 3: Install official kokoro package ────────────────────────
info "Installing official kokoro package (hexgrad) via pip..."
pip install --upgrade pip --quiet
pip install "kokoro>=0.9.4" soundfile "misaki[en]" --quiet
info "kokoro installed: $(pip show kokoro | grep Version)"

# ── Step 4: Smoke test ─────────────────────────────────────────────
info "Running smoke test with bm_daniel voice..."
python -c "
from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code='b')
generator = pipeline('[Kokoro](/kˈOkəɹO/) setup is complete.', voice='bm_daniel')
for i, (gs, ps, audio) in enumerate(generator):
    sf.write('test_output.wav', audio, 24000)
    break
print('Smoke test passed!')
"

if [ -f "test_output.wav" ] && [ -s "test_output.wav" ]; then
    info "Smoke test passed! test_output.wav created successfully."
else
    warn "Smoke test did not produce output. Try running manually."
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN} Official Kokoro TTS setup complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Source: https://github.com/hexgrad/kokoro (Apache 2.0)"
echo "  G2P: misaki — supports [word](/phonemes/) pronunciation overrides"
echo ""
echo "  Usage:"
echo "    cd $SCRIPT_DIR"
echo "    source .venv/bin/activate"
echo "    python kokoro-generate.py input.txt output.wav --voice bm_daniel"
echo ""
