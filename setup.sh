#!/usr/bin/env bash
# setup.sh — Install Official Kokoro TTS (hexgrad) on Ubuntu
# After running: uv run kokoro-txt-to-wav.py input.txt output.wav --voice am_liam
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
info "Installing system dependencies (espeak-ng, ffmpeg)..."
sudo apt-get update -qq
sudo apt-get install -y espeak-ng ffmpeg

# ── Step 2: uv ─────────────────────────────────────────────────────
# uv manages the Python interpreter and the project virtualenv. It reads
# .python-version (3.12 — kokoro requires Python <3.13) and downloads that
# interpreter into its shared store if the machine doesn't already have one.
if ! command -v uv >/dev/null 2>&1; then
    error "uv is required but was not found in PATH."
    error "Install it from https://docs.astral.sh/uv/ or via: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ── Step 3: Install Python dependencies ────────────────────────────
# --locked installs exactly what uv.lock records (and fails if pyproject.toml
# has drifted from it), so re-running setup.sh converges .venv back onto the
# known-good set.
info "Installing Python dependencies from uv.lock..."
uv sync --locked
info "Python: $(uv run python --version)"

# ── Step 4: Smoke test ─────────────────────────────────────────────
info "Running smoke test with am_liam voice..."
uv run python -c "
from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code='a')
generator = pipeline('[Kokoro](/kˈOkəɹO/) setup is complete.', voice='am_liam')
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
echo "    uv run create-video.py <base-folder> <subfolder-name>"
echo "    uv run kokoro-txt-to-wav.py input.txt output.wav --voice am_liam"
echo ""
