#!/usr/bin/env bash
# setup.sh — Set up kocreator on Ubuntu.
#
# Speech comes from a TTS plugin folder (normally a sibling such as
# ../kocreator-kokoro), chosen by kocreator.toml. Set up the plugin first with
# its own ./setup.sh, then run this. Usage:
#   ./setup.sh                       # uses the plugin from kocreator.toml
#   ./setup.sh --plugin <dir>        # check and smoke-test another plugin
# After running: uv run create-video.py <base-folder> <subfolder-name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# Per-machine venv (.venv-<hostname>) — see uv-env.sh
source "$SCRIPT_DIR/uv-env.sh"
# An activated virtualenv (another project's, or one VS Code activated) only
# makes uv warn that VIRTUAL_ENV "does not match the project environment"
# before ignoring it, so drop it rather than read past the warning.
unset VIRTUAL_ENV

PLUGIN_ARG=""
if [ "${1:-}" = "--plugin" ] && [ -n "${2:-}" ]; then
    PLUGIN_ARG="$2"
fi

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Step 1: System dependencies ────────────────────────────────────
info "Installing system dependencies (ffmpeg)..."
sudo apt-get update -qq
sudo apt-get install -y ffmpeg

# ── Step 2: uv ─────────────────────────────────────────────────────
# uv manages the Python interpreter and the project virtualenv. It reads
# .python-version (3.12) and downloads that interpreter into its shared store
# if the machine doesn't already have one.
if ! command -v uv >/dev/null 2>&1; then
    error "uv is required but was not found in PATH."
    error "Install it from https://docs.astral.sh/uv/ or via: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ── Step 3: Install Python dependencies ────────────────────────────
# --locked installs exactly what uv.lock records (and fails if pyproject.toml
# has drifted from it), so re-running setup.sh converges the venv back onto the
# known-good set.
info "Installing Python dependencies from uv.lock..."
uv sync --locked
info "Venv:   $UV_PROJECT_ENVIRONMENT"
info "Python: $(uv run python --version)"

# ── Step 4: TTS plugin ─────────────────────────────────────────────
# Same precedence as create-video.py: --plugin, $KOCREATOR_TTS_PLUGIN, then
# tts_plugin in kocreator.toml. Relative paths are resolved against this folder.
PLUGIN="${PLUGIN_ARG:-${KOCREATOR_TTS_PLUGIN:-}}"
if [ -z "$PLUGIN" ]; then
    PLUGIN="$(uv run python -c 'import tomllib; print(tomllib.load(open("kocreator.toml", "rb")).get("tts_plugin", ""))')"
fi
if [ -z "$PLUGIN" ]; then
    error "No TTS plugin configured: set tts_plugin in kocreator.toml."
    exit 1
fi
case "$PLUGIN" in
    /*) ;;
    *) PLUGIN="$SCRIPT_DIR/$PLUGIN" ;;
esac
if [ ! -x "$PLUGIN/tts.sh" ]; then
    error "TTS plugin not found: $PLUGIN/tts.sh"
    error "Check out a plugin (kocreator-kokoro, Buffy or kocreator-qwen)"
    error "next to this folder, run its ./setup.sh, and point kocreator.toml at it."
    exit 1
fi
PLUGIN="$(cd "$PLUGIN" && pwd)"
info "Plugin: $PLUGIN"

# Plugins run their engines in their own environments, never in ours.
unset UV_PROJECT_ENVIRONMENT

if ! "$PLUGIN/tts.sh" check; then
    error "The TTS plugin is not ready (see above). Run: $PLUGIN/setup.sh"
    exit 1
fi
info "Plugin settings: $("$PLUGIN/tts.sh" settings)"

# ── Step 5: Smoke test ─────────────────────────────────────────────
info "Running smoke test through the TTS plugin..."
# Speaking is batch-only, so the smoke test is a batch of one: a jobs file
# with a single input<TAB>output<TAB>seed line (no seed).
SMOKE_TXT="$(mktemp --suffix=.txt)"
SMOKE_JOBS="$(mktemp --suffix=.tsv)"
trap 'rm -f "$SMOKE_TXT" "$SMOKE_JOBS"' EXIT
echo "Text to speech setup is complete." > "$SMOKE_TXT"
printf '%s\t%s\t\n' "$SMOKE_TXT" "$SCRIPT_DIR/test_output.wav" > "$SMOKE_JOBS"
rm -f "$SCRIPT_DIR/test_output.wav"
"$PLUGIN/tts.sh" speak "$SMOKE_JOBS" \
    || warn "The TTS plugin exited with an error (see its output above)."

if [ -s "$SCRIPT_DIR/test_output.wav" ]; then
    info "Smoke test passed! test_output.wav created successfully."
else
    warn "Smoke test did not produce output. Did you run $PLUGIN/setup.sh?"
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN} kocreator setup complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  TTS plugin: $PLUGIN"
echo ""
echo "  Usage:"
echo "    cd $SCRIPT_DIR && source uv-env.sh"
echo "    uv run create-video.py <base-folder> <subfolder-name>"
echo ""
