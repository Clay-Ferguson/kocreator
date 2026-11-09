#!/bin/bash
# Quick test: generate demo videos from the test screenshots folder.
# Extra arguments are passed to create-video.py, e.g.:
#   ./test-generate.sh --plugin ../kocreator-underdog
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/uv-env.sh"
# Avoid uv's "VIRTUAL_ENV does not match" warning — see setup.sh.
unset VIRTUAL_ENV

uv run --directory "$SCRIPT_DIR" python "$SCRIPT_DIR/create-video.py" "$SCRIPT_DIR/test" create-mermaid-demo "$@"
