#!/bin/bash
# Quick test: generate demo videos from the test screenshots folder.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

python "$SCRIPT_DIR/create-video.py" "$SCRIPT_DIR/test" create-mermaid-demo
