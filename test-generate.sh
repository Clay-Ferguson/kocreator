#!/bin/bash
# Quick test: generate demo videos from the test screenshots folder.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv run --directory "$SCRIPT_DIR" python "$SCRIPT_DIR/create-video.py" "$SCRIPT_DIR/test" create-mermaid-demo
