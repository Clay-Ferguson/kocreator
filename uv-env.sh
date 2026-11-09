# uv-env.sh — source this (don't execute it) before any `uv` command for kocreator.
#
# This folder is shared between machines (e.g. a host and a VM mounting the same
# disk). A uv virtualenv is machine-specific: .venv/bin/python is a symlink into
# that machine's uv interpreter store, so a venv built on one machine is a broken
# symlink on the other. Giving each machine its own venv (.venv-<hostname>) lets
# them coexist instead of clobbering each other on every `uv sync` / `uv run`.
#
# An explicit UV_PROJECT_ENVIRONMENT from the caller's environment still wins.
KOCREATOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$KOCREATOR_DIR/.venv-$(hostname -s)}"
