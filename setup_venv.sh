#!/usr/bin/env bash
# Creates a local virtual environment (.venv) and installs requirements.txt into it.
# Usage: ./setup_venv.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python was not found on PATH. Install Python 3 and try again." >&2
    exit 1
fi

if [ ! -d ".venv" ]; then
    "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo ""
echo "Virtual environment ready. Activate it with:"
echo "  source .venv/bin/activate"
