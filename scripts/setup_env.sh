#!/usr/bin/env bash
# Environment setup for the AI Dataset Intelligence Platform (backend)
# Usage: bash scripts/setup_env.sh
set -e

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=".venv"

echo "==> Using $($PYTHON_BIN --version)"

if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtual environment in $VENV_DIR"
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing backend requirements"
pip install -r backend/requirements.txt

echo "==> Done. Activate with: source $VENV_DIR/bin/activate"
