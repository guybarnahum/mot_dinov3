#!/usr/bin/env bash
set -euo pipefail

echo ">>> Removing .venv"
rm -rf .venv

echo ">>> Cleaning Python caches"
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

echo ">>> Done. Outputs left in ./outputs"

