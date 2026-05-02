#!/usr/bin/env bash

set -euo pipefail

echo "Searching TODOs in repository..."

grep -n "TODO" -R . \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude=find_todos.sh \
  || echo "No TODO found."
