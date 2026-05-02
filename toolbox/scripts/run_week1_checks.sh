#!/usr/bin/env bash

set -euo pipefail

echo "Running Week 1 checks..."
echo

echo "1) TODO scan"
./toolbox/scripts/find_todos.sh
echo

echo "2) Vector utilities"
python3 labs/lin_alg/vec.py
echo

echo "3) Debug examples"
python3 labs/lin_alg/debug_examples.py
echo

echo "Week 1 checks completed."
