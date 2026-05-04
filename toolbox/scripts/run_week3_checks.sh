#!/usr/bin/env bash

set -euo pipefail

echo "Running Week 3 checks..."
echo

echo "1) Pandas basics"
python labs/data/pandas_basics.py
echo

echo "2) Data cleaning"
python labs/data/clean_students.py
echo

echo "3) SQL basics"
python labs/data/sql_basics.py
echo

echo "4) EDA report"
python labs/data/eda_report.py
echo

echo "5) Feature dataset"
python labs/data/build_features.py
echo

echo "6) Baseline ML classifier"
python labs/ml/train_baseline_classifier.py
echo

echo "7) Unit tests"
pytest -v
echo

echo "8) Static type checking"
mypy src examples tests labs
echo

echo "Week 3 checks completed."
