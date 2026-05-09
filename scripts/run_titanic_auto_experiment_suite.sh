#!/usr/bin/env bash
set -euo pipefail

cd /home/alperen/ai-engineering-journey

python competitions/titanic/src/auto_experiment_suite.py \
  --seeds 42,7,123 \
  --n-splits 5 \
  --min-cv-mean 0.838 \
  --max-cv-std 0.018 \
  --min-risk-adjusted 0.820 \
  --max-total-changed 12 \
  --max-0-to-1 6 \
  --max-pclass3-female-0-to-1 3 \
  --max-miss-mrs-0-to-1 4

cat competitions/titanic/reports/auto_experiment_suite/auto_experiment_suite_report.md
cat competitions/titanic/reports/auto_experiment_suite/auto_experiment_qualified.csv
