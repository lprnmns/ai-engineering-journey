# Week 3 Summary — Data and Baseline ML

## Goal

Build the first complete data-to-model pipeline.

This week focused on:

- loading CSV data with pandas
- detecting missing values
- cleaning raw data
- saving processed datasets
- querying data with SQLite
- generating an EDA report
- splitting features and target
- training a baseline Logistic Regression classifier

## Completed Work

### W3D1 — Pandas Basics

Created a toy student dataset:

- `data/raw/students.csv`

Added:

- `labs/data/pandas_basics.py`
- `tests/test_pandas_basics.py`

Main idea:

```text
Before modeling, understand the data.
```

### W3D2 — Data Cleaning

Added a cleaning pipeline:

- `labs/data/clean_students.py`
- `data/processed/students_clean.csv`
- `tests/test_clean_students.py`

Main idea:

```text
Raw data should not go directly into a model.
```

### W3D3 — SQL Basics

Added SQLite loading and query examples:

- `labs/data/sql_basics.py`
- `data/processed/students.db`
- `tests/test_sql_basics.py`

Covered:

- SELECT
- WHERE
- ORDER BY
- LIMIT
- GROUP BY

Main idea:

```text
AI/ML engineers must know how to retrieve the right data.
```

### W3D4 — EDA Report

Generated a Markdown EDA report:

- `labs/data/eda_report.py`
- `docs/w3_eda_report.md`
- `tests/test_eda_report.py`

Main idea:

```text
Do not train models blindly. Write down what the data says.
```

### W3D5 — Feature Dataset

Built feature and target splits:

- `labs/data/build_features.py`
- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `tests/test_build_features.py`

Main idea:

```text
Model training starts with X and y.
```

### W3D6 — Baseline ML Classifier

Trained the first ML model:

- `labs/ml/train_baseline_classifier.py`
- `docs/w3_baseline_model_report.md`
- `tests/test_baseline_classifier.py`

Main idea:

```text
A baseline model is the first working reference point.
```

## Pipeline

```text
raw CSV
→ pandas load
→ missing value analysis
→ cleaning
→ processed CSV
→ SQLite queries
→ EDA report
→ feature/target split
→ train/test split
→ baseline Logistic Regression
→ evaluation report
```

## Main Artifacts

- `data/raw/students.csv`
- `data/processed/students_clean.csv`
- `data/processed/students.db`
- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `labs/data/pandas_basics.py`
- `labs/data/clean_students.py`
- `labs/data/sql_basics.py`
- `labs/data/eda_report.py`
- `labs/data/build_features.py`
- `labs/ml/train_baseline_classifier.py`
- `docs/w3_eda_report.md`
- `docs/w3_baseline_model_report.md`
- `toolbox/scripts/run_week3_checks.sh`

## How to Run Week 3 Checks

```bash
./toolbox/scripts/run_week3_checks.sh
```

## Reflection

This week moved the repository from Python engineering practice into the first real ML pipeline.

The important lesson:

```text
Modeling is not just model.fit().
A real ML workflow starts with data quality, cleaning, analysis, feature preparation, and reproducible evaluation.
```

The dataset is intentionally tiny, so the model score is not meaningful. The pipeline discipline is the real artifact.
