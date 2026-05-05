# Titanic Kaggle Workflow

## Goal

This project is not only about Titanic.

The real goal is to learn a reusable Kaggle competition workflow:

- read the competition description
- understand the metric
- inspect train/test/submission files
- build a validation strategy
- create a baseline
- track experiments
- generate a valid submission

## Competition Files

Expected raw files:

- `train.csv`
- `test.csv`
- `gender_submission.csv`

## Core ML Thinking

```text
train.csv  -> has features + target
test.csv   -> has features, no target
submission -> required output format
```

## Titanic Target

The target column is:

```text
Survived
```

Meaning:

- `1` means survived
- `0` means did not survive

## Important Rule

Do not look at the test labels, because Kaggle test labels are hidden.

A proper workflow is:

```text
train.csv
  -> split into local train/validation
  -> train model on local train
  -> validate on local validation
  -> train final model
  -> predict test.csv
  -> create submission.csv
```

## First Questions

Before modeling, answer:

1. What is the target?
2. What is the metric?
3. What files are provided?
4. Which columns are numerical?
5. Which columns are categorical?
6. Which columns have missing values?
7. Which features may be useful?
8. Which features may leak target information?
9. What is the simplest baseline?
10. How will experiments be tracked?

## Folder Structure

```text
competitions/titanic/
  data/raw/
  data/processed/
  src/
  reports/
  submissions/
  notebooks/
  experiments/
```

## W4D2 — Titanic EDA

Added:

- `competitions/titanic/src/eda.py`
- `competitions/titanic/reports/titanic_eda_report.md`
- `tests/test_titanic_eda.py`

Main idea:

EDA is not decoration. EDA is where we identify target, missing values, column types, train/test alignment, and first baseline direction.

## W4D3 — Titanic Preprocessing

Added:

- `competitions/titanic/src/preprocess.py`
- `tests/test_titanic_preprocess.py`
- processed baseline feature files under `competitions/titanic/data/processed/`

Main idea:

Missing values are not blindly filled. They are handled according to feature type and baseline strategy:

- numeric features use median imputation
- categorical features use most-frequent imputation
- categorical features are one-hot encoded

## W4D4 — Titanic Baseline Submission

Added:

- `competitions/titanic/src/train_baseline.py`
- `competitions/titanic/reports/titanic_baseline_report.md`
- `competitions/titanic/submissions/baseline_logreg_submission.csv`
- `tests/test_titanic_baseline.py`

Main idea:

A Kaggle baseline is not complete until it creates a valid submission file and is submitted for scoring.

## W4D5 — Cross-Validation and Experiment Logging

Added:

- `competitions/titanic/src/cross_validate.py`
- `competitions/titanic/src/experiment_logging.py`
- `competitions/titanic/reports/titanic_cross_validation_report.md`
- updated `competitions/titanic/experiments/experiments.csv`
- `tests/test_titanic_cross_validation.py`
- `tests/test_titanic_experiment_logging.py`

Main idea:

A Kaggle score is only useful when compared against a stable local validation strategy. We now track CV mean, CV std, public leaderboard score, and experiment notes.

