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

## W4D6 — Feature Engineering v1

Added:

- `competitions/titanic/src/feature_v1.py`
- `competitions/titanic/src/log_feature_v1_experiment.py`
- `competitions/titanic/reports/titanic_feature_v1_report.md`
- `competitions/titanic/submissions/feature_v1_logreg_submission.csv`
- `tests/test_titanic_feature_v1.py`

Main idea:

The first real Kaggle improvement loop is feature engineering, not model shopping.

Feature v1 adds:

- FamilySize
- IsAlone
- HasCabin
- Title

## W4D7 — Feature Ablation Study

Added:

- `competitions/titanic/src/ablation_study.py`
- `competitions/titanic/reports/titanic_ablation_report.md`
- `competitions/titanic/reports/titanic_ablation_results.csv`
- `tests/test_titanic_ablation_study.py`

Main idea:

Feature engineering should be measured. Ablation study checks whether each engineered feature or feature group actually improves cross-validation score.

## W4D8 — Model Comparison

Added:

- `competitions/titanic/src/model_comparison.py`
- `competitions/titanic/src/log_model_comparison_experiment.py`
- `competitions/titanic/reports/titanic_model_comparison_report.md`
- `competitions/titanic/reports/titanic_model_comparison_results.csv`
- `competitions/titanic/submissions/model_comparison_best_submission.csv`
- `tests/test_titanic_model_comparison.py`

Main idea:

After ablation, keep the best simple feature set fixed and compare models. Controlled experiments change one thing at a time.

Result:

- Best local model: `hist_gradient_boosting`
- CV mean: `0.83725`
- Kaggle public score: `0.77272`

## W4D10 — Age Imputation v2

Added:

- `competitions/titanic/src/age_imputation_v2.py`
- `competitions/titanic/reports/titanic_age_imputation_v2_report.md`
- `competitions/titanic/reports/titanic_age_imputation_v2_results.csv`
- `competitions/titanic/submissions/age_imputation_v2_hgb_submission.csv`
- `tests/test_titanic_age_imputation_v2.py`

Main idea:

Test whether grouped Age imputation by `Title + Pclass + Sex` plus `AgeBin` improves over the current best `plus_title + HistGradientBoosting` pipeline.

Result:

- CV mean: `0.83499`
- CV std: `0.01884`
- Current best reference: `0.83725 ± 0.01390`
- Decision: do not submit yet; keep as a negative/diagnostic experiment.

## W4D11 — Deck, Ticket Group, and Fare Per Person

Added:

- `competitions/titanic/src/deck_ticket_fare.py`
- `competitions/titanic/reports/titanic_deck_ticket_fare_report.md`
- `competitions/titanic/reports/titanic_deck_ticket_fare_results.csv`
- `competitions/titanic/submissions/deck_ticket_fare_hgb_submission.csv`
- `tests/test_titanic_deck_ticket_fare.py`

Main idea:

Test whether Titanic-specific group/location features improve the current best pipeline.

Features:

- `Deck`
- `TicketGroupSize`
- `FarePerPerson`

Result:

- CV mean: `0.83611`
- CV std: `0.01913`
- Public LB: `0.75837`
- Current best reference: `0.77272`

Decision:

Negative experiment. Do not use this feature set as the new best. It changed 20 predictions but moved public leaderboard in the wrong direction.

## W4D12 — HGB Tuning

Added:

- `competitions/titanic/src/hgb_tuning.py`
- `competitions/titanic/reports/titanic_hgb_tuning_report.md`
- `competitions/titanic/reports/titanic_hgb_tuning_results.csv`
- `competitions/titanic/submissions/hgb_tuned_submission.csv`
- `tests/test_titanic_hgb_tuning.py`

Main idea:

Tune `HistGradientBoostingClassifier` while keeping the current best feature set fixed.

Best local config:

- Config: `more_min_samples_leaf`
- CV mean: `0.84845`
- CV std: `0.02347`
- Public LB: `0.75837`

Decision:

Negative experiment. The tuned model improved local CV but dropped sharply on public leaderboard. High CV std suggests instability or overfitting.

Current best remains:

- `plus_title + HistGradientBoosting`
- Public LB: `0.77272`

