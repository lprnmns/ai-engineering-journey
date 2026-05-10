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

## W4D16 — Submission Difference Analysis

Added:

- `competitions/titanic/src/submission_difference_analysis.py`
- `competitions/titanic/reports/titanic_submission_difference_analysis.md`
- `competitions/titanic/reports/titanic_submission_difference_rows.csv`
- `competitions/titanic/reports/titanic_submission_difference_segments.csv`
- `tests/test_titanic_submission_difference_analysis.py`

Main idea:

Compare failed submissions against the current best submission to understand which passenger predictions were changed.

Key finding:

Failed submissions mostly changed current-best `0` predictions into `1`, especially around `Pclass=3`, female passengers, and `Miss/Mrs`-like groups.

Decision:

Before future submissions, inspect whether a new model is flipping the same risky passenger groups.

## W4D17 — Risky Flip Guard

Added:

- `competitions/titanic/src/risky_flip_guard.py`
- `competitions/titanic/reports/titanic_risky_flip_guard_report.md`
- `competitions/titanic/reports/titanic_risky_flip_guard_summary.csv`
- `competitions/titanic/reports/titanic_risky_flip_guard_rows.csv`
- `tests/test_titanic_risky_flip_guard.py`

Main idea:

Create a pre-submission guardrail that compares a candidate submission against the current best public submission.

The guard checks whether a candidate makes risky prediction flips, especially:

- current-best `0` → candidate `1`
- `Pclass=3 female` 0→1 flips
- `Miss/Mrs` 0→1 flips
- `Embarked=S/C` 0→1 flips

Result:

- `deck_ticket_fare`: `DO_NOT_SUBMIT_WITHOUT_REVIEW`
- `hgb_tuned`: `DO_NOT_SUBMIT_WITHOUT_REVIEW`
- `rare_title_interactions`: `DO_NOT_SUBMIT_WITHOUT_REVIEW`
- `repeated_cv_best`: `NO_NEW_INFORMATION`

Decision:

Use this guard before future Kaggle submissions. CV improvement alone is not enough.

## W4D18 — PassengerId and Surname Analysis

Added:

- `competitions/titanic/src/id_and_surname_analysis.py`
- `competitions/titanic/reports/titanic_passenger_id_bins.csv`
- `competitions/titanic/reports/titanic_surname_survival.csv`
- `competitions/titanic/reports/titanic_id_and_surname_analysis.md`
- `tests/test_titanic_id_and_surname_analysis.py`

Main idea:

Analyze whether PassengerId ranges and surname/family groups reveal useful structure.

Key findings:

- PassengerId bins show different survival rates, but PassengerId should be treated carefully because it may encode dataset ordering rather than a real-world feature.
- Surname groups show strong family-level patterns. Some families have very low survival rates, while others have high survival rates.
- Safe future features may include `SurnameGroupSize`, `IsLargeFamily`, or train/test combined surname group counts.
- Direct surname target survival rate is risky because it can introduce target leakage.

## W4D19 — Name, Family, and PassengerId Pattern Analysis

Added:

- `competitions/titanic/src/name_family_id_pattern_analysis.py`
- `competitions/titanic/reports/titanic_name_family_id_pattern_analysis.md`
- `competitions/titanic/reports/titanic_safe_family_features_train.csv`
- `competitions/titanic/reports/titanic_safe_family_features_test.csv`
- `competitions/titanic/reports/titanic_surname_feature_analysis.csv`
- `competitions/titanic/reports/titanic_passenger_id_jump_segments.csv`
- `competitions/titanic/reports/titanic_raw_title_analysis.csv`
- `competitions/titanic/reports/titanic_title_group_analysis.csv`
- `competitions/titanic/reports/titanic_first_given_token_analysis.csv`
- `tests/test_titanic_name_family_id_pattern_analysis.py`

Main idea:

Analyze safe family/surname structural features, PassengerId jump segments, raw title/status groups, and first given-name tokens.

Key findings:

- `SurnameTicketGroupSize` shows strong non-linear signal. Small surname-ticket groups have much higher survival, while very large groups have very low survival.
- PassengerId jump segments show survival-rate shifts, but PassengerId is risky as a model feature because it may encode dataset ordering.
- Raw title/status is very predictive: `Mr` is low survival, `Mrs/Miss` high survival, `Master` moderate, `Rev` very low.
- First given-name tokens show patterns but are likely noisy and overfit-prone.

Safe future feature candidates:

- `SurnameGroupSize`
- `SurnamePclassGroupSize`
- `SurnameTicketGroupSize`
- `IsSoloSurname`
- `IsLargeSurnameGroup`
- `TitleGroup`
- `HasParenthesesName`
- `HasQuotedNickname`

Risky candidates:

- `PassengerIdJumpSegment`
- `FirstGivenToken`
- Any target-mean survival feature by surname/title/token.

## W4D20 — Auto Experiment Suite

Added:

- `competitions/titanic/src/auto_experiment_suite.py`
- `tests/test_titanic_auto_experiment_suite.py`
- `scripts/run_titanic_auto_experiment_suite.sh`
- `scripts/run_titanic_auto_experiment_suite_dry_submit.sh`
- `scripts/run_titanic_auto_experiment_suite_submit.sh`
- `competitions/titanic/reports/auto_experiment_suite/auto_experiment_results.csv`
- `competitions/titanic/reports/auto_experiment_suite/auto_experiment_qualified.csv`
- `competitions/titanic/reports/auto_experiment_suite/auto_experiment_guard_summary.csv`
- `competitions/titanic/reports/auto_experiment_suite/auto_experiment_suite_report.md`
- `competitions/titanic/submissions/auto_experiment_suite/`

Main idea:

Run a controlled experiment suite across multiple safe feature sets, model configurations, repeated CV seeds, and current-best flip guard checks.

Result:

- Total experiments: `42`
- Qualified experiments: `0`
- Tests: `126 passed`
- Type check: `mypy Success`

Best candidate:

- `auto_008_safe_family_counts_hgb_reference`
- CV mean: `0.83875`
- CV std: `0.01789`
- Risk-adjusted score: `0.82086`
- Changed predictions vs current best: `13`
- 0→1 flips: `7`
- Guard decision: `DO_NOT_SUBMIT_WITHOUT_REVIEW`

Decision:

Do not submit automatically. The best candidate passed the CV/risk-adjusted signal but failed the guard by a small margin. This suggests family count features are promising, but still too risky to submit without deeper changed-row analysis.

## W4D18 — Model Zoo and Ensemble

Added:

- `competitions/titanic/src/model_zoo_ensemble.py`
- `competitions/titanic/reports/titanic_model_zoo_ensemble_report.md`
- `competitions/titanic/reports/titanic_model_zoo_ensemble_results.csv`
- `competitions/titanic/submissions/model_zoo_ensemble_best_submission.csv`
- `tests/test_titanic_model_zoo_ensemble.py`

Main idea:

Expand beyond the HGB-only approach by testing a broader tabular model zoo and voting ensembles.

Feature sets:

- `plus_title`
- `binned_competition`
- `wide_competition`
- `nb_competition_ordinal`

Models:

- GaussianNB
- LogisticRegression
- KNN
- SVC
- DecisionTree
- RandomForest
- ExtraTrees
- AdaBoost
- GradientBoosting
- HistGradientBoosting
- XGBoost when available

Best local candidate:

- Candidate: `plus_title__top3_hard_voting`
- CV mean: `0.84397`
- CV std: `0.01504`

Guard result:

- `model_zoo`: `DO_NOT_SUBMIT_WITHOUT_REVIEW`
- Reason: risky `0_to_1` flips, especially `Pclass=3 female` and `Miss/Mrs` groups.

Decision:

Do not submit yet. The local CV improved, but the risky flip guard detected the same failure pattern that damaged recent public leaderboard submissions.

