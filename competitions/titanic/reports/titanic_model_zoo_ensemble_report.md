# Titanic Model Zoo + Ensemble Report

## Goal

Beat the current HGB-only approach by testing a broad tabular model zoo and voting ensembles.

Current best public reference:

```text
model_comparison_best_submission.csv
public LB = 0.77272
```

## What This Sprint Adds

```text
Feature sets:
- plus_title
- binned_competition
- wide_competition
- nb_competition_ordinal

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
- XGBoost if installed

Ensembles:
- top3 hard voting
- top5 hard voting
- AdaBoost + ExtraTrees + GaussianNB weighted hard voting
```

## Top Results

| Rank | Candidate | Type | Feature set | Model | CV mean | CV std |
|---:|---|---|---|---|---:|---:|
| 1 | plus_title__top3_hard_voting | voting | plus_title | top3_hard_voting | 0.84397 | 0.01504 |
| 2 | wide_competition__top3_hard_voting | voting | wide_competition | top3_hard_voting | 0.84285 | 0.01861 |
| 3 | wide_competition__top5_hard_voting | voting | wide_competition | top5_hard_voting | 0.84285 | 0.01861 |
| 4 | plus_title__top5_hard_voting | voting | plus_title | top5_hard_voting | 0.84285 | 0.01649 |
| 5 | plus_title__xgboost | single_model | plus_title | xgboost | 0.84061 | 0.01327 |
| 6 | wide_competition__hist_gradient_boosting | single_model | wide_competition | hist_gradient_boosting | 0.84061 | 0.01912 |
| 7 | nb_competition_ordinal__top5_hard_voting | voting | nb_competition_ordinal | top5_hard_voting | 0.84061 | 0.02274 |
| 8 | wide_competition__xgboost | single_model | wide_competition | xgboost | 0.83948 | 0.01553 |
| 9 | plus_title__svc_rbf | single_model | plus_title | svc_rbf | 0.83838 | 0.01859 |
| 10 | nb_competition_ordinal__xgboost | single_model | nb_competition_ordinal | xgboost | 0.83837 | 0.01936 |
| 11 | plus_title__hist_gradient_boosting | single_model | plus_title | hist_gradient_boosting | 0.83725 | 0.01390 |
| 12 | plus_title__gradient_boosting | single_model | plus_title | gradient_boosting | 0.83724 | 0.01253 |
| 13 | nb_competition_ordinal__top3_hard_voting | voting | nb_competition_ordinal | top3_hard_voting | 0.83612 | 0.01974 |
| 14 | wide_competition__svc_rbf | single_model | wide_competition | svc_rbf | 0.83501 | 0.01840 |
| 15 | nb_competition_ordinal__gradient_boosting | single_model | nb_competition_ordinal | gradient_boosting | 0.83498 | 0.02739 |
| 16 | binned_competition__random_forest | single_model | binned_competition | random_forest | 0.83389 | 0.00771 |
| 17 | plus_title__random_forest | single_model | plus_title | random_forest | 0.83277 | 0.01250 |
| 18 | binned_competition__top3_hard_voting | voting | binned_competition | top3_hard_voting | 0.83277 | 0.00669 |
| 19 | wide_competition__random_forest | single_model | wide_competition | random_forest | 0.83277 | 0.00669 |
| 20 | wide_competition__gradient_boosting | single_model | wide_competition | gradient_boosting | 0.83274 | 0.02438 |
| 21 | binned_competition__svc_rbf | single_model | binned_competition | svc_rbf | 0.83164 | 0.01196 |
| 22 | nb_competition_ordinal__hist_gradient_boosting | single_model | nb_competition_ordinal | hist_gradient_boosting | 0.83163 | 0.01793 |
| 23 | binned_competition__extra_trees | single_model | binned_competition | extra_trees | 0.83052 | 0.00675 |
| 24 | wide_competition__extra_trees | single_model | wide_competition | extra_trees | 0.83052 | 0.00675 |
| 25 | binned_competition__top5_hard_voting | voting | binned_competition | top5_hard_voting | 0.83052 | 0.01101 |
| 26 | plus_title__decision_tree | single_model | plus_title | decision_tree | 0.83051 | 0.01808 |
| 27 | plus_title__logistic_regression | single_model | plus_title | logistic_regression | 0.82941 | 0.01152 |
| 28 | binned_competition__xgboost | single_model | binned_competition | xgboost | 0.82938 | 0.01382 |
| 29 | nb_competition_ordinal__random_forest | single_model | nb_competition_ordinal | random_forest | 0.82827 | 0.01171 |
| 30 | binned_competition__gradient_boosting | single_model | binned_competition | gradient_boosting | 0.82826 | 0.01514 |

## Best Local Candidate

- Candidate: `plus_title__top3_hard_voting`
- Type: `voting`
- Feature set: `plus_title`
- Model: `top3_hard_voting`
- CV mean: `0.84397`
- CV std: `0.01504`

## Submission

- Path: `competitions/titanic/submissions/model_zoo_ensemble_best_submission.csv`

## Required Next Step

Before submitting:

```bash
python competitions/titanic/src/risky_flip_guard.py \
  --candidate model_zoo=competitions/titanic/submissions/model_zoo_ensemble_best_submission.csv
```

Only submit if the guard report is acceptable.
