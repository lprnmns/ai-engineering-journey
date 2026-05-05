# Titanic Model Comparison Report

## Goal

Compare multiple models on the same `plus_title` feature set.

The feature set is fixed. Only the model changes.

## Feature Set

Numeric:

Pclass, Age, SibSp, Parch, Fare

Categorical:

Sex, Embarked, Title

## Validation Setup

- Metric: accuracy
- Validation: 5-fold StratifiedKFold
- Random state: 42

## Results

| Rank | Model | CV mean | CV std |
|---:|---|---:|---:|
| 1 | hist_gradient_boosting | 0.83725 | 0.01390 |
| 2 | random_forest | 0.83277 | 0.01250 |
| 3 | logistic_regression | 0.82941 | 0.01152 |

## Best Local Model

- Model: hist_gradient_boosting
- CV mean: 0.83725
- CV std: 0.01390

## Submission

- Best-model submission path: `competitions/titanic/submissions/model_comparison_best_submission.csv`

## Competition Thinking

This is a controlled experiment:

```text
same features
same CV
different model
```

If a model improves CV but fails on public leaderboard, we do not blindly trust it.
We compare CV, public LB, and experiment notes together.
