# Titanic Feature Engineering v1 Report

## Feature Set

Baseline features plus:

- FamilySize = SibSp + Parch + 1
- IsAlone = FamilySize == 1
- HasCabin = Cabin is not missing
- Title extracted from Name

## Model

- Algorithm: Logistic Regression
- Preprocessing:
  - numeric median imputation
  - categorical most-frequent imputation
  - one-hot encoding
- Metric: Accuracy

## Single Split Validation

- Train rows: 712
- Validation rows: 179
- Single split accuracy: 0.83240

## Cross-Validation

- Fold 1: 0.84358
- Fold 2: 0.81461
- Fold 3: 0.82584
- Fold 4: 0.83146
- Fold 5: 0.83146

- CV mean accuracy: 0.82939
- CV std accuracy: 0.00939

## Classification Report

```text
              precision    recall  f1-score   support

           0       0.85      0.88      0.87       110
           1       0.80      0.75      0.78        69

    accuracy                           0.83       179
   macro avg       0.83      0.82      0.82       179
weighted avg       0.83      0.83      0.83       179

```

## Submission

- Path: `competitions/titanic/submissions/feature_v1_logreg_submission.csv`
- Test rows: 418

## Competition Thinking

This experiment tests whether simple human-designed Titanic features improve local validation.

The most important comparison is not only against the previous single split score.

We compare:

```text
exp_001 baseline CV / public LB
vs
exp_002 feature_v1 CV / public LB
```

If CV improves but public leaderboard drops, the feature may be overfitting local validation or public leaderboard noise may be involved.
