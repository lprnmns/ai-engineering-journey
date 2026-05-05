# Titanic Feature Ablation Study

## Goal

Understand which engineered features actually help the model.

Feature v1 added multiple features at once:

- FamilySize
- IsAlone
- HasCabin
- Title

This report tests individual and grouped feature contributions using the same model and validation strategy.

## Validation Setup

- Model: LogisticRegression
- Metric: accuracy
- Validation: 5-fold StratifiedKFold
- Random state: 42

## Results

| Rank | Feature set | CV mean | CV std | Delta vs baseline |
|---:|---|---:|---:|---:|
| 1 | plus_title | 0.82941 | 0.01152 | +0.03702 |
| 2 | plus_family_size_title | 0.82941 | 0.01152 | +0.03702 |
| 3 | feature_v1_all | 0.82939 | 0.00939 | +0.03700 |
| 4 | feature_v1_without_has_cabin | 0.82827 | 0.00784 | +0.03588 |
| 5 | plus_has_cabin_title | 0.82601 | 0.01491 | +0.03362 |
| 6 | feature_v1_without_is_alone | 0.82489 | 0.01423 | +0.03250 |
| 7 | plus_is_alone | 0.80024 | 0.01662 | +0.00785 |
| 8 | plus_family_size_is_alone | 0.80024 | 0.01662 | +0.00785 |
| 9 | plus_family_size | 0.79352 | 0.01721 | +0.00112 |
| 10 | baseline | 0.79239 | 0.01744 | +0.00000 |
| 11 | plus_has_cabin | 0.79238 | 0.02264 | -0.00001 |

## Best Local Feature Set

- Name: plus_title
- CV mean: 0.82941
- CV std: 0.01152
- Delta vs baseline: +0.03702

## Baseline Reference

- Name: baseline
- CV mean: 0.79239
- CV std: 0.01744

## Interpretation

Ablation study answers this question:

```text
When a score improves after adding several features,
which feature actually caused the improvement?
```

Important warning:

A feature can look good locally but fail on Kaggle public leaderboard.
Therefore, we use ablation to choose candidates, not to blindly trust local CV.

## Next Step

Use the best local feature set for model comparison:

- LogisticRegression
- RandomForest
- HistGradientBoosting
