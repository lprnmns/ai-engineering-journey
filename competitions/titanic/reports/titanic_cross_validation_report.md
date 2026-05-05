# Titanic Cross-Validation Report

## Model

- Algorithm: Logistic Regression
- Feature set: W4D3 baseline preprocessed features
- Validation: StratifiedKFold
- Number of folds: 5
- Metric: accuracy

## Fold Scores

- Fold 1: 0.77095
- Fold 2: 0.80337
- Fold 3: 0.78652
- Fold 4: 0.78090
- Fold 5: 0.82022

## Summary

- CV mean accuracy: 0.79239
- CV std accuracy: 0.01744
- Kaggle public score: 0.76555
- CV minus public score gap: 0.02684

## Interpretation

A single train/validation split can be noisy.

Cross-validation gives a more stable local estimate by evaluating the model on multiple validation folds.

The important competition question is not only:

"Is my model good?"

It is also:

"Does my local validation score correlate with Kaggle public/private leaderboard?"

## Next Improvements

- Add FamilySize feature
- Add IsAlone feature
- Extract Title from Name
- Add HasCabin feature
- Compare LogisticRegression, RandomForest, and gradient boosting models
