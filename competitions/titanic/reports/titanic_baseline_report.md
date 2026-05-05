# Titanic Baseline Model Report

## Model

- Algorithm: Logistic Regression
- Features: preprocessed baseline features from W4D3
- Validation split: 80/20 stratified split
- Metric: Accuracy

## Local Validation

- Train rows: 712
- Validation rows: 179
- Validation accuracy: 0.8045

## Classification Report

```text
              precision    recall  f1-score   support

           0       0.81      0.89      0.85       110
           1       0.79      0.67      0.72        69

    accuracy                           0.80       179
   macro avg       0.80      0.78      0.79       179
weighted avg       0.80      0.80      0.80       179

```

## Submission

- Submission path: `competitions/titanic/submissions/baseline_logreg_submission.csv`
- Expected columns: `PassengerId`, `Survived`
- Expected test rows: 418

## Notes

This is the first simple baseline.

The purpose is not to maximize leaderboard score yet.
The purpose is to create a valid end-to-end Kaggle workflow:

```text
preprocess -> train -> validate -> predict test -> create submission -> submit to Kaggle
```

Next improvements may include:

- title extraction from Name
- FamilySize feature
- IsAlone feature
- HasCabin feature
- better Age imputation
- model comparison with RandomForest / CatBoost / LightGBM
