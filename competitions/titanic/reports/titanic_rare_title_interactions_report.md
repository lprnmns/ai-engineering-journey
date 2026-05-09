# Titanic Rare Title + Interaction Features Report

## Hypothesis

The visual error analysis showed that the `Rare` title bucket has the highest OOF error rate.

The previous title logic compressed many different social roles into one label:

```text
Dr, Rev, Major, Col, Capt, Countess, Lady, Sir, Don, Jonkheer, etc. → Rare
```

This experiment splits rare titles into more meaningful groups and adds interaction features.

## Added Features

```text
TitleV2
TitleV2_Pclass
Sex_Pclass
Embarked_Pclass
```

## TitleV2 Counts

- Mr: 517
- Miss: 185
- Mrs: 126
- Master: 40
- Professional: 7
- Clergy: 6
- Military: 5
- NobleMale: 3
- NobleFemale: 1
- RareOther: 1

## Model

- Model: HistGradientBoostingClassifier
- Feature set: plus_title replacement with TitleV2 + interaction features
- Metric: accuracy
- Validation: 5-fold StratifiedKFold

## Results

- Fold 1: 0.86592
- Fold 2: 0.85955
- Fold 3: 0.83708
- Fold 4: 0.83708
- Fold 5: 0.84831

- CV mean: 0.84959
- CV std: 0.01167
- Delta vs current best CV: +0.01234

## Current Best Reference

- CV mean: 0.83725
- Public LB: 0.77272

## Submission

- Path: `competitions/titanic/submissions/rare_title_interactions_hgb_submission.csv`

## Competition Interpretation

This experiment is directly driven by visual error analysis.

Decision rule:

```text
If CV improves or prediction difference is meaningful, consider Kaggle submission.
If CV drops and predictions barely change, treat as negative feature engineering.
```
