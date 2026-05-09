# Titanic 0.80 Gap Analysis

## Goal

Understand what is missing between the current best honest score and the 0.80+ target.

## Current Best Experiment

- Experiment: `titanic_exp_003_model_comparison_hgb`
- Model: `hist_gradient_boosting`
- Public LB: `0.77272`
- Features: `plus_title: Pclass, Age, SibSp, Parch, Fare, Sex, Embarked, Title`

## Score Gap

```text
Test rows: 418
Current public score: 0.77272
Target public score: 0.80000
Estimated current correct predictions: 323
Estimated target correct predictions: 335
Estimated extra correct predictions needed: 12
```

## Existing Experiments

| Experiment | Model | Public LB | Local score |
|---|---|---:|---:|
| titanic_exp_001_logreg_baseline | LogisticRegression | 0.76555 | 0.79239 +/- 0.01744 |
| titanic_exp_002_feature_v1_logreg | LogisticRegression | 0.77030 | 0.82939 +/- 0.00939 |
| titanic_exp_003_model_comparison_hgb | hist_gradient_boosting | 0.77272 | 0.83725 +/- 0.01390 |
| titanic_exp_004_age_imputation_v2 | hist_gradient_boosting | 0.00000 | 0.83499 +/- 0.01884 |
| titanic_exp_005_deck_ticket_fare | hist_gradient_boosting | 0.75837 | 0.83611 +/- 0.01913 |

## Diagnosis

The gap is not huge, but it is meaningful.

We probably need around `12` additional correct predictions on the 418-row Kaggle test set.

This means we should not randomly throw models at the problem.

We need targeted experiments:

1. Better missing value handling
2. Stronger Titanic-specific features
3. More stable validation
4. Careful model tuning
5. Small number of meaningful submissions

## Priority Experiment Backlog

| Priority | Experiment | Hypothesis |
|---:|---|---|
| 1 | titanic_exp_004_age_imputation_v2 | Age global median is too crude. Title + Pclass + Sex grouped age imputation may improve generalization. |
| 2 | titanic_exp_005_deck_ticket_fare_person | Deck, TicketGroupSize, and FarePerPerson may capture status/group travel better than raw Cabin/Fare. |
| 3 | titanic_exp_006_hgb_tuning | HistGradientBoosting is the current best model; tuning may improve public score. |
| 4 | titanic_exp_007_catboost_lightgbm | Gradient boosting libraries may handle tabular patterns better than sklearn defaults. |
| 5 | titanic_exp_008_simple_ensemble | Different models may make different errors; voting or averaging can improve stability. |

## Competition Thinking

The rule is:

```text
hypothesis
→ implementation
→ CV
→ public LB if promising
→ experiment log
→ decision
```

We do not trust public leaderboard alone.

We do not trust CV alone.

We compare both.

## Next Immediate Experiment

Start with:

`titanic_exp_004_age_imputation_v2`

Reason:

Age is currently imputed too simply.

A more targeted age strategy may correct a small number of important passengers, especially children and title-based groups.
