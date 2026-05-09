# Titanic Age Imputation v2 Report

## Hypothesis

Global median Age imputation may be too crude.

Instead, missing Age values are filled by:

```text
Title + Pclass + Sex group median
fallback: Title median
fallback: global median
```

Then `AgeBin` is added:

```text
child / teen / adult / senior
```

## Model

- Model: HistGradientBoostingClassifier
- Feature set: plus_title + AgeBin
- Metric: accuracy
- Validation: 5-fold StratifiedKFold

## Results

- Fold 1: 0.85475
- Fold 2: 0.85393
- Fold 3: 0.80337
- Fold 4: 0.83146
- Fold 5: 0.83146

- CV mean: 0.83499
- CV std: 0.01884

## Baseline Reference

exp_003 plus_title hgb: CV 0.83725, public LB 0.77272

## Submission

- Path: `competitions/titanic/submissions/age_imputation_v2_hgb_submission.csv`

## Competition Interpretation

This experiment tests whether smarter Age handling can recover some of the estimated 12 missing correct predictions needed to reach 0.80 public LB.

Important:

If CV improves but public LB drops, this may mean the Age strategy overfits local validation.
