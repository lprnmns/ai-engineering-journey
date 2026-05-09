# Titanic Visual EDA and Error Analysis

## Goal

Use graphs and out-of-fold error analysis to understand where the current best model succeeds and fails.

Current best model:

```text
plus_title + HistGradientBoosting
```

## OOF Performance

- Rows: 891
- OOF accuracy: 0.83726
- OOF errors: 145

## Top Error Segments

| Segment | Value | Rows | Errors | Error rate | Survival rate | Predicted survival rate |
|---|---|---:|---:|---:|---:|---:|
| Title | Rare | 23 | 8 | 0.348 | 0.348 | 0.522 |
| Pclass | 1.0 | 216 | 50 | 0.231 | 0.630 | 0.602 |
| Embarked | C | 168 | 33 | 0.196 | 0.554 | 0.488 |
| Embarked | Q | 77 | 15 | 0.195 | 0.390 | 0.429 |
| AgeBin | teen | 70 | 13 | 0.186 | 0.429 | 0.414 |
| FareBin | (14.454, 31.0] | 222 | 41 | 0.185 | 0.455 | 0.441 |
| AgeBin | senior | 22 | 4 | 0.182 | 0.227 | 0.136 |
| FareBin | (31.0, 512.329] | 222 | 40 | 0.180 | 0.581 | 0.554 |
| AgeBin | young_adult | 358 | 63 | 0.176 | 0.383 | 0.324 |
| AgeBin | adult | 195 | 33 | 0.169 | 0.400 | 0.364 |

## Generated Figures

- `competitions/titanic/reports/figures/survival_by_sex.png`
- `competitions/titanic/reports/figures/survival_by_pclass.png`
- `competitions/titanic/reports/figures/survival_by_title.png`
- `competitions/titanic/reports/figures/survival_by_age_bin.png`
- `competitions/titanic/reports/figures/survival_by_fare_bin.png`
- `competitions/titanic/reports/figures/age_distribution_by_survival.png`
- `competitions/titanic/reports/figures/fare_distribution_by_survival.png`
- `competitions/titanic/reports/figures/experiment_public_scores.png`
- `competitions/titanic/reports/figures/experiment_cv_vs_public.png`
- `competitions/titanic/reports/figures/oof_errors_by_title.png`
- `competitions/titanic/reports/figures/oof_errors_by_pclass.png`
- `competitions/titanic/reports/figures/oof_errors_by_sex.png`

## Interpretation Guide

Use these figures to answer:

```text
Which features separate survival strongly?
Which segments have high model error?
Which segments are under-predicted or over-predicted?
Where could the next +12 correct predictions come from?
```

## Important Warning

A chart is useful only if it changes the next experiment decision.

The next experiment should come from a concrete observation in this report, not from random feature guessing.
