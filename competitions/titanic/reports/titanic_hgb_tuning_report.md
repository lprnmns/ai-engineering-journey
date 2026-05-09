# Titanic HGB Tuning Report

## Goal

Tune HistGradientBoostingClassifier on the current best honest feature set.

Feature set is fixed:

```text
Pclass, Age, SibSp, Parch, Fare, Sex, Embarked, Title
```

Only model hyperparameters change.

## Current Best Reference

- CV mean: 0.83725
- Public LB: 0.77272

## Results

| Rank | Config | CV mean | CV std | Delta vs current best |
|---:|---|---:|---:|---:|
| 1 | more_min_samples_leaf | 0.84845 | 0.02347 | +0.01120 |
| 2 | larger_trees | 0.84511 | 0.01056 | +0.00786 |
| 3 | strong_l2_regularization | 0.84400 | 0.01523 | +0.00675 |
| 4 | large_tree_regularized | 0.84283 | 0.03051 | +0.00558 |
| 5 | less_min_samples_leaf | 0.84062 | 0.01582 | +0.00337 |
| 6 | slower_learning_more_iters | 0.84061 | 0.01707 | +0.00336 |
| 7 | light_l2_regularization | 0.83948 | 0.01780 | +0.00223 |
| 8 | current_best_reference | 0.83725 | 0.01390 | -0.00000 |
| 9 | faster_learning_fewer_iters | 0.83725 | 0.01296 | -0.00000 |
| 10 | small_tree_l2 | 0.82940 | 0.02181 | -0.00785 |
| 11 | smaller_trees_more_regular | 0.82826 | 0.01918 | -0.00899 |

## Best Local Config

- Name: more_min_samples_leaf
- CV mean: 0.84845
- CV std: 0.02347
- Delta vs current best: +0.01120

## Submission

- Path: `competitions/titanic/submissions/hgb_tuned_submission.csv`

## Competition Interpretation

This is a controlled tuning experiment.

If the best tuned configuration improves CV but only by a tiny amount, public leaderboard may still move down due to noise.

Submission decision should consider:

```text
CV mean
CV std
delta vs current best
prediction difference vs current best submission
```
