# Titanic Repeated CV Selection Report

## Goal

Select a more reliable HGB configuration using repeated cross-validation.

The previous tuning experiment selected by single-seed CV mean and failed on public leaderboard.

## Current Best Public Reference

- Experiment: `titanic_exp_003_model_comparison_hgb`
- Public LB: `0.77272`

## Validation Setup

```text
seeds = [42, 7, 123, 2026, 99]
folds per seed = 5
total validation scores per config = 25
selection score = repeated_cv_mean - repeated_cv_std
```

## Results

| Rank | Config | Mean | Std | Risk-adjusted | Min | Max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | current_best_reference | 0.83637 | 0.02166 | 0.81471 | 0.80337 | 0.88202 |
| 2 | light_l2_regularization | 0.83547 | 0.02157 | 0.81390 | 0.80337 | 0.88202 |
| 3 | larger_trees | 0.83300 | 0.02014 | 0.81286 | 0.79330 | 0.88202 |
| 4 | faster_learning_fewer_iters | 0.83569 | 0.02314 | 0.81255 | 0.79775 | 0.88764 |
| 5 | strong_l2_regularization | 0.83210 | 0.02051 | 0.81159 | 0.79775 | 0.87079 |
| 6 | slower_learning_more_iters | 0.83277 | 0.02128 | 0.81149 | 0.79888 | 0.87079 |
| 7 | more_min_samples_leaf | 0.83682 | 0.02564 | 0.81118 | 0.77654 | 0.87709 |
| 8 | less_min_samples_leaf | 0.83345 | 0.02269 | 0.81077 | 0.77654 | 0.87079 |
| 9 | large_tree_regularized | 0.83546 | 0.02665 | 0.80881 | 0.79213 | 0.88202 |
| 10 | small_tree_l2 | 0.82985 | 0.02218 | 0.80767 | 0.79213 | 0.87079 |
| 11 | smaller_trees_more_regular | 0.82805 | 0.02187 | 0.80617 | 0.78771 | 0.87079 |

## Best Risk-Adjusted Config

- Name: `current_best_reference`
- Repeated CV mean: `0.83637`
- Repeated CV std: `0.02166`
- Risk-adjusted score: `0.81471`

## Submission

- Path: `competitions/titanic/submissions/repeated_cv_best_submission.csv`

## Competition Interpretation

This experiment tests whether selecting a more stable configuration is safer than selecting the highest single CV mean.

A config with slightly lower mean but much lower std may be better for public/private leaderboard generalization.
