# Titanic Auto Experiment Suite Report

## Goal

Run a controlled suite of Titanic experiments:

1. Multiple safe feature sets.
2. Multiple model/parameter configurations.
3. Repeated CV across multiple seeds.
4. Current-best flip guard.
5. Optional Kaggle submission for only threshold-qualified candidates.

## Threshold Logic

```text
min_cv_mean
max_cv_std
min_risk_adjusted
max_total_changed
max_0_to_1
max_pclass3_female_0_to_1
max_miss_mrs_0_to_1
guard_decision != DO_NOT_SUBMIT_WITHOUT_REVIEW
guard_decision != NO_NEW_INFORMATION
```

## Top Results

| experiment_id | feature_set_name | model_name | cv_mean | cv_std | risk_adjusted_score | changed_count | total_0_to_1 | guard_decision | qualifies |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n| auto_008_safe_family_counts_hgb_reference | safe_family_counts | hgb_reference | 0.8387483522691608 | 0.017887414188488486 | 0.8208609380806723 | 13 | 7 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_001_current_best_plus_title_hgb_reference | current_best_plus_title | hgb_reference | 0.8379992885987487 | 0.019405151981620523 | 0.8185941366171282 | 4 | 2 | REVIEW_OK | False |\n| auto_015_safe_family_flags_hgb_reference | safe_family_flags | hgb_reference | 0.8342665243864165 | 0.01739240703962144 | 0.816874117346795 | 11 | 8 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_038_name_structure_family_hgb_large_regular | name_structure_family | hgb_large_regular | 0.8335090912895193 | 0.017217639413260787 | 0.8162914518762585 | 28 | 14 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_029_title_group_family_counts_hgb_reference | title_group_family_counts | hgb_reference | 0.8353775657523068 | 0.019724477960026045 | 0.8156530877922807 | 13 | 7 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_019_safe_family_flags_random_forest_conservative | safe_family_flags | random_forest_conservative | 0.8327516581926223 | 0.01770504969390305 | 0.8150466084987192 | 43 | 29 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_010_safe_family_counts_hgb_large_regular | safe_family_counts | hgb_large_regular | 0.8342686167430376 | 0.019524337414883453 | 0.8147442793281542 | 22 | 10 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_040_name_structure_family_random_forest_conservative | name_structure_family | random_forest_conservative | 0.8338752536982403 | 0.019136614502492162 | 0.8147386391957482 | 45 | 30 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_036_name_structure_family_hgb_reference | name_structure_family | hgb_reference | 0.8357541899441341 | 0.021424811880660487 | 0.8143293780634736 | 19 | 11 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_012_safe_family_counts_random_forest_conservative | safe_family_counts | random_forest_conservative | 0.8312514384951771 | 0.017887922528363125 | 0.813363515966814 | 39 | 28 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_018_safe_family_flags_hgb_conservative | safe_family_flags | hgb_conservative | 0.8338731613416192 | 0.021017972321227325 | 0.8128551890203918 | 29 | 17 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_033_title_group_family_counts_random_forest_conservative | title_group_family_counts | random_forest_conservative | 0.8312577155650409 | 0.018529345557625178 | 0.8127283700074157 | 45 | 29 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_022_safe_family_counts_flags_hgb_reference | safe_family_counts_flags | hgb_reference | 0.8346285020818949 | 0.021964819360798526 | 0.8126636827210963 | 14 | 8 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_026_safe_family_counts_flags_random_forest_conservative | safe_family_counts_flags | random_forest_conservative | 0.8305107442512502 | 0.01838125871982157 | 0.8121294855314286 | 45 | 29 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_017_safe_family_flags_hgb_large_regular | safe_family_flags | hgb_large_regular | 0.8353838428221706 | 0.02391245899280672 | 0.8114713838293639 | 24 | 12 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_009_safe_family_counts_hgb_small_regular | safe_family_counts | hgb_small_regular | 0.830508651894629 | 0.01960286697916623 | 0.8109057849154627 | 16 | 6 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_003_current_best_plus_title_hgb_large_regular | current_best_plus_title | hgb_large_regular | 0.8346347791517587 | 0.024015555409983753 | 0.8106192237417749 | 15 | 8 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_025_safe_family_counts_flags_hgb_conservative | safe_family_counts_flags | hgb_conservative | 0.8312577155650409 | 0.020657949984542994 | 0.810599765580498 | 31 | 19 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_032_title_group_family_counts_hgb_conservative | title_group_family_counts | hgb_conservative | 0.8308831837298349 | 0.02059986609486804 | 0.8102833176349669 | 29 | 17 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |\n| auto_030_title_group_family_counts_hgb_small_regular | title_group_family_counts | hgb_small_regular | 0.829010524553805 | 0.01926484271067005 | 0.8097456818431349 | 16 | 5 | DO_NOT_SUBMIT_WITHOUT_REVIEW | False |

## Qualified Results

_No rows._

## Submitted Experiment IDs

_None._

## Output Files

- `competitions/titanic/reports/auto_experiment_suite/auto_experiment_results.csv`
- `competitions/titanic/reports/auto_experiment_suite/auto_experiment_qualified.csv`
- `competitions/titanic/reports/auto_experiment_suite/auto_experiment_guard_summary.csv`
- `competitions/titanic/reports/auto_experiment_suite/auto_experiment_suite_report.md`
- `competitions/titanic/submissions/auto_experiment_suite`

## Important Warning

This suite can submit to Kaggle if `--submit-qualified` is used.

Do not use that flag until you inspect the generated CSV/report.
