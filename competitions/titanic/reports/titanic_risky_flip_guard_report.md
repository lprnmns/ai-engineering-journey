# Titanic Risky Flip Guard Report

## Goal

Create a pre-submission guardrail against risky prediction flips.

Current best public submission:

```text
model_comparison_best_submission.csv
public LB = 0.77272
```

Recent failed submissions showed a pattern:

```text
Too many current-best 0 predictions were flipped to 1.
Many risky flips involved Pclass=3, female, Miss/Mrs, and Embarked S/C groups.
```

## Guard Thresholds

```text
MAX_TOTAL_CHANGED_FOR_SAFE_SUBMIT = 12
MAX_RISKY_0_TO_1_FOR_SAFE_SUBMIT = 6
MAX_PCLASS3_FEMALE_0_TO_1_FOR_SAFE_SUBMIT = 3
MAX_MISS_MRS_0_TO_1_FOR_SAFE_SUBMIT = 4
MAX_EMBARKED_SC_0_TO_1_FOR_SAFE_SUBMIT = 6
```

## Guard Summary

| Candidate | Changed | 0->1 | 1->0 | P3 female 0->1 | Miss/Mrs 0->1 | Embarked S/C 0->1 | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| deck_ticket_fare | 20 | 15 | 5 | 9 | 9 | 13 | DO_NOT_SUBMIT_WITHOUT_REVIEW |
| hgb_tuned | 18 | 11 | 7 | 8 | 8 | 7 | DO_NOT_SUBMIT_WITHOUT_REVIEW |
| rare_title_interactions | 11 | 7 | 4 | 6 | 6 | 5 | DO_NOT_SUBMIT_WITHOUT_REVIEW |
| repeated_cv_best | 0 | 0 | 0 | 0 | 0 | 0 | NO_NEW_INFORMATION |

## Top Risky Rows

| Candidate | PassengerId | Pclass | Sex | Title | Embarked | Change |
|---|---:|---:|---|---|---|---|
| deck_ticket_fare | 898 | 3 | female | Miss | Q | 0_to_1 |
| deck_ticket_fare | 910 | 3 | female | Miss | S | 0_to_1 |
| deck_ticket_fare | 925 | 3 | female | Mrs | S | 0_to_1 |
| deck_ticket_fare | 931 | 3 | male | Mr | S | 0_to_1 |
| deck_ticket_fare | 965 | 1 | male | Mr | C | 0_to_1 |
| deck_ticket_fare | 982 | 3 | female | Mrs | S | 0_to_1 |
| deck_ticket_fare | 1045 | 3 | female | Mrs | S | 0_to_1 |
| deck_ticket_fare | 1098 | 3 | female | Miss | Q | 0_to_1 |
| deck_ticket_fare | 1134 | 1 | male | Mr | C | 0_to_1 |
| deck_ticket_fare | 1136 | 3 | male | Master | S | 0_to_1 |
| deck_ticket_fare | 1144 | 1 | male | Mr | C | 0_to_1 |
| deck_ticket_fare | 1223 | 1 | male | Mr | C | 0_to_1 |
| deck_ticket_fare | 1239 | 3 | female | Mrs | C | 0_to_1 |
| deck_ticket_fare | 1251 | 3 | female | Mrs | S | 0_to_1 |
| deck_ticket_fare | 1274 | 3 | female | Mrs | S | 0_to_1 |
| hgb_tuned | 898 | 3 | female | Miss | Q | 0_to_1 |
| hgb_tuned | 911 | 3 | female | Mrs | C | 0_to_1 |
| hgb_tuned | 1045 | 3 | female | Mrs | S | 0_to_1 |
| hgb_tuned | 1094 | 1 | male | Rare | C | 0_to_1 |
| hgb_tuned | 1098 | 3 | female | Miss | Q | 0_to_1 |
| hgb_tuned | 1126 | 1 | male | Mr | C | 0_to_1 |
| hgb_tuned | 1183 | 3 | female | Miss | Q | 0_to_1 |
| hgb_tuned | 1205 | 3 | female | Miss | Q | 0_to_1 |
| hgb_tuned | 1239 | 3 | female | Mrs | C | 0_to_1 |
| hgb_tuned | 1251 | 3 | female | Mrs | S | 0_to_1 |

## Interpretation

This guard does not know the real Kaggle labels.

It is not a replacement for validation.

It is a submission safety check based on observed public leaderboard failures.

## Decision Rule

```text
If decision = DO_NOT_SUBMIT_WITHOUT_REVIEW:
    Do not submit just because CV improved.

If decision = REVIEW_OK:
    Submission may be reasonable, but still inspect changed rows.

If decision = NO_NEW_INFORMATION:
    Submission is identical to current best and should not be sent.
```

## Output Files

- `competitions/titanic/reports/titanic_risky_flip_guard_summary.csv`
- `competitions/titanic/reports/titanic_risky_flip_guard_rows.csv`
