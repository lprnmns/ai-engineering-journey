# Titanic Submission Difference Analysis

## Goal

Understand why recent submissions damaged public leaderboard score.

We compare each weaker submission against the current best submission:

```text
current best = model_comparison_best_submission.csv
public LB = 0.77272
```

The key question:

```text
Which passengers did the weaker models change?
What do those changed passengers have in common?
```

## Candidate Difference Summary

| Candidate | Changed rows | Changed rate | Best 1->Candidate 0 | Best 0->Candidate 1 |
|---|---:|---:|---:|---:|
| deck_ticket_fare | 20 | 4.78% | 5 | 15 |
| hgb_tuned | 18 | 4.31% | 7 | 11 |
| rare_title_interactions | 11 | 2.63% | 4 | 7 |

## Top Changed Segments

| Candidate | Segment | Value | Changed rows |
|---|---|---|---:|
| deck_ticket_fare | change_direction | 0_to_1 | 15 |
| deck_ticket_fare | Pclass | 3 | 13 |
| deck_ticket_fare | Sex | female | 11 |
| deck_ticket_fare | Embarked | S | 10 |
| deck_ticket_fare | Sex | male | 9 |
| deck_ticket_fare | Embarked | C | 8 |
| deck_ticket_fare | Title | Mr | 8 |
| deck_ticket_fare | Pclass | 1 | 7 |
| hgb_tuned | Pclass | 3 | 12 |
| hgb_tuned | change_direction | 0_to_1 | 11 |
| hgb_tuned | Sex | female | 10 |
| hgb_tuned | Embarked | C | 8 |
| hgb_tuned | Sex | male | 8 |
| hgb_tuned | change_direction | 1_to_0 | 7 |
| hgb_tuned | Embarked | S | 6 |
| hgb_tuned | Pclass | 1 | 6 |
| rare_title_interactions | Pclass | 3 | 8 |
| rare_title_interactions | Sex | female | 8 |
| rare_title_interactions | change_direction | 0_to_1 | 7 |
| rare_title_interactions | Embarked | S | 5 |
| rare_title_interactions | Embarked | C | 4 |
| rare_title_interactions | Title | Miss | 4 |
| rare_title_interactions | Title | Mrs | 4 |
| rare_title_interactions | change_direction | 1_to_0 | 4 |

## Interpretation

If a candidate changes many predictions and public LB drops, those changes are probably moving in the wrong direction.

This analysis does not reveal true Kaggle labels.

It helps identify dangerous feature/model changes that are repeatedly flipping similar passenger groups.

## Next Decision Rule

Do not trust a new experiment only because CV improves.

Before submitting, compare changed rows against current best:

```text
small meaningful change + stable CV = possible submission
large unstable change + CV/public mismatch history = high risk
```

## Output Files

- `competitions/titanic/reports/titanic_submission_difference_rows.csv`
- `competitions/titanic/reports/titanic_submission_difference_segments.csv`
