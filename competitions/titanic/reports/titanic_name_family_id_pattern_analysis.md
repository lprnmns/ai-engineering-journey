# Titanic Name, Family, and PassengerId Pattern Analysis

## Goal

Explore three possible pattern sources:

1. Safe surname/family features that do not use target labels directly.
2. PassengerId jump segments based on large survival-rate changes across ID ranges.
3. Name/title tokens, including raw title and first given token.

## Safe Family Feature Candidates

These features are safe because they are based on train+test structure/counts, not on target mean survival.

Examples:

```text
SurnameGroupSize
SurnamePclassGroupSize
SurnameTicketGroupSize
IsSoloSurname
IsLargeSurnameGroup
```

### Family Feature Survival Analysis

| feature | value | passenger_count | survived_count | survival_rate |
| --- | --- | --- | --- | --- |
| SurnameGroupSize | 1 | 447 | 149.0 | 0.3333333333333333 |
| SurnameGroupSize | 2 | 168 | 80.0 | 0.47619047619047616 |
| SurnameGroupSize | 3 | 121 | 63.0 | 0.5206611570247934 |
| SurnameGroupSize | 4 | 62 | 24.0 | 0.3870967741935484 |
| SurnameGroupSize | 6 | 45 | 13.0 | 0.28888888888888886 |
| SurnameGroupSize | 5 | 19 | 7.0 | 0.3684210526315789 |
| SurnameGroupSize | 11 | 16 | 2.0 | 0.125 |
| SurnameGroupSize | 8 | 10 | 3.0 | 0.3 |
| SurnameGroupSize | 7 | 3 | 1.0 | 0.3333333333333333 |
| SurnamePclassGroupSize | 1 | 473 | 159.0 | 0.3361522198731501 |
| SurnamePclassGroupSize | 2 | 174 | 85.0 | 0.4885057471264368 |
| SurnamePclassGroupSize | 3 | 113 | 60.0 | 0.5309734513274337 |
| SurnamePclassGroupSize | 4 | 63 | 25.0 | 0.3968253968253968 |
| SurnamePclassGroupSize | 6 | 31 | 5.0 | 0.16129032258064516 |
| SurnamePclassGroupSize | 11 | 16 | 2.0 | 0.125 |
| SurnamePclassGroupSize | 5 | 11 | 3.0 | 0.2727272727272727 |
| SurnamePclassGroupSize | 8 | 10 | 3.0 | 0.3 |
| SurnameTicketGroupSize | 1 | 577 | 183.0 | 0.317157712305026 |
| SurnameTicketGroupSize | 2 | 155 | 85.0 | 0.5483870967741935 |
| SurnameTicketGroupSize | 3 | 75 | 49.0 | 0.6533333333333333 |
| SurnameTicketGroupSize | 4 | 25 | 18.0 | 0.72 |
| SurnameTicketGroupSize | 6 | 21 | 2.0 | 0.09523809523809523 |
| SurnameTicketGroupSize | 5 | 14 | 2.0 | 0.14285714285714285 |
| SurnameTicketGroupSize | 7 | 11 | 3.0 | 0.2727272727272727 |
| SurnameTicketGroupSize | 11 | 7 | 0.0 | 0.0 |
| SurnameTicketGroupSize | 8 | 6 | 0.0 | 0.0 |
| IsSoloSurname | 1 | 447 | 149.0 | 0.3333333333333333 |
| IsSoloSurname | 0 | 444 | 193.0 | 0.4346846846846847 |
| IsSmallSurnameGroup | 0 | 602 | 199.0 | 0.33056478405315615 |
| IsSmallSurnameGroup | 1 | 289 | 143.0 | 0.49480968858131485 |
| IsLargeSurnameGroup | 0 | 736 | 292.0 | 0.3967391304347826 |
| IsLargeSurnameGroup | 1 | 155 | 50.0 | 0.3225806451612903 |
| IsLargeSurnamePclassGroup | 0 | 760 | 304.0 | 0.4 |
| IsLargeSurnamePclassGroup | 1 | 131 | 38.0 | 0.2900763358778626 |

## PassengerId Jump Segments

PassengerId may reveal dataset-order patterns, but it is risky as a model feature.

These segments are exploratory. They are created by first splitting PassengerId into 20 small bins, then placing a boundary where adjacent bin survival rates jump by at least 0.12.

| PassengerIdJumpSegment | passenger_count | start_id | end_id | survived_count | survival_rate | method |
| --- | --- | --- | --- | --- | --- | --- |
| 1-90 | 90 | 1 | 90 | 39 | 0.43333333333333335 | 20 base bins, boundary when adjacent survival delta >= 0.12 |
| 91-179 | 89 | 91 | 179 | 20 | 0.2247191011235955 | 20 base bins, boundary when adjacent survival delta >= 0.12 |
| 180-268 | 89 | 180 | 268 | 34 | 0.38202247191011235 | 20 base bins, boundary when adjacent survival delta >= 0.12 |
| 269-535 | 267 | 269 | 535 | 114 | 0.42696629213483145 | 20 base bins, boundary when adjacent survival delta >= 0.12 |
| 536-891 | 356 | 536 | 891 | 135 | 0.3792134831460674 | 20 base bins, boundary when adjacent survival delta >= 0.12 |

## Raw Title Analysis

RawTitle is the title/status text between comma and period in the Titanic name.

Example:

```text
"Carter, Rev. Ernest Courtenay" -> RawTitle = Rev
```

| RawTitle | passenger_count | survived_count | survival_rate |
| --- | --- | --- | --- |
| Mr | 517 | 81 | 0.15667311411992263 |
| Miss | 182 | 127 | 0.6978021978021978 |
| Mrs | 125 | 99 | 0.792 |
| Master | 40 | 23 | 0.575 |
| Dr | 7 | 3 | 0.42857142857142855 |
| Rev | 6 | 0 | 0.0 |
| Mlle | 2 | 2 | 1.0 |
| Col | 2 | 1 | 0.5 |
| Major | 2 | 1 | 0.5 |
| Lady | 1 | 1 | 1.0 |
| Mme | 1 | 1 | 1.0 |
| Ms | 1 | 1 | 1.0 |
| Sir | 1 | 1 | 1.0 |
| the Countess | 1 | 1 | 1.0 |
| Capt | 1 | 0 | 0.0 |
| Don | 1 | 0 | 0.0 |
| Jonkheer | 1 | 0 | 0.0 |

## Title Group Analysis

Raw titles are mapped into broader status groups.

| TitleGroup | passenger_count | survived_count | survival_rate |
| --- | --- | --- | --- |
| CommonMale | 517 | 81 | 0.15667311411992263 |
| CommonFemale | 311 | 230 | 0.7395498392282959 |
| ChildMale | 40 | 23 | 0.575 |
| Professional | 7 | 3 | 0.42857142857142855 |
| Clergy | 6 | 0 | 0.0 |
| Military | 5 | 2 | 0.4 |
| NobleMale | 3 | 1 | 0.3333333333333333 |
| NobleFemale | 1 | 1 | 1.0 |
| RareOther | 1 | 1 | 1.0 |

## First Given Token Analysis

This extracts the first token after the title.

Example:

```text
"Carter, Rev. Ernest Courtenay" -> FirstGivenToken = Ernest
```

This is exploratory and can be noisy. Use it for pattern discovery, not direct modeling unless validated carefully.

| FirstGivenToken | passenger_count | survived_count | survival_rate |
| --- | --- | --- | --- |
| William | 48 | 16 | 0.3333333333333333 |
| John | 31 | 8 | 0.25806451612903225 |
| Thomas | 19 | 7 | 0.3684210526315789 |
| Charles | 16 | 6 | 0.375 |
| George | 16 | 6 | 0.375 |
| Henry | 15 | 6 | 0.4 |
| James | 15 | 5 | 0.3333333333333333 |
| Edward | 13 | 6 | 0.46153846153846156 |
| Frederick | 13 | 5 | 0.38461538461538464 |
| Mary | 11 | 7 | 0.6363636363636364 |
| Richard | 11 | 3 | 0.2727272727272727 |
| Johan | 10 | 1 | 0.1 |
| Anna | 9 | 9 | 1.0 |
| Samuel | 9 | 4 | 0.4444444444444444 |
| Karl | 9 | 3 | 0.3333333333333333 |
| Joseph | 9 | 1 | 0.1111111111111111 |
| Elizabeth | 8 | 8 | 1.0 |
| Albert | 7 | 4 | 0.5714285714285714 |
| Arthur | 7 | 1 | 0.14285714285714285 |
| Alfred | 7 | 0 | 0.0 |
| Margaret | 6 | 6 | 1.0 |
| Peter | 6 | 2 | 0.3333333333333333 |
| Victor | 6 | 2 | 0.3333333333333333 |
| Robert | 6 | 1 | 0.16666666666666666 |
| Ernest | 6 | 0 | 0.0 |
| Alice | 5 | 4 | 0.8 |
| Frank | 5 | 3 | 0.6 |
| Harry | 5 | 2 | 0.4 |
| Walter | 5 | 2 | 0.4 |
| Alexander | 5 | 1 | 0.2 |
| Ernst | 5 | 1 | 0.2 |
| Bertha | 4 | 4 | 1.0 |
| Ellen | 4 | 3 | 0.75 |
| Helen | 4 | 3 | 0.75 |
| Sidney | 4 | 2 | 0.5 |
| August | 4 | 1 | 0.25 |
| Benjamin | 4 | 1 | 0.25 |
| Ivan | 4 | 1 | 0.25 |
| Martin | 4 | 1 | 0.25 |
| Percival | 4 | 1 | 0.25 |
| Daniel | 4 | 0 | 0.0 |
| David | 4 | 0 | 0.0 |
| Hans | 4 | 0 | 0.0 |
| Nils | 4 | 0 | 0.0 |
| Patrick | 4 | 0 | 0.0 |
| Amelia | 3 | 3 | 1.0 |
| Carl | 3 | 3 | 1.0 |
| Kate | 3 | 3 | 1.0 |
| Katherine | 3 | 3 | 1.0 |
| Annie | 3 | 2 | 0.6666666666666666 |

## Strong Warning

Do not directly use survival_rate columns as model features.

That would use the answer key from train labels and can become target leakage.

Safer next modeling candidates:

```text
SurnameGroupSize
SurnamePclassGroupSize
SurnameTicketGroupSize
TitleGroup
HasParenthesesName
HasQuotedNickname
```

Riskier candidates:

```text
PassengerIdJumpSegment
FirstGivenToken
Any target mean survival by Surname/Title/Token
```
