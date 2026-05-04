# W3 EDA Report — Students Dataset

## Dataset Overview

- Rows: 7
- Columns: 5
- Columns list: name, age, study_hours, previous_score, passed

## Missing Values

```text
name              0
age               0
study_hours       0
previous_score    0
passed            0
```

## Target Distribution

The target column is `passed`.

- Pass rate: 57.14%
- Failed rate: 42.86%

## Average Study Hours by Target

```text
passed
False    3.500
True     6.125
```

## Average Previous Score by Target

```text
passed
False    53.333333
True     78.750000
```

## Top Student by Previous Score

- Name: Can
- Previous score: 90
- Study hours: 8.0
- Passed: True

## Initial Insights

1. Students who passed studied more hours on average.
2. Students who passed also had a higher previous score on average.
3. The cleaned dataset has no missing values.
4. The dataset is tiny, so these insights are not statistically strong.
5. This dataset is ready for a small baseline ML experiment.

## Modeling Readiness

This dataset can be used for a toy classification task:

```text
features: age, study_hours, previous_score
target: passed
```

The next step is to build a simple baseline model.
