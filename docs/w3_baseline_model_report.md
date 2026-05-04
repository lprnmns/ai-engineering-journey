# W3 Baseline Model Report — Logistic Regression

## Model

- Algorithm: Logistic Regression
- Task: Binary classification
- Target: `passed`
- Features: `age`, `study_hours`, `previous_score`

## Dataset Split

- Train rows: 4
- Test rows: 3

## Metrics

- Accuracy: 100.00%

## Classification Report

```text
              precision    recall  f1-score   support

       False       1.00      1.00      1.00         1
        True       1.00      1.00      1.00         2

    accuracy                           1.00         3
   macro avg       1.00      1.00      1.00         3
weighted avg       1.00      1.00      1.00         3

```

## Predictions

- Row 1: actual=False, predicted=False
- Row 2: actual=True, predicted=True
- Row 3: actual=True, predicted=True

## Notes

This is a tiny toy dataset, so the metric is not statistically reliable.

The goal is to build the first end-to-end ML pipeline:

```text
clean data -> feature dataset -> train model -> evaluate model -> report result
```

This baseline gives us a reference point before trying larger datasets or more advanced models.
