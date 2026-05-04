from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, classification_report  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")

REPORT_PATH = Path("docs/w3_baseline_model_report.md")


@dataclass
class BaselineModelResult:
    accuracy: float
    train_rows: int
    test_rows: int
    predictions: list[bool]
    actuals: list[bool]
    report_text: str


def load_feature_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]
    y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]

    return X_train, X_test, y_train, y_test


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_rows: int,
) -> BaselineModelResult:
    predictions_raw = model.predict(X_test)

    predictions = [bool(value) for value in predictions_raw]
    actuals = [bool(value) for value in y_test.tolist()]

    accuracy = float(accuracy_score(y_test, predictions_raw))
    report_text = str(classification_report(y_test, predictions_raw, zero_division=0))

    return BaselineModelResult(
        accuracy=accuracy,
        train_rows=train_rows,
        test_rows=len(X_test),
        predictions=predictions,
        actuals=actuals,
        report_text=report_text,
    )


def build_report(result: BaselineModelResult) -> str:
    prediction_lines = []
    for index, (actual, predicted) in enumerate(
        zip(result.actuals, result.predictions),
        start=1,
    ):
        prediction_lines.append(
            f"- Row {index}: actual={actual}, predicted={predicted}"
        )

    predictions_text = "\n".join(prediction_lines)

    return f"""# W3 Baseline Model Report — Logistic Regression

## Model

- Algorithm: Logistic Regression
- Task: Binary classification
- Target: `passed`
- Features: `age`, `study_hours`, `previous_score`

## Dataset Split

- Train rows: {result.train_rows}
- Test rows: {result.test_rows}

## Metrics

- Accuracy: {result.accuracy:.2%}

## Classification Report

```text
{result.report_text}
```

## Predictions

{predictions_text}

## Notes

This is a tiny toy dataset, so the metric is not statistically reliable.

The goal is to build the first end-to-end ML pipeline:

```text
clean data -> feature dataset -> train model -> evaluate model -> report result
```

This baseline gives us a reference point before trying larger datasets or more advanced models.
"""


def save_report(report: str, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run_pipeline() -> BaselineModelResult:
    X_train, X_test, y_train, y_test = load_feature_dataset()

    model = train_baseline_model(X_train, y_train)
    result = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        train_rows=len(X_train),
    )

    report = build_report(result)
    save_report(report)

    return result


def main() -> None:
    result = run_pipeline()

    print("=== Baseline Logistic Regression ===")
    print(f"Accuracy: {result.accuracy:.2%}")
    print(f"Train rows: {result.train_rows}")
    print(f"Test rows: {result.test_rows}")
    print()

    print("Predictions:")
    for actual, predicted in zip(result.actuals, result.predictions):
        print(f"actual={actual}, predicted={predicted}")

    print()
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
