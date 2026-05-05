from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, classification_report  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


RAW_DATA_DIR = Path("competitions/titanic/data/raw")
PROCESSED_DATA_DIR = Path("competitions/titanic/data/processed")
SUBMISSIONS_DIR = Path("competitions/titanic/submissions")
REPORT_PATH = Path("competitions/titanic/reports/titanic_baseline_report.md")

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_baseline.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_baseline.csv"
TEST_RAW_PATH = RAW_DATA_DIR / "test.csv"

SUBMISSION_PATH = SUBMISSIONS_DIR / "baseline_logreg_submission.csv"


@dataclass
class TitanicBaselineResult:
    validation_accuracy: float
    train_rows: int
    validation_rows: int
    test_rows: int
    report_text: str


def load_processed_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    X = pd.read_csv(X_TRAIN_PATH)
    y = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]
    X_test = pd.read_csv(X_TEST_PATH)
    test_raw = pd.read_csv(TEST_RAW_PATH)

    return X, y, X_test, test_raw


def build_model() -> Any:
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
    )


def evaluate_local_validation(
    X: pd.DataFrame,
    y: pd.Series,
) -> TitanicBaselineResult:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    validation_predictions = model.predict(X_valid)

    validation_accuracy = float(accuracy_score(y_valid, validation_predictions))
    report_text = str(
        classification_report(
            y_valid,
            validation_predictions,
            zero_division=0,
        )
    )

    return TitanicBaselineResult(
        validation_accuracy=validation_accuracy,
        train_rows=len(X_train),
        validation_rows=len(X_valid),
        test_rows=0,
        report_text=report_text,
    )


def train_final_model_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
) -> list[int]:
    model = build_model()
    model.fit(X, y)

    predictions_raw = model.predict(X_test)

    return [int(value) for value in predictions_raw]


def build_submission(
    test_raw: pd.DataFrame,
    predictions: list[int],
) -> pd.DataFrame:
    submission = pd.DataFrame(
        {
            "PassengerId": test_raw["PassengerId"],
            "Survived": predictions,
        }
    )

    return submission


def validate_submission(submission: pd.DataFrame, expected_rows: int) -> None:
    expected_columns = ["PassengerId", "Survived"]

    if list(submission.columns) != expected_columns:
        raise ValueError(
            f"Submission columns must be {expected_columns}, got {list(submission.columns)}"
        )

    if len(submission) != expected_rows:
        raise ValueError(
            f"Submission must have {expected_rows} rows, got {len(submission)}"
        )

    invalid_values = set(submission["Survived"].unique()) - {0, 1}
    if invalid_values:
        raise ValueError(f"Survived contains invalid values: {invalid_values}")


def save_submission(submission: pd.DataFrame, path: Path = SUBMISSION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)


def build_report(result: TitanicBaselineResult, submission_path: Path) -> str:
    return f"""# Titanic Baseline Model Report

## Model

- Algorithm: Logistic Regression
- Features: preprocessed baseline features from W4D3
- Validation split: 80/20 stratified split
- Metric: Accuracy

## Local Validation

- Train rows: {result.train_rows}
- Validation rows: {result.validation_rows}
- Validation accuracy: {result.validation_accuracy:.4f}

## Classification Report

```text
{result.report_text}
```

## Submission

- Submission path: `{submission_path}`
- Expected columns: `PassengerId`, `Survived`
- Expected test rows: 418

## Notes

This is the first simple baseline.

The purpose is not to maximize leaderboard score yet.
The purpose is to create a valid end-to-end Kaggle workflow:

```text
preprocess -> train -> validate -> predict test -> create submission -> submit to Kaggle
```

Next improvements may include:

- title extraction from Name
- FamilySize feature
- IsAlone feature
- HasCabin feature
- better Age imputation
- model comparison with RandomForest / CatBoost / LightGBM
"""


def save_report(report: str, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run_pipeline() -> TitanicBaselineResult:
    X, y, X_test, test_raw = load_processed_data()

    local_result = evaluate_local_validation(X, y)

    predictions = train_final_model_and_predict(X, y, X_test)
    submission = build_submission(test_raw, predictions)
    validate_submission(submission, expected_rows=len(test_raw))
    save_submission(submission)

    final_result = TitanicBaselineResult(
        validation_accuracy=local_result.validation_accuracy,
        train_rows=local_result.train_rows,
        validation_rows=local_result.validation_rows,
        test_rows=len(X_test),
        report_text=local_result.report_text,
    )

    report = build_report(final_result, SUBMISSION_PATH)
    save_report(report)

    return final_result


def main() -> None:
    result = run_pipeline()

    print("=== Titanic Baseline Logistic Regression ===")
    print(f"Validation accuracy: {result.validation_accuracy:.4f}")
    print(f"Train rows: {result.train_rows}")
    print(f"Validation rows: {result.validation_rows}")
    print(f"Test rows: {result.test_rows}")
    print()
    print(f"Saved submission to: {SUBMISSION_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
