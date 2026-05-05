from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold, cross_val_score  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


PROCESSED_DATA_DIR = Path("competitions/titanic/data/processed")
REPORT_PATH = Path("competitions/titanic/reports/titanic_cross_validation_report.md")

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_baseline.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"


@dataclass(frozen=True)
class CrossValidationResult:
    scores: list[float]
    mean_score: float
    std_score: float
    n_splits: int
    metric: str


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(X_TRAIN_PATH)
    y = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]

    return X, y


def build_model() -> Any:
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
    )


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> CrossValidationResult:
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    model = build_model()

    raw_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    scores = [float(score) for score in raw_scores]

    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    std_score = variance ** 0.5

    return CrossValidationResult(
        scores=scores,
        mean_score=mean_score,
        std_score=std_score,
        n_splits=n_splits,
        metric="accuracy",
    )


def build_report(result: CrossValidationResult, public_score: float | None = None) -> str:
    fold_lines = []
    for index, score in enumerate(result.scores, start=1):
        fold_lines.append(f"- Fold {index}: {score:.5f}")

    public_score_text = "Not submitted yet"
    cv_lb_gap_text = "N/A"

    if public_score is not None:
        public_score_text = f"{public_score:.5f}"
        cv_lb_gap_text = f"{result.mean_score - public_score:.5f}"

    return f"""# Titanic Cross-Validation Report

## Model

- Algorithm: Logistic Regression
- Feature set: W4D3 baseline preprocessed features
- Validation: StratifiedKFold
- Number of folds: {result.n_splits}
- Metric: {result.metric}

## Fold Scores

{chr(10).join(fold_lines)}

## Summary

- CV mean accuracy: {result.mean_score:.5f}
- CV std accuracy: {result.std_score:.5f}
- Kaggle public score: {public_score_text}
- CV minus public score gap: {cv_lb_gap_text}

## Interpretation

A single train/validation split can be noisy.

Cross-validation gives a more stable local estimate by evaluating the model on multiple validation folds.

The important competition question is not only:

"Is my model good?"

It is also:

"Does my local validation score correlate with Kaggle public/private leaderboard?"

## Next Improvements

- Add FamilySize feature
- Add IsAlone feature
- Extract Title from Name
- Add HasCabin feature
- Compare LogisticRegression, RandomForest, and gradient boosting models
"""


def save_report(report: str, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run_pipeline(public_score: float | None = 0.76555) -> CrossValidationResult:
    X, y = load_training_data()
    result = run_cross_validation(X, y)

    report = build_report(result, public_score=public_score)
    save_report(report)

    return result


def main() -> None:
    result = run_pipeline(public_score=0.76555)

    print("=== Titanic Cross-Validation ===")
    for index, score in enumerate(result.scores, start=1):
        print(f"Fold {index}: {score:.5f}")

    print()
    print(f"CV mean accuracy: {result.mean_score:.5f}")
    print(f"CV std accuracy: {result.std_score:.5f}")
    print("Kaggle public score: 0.76555")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
