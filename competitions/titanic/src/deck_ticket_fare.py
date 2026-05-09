from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from competitions.titanic.src.feature_v1 import add_feature_v1_columns
from competitions.titanic.src.model_comparison import (
    build_submission,
    validate_submission,
)


RAW_DATA_DIR = Path("competitions/titanic/data/raw")
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

REPORT_PATH = Path("competitions/titanic/reports/titanic_deck_ticket_fare_report.md")
RESULTS_PATH = Path("competitions/titanic/reports/titanic_deck_ticket_fare_results.csv")
SUBMISSION_PATH = Path("competitions/titanic/submissions/deck_ticket_fare_hgb_submission.csv")

TARGET_COLUMN = "Survived"

NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "TicketGroupSize",
    "FarePerPerson",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "Deck",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class DeckTicketFareResult:
    model_name: str
    cv_scores: list[float]
    cv_mean: float
    cv_std: float
    baseline_reference: str


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    return train_raw, test_raw


def extract_deck(cabin: object) -> str:
    if cabin is None:
        return "Unknown"

    cabin_text = str(cabin).strip()
    if cabin_text == "" or cabin_text.lower() == "nan" or cabin_text == "<NA>":
        return "Unknown"

    return cabin_text[0].upper()


def normalize_ticket(ticket: object) -> str:
    if ticket is None:
        return "UNKNOWN"

    ticket_text = str(ticket).strip()
    if ticket_text == "" or ticket_text.lower() == "nan" or ticket_text == "<NA>":
        return "UNKNOWN"

    return ticket_text


def add_deck_ticket_fare_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_featured = add_feature_v1_columns(train_raw)
    test_featured = add_feature_v1_columns(test_raw)

    train_featured = train_featured.copy()
    test_featured = test_featured.copy()

    train_featured["_dataset"] = "train"
    test_featured["_dataset"] = "test"

    combined = pd.concat(
        [train_featured, test_featured],
        ignore_index=True,
        sort=False,
    )

    combined["Deck"] = combined["Cabin"].apply(extract_deck)
    combined["TicketNormalized"] = combined["Ticket"].apply(normalize_ticket)

    ticket_counts = combined["TicketNormalized"].value_counts()
    combined["TicketGroupSize"] = combined["TicketNormalized"].map(ticket_counts).astype(int)

    safe_ticket_group_size = combined["TicketGroupSize"].replace(0, 1)
    combined["FarePerPerson"] = combined["Fare"] / safe_ticket_group_size

    train_result = combined[combined["_dataset"] == "train"].copy()
    test_result = combined[combined["_dataset"] == "test"].copy()

    train_result = train_result.drop(columns=["_dataset"])
    test_result = test_result.drop(columns=["_dataset"])

    return train_result, test_result


def build_model_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def run_cross_validation(
    train_df: pd.DataFrame,
    n_splits: int = 5,
) -> DeckTicketFareResult:
    X = train_df[FEATURES].copy()
    y = train_df[TARGET_COLUMN].copy()

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    scores: list[float] = []

    for train_index, valid_index in cv.split(X, y):
        X_train = X.iloc[train_index].copy()
        X_valid = X.iloc[valid_index].copy()
        y_train = y.iloc[train_index].copy()
        y_valid = y.iloc[valid_index].copy()

        pipeline = build_model_pipeline()
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_valid)
        score = float(accuracy_score(y_valid, predictions))
        scores.append(score)

    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    std_score = variance ** 0.5

    return DeckTicketFareResult(
        model_name="hist_gradient_boosting",
        cv_scores=scores,
        cv_mean=mean_score,
        cv_std=std_score,
        baseline_reference="exp_003 plus_title hgb: CV 0.83725, public LB 0.77272",
    )


def train_final_and_create_submission(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    X = train_df[FEATURES].copy()
    y = train_df[TARGET_COLUMN].copy()
    X_test = test_df[FEATURES].copy()

    pipeline = build_model_pipeline()
    pipeline.fit(X, y)

    predictions = [int(value) for value in pipeline.predict(X_test)]

    submission = build_submission(
        test_raw=test_df,
        predictions=predictions,
    )
    validate_submission(submission, expected_rows=len(test_df))

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    return submission


def results_to_dataframe(result: DeckTicketFareResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": "titanic_exp_005_deck_ticket_fare",
                "model_name": result.model_name,
                "cv_mean": result.cv_mean,
                "cv_std": result.cv_std,
                "fold_scores": ", ".join(f"{score:.5f}" for score in result.cv_scores),
                "baseline_reference": result.baseline_reference,
            }
        ]
    )


def save_results(result: DeckTicketFareResult) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_to_dataframe(result).to_csv(RESULTS_PATH, index=False)


def build_report(result: DeckTicketFareResult) -> str:
    code_fence = chr(96) * 3

    fold_lines = []
    for index, score in enumerate(result.cv_scores, start=1):
        fold_lines.append(f"- Fold {index}: {score:.5f}")

    return f"""# Titanic Deck + Ticket + Fare Report

## Hypothesis

The current best model may be missing Titanic-specific group and location signals.

This experiment adds:

{code_fence}text
Deck = first letter of Cabin
TicketGroupSize = number of passengers sharing the same Ticket
FarePerPerson = Fare / TicketGroupSize
{code_fence}

## Why These Features?

### Deck

Cabin location may proxy ship location and passenger class.

### TicketGroupSize

Passengers sharing a ticket may have travelled together.

### FarePerPerson

Fare can represent a group ticket price, so dividing by ticket group size may create a cleaner economic signal.

## Model

- Model: HistGradientBoostingClassifier
- Feature set: plus_title + Deck + TicketGroupSize + FarePerPerson
- Metric: accuracy
- Validation: 5-fold StratifiedKFold

## Results

{chr(10).join(fold_lines)}

- CV mean: {result.cv_mean:.5f}
- CV std: {result.cv_std:.5f}

## Baseline Reference

{result.baseline_reference}

## Submission

- Path: `{SUBMISSION_PATH}`

## Competition Interpretation

This experiment checks whether domain-specific Titanic features move us closer to the estimated 12 additional correct predictions needed for 0.80 public LB.

Important:

TicketGroupSize is computed from train+test features only, not target labels. This is targetless feature engineering, not label leakage.
"""


def save_report(result: DeckTicketFareResult) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(result), encoding="utf-8")


def run_pipeline() -> DeckTicketFareResult:
    train_raw, test_raw = load_raw_data()

    train_df, test_df = add_deck_ticket_fare_features(train_raw, test_raw)

    result = run_cross_validation(train_df)
    save_results(result)
    save_report(result)
    train_final_and_create_submission(train_df, test_df)

    return result


def main() -> None:
    result = run_pipeline()

    print("=== Titanic Deck + Ticket + Fare ===")
    print(f"Model: {result.model_name}")
    print(f"CV mean: {result.cv_mean:.5f}")
    print(f"CV std: {result.cv_std:.5f}")
    print(f"Saved results to: {RESULTS_PATH}")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Saved submission to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
