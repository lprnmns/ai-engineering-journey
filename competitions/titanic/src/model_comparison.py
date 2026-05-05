from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold, cross_val_score  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from competitions.titanic.src.feature_v1 import add_feature_v1_columns


RAW_DATA_DIR = Path("competitions/titanic/data/raw")
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

REPORT_PATH = Path("competitions/titanic/reports/titanic_model_comparison_report.md")
RESULTS_PATH = Path("competitions/titanic/reports/titanic_model_comparison_results.csv")
SUBMISSION_PATH = Path("competitions/titanic/submissions/model_comparison_best_submission.csv")

TARGET_COLUMN = "Survived"

PLUS_TITLE_NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]

PLUS_TITLE_CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
]

PLUS_TITLE_FEATURES = PLUS_TITLE_NUMERIC_FEATURES + PLUS_TITLE_CATEGORICAL_FEATURES


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    hypothesis: str


@dataclass(frozen=True)
class ModelComparisonResult:
    model_name: str
    hypothesis: str
    cv_scores: list[float]
    cv_mean: float
    cv_std: float


def load_train_test() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    train_featured = add_feature_v1_columns(train_raw)
    test_featured = add_feature_v1_columns(test_raw)

    X = train_featured[PLUS_TITLE_FEATURES].copy()
    y = train_featured[TARGET_COLUMN].copy()
    X_test = test_featured[PLUS_TITLE_FEATURES].copy()

    return X, y, X_test, test_raw


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            hypothesis="Linear baseline. Strong if features are mostly linearly separable.",
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
            hypothesis="Tree ensemble can capture non-linear feature interactions.",
        ),
        ModelSpec(
            name="hist_gradient_boosting",
            estimator=HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.05,
                max_leaf_nodes=15,
                random_state=42,
            ),
            hypothesis="Boosting may improve tabular performance by sequentially correcting errors.",
        ),
    ]


def build_preprocessor() -> ColumnTransformer:
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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, PLUS_TITLE_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, PLUS_TITLE_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    spec: ModelSpec,
    n_splits: int = 5,
) -> ModelComparisonResult:
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    pipeline = build_pipeline(spec.estimator)

    raw_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    scores = [float(score) for score in raw_scores]
    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    std_score = variance ** 0.5

    return ModelComparisonResult(
        model_name=spec.name,
        hypothesis=spec.hypothesis,
        cv_scores=scores,
        cv_mean=mean_score,
        cv_std=std_score,
    )


def run_model_comparison() -> list[ModelComparisonResult]:
    X, y, _, _ = load_train_test()

    results = [
        evaluate_model(X, y, spec)
        for spec in build_model_specs()
    ]

    return sorted(results, key=lambda result: result.cv_mean, reverse=True)


def results_to_dataframe(results: list[ModelComparisonResult]) -> pd.DataFrame:
    rows = []

    for result in results:
        rows.append(
            {
                "model_name": result.model_name,
                "cv_mean": result.cv_mean,
                "cv_std": result.cv_std,
                "hypothesis": result.hypothesis,
                "fold_scores": ", ".join(f"{score:.5f}" for score in result.cv_scores),
            }
        )

    return pd.DataFrame(rows)


def save_results_csv(results: list[ModelComparisonResult]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_to_dataframe(results).to_csv(RESULTS_PATH, index=False)


def find_model_spec(model_name: str) -> ModelSpec:
    for spec in build_model_specs():
        if spec.name == model_name:
            return spec

    raise ValueError(f"Unknown model name: {model_name}")


def build_submission(
    test_raw: pd.DataFrame,
    predictions: list[int],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": test_raw["PassengerId"],
            "Survived": predictions,
        }
    )


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


def train_best_model_and_save_submission(best_model_name: str) -> None:
    X, y, X_test, test_raw = load_train_test()
    spec = find_model_spec(best_model_name)

    pipeline = build_pipeline(spec.estimator)
    pipeline.fit(X, y)

    predictions = [int(value) for value in pipeline.predict(X_test)]
    submission = build_submission(test_raw, predictions)

    validate_submission(submission, expected_rows=len(test_raw))

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)


def build_report(results: list[ModelComparisonResult]) -> str:
    code_fence = chr(96) * 3

    table_lines = [
        "| Rank | Model | CV mean | CV std |",
        "|---:|---|---:|---:|",
    ]

    for rank, result in enumerate(results, start=1):
        table_lines.append(
            f"| {rank} | {result.model_name} | {result.cv_mean:.5f} | {result.cv_std:.5f} |"
        )

    best = results[0]

    numeric_text = ", ".join(PLUS_TITLE_NUMERIC_FEATURES)
    categorical_text = ", ".join(PLUS_TITLE_CATEGORICAL_FEATURES)

    return f"""# Titanic Model Comparison Report

## Goal

Compare multiple models on the same `plus_title` feature set.

The feature set is fixed. Only the model changes.

## Feature Set

Numeric:

{numeric_text}

Categorical:

{categorical_text}

## Validation Setup

- Metric: accuracy
- Validation: 5-fold StratifiedKFold
- Random state: 42

## Results

{chr(10).join(table_lines)}

## Best Local Model

- Model: {best.model_name}
- CV mean: {best.cv_mean:.5f}
- CV std: {best.cv_std:.5f}

## Submission

- Best-model submission path: `{SUBMISSION_PATH}`

## Competition Thinking

This is a controlled experiment:

{code_fence}text
same features
same CV
different model
{code_fence}

If a model improves CV but fails on public leaderboard, we do not blindly trust it.
We compare CV, public LB, and experiment notes together.
"""


def save_report(results: list[ModelComparisonResult]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(results), encoding="utf-8")


def run_pipeline() -> list[ModelComparisonResult]:
    results = run_model_comparison()
    save_results_csv(results)
    save_report(results)
    train_best_model_and_save_submission(results[0].model_name)

    return results


def main() -> None:
    results = run_pipeline()

    print("=== Titanic Model Comparison ===")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result.model_name} | "
            f"cv_mean={result.cv_mean:.5f} | "
            f"cv_std={result.cv_std:.5f}"
        )

    print()
    print(f"Best model: {results[0].model_name}")
    print(f"Saved CSV to: {RESULTS_PATH}")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Saved submission to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
