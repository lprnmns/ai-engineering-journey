from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold, cross_val_score  # type: ignore[import-untyped]
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

REPORT_PATH = Path("competitions/titanic/reports/titanic_hgb_tuning_report.md")
RESULTS_PATH = Path("competitions/titanic/reports/titanic_hgb_tuning_results.csv")
SUBMISSION_PATH = Path("competitions/titanic/submissions/hgb_tuned_submission.csv")

TARGET_COLUMN = "Survived"

NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CURRENT_BEST_CV_MEAN = 0.83725
CURRENT_BEST_PUBLIC_LB = 0.77272


@dataclass(frozen=True)
class HgbConfig:
    name: str
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float


@dataclass(frozen=True)
class HgbTuningResult:
    config: HgbConfig
    cv_scores: list[float]
    cv_mean: float
    cv_std: float
    delta_vs_current_best: float


def load_train_test() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    train_featured = add_feature_v1_columns(train_raw)
    test_featured = add_feature_v1_columns(test_raw)

    X = train_featured[FEATURES].copy()
    y = train_featured[TARGET_COLUMN].copy()
    X_test = test_featured[FEATURES].copy()

    return X, y, X_test, test_raw


def build_configs() -> list[HgbConfig]:
    return [
        HgbConfig(
            name="current_best_reference",
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.0,
        ),
        HgbConfig(
            name="smaller_trees_more_regular",
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=0.0,
        ),
        HgbConfig(
            name="slower_learning_more_iters",
            max_iter=200,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.0,
        ),
        HgbConfig(
            name="faster_learning_fewer_iters",
            max_iter=70,
            learning_rate=0.08,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.0,
        ),
        HgbConfig(
            name="larger_trees",
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
        ),
        HgbConfig(
            name="more_min_samples_leaf",
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=0.0,
        ),
        HgbConfig(
            name="less_min_samples_leaf",
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=0.0,
        ),
        HgbConfig(
            name="light_l2_regularization",
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.1,
        ),
        HgbConfig(
            name="strong_l2_regularization",
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
        ),
        HgbConfig(
            name="small_tree_l2",
            max_iter=150,
            learning_rate=0.04,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=0.1,
        ),
        HgbConfig(
            name="large_tree_regularized",
            max_iter=150,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=0.1,
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
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(config: HgbConfig) -> Pipeline:
    model = HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def evaluate_config(
    X: pd.DataFrame,
    y: pd.Series,
    config: HgbConfig,
    n_splits: int = 5,
) -> HgbTuningResult:
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    pipeline = build_pipeline(config)

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

    return HgbTuningResult(
        config=config,
        cv_scores=scores,
        cv_mean=mean_score,
        cv_std=std_score,
        delta_vs_current_best=mean_score - CURRENT_BEST_CV_MEAN,
    )


def run_tuning() -> list[HgbTuningResult]:
    X, y, _, _ = load_train_test()

    results = [
        evaluate_config(X, y, config)
        for config in build_configs()
    ]

    return sorted(results, key=lambda result: result.cv_mean, reverse=True)


def results_to_dataframe(results: list[HgbTuningResult]) -> pd.DataFrame:
    rows = []

    for result in results:
        config = result.config
        rows.append(
            {
                "config_name": config.name,
                "cv_mean": result.cv_mean,
                "cv_std": result.cv_std,
                "delta_vs_current_best": result.delta_vs_current_best,
                "max_iter": config.max_iter,
                "learning_rate": config.learning_rate,
                "max_leaf_nodes": config.max_leaf_nodes,
                "min_samples_leaf": config.min_samples_leaf,
                "l2_regularization": config.l2_regularization,
                "fold_scores": ", ".join(f"{score:.5f}" for score in result.cv_scores),
            }
        )

    return pd.DataFrame(rows)


def save_results_csv(results: list[HgbTuningResult]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_to_dataframe(results).to_csv(RESULTS_PATH, index=False)


def train_best_and_save_submission(best_config: HgbConfig) -> None:
    X, y, X_test, test_raw = load_train_test()

    pipeline = build_pipeline(best_config)
    pipeline.fit(X, y)

    predictions = [int(value) for value in pipeline.predict(X_test)]

    submission = build_submission(
        test_raw=test_raw,
        predictions=predictions,
    )
    validate_submission(submission, expected_rows=len(test_raw))

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)


def build_report(results: list[HgbTuningResult]) -> str:
    table_lines = [
        "| Rank | Config | CV mean | CV std | Delta vs current best |",
        "|---:|---|---:|---:|---:|",
    ]

    for rank, result in enumerate(results, start=1):
        table_lines.append(
            f"| {rank} | {result.config.name} | {result.cv_mean:.5f} | "
            f"{result.cv_std:.5f} | {result.delta_vs_current_best:+.5f} |"
        )

    best = results[0]
    code_fence = chr(96) * 3

    return f"""# Titanic HGB Tuning Report

## Goal

Tune HistGradientBoostingClassifier on the current best honest feature set.

Feature set is fixed:

{code_fence}text
Pclass, Age, SibSp, Parch, Fare, Sex, Embarked, Title
{code_fence}

Only model hyperparameters change.

## Current Best Reference

- CV mean: {CURRENT_BEST_CV_MEAN:.5f}
- Public LB: {CURRENT_BEST_PUBLIC_LB:.5f}

## Results

{chr(10).join(table_lines)}

## Best Local Config

- Name: {best.config.name}
- CV mean: {best.cv_mean:.5f}
- CV std: {best.cv_std:.5f}
- Delta vs current best: {best.delta_vs_current_best:+.5f}

## Submission

- Path: `{SUBMISSION_PATH}`

## Competition Interpretation

This is a controlled tuning experiment.

If the best tuned configuration improves CV but only by a tiny amount, public leaderboard may still move down due to noise.

Submission decision should consider:

{code_fence}text
CV mean
CV std
delta vs current best
prediction difference vs current best submission
{code_fence}
"""


def save_report(results: list[HgbTuningResult]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(results), encoding="utf-8")


def run_pipeline() -> list[HgbTuningResult]:
    results = run_tuning()
    save_results_csv(results)
    save_report(results)
    train_best_and_save_submission(results[0].config)

    return results


def main() -> None:
    results = run_pipeline()

    print("=== Titanic HGB Tuning ===")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result.config.name} | "
            f"cv_mean={result.cv_mean:.5f} | "
            f"cv_std={result.cv_std:.5f} | "
            f"delta={result.delta_vs_current_best:+.5f}"
        )

    print()
    print(f"Best config: {results[0].config.name}")
    print(f"Saved results to: {RESULTS_PATH}")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Saved submission to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
