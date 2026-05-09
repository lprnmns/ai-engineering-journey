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

REPORT_PATH = Path("competitions/titanic/reports/titanic_age_imputation_v2_report.md")
RESULTS_PATH = Path("competitions/titanic/reports/titanic_age_imputation_v2_results.csv")
SUBMISSION_PATH = Path("competitions/titanic/submissions/age_imputation_v2_hgb_submission.csv")

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
    "AgeBin",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
AGE_GROUP_COLUMNS = ["Title", "Pclass", "Sex"]


@dataclass(frozen=True)
class GroupAgeStats:
    group_medians: pd.Series
    title_medians: pd.Series
    global_median: float


@dataclass(frozen=True)
class AgeImputationResult:
    model_name: str
    cv_scores: list[float]
    cv_mean: float
    cv_std: float
    baseline_reference: str


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    return train_raw, test_raw


def prepare_base_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = add_feature_v1_columns(df)

    return featured


def fit_group_age_stats(train_df: pd.DataFrame) -> GroupAgeStats:
    known_age = train_df.dropna(subset=["Age"])

    group_medians = known_age.groupby(AGE_GROUP_COLUMNS)["Age"].median()
    title_medians = known_age.groupby("Title")["Age"].median()
    global_median = float(known_age["Age"].median())

    return GroupAgeStats(
        group_medians=group_medians,
        title_medians=title_medians,
        global_median=global_median,
    )


def lookup_group_age(row: pd.Series, stats: GroupAgeStats) -> float:
    group_key = (
        str(row["Title"]),
        int(row["Pclass"]),
        str(row["Sex"]),
    )

    if group_key in stats.group_medians.index:
        return float(stats.group_medians.loc[group_key])

    title = str(row["Title"])
    if title in stats.title_medians.index:
        return float(stats.title_medians.loc[title])

    return stats.global_median


def apply_group_age_imputation(
    df: pd.DataFrame,
    stats: GroupAgeStats,
) -> pd.DataFrame:
    result = df.copy()

    missing_age_mask = result["Age"].isna()
    if missing_age_mask.any():
        result.loc[missing_age_mask, "Age"] = result.loc[missing_age_mask].apply(
            lambda row: lookup_group_age(row, stats),
            axis=1,
        )

    return result


def add_age_bin(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    bins = [-1.0, 12.0, 18.0, 60.0, 200.0]
    labels = ["child", "teen", "adult", "senior"]

    result["AgeBin"] = pd.cut(
        result["Age"],
        bins=bins,
        labels=labels,
    ).astype(str)

    return result


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
) -> AgeImputationResult:
    y = train_df[TARGET_COLUMN].copy()

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    scores: list[float] = []

    for train_index, valid_index in cv.split(train_df, y):
        fold_train = train_df.iloc[train_index].copy()
        fold_valid = train_df.iloc[valid_index].copy()

        y_train = fold_train[TARGET_COLUMN].copy()
        y_valid = fold_valid[TARGET_COLUMN].copy()

        age_stats = fit_group_age_stats(fold_train)

        fold_train_features = apply_group_age_imputation(fold_train, age_stats)
        fold_valid_features = apply_group_age_imputation(fold_valid, age_stats)

        fold_train_features = add_age_bin(fold_train_features)
        fold_valid_features = add_age_bin(fold_valid_features)

        X_train = fold_train_features[FEATURES].copy()
        X_valid = fold_valid_features[FEATURES].copy()

        pipeline = build_model_pipeline()
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_valid)
        score = float(accuracy_score(y_valid, predictions))
        scores.append(score)

    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    std_score = variance ** 0.5

    return AgeImputationResult(
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
    age_stats = fit_group_age_stats(train_df)

    train_imputed = apply_group_age_imputation(train_df, age_stats)
    test_imputed = apply_group_age_imputation(test_df, age_stats)

    train_imputed = add_age_bin(train_imputed)
    test_imputed = add_age_bin(test_imputed)

    X = train_imputed[FEATURES].copy()
    y = train_imputed[TARGET_COLUMN].copy()
    X_test = test_imputed[FEATURES].copy()

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


def results_to_dataframe(result: AgeImputationResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": "titanic_exp_004_age_imputation_v2",
                "model_name": result.model_name,
                "cv_mean": result.cv_mean,
                "cv_std": result.cv_std,
                "fold_scores": ", ".join(f"{score:.5f}" for score in result.cv_scores),
                "baseline_reference": result.baseline_reference,
            }
        ]
    )


def save_results(result: AgeImputationResult) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_to_dataframe(result).to_csv(RESULTS_PATH, index=False)


def build_report(result: AgeImputationResult) -> str:
    code_fence = chr(96) * 3

    fold_lines = []
    for index, score in enumerate(result.cv_scores, start=1):
        fold_lines.append(f"- Fold {index}: {score:.5f}")

    return f"""# Titanic Age Imputation v2 Report

## Hypothesis

Global median Age imputation may be too crude.

Instead, missing Age values are filled by:

{code_fence}text
Title + Pclass + Sex group median
fallback: Title median
fallback: global median
{code_fence}

Then `AgeBin` is added:

{code_fence}text
child / teen / adult / senior
{code_fence}

## Model

- Model: HistGradientBoostingClassifier
- Feature set: plus_title + AgeBin
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

This experiment tests whether smarter Age handling can recover some of the estimated 12 missing correct predictions needed to reach 0.80 public LB.

Important:

If CV improves but public LB drops, this may mean the Age strategy overfits local validation.
"""


def save_report(result: AgeImputationResult) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(result), encoding="utf-8")


def run_pipeline() -> AgeImputationResult:
    train_raw, test_raw = load_raw_data()

    train_df = prepare_base_features(train_raw)
    test_df = prepare_base_features(test_raw)

    result = run_cross_validation(train_df)
    save_results(result)
    save_report(result)
    train_final_and_create_submission(train_df, test_df)

    return result


def main() -> None:
    result = run_pipeline()

    print("=== Titanic Age Imputation v2 ===")
    print(f"Model: {result.model_name}")
    print(f"CV mean: {result.cv_mean:.5f}")
    print(f"CV std: {result.cv_std:.5f}")
    print(f"Saved results to: {RESULTS_PATH}")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Saved submission to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
