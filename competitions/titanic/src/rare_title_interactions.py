from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

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
from competitions.titanic.src.model_comparison import build_submission, validate_submission


RAW_DATA_DIR = Path("competitions/titanic/data/raw")
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

REPORT_PATH = Path("competitions/titanic/reports/titanic_rare_title_interactions_report.md")
RESULTS_PATH = Path("competitions/titanic/reports/titanic_rare_title_interactions_results.csv")
SUBMISSION_PATH = Path("competitions/titanic/submissions/rare_title_interactions_hgb_submission.csv")

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
    "TitleV2",
    "TitleV2_Pclass",
    "Sex_Pclass",
    "Embarked_Pclass",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CURRENT_BEST_PUBLIC_LB = 0.77272
CURRENT_BEST_CV_MEAN = 0.83725


@dataclass(frozen=True)
class RareTitleInteractionResult:
    model_name: str
    cv_scores: list[float]
    cv_mean: float
    cv_std: float
    delta_vs_current_best: float


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    return train_raw, test_raw


def extract_raw_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", name)
    if match is None:
        return "Unknown"

    return match.group(1).strip()


def map_title_v2(raw_title: str) -> str:
    normalized = raw_title.strip()

    direct_mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }

    normalized = direct_mapping.get(normalized, normalized)

    if normalized in {"Mr", "Mrs", "Miss", "Master"}:
        return normalized

    if normalized == "Rev":
        return "Clergy"

    if normalized == "Dr":
        return "Professional"

    if normalized in {"Col", "Major", "Capt"}:
        return "Military"

    if normalized in {"Don", "Sir", "Jonkheer"}:
        return "NobleMale"

    if normalized in {"Lady", "Countess", "Dona"}:
        return "NobleFemale"

    return "RareOther"


def add_rare_title_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    result = add_feature_v1_columns(df)

    result["RawTitle"] = result["Name"].astype(str).apply(extract_raw_title)
    result["TitleV2"] = result["RawTitle"].apply(map_title_v2)

    result["TitleV2_Pclass"] = result["TitleV2"].astype(str) + "_P" + result["Pclass"].astype(str)
    result["Sex_Pclass"] = result["Sex"].astype(str) + "_P" + result["Pclass"].astype(str)
    result["Embarked_Pclass"] = result["Embarked"].astype(str) + "_P" + result["Pclass"].astype(str)

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
        min_samples_leaf=20,
        l2_regularization=0.0,
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
) -> RareTitleInteractionResult:
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

    return RareTitleInteractionResult(
        model_name="hist_gradient_boosting",
        cv_scores=scores,
        cv_mean=mean_score,
        cv_std=std_score,
        delta_vs_current_best=mean_score - CURRENT_BEST_CV_MEAN,
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


def results_to_dataframe(result: RareTitleInteractionResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": "titanic_exp_008_rare_title_interactions",
                "model_name": result.model_name,
                "cv_mean": result.cv_mean,
                "cv_std": result.cv_std,
                "delta_vs_current_best": result.delta_vs_current_best,
                "fold_scores": ", ".join(f"{score:.5f}" for score in result.cv_scores),
                "current_best_cv_reference": CURRENT_BEST_CV_MEAN,
                "current_best_public_lb_reference": CURRENT_BEST_PUBLIC_LB,
            }
        ]
    )


def save_results(result: RareTitleInteractionResult) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_to_dataframe(result).to_csv(RESULTS_PATH, index=False)


def build_report(result: RareTitleInteractionResult, train_df: pd.DataFrame) -> str:
    code_fence = chr(96) * 3

    title_counts = train_df["TitleV2"].value_counts().to_dict()
    title_count_lines = [
        f"- {title}: {count}"
        for title, count in title_counts.items()
    ]

    fold_lines = [
        f"- Fold {index}: {score:.5f}"
        for index, score in enumerate(result.cv_scores, start=1)
    ]

    return f"""# Titanic Rare Title + Interaction Features Report

## Hypothesis

The visual error analysis showed that the `Rare` title bucket has the highest OOF error rate.

The previous title logic compressed many different social roles into one label:

{code_fence}text
Dr, Rev, Major, Col, Capt, Countess, Lady, Sir, Don, Jonkheer, etc. → Rare
{code_fence}

This experiment splits rare titles into more meaningful groups and adds interaction features.

## Added Features

{code_fence}text
TitleV2
TitleV2_Pclass
Sex_Pclass
Embarked_Pclass
{code_fence}

## TitleV2 Counts

{chr(10).join(title_count_lines)}

## Model

- Model: HistGradientBoostingClassifier
- Feature set: plus_title replacement with TitleV2 + interaction features
- Metric: accuracy
- Validation: 5-fold StratifiedKFold

## Results

{chr(10).join(fold_lines)}

- CV mean: {result.cv_mean:.5f}
- CV std: {result.cv_std:.5f}
- Delta vs current best CV: {result.delta_vs_current_best:+.5f}

## Current Best Reference

- CV mean: {CURRENT_BEST_CV_MEAN:.5f}
- Public LB: {CURRENT_BEST_PUBLIC_LB:.5f}

## Submission

- Path: `{SUBMISSION_PATH}`

## Competition Interpretation

This experiment is directly driven by visual error analysis.

Decision rule:

{code_fence}text
If CV improves or prediction difference is meaningful, consider Kaggle submission.
If CV drops and predictions barely change, treat as negative feature engineering.
{code_fence}
"""


def save_report(result: RareTitleInteractionResult, train_df: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(result, train_df), encoding="utf-8")


def run_pipeline() -> RareTitleInteractionResult:
    train_raw, test_raw = load_raw_data()

    train_df = add_rare_title_interaction_features(train_raw)
    test_df = add_rare_title_interaction_features(test_raw)

    result = run_cross_validation(train_df)
    save_results(result)
    save_report(result, train_df)
    train_final_and_create_submission(train_df, test_df)

    return result


def main() -> None:
    result = run_pipeline()

    print("=== Titanic Rare Title + Interactions ===")
    print(f"Model: {result.model_name}")
    print(f"CV mean: {result.cv_mean:.5f}")
    print(f"CV std: {result.cv_std:.5f}")
    print(f"Delta vs current best: {result.delta_vs_current_best:+.5f}")
    print(f"Saved results to: {RESULTS_PATH}")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Saved submission to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
