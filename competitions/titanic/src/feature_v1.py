from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, classification_report  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


RAW_DATA_DIR = Path("competitions/titanic/data/raw")
PROCESSED_DATA_DIR = Path("competitions/titanic/data/processed")
REPORT_PATH = Path("competitions/titanic/reports/titanic_feature_v1_report.md")
SUBMISSION_PATH = Path("competitions/titanic/submissions/feature_v1_logreg_submission.csv")

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

X_TRAIN_V1_PATH = PROCESSED_DATA_DIR / "X_train_feature_v1.csv"
Y_TRAIN_V1_PATH = PROCESSED_DATA_DIR / "y_train_feature_v1.csv"
X_TEST_V1_PATH = PROCESSED_DATA_DIR / "X_test_feature_v1.csv"

TARGET_COLUMN = "Survived"

NUMERIC_FEATURES_V1 = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "HasCabin",
]

CATEGORICAL_FEATURES_V1 = [
    "Sex",
    "Embarked",
    "Title",
]

MODEL_FEATURES_V1 = NUMERIC_FEATURES_V1 + CATEGORICAL_FEATURES_V1


@dataclass(frozen=True)
class FeatureV1Result:
    single_split_accuracy: float
    cv_scores: list[float]
    cv_mean: float
    cv_std: float
    train_rows: int
    validation_rows: int
    test_rows: int
    report_text: str


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", name)
    if match is None:
        return "Unknown"

    title = match.group(1).strip()

    title_mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }

    normalized_title = title_mapping.get(title, title)

    common_titles = {"Mr", "Mrs", "Miss", "Master"}
    if normalized_title not in common_titles:
        return "Rare"

    return normalized_title


def add_feature_v1_columns(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    featured["FamilySize"] = featured["SibSp"] + featured["Parch"] + 1
    featured["IsAlone"] = (featured["FamilySize"] == 1).astype(int)
    featured["HasCabin"] = featured["Cabin"].notna().astype(int)
    featured["Title"] = featured["Name"].astype(str).apply(extract_title)

    return featured


def split_features_target(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    X = train_df[MODEL_FEATURES_V1].copy()
    y = train_df[TARGET_COLUMN].copy()

    return X, y


def select_test_features(test_df: pd.DataFrame) -> pd.DataFrame:
    return test_df[MODEL_FEATURES_V1].copy()


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
            ("numeric", numeric_pipeline, NUMERIC_FEATURES_V1),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES_V1),
        ],
        remainder="drop",
    )


def build_model_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> tuple[list[float], float, float]:
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    model_pipeline = build_model_pipeline()

    raw_scores = cross_val_score(
        model_pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    scores = [float(score) for score in raw_scores]
    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    std_score = variance ** 0.5

    return scores, mean_score, std_score


def evaluate_single_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[float, str, int, int]:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_pipeline = build_model_pipeline()
    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_valid)

    accuracy = float(accuracy_score(y_valid, predictions))
    report_text = str(
        classification_report(
            y_valid,
            predictions,
            zero_division=0,
        )
    )

    return accuracy, report_text, len(X_train), len(X_valid)


def save_processed_feature_tables(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X.to_csv(X_TRAIN_V1_PATH, index=False)
    y.to_csv(Y_TRAIN_V1_PATH, index=False)
    X_test.to_csv(X_TEST_V1_PATH, index=False)


def train_final_model_and_create_submission(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> pd.DataFrame:
    model_pipeline = build_model_pipeline()
    model_pipeline.fit(X, y)

    predictions = [int(value) for value in model_pipeline.predict(X_test)]

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


def save_submission(submission: pd.DataFrame) -> None:
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)


def build_report(result: FeatureV1Result) -> str:
    fold_lines = []
    for index, score in enumerate(result.cv_scores, start=1):
        fold_lines.append(f"- Fold {index}: {score:.5f}")

    return f"""# Titanic Feature Engineering v1 Report

## Feature Set

Baseline features plus:

- FamilySize = SibSp + Parch + 1
- IsAlone = FamilySize == 1
- HasCabin = Cabin is not missing
- Title extracted from Name

## Model

- Algorithm: Logistic Regression
- Preprocessing:
  - numeric median imputation
  - categorical most-frequent imputation
  - one-hot encoding
- Metric: Accuracy

## Single Split Validation

- Train rows: {result.train_rows}
- Validation rows: {result.validation_rows}
- Single split accuracy: {result.single_split_accuracy:.5f}

## Cross-Validation

{chr(10).join(fold_lines)}

- CV mean accuracy: {result.cv_mean:.5f}
- CV std accuracy: {result.cv_std:.5f}

## Classification Report

```text
{result.report_text}
```

## Submission

- Path: `{SUBMISSION_PATH}`
- Test rows: {result.test_rows}

## Competition Thinking

This experiment tests whether simple human-designed Titanic features improve local validation.

The most important comparison is not only against the previous single split score.

We compare:

```text
exp_001 baseline CV / public LB
vs
exp_002 feature_v1 CV / public LB
```

If CV improves but public leaderboard drops, the feature may be overfitting local validation or public leaderboard noise may be involved.
"""


def save_report(report: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_pipeline() -> FeatureV1Result:
    train_raw, test_raw = load_raw_data()

    train_featured = add_feature_v1_columns(train_raw)
    test_featured = add_feature_v1_columns(test_raw)

    X, y = split_features_target(train_featured)
    X_test = select_test_features(test_featured)

    save_processed_feature_tables(X, y, X_test)

    single_split_accuracy, report_text, train_rows, validation_rows = evaluate_single_split(X, y)
    cv_scores, cv_mean, cv_std = run_cross_validation(X, y)

    submission = train_final_model_and_create_submission(X, y, X_test, test_raw)
    validate_submission(submission, expected_rows=len(test_raw))
    save_submission(submission)

    result = FeatureV1Result(
        single_split_accuracy=single_split_accuracy,
        cv_scores=cv_scores,
        cv_mean=cv_mean,
        cv_std=cv_std,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=len(X_test),
        report_text=report_text,
    )

    save_report(build_report(result))

    return result


def main() -> None:
    result = run_pipeline()

    print("=== Titanic Feature Engineering v1 ===")
    print(f"Single split accuracy: {result.single_split_accuracy:.5f}")
    print(f"CV mean accuracy: {result.cv_mean:.5f}")
    print(f"CV std accuracy: {result.cv_std:.5f}")
    print(f"Test rows: {result.test_rows}")
    print()
    print(f"Saved submission to: {SUBMISSION_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
