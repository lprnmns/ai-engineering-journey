from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast

import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


RAW_DATA_DIR = Path("competitions/titanic/data/raw")
PROCESSED_DATA_DIR = Path("competitions/titanic/data/processed")

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_baseline.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_baseline.csv"

TARGET_COLUMN = "Survived"
ID_COLUMN = "PassengerId"

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
]

BASELINE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def split_train_features_target(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    X_train = train_df[BASELINE_FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    return X_train, y_train


def select_test_features(test_df: pd.DataFrame) -> pd.DataFrame:
    return test_df[BASELINE_FEATURES].copy()


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

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    raw_feature_names = preprocessor.get_feature_names_out()
    return [str(feature_name) for feature_name in raw_feature_names]


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preprocessor = build_preprocessor()

    X_train_array = preprocessor.fit_transform(X_train)
    X_test_array = preprocessor.transform(X_test)

    feature_names = get_output_feature_names(preprocessor)

    X_train_processed = pd.DataFrame(
        cast(Any, X_train_array),
        columns=feature_names,
        index=X_train.index,
    )
    X_test_processed = pd.DataFrame(
        cast(Any, X_test_array),
        columns=feature_names,
        index=X_test.index,
    )

    return X_train_processed, X_test_processed


def save_processed_data(
    X_train_processed: pd.DataFrame,
    y_train: pd.Series,
    X_test_processed: pd.DataFrame,
) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_train_processed.to_csv(X_TRAIN_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    X_test_processed.to_csv(X_TEST_PATH, index=False)


def run_preprocessing() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_df, test_df = load_raw_data()

    X_train_raw, y_train = split_train_features_target(train_df)
    X_test_raw = select_test_features(test_df)

    X_train_processed, X_test_processed = preprocess_features(
        X_train_raw,
        X_test_raw,
    )

    save_processed_data(
        X_train_processed,
        y_train,
        X_test_processed,
    )

    return X_train_processed, y_train, X_test_processed


def main() -> None:
    X_train_processed, y_train, X_test_processed = run_preprocessing()

    print("=== Baseline features ===")
    print(BASELINE_FEATURES)
    print()

    print("=== Processed shapes ===")
    print("X_train:", X_train_processed.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test_processed.shape)
    print()

    print("=== Processed columns ===")
    print(list(X_train_processed.columns))
    print()

    print("=== Missing values after preprocessing ===")
    print("X_train missing:", int(X_train_processed.isna().sum().sum()))
    print("X_test missing:", int(X_test_processed.isna().sum().sum()))
    print()

    print("Saved:")
    print("-", X_TRAIN_PATH)
    print("-", Y_TRAIN_PATH)
    print("-", X_TEST_PATH)


if __name__ == "__main__":
    main()
