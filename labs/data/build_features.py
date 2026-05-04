from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from labs.data.clean_students import PROCESSED_DATA_PATH


if TYPE_CHECKING:
    SeriesAny: TypeAlias = pd.Series[Any]
else:
    SeriesAny = pd.Series


FEATURE_COLUMNS = ["age", "study_hours", "previous_score"]
TARGET_COLUMN = "passed"

FEATURES_TRAIN_PATH = Path("data/processed/X_train.csv")
FEATURES_TEST_PATH = Path("data/processed/X_test.csv")
TARGET_TRAIN_PATH = Path("data/processed/y_train.csv")
TARGET_TEST_PATH = Path("data/processed/y_test.csv")


def load_clean_students(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, SeriesAny]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def build_train_test_split(
    X: pd.DataFrame,
    y: SeriesAny,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, SeriesAny, SeriesAny]:
    split_result = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_result

    return (
        cast(pd.DataFrame, X_train_raw),
        cast(pd.DataFrame, X_test_raw),
        cast(SeriesAny, y_train_raw),
        cast(SeriesAny, y_test_raw),
    )


def save_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: SeriesAny,
    y_test: SeriesAny,
) -> None:
    FEATURES_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(FEATURES_TRAIN_PATH, index=False)
    X_test.to_csv(FEATURES_TEST_PATH, index=False)
    y_train.to_csv(TARGET_TRAIN_PATH, index=False)
    y_test.to_csv(TARGET_TEST_PATH, index=False)


def main() -> None:
    df = load_clean_students()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = build_train_test_split(X, y)

    save_split(X_train, X_test, y_train, y_test)

    print("=== Feature columns ===")
    print(FEATURE_COLUMNS)
    print()

    print("=== Target column ===")
    print(TARGET_COLUMN)
    print()

    print("=== Shapes ===")
    print("X:", X.shape)
    print("y:", y.shape)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    print()

    print("Saved:")
    print("-", FEATURES_TRAIN_PATH)
    print("-", FEATURES_TEST_PATH)
    print("-", TARGET_TRAIN_PATH)
    print("-", TARGET_TEST_PATH)


if __name__ == "__main__":
    main()
