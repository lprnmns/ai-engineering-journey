import pandas as pd

from labs.data.build_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_train_test_split,
    split_features_target,
)


def build_sample_students() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Ali", "Ayse", "Can", "Ece", "Zeynep", "Mehmet", "Deniz", "Ada", "Berk", "Cem"],
            "age": [21.0, 23.0, 21.0, 22.0, 22.0, 20.0, 21.5, 19.0, 24.0, 25.0],
            "study_hours": [5.5, 3.5, 8.0, 2.0, 7.0, 5.0, 4.0, 6.0, 1.0, 9.0],
            "previous_score": [72, 55, 90, 45, 85, 60, 68, 75, 40, 95],
            "passed": [True, False, True, False, True, False, True, True, False, True],
        }
    )


def test_split_features_target_returns_expected_columns() -> None:
    df = build_sample_students()

    X, y = split_features_target(df)

    assert list(X.columns) == FEATURE_COLUMNS
    assert y.name == TARGET_COLUMN


def test_split_features_target_returns_expected_shapes() -> None:
    df = build_sample_students()

    X, y = split_features_target(df)

    assert X.shape == (10, 3)
    assert y.shape == (10,)


def test_build_train_test_split_preserves_total_row_count() -> None:
    df = build_sample_students()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = build_train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
