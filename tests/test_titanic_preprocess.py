import pandas as pd

from competitions.titanic.src.preprocess import (
    BASELINE_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    preprocess_features,
    select_test_features,
    split_train_features_target,
)


def build_sample_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Survived": [0, 1, 1],
            "Pclass": [3, 1, 3],
            "Name": ["A", "B", "C"],
            "Sex": ["male", "female", "female"],
            "Age": [22.0, 38.0, None],
            "SibSp": [1, 1, 0],
            "Parch": [0, 0, 0],
            "Ticket": ["A/5", "PC", "STON"],
            "Fare": [7.25, 71.28, 7.92],
            "Cabin": [None, "C85", None],
            "Embarked": ["S", "C", None],
        }
    )


def build_sample_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [4, 5],
            "Pclass": [3, 2],
            "Name": ["D", "E"],
            "Sex": ["male", "female"],
            "Age": [35.0, None],
            "SibSp": [0, 1],
            "Parch": [0, 0],
            "Ticket": ["A", "B"],
            "Fare": [8.05, None],
            "Cabin": [None, None],
            "Embarked": ["S", "Q"],
        }
    )


def test_baseline_features_are_expected() -> None:
    assert NUMERIC_FEATURES == [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
    ]
    assert CATEGORICAL_FEATURES == [
        "Sex",
        "Embarked",
    ]
    assert BASELINE_FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_split_train_features_target() -> None:
    train_df = build_sample_train()

    X_train, y_train = split_train_features_target(train_df)

    assert list(X_train.columns) == BASELINE_FEATURES
    assert y_train.name == TARGET_COLUMN
    assert X_train.shape == (3, 7)
    assert y_train.shape == (3,)


def test_select_test_features() -> None:
    test_df = build_sample_test()

    X_test = select_test_features(test_df)

    assert list(X_test.columns) == BASELINE_FEATURES
    assert X_test.shape == (2, 7)


def test_preprocess_features_removes_missing_values() -> None:
    train_df = build_sample_train()
    test_df = build_sample_test()

    X_train_raw, _ = split_train_features_target(train_df)
    X_test_raw = select_test_features(test_df)

    X_train_processed, X_test_processed = preprocess_features(
        X_train_raw,
        X_test_raw,
    )

    assert X_train_processed.isna().sum().sum() == 0
    assert X_test_processed.isna().sum().sum() == 0
    assert X_train_processed.shape[0] == 3
    assert X_test_processed.shape[0] == 2
