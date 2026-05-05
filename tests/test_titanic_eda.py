import pandas as pd

from competitions.titanic.src.eda import (
    TARGET_COLUMN,
    build_eda_report,
    get_categorical_columns,
    get_missing_values,
    get_numeric_columns,
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
            "Embarked": ["S", "C", "S"],
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


def build_sample_submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [4, 5],
            "Survived": [0, 1],
        }
    )


def test_target_column_is_survived() -> None:
    assert TARGET_COLUMN == "Survived"


def test_get_missing_values_counts_nulls() -> None:
    train_df = build_sample_train()

    missing = get_missing_values(train_df)

    assert missing["Age"] == 1
    assert missing["Cabin"] == 2


def test_column_type_helpers_detect_numeric_and_categorical_columns() -> None:
    train_df = build_sample_train()

    numeric_columns = get_numeric_columns(train_df)
    categorical_columns = get_categorical_columns(train_df)

    assert "Age" in numeric_columns
    assert "Fare" in numeric_columns
    assert "Name" in categorical_columns
    assert "Sex" in categorical_columns


def test_build_eda_report_contains_core_sections() -> None:
    report = build_eda_report(
        build_sample_train(),
        build_sample_test(),
        build_sample_submission(),
    )

    assert "# Titanic EDA Report" in report
    assert "Survival rate" in report
    assert "Missing Values" in report
    assert "Train/Test Feature Alignment" in report
