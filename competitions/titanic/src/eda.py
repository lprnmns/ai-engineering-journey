from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


RAW_DATA_DIR = Path("competitions/titanic/data/raw")
REPORT_PATH = Path("competitions/titanic/reports/titanic_eda_report.md")

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
SUBMISSION_PATH = RAW_DATA_DIR / "gender_submission.csv"

TARGET_COLUMN = "Survived"


def load_titanic_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    submission_df = pd.read_csv(SUBMISSION_PATH)

    return train_df, test_df, submission_df


def get_missing_values(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum().sort_values(ascending=False)


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=["number"]).columns)


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=["object", "category", "bool"]).columns)


def build_eda_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    submission_df: pd.DataFrame,
) -> str:
    train_rows, train_columns = train_df.shape
    test_rows, test_columns = test_df.shape
    submission_rows, submission_columns = submission_df.shape

    train_missing = get_missing_values(train_df)
    test_missing = get_missing_values(test_df)

    numeric_columns = get_numeric_columns(train_df)
    categorical_columns = get_categorical_columns(train_df)

    target_distribution = train_df[TARGET_COLUMN].value_counts().sort_index()
    target_rate = float(train_df[TARGET_COLUMN].mean())

    train_feature_columns = set(train_df.columns) - {TARGET_COLUMN}
    test_feature_columns = set(test_df.columns)
    missing_in_test = sorted(train_feature_columns - test_feature_columns)
    extra_in_test = sorted(test_feature_columns - train_feature_columns)

    report = f"""# Titanic EDA Report

## Dataset Shapes

- Train shape: {train_rows} rows, {train_columns} columns
- Test shape: {test_rows} rows, {test_columns} columns
- Submission shape: {submission_rows} rows, {submission_columns} columns

## Target

Target column: {TARGET_COLUMN}

Target distribution:

{target_distribution.to_string()}

Survival rate: {target_rate:.2%}

## Columns

Train columns:

{", ".join(train_df.columns)}

Numeric columns:

{", ".join(numeric_columns)}

Categorical columns:

{", ".join(categorical_columns)}

## Missing Values — Train

{train_missing.to_string()}

## Missing Values — Test

{test_missing.to_string()}

## Train/Test Feature Alignment

Columns present in train features but missing in test:

{missing_in_test}

Columns present in test but not train features:

{extra_in_test}

## Initial Competition Thinking

1. Survived is the target and exists only in train.csv.
2. test.csv does not contain Survived, so it must be predicted.
3. PassengerId is required for submission but should not be treated as a meaningful predictive signal by default.
4. Cabin has many missing values and needs careful handling.
5. Age has missing values and will need imputation.
6. Embarked has a small number of missing values in train.
7. Fare has missing values in test.
8. Sex, Pclass, Age, Fare, SibSp, Parch, Embarked are likely useful baseline features.
9. Name, Ticket, Cabin may contain useful engineered features later.
10. First baseline should be simple and reproducible before advanced feature engineering.

## Next Step

Build a preprocessing pipeline:

- impute missing Age
- impute missing Fare
- impute missing Embarked
- encode categorical columns
- create first baseline model
"""

    return report


def save_report(report: str, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    train_df, test_df, submission_df = load_titanic_data()
    report = build_eda_report(train_df, test_df, submission_df)
    save_report(report)

    print(report)
    print()
    print(f"Saved Titanic EDA report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
