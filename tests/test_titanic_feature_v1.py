import pandas as pd

from competitions.titanic.src.feature_v1 import (
    CATEGORICAL_FEATURES_V1,
    MODEL_FEATURES_V1,
    NUMERIC_FEATURES_V1,
    add_feature_v1_columns,
    extract_title,
    select_test_features,
    split_features_target,
    validate_submission,
)


def build_sample_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Survived": [0, 1, 1],
            "Pclass": [3, 1, 3],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
            ],
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


def test_extract_title_handles_common_titles() -> None:
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"
    assert extract_title("Cumings, Mrs. John Bradley") == "Mrs"
    assert extract_title("Heikkinen, Miss. Laina") == "Miss"
    assert extract_title("Palsson, Master. Gosta Leonard") == "Master"


def test_extract_title_maps_rare_titles_to_rare() -> None:
    assert extract_title("Minahan, Dr. William Edward") == "Rare"
    assert extract_title("Butt, Major. Archibald Willingham") == "Rare"


def test_add_feature_v1_columns_creates_expected_features() -> None:
    df = build_sample_train()

    featured = add_feature_v1_columns(df)

    assert "FamilySize" in featured.columns
    assert "IsAlone" in featured.columns
    assert "HasCabin" in featured.columns
    assert "Title" in featured.columns

    assert featured.loc[0, "FamilySize"] == 2
    assert featured.loc[2, "IsAlone"] == 1
    assert featured.loc[1, "HasCabin"] == 1
    assert featured.loc[0, "Title"] == "Mr"


def test_model_features_include_engineered_columns() -> None:
    assert "FamilySize" in NUMERIC_FEATURES_V1
    assert "IsAlone" in NUMERIC_FEATURES_V1
    assert "HasCabin" in NUMERIC_FEATURES_V1
    assert "Title" in CATEGORICAL_FEATURES_V1
    assert MODEL_FEATURES_V1 == NUMERIC_FEATURES_V1 + CATEGORICAL_FEATURES_V1


def test_split_features_target_uses_feature_v1_columns() -> None:
    df = add_feature_v1_columns(build_sample_train())

    X, y = split_features_target(df)

    assert list(X.columns) == MODEL_FEATURES_V1
    assert y.name == "Survived"


def test_select_test_features_uses_feature_v1_columns() -> None:
    df = add_feature_v1_columns(build_sample_train().drop(columns=["Survived"]))

    X_test = select_test_features(df)

    assert list(X_test.columns) == MODEL_FEATURES_V1


def test_validate_submission_accepts_valid_submission() -> None:
    submission = pd.DataFrame(
        {
            "PassengerId": [892, 893],
            "Survived": [0, 1],
        }
    )

    validate_submission(submission, expected_rows=2)
