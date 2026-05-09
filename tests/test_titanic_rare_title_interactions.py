import pandas as pd

from competitions.titanic.src.rare_title_interactions import (
    add_rare_title_interaction_features,
    extract_raw_title,
    map_title_v2,
    results_to_dataframe,
    RareTitleInteractionResult,
)


def build_sample_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived": [0, 1, 1, 0],
            "Pclass": [3, 1, 3, 2],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Minahan, Dr. William Edward",
                "Bateman, Rev. Robert James",
            ],
            "Sex": ["male", "female", "male", "male"],
            "Age": [22.0, 38.0, 44.0, 51.0],
            "SibSp": [1, 1, 0, 0],
            "Parch": [0, 0, 0, 0],
            "Ticket": ["A/5", "PC", "STON", "373450"],
            "Fare": [7.25, 71.28, 7.92, 8.05],
            "Cabin": [None, "C85", None, None],
            "Embarked": ["S", "C", "S", "S"],
        }
    )


def test_extract_raw_title_extracts_title_from_name() -> None:
    assert extract_raw_title("Braund, Mr. Owen Harris") == "Mr"
    assert extract_raw_title("Minahan, Dr. William Edward") == "Dr"


def test_map_title_v2_splits_rare_titles() -> None:
    assert map_title_v2("Mr") == "Mr"
    assert map_title_v2("Mlle") == "Miss"
    assert map_title_v2("Rev") == "Clergy"
    assert map_title_v2("Dr") == "Professional"
    assert map_title_v2("Major") == "Military"
    assert map_title_v2("Countess") == "NobleFemale"


def test_add_rare_title_interaction_features_creates_expected_columns() -> None:
    df = add_rare_title_interaction_features(build_sample_raw_data())

    assert "TitleV2" in df.columns
    assert "TitleV2_Pclass" in df.columns
    assert "Sex_Pclass" in df.columns
    assert "Embarked_Pclass" in df.columns
    assert "Professional_P3" in set(df["TitleV2_Pclass"])
    assert "Clergy_P2" in set(df["TitleV2_Pclass"])


def test_results_to_dataframe_contains_expected_columns() -> None:
    result = RareTitleInteractionResult(
        model_name="hist_gradient_boosting",
        cv_scores=[0.8, 0.9],
        cv_mean=0.85,
        cv_std=0.05,
        delta_vs_current_best=0.01,
    )

    df = results_to_dataframe(result)

    assert "experiment_id" in df.columns
    assert "cv_mean" in df.columns
    assert df.loc[0, "experiment_id"] == "titanic_exp_008_rare_title_interactions"
