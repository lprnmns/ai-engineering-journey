import pandas as pd

from competitions.titanic.src.visual_error_analysis import (
    add_analysis_bins,
    parse_first_float,
    prepare_analysis_dataframe,
    summarize_errors_by_segment,
)


def build_sample_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived": [0, 1, 1, 0],
            "Pclass": [3, 1, 3, 2],
            "Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley", "Heikkinen, Miss. Laina", "Allen, Mr. William Henry"],
            "Sex": ["male", "female", "female", "male"],
            "Age": [22.0, 38.0, 26.0, 35.0],
            "SibSp": [1, 1, 0, 0],
            "Parch": [0, 0, 0, 0],
            "Ticket": ["A/5", "PC", "STON", "373450"],
            "Fare": [7.25, 71.28, 7.92, 8.05],
            "Cabin": [None, "C85", None, None],
            "Embarked": ["S", "C", "S", "S"],
        }
    )


def test_parse_first_float_extracts_score() -> None:
    assert parse_first_float("0.83725 +/- 0.01390") == 0.83725


def test_parse_first_float_returns_none_without_float() -> None:
    assert parse_first_float("not_submitted") is None


def test_add_analysis_bins_creates_age_and_fare_bins() -> None:
    df = pd.DataFrame({"Age": [5.0, 16.0, 25.0, 45.0, 70.0], "Fare": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = add_analysis_bins(df)
    assert "AgeBin" in result.columns
    assert "FareBin" in result.columns


def test_prepare_analysis_dataframe_adds_title_and_bins() -> None:
    df = build_sample_raw_data()
    result = prepare_analysis_dataframe(df)
    assert "Title" in result.columns
    assert "AgeBin" in result.columns
    assert "FareBin" in result.columns


def test_summarize_errors_by_segment_returns_error_rates() -> None:
    df = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Sex": ["male", "female", "female", "male"],
            "Pclass": [3, 1, 3, 2],
            "Title": ["Mr", "Mrs", "Miss", "Mr"],
            "AgeBin": ["adult", "adult", "adult", "adult"],
            "FareBin": ["low", "high", "low", "low"],
            "Embarked": ["S", "C", "S", "S"],
            "Survived": [0, 1, 1, 0],
            "oof_prediction": [0, 0, 1, 1],
            "oof_error": [False, True, False, True],
        }
    )
    result = summarize_errors_by_segment(df, segment_columns=["Sex"])
    assert "error_rate" in result.columns
    assert set(result["segment"]) == {"Sex"}
