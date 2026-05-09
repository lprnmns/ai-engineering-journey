import pandas as pd

from competitions.titanic.src.id_and_surname_analysis import (
    add_passenger_id_bins,
    add_surname,
    build_passenger_id_bin_summary,
    build_surname_survival_summary,
    extract_surname,
)


def build_sample_train_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5, 6],
            "Survived": [0, 1, 1, 0, 0, 1],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Braund, Miss. Example",
                "Smith, Mrs. Example",
                "Smith, Mr. Example",
                "Allen, Mr. Example",
                "Allen, Miss. Example",
            ],
        }
    )


def test_extract_surname_gets_text_before_first_comma() -> None:
    assert extract_surname("Braund, Mr. Owen Harris") == "Braund"


def test_add_surname_adds_surname_column() -> None:
    df = build_sample_train_data()

    result = add_surname(df)

    assert "Surname" in result.columns
    assert list(result["Surname"].head(2)) == ["Braund", "Braund"]


def test_add_passenger_id_bins_creates_bin_column() -> None:
    df = build_sample_train_data()

    result = add_passenger_id_bins(df, n_bins=3)

    assert "PassengerIdBin" in result.columns


def test_build_passenger_id_bin_summary_returns_survival_rate() -> None:
    df = build_sample_train_data()

    result = build_passenger_id_bin_summary(df)

    assert "PassengerIdRange" in result.columns
    assert "survival_rate" in result.columns
    assert result["passenger_count"].sum() == 6


def test_build_surname_survival_summary_groups_by_surname() -> None:
    df = build_sample_train_data()

    result = build_surname_survival_summary(df)

    assert "Surname" in result.columns
    assert "passenger_count" in result.columns
    assert "survival_rate" in result.columns
    assert result["passenger_count"].max() == 2
