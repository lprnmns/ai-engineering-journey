import pandas as pd

from competitions.titanic.src.deck_ticket_fare import (
    add_deck_ticket_fare_features,
    extract_deck,
    normalize_ticket,
)


def build_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "Survived": [0, 1],
            "Pclass": [3, 1],
            "Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley"],
            "Sex": ["male", "female"],
            "Age": [22.0, 38.0],
            "SibSp": [1, 1],
            "Parch": [0, 0],
            "Ticket": ["A/5 21171", "PC 17599"],
            "Fare": [7.25, 71.28],
            "Cabin": [None, "C85"],
            "Embarked": ["S", "C"],
        }
    )


def build_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [3, 4],
            "Pclass": [3, 1],
            "Name": ["Heikkinen, Miss. Laina", "Allen, Mr. William Henry"],
            "Sex": ["female", "male"],
            "Age": [26.0, 35.0],
            "SibSp": [0, 0],
            "Parch": [0, 0],
            "Ticket": ["A/5 21171", "373450"],
            "Fare": [7.92, 8.05],
            "Cabin": [None, "E12"],
            "Embarked": ["S", "S"],
        }
    )


def test_extract_deck_handles_missing_and_known_cabin() -> None:
    assert extract_deck(None) == "Unknown"
    assert extract_deck("C85") == "C"
    assert extract_deck(" e12 ") == "E"


def test_normalize_ticket_handles_missing_and_text() -> None:
    assert normalize_ticket(None) == "UNKNOWN"
    assert normalize_ticket(" A/5 21171 ") == "A/5 21171"


def test_add_deck_ticket_fare_features_creates_expected_columns() -> None:
    train_df, test_df = add_deck_ticket_fare_features(
        build_train(),
        build_test(),
    )

    for df in [train_df, test_df]:
        assert "Deck" in df.columns
        assert "TicketGroupSize" in df.columns
        assert "FarePerPerson" in df.columns
        assert "Title" in df.columns

    first_train = train_df.iloc[0]
    first_test = test_df.iloc[0]

    assert first_train["TicketGroupSize"] == 2
    assert first_test["TicketGroupSize"] == 2
    assert first_train["Deck"] == "Unknown"
    assert train_df.iloc[1]["Deck"] == "C"
    assert test_df.iloc[1]["Deck"] == "E"
