import pandas as pd

from competitions.titanic.src.model_zoo_ensemble import (
    add_competition_features,
    build_feature_sets,
    build_model_specs,
    extract_raw_title,
    extract_ticket_number,
    map_title_v2,
    ticket_has_letters,
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
            "Ticket": ["STON/O2. 3101282", "373450"],
            "Fare": [7.92, 8.05],
            "Cabin": [None, None],
            "Embarked": ["S", "S"],
        }
    )


def test_extract_raw_title_extracts_title() -> None:
    assert extract_raw_title("Braund, Mr. Owen Harris") == "Mr"


def test_map_title_v2_maps_rare_groups() -> None:
    assert map_title_v2("Mlle") == "Miss"
    assert map_title_v2("Rev") == "Clergy"
    assert map_title_v2("Major") == "Military"


def test_ticket_helpers_extract_number_and_letters() -> None:
    assert extract_ticket_number("A/5 21171") == "21171"
    assert ticket_has_letters("A/5 21171") == 1
    assert ticket_has_letters("373450") == 0


def test_add_competition_features_creates_expected_columns() -> None:
    train_df, test_df = add_competition_features(build_train(), build_test())
    expected_columns = {
        "FamilySize",
        "FamilySizeGrouped",
        "AgeBin5",
        "FareBin5",
        "NameLength",
        "NameLengthGroup",
        "TitleV2",
        "TicketNumberCount",
        "TicketHasLetters",
        "CabinAssigned",
        "CabinLetter",
        "Sex_Pclass",
        "Title_Pclass",
    }
    for column in expected_columns:
        assert column in train_df.columns
        assert column in test_df.columns


def test_build_feature_sets_contains_expected_sets() -> None:
    names = [feature_set.name for feature_set in build_feature_sets()]
    assert "plus_title" in names
    assert "binned_competition" in names
    assert "wide_competition" in names


def test_build_model_specs_contains_gaussian_nb_and_extra_trees() -> None:
    names = [model_spec.name for model_spec in build_model_specs()]
    assert "gaussian_nb" in names
    assert "extra_trees" in names
    assert "adaboost" in names
