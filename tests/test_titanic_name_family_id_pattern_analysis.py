import pandas as pd

from competitions.titanic.src.name_family_id_pattern_analysis import (
    add_name_parts,
    assign_passenger_id_jump_segment,
    build_passenger_id_jump_segments,
    build_safe_family_features,
    detect_passenger_id_jump_boundaries,
    extract_first_given_token,
    extract_raw_title,
    extract_surname,
    map_title_group,
)


def build_sample_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5, 6],
            "Survived": [0, 1, 1, 0, 0, 1],
            "Pclass": [3, 1, 1, 3, 3, 1],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Carter, Rev. Ernest Courtenay",
                "Minahan, Dr. William Edward",
                "Allen, Mr. William Henry",
                "Allen, Miss. Elisabeth Walton",
                "Allen, Mrs. John Example (Mary Smith)",
            ],
            "Ticket": ["A", "B", "C", "D", "D", "D"],
        }
    )


def build_sample_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [7, 8],
            "Pclass": [3, 1],
            "Name": [
                "Allen, Master. George Example",
                "Smith, Col. John Example",
            ],
            "Ticket": ["D", "E"],
        }
    )


def test_extract_name_parts() -> None:
    assert extract_surname("Carter, Rev. Ernest Courtenay") == "Carter"
    assert extract_raw_title("Carter, Rev. Ernest Courtenay") == "Rev"
    assert extract_first_given_token("Carter, Rev. Ernest Courtenay") == "Ernest"


def test_map_title_group_maps_status_titles() -> None:
    assert map_title_group("Mr") == "CommonMale"
    assert map_title_group("Rev") == "Clergy"
    assert map_title_group("Dr") == "Professional"
    assert map_title_group("Col") == "Military"


def test_add_name_parts_creates_expected_columns() -> None:
    result = add_name_parts(build_sample_train())

    assert "Surname" in result.columns
    assert "RawTitle" in result.columns
    assert "TitleGroup" in result.columns
    assert "FirstGivenToken" in result.columns
    assert "HasParenthesesName" in result.columns


def test_build_safe_family_features_uses_train_test_combined_counts() -> None:
    train_features, test_features = build_safe_family_features(
        build_sample_train(),
        build_sample_test(),
    )

    allen_train = train_features[train_features["Surname"] == "Allen"]
    allen_test = test_features[test_features["Surname"] == "Allen"]

    assert int(allen_train["SurnameGroupSize"].iloc[0]) == 4
    assert int(allen_test["SurnameGroupSize"].iloc[0]) == 4


def test_detect_passenger_id_jump_boundaries() -> None:
    base_bins = pd.DataFrame(
        {
            "end_id": [10, 20, 30],
            "survival_rate": [0.2, 0.7, 0.72],
            "delta_from_previous": [0.0, 0.5, 0.02],
        }
    )

    assert detect_passenger_id_jump_boundaries(base_bins, min_delta=0.12) == [10]


def test_assign_passenger_id_jump_segment() -> None:
    assert assign_passenger_id_jump_segment(5, [10, 20]) == "1-10"
    assert assign_passenger_id_jump_segment(15, [10, 20]) == "11-20"
    assert assign_passenger_id_jump_segment(25, [10, 20]) == "21-891"


def test_build_passenger_id_jump_segments_returns_summary() -> None:
    result = build_passenger_id_jump_segments(
        build_sample_train(),
        n_bins=3,
        min_delta=0.1,
    )

    assert "survival_rate" in result.columns
    assert "method" in result.columns
