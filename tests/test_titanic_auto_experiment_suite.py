import pandas as pd

from competitions.titanic.src.auto_experiment_suite import (
    FeatureSetSpec,
    calculate_mean,
    calculate_std,
    extract_raw_title,
    extract_surname,
    map_title_group,
    qualify_result,
    sanitize_categorical_columns,
    unique_columns,
)


def test_extract_surname() -> None:
    assert extract_surname("Carter, Rev. Ernest Courtenay") == "Carter"


def test_extract_raw_title() -> None:
    assert extract_raw_title("Carter, Rev. Ernest Courtenay") == "Rev"


def test_map_title_group() -> None:
    assert map_title_group("Mr") == "CommonMale"
    assert map_title_group("Miss") == "CommonFemale"
    assert map_title_group("Rev") == "Clergy"
    assert map_title_group("Col") == "Military"


def test_calculate_mean_and_std() -> None:
    values = [0.8, 0.9, 1.0]

    assert calculate_mean(values) == 0.9
    assert round(calculate_std(values), 5) == 0.08165


def test_qualify_result_accepts_good_candidate() -> None:
    qualifies, reason = qualify_result(
        cv_mean=0.84,
        cv_std=0.01,
        risk_adjusted_score=0.83,
        changed_count=5,
        total_0_to_1=2,
        pclass3_female_0_to_1=1,
        miss_mrs_0_to_1=1,
        guard_decision="REVIEW_OK",
        min_cv_mean=0.838,
        max_cv_std=0.018,
        min_risk_adjusted=0.82,
        max_total_changed=12,
        max_0_to_1=6,
        max_pclass3_female_0_to_1=3,
        max_miss_mrs_0_to_1=4,
    )

    assert qualifies is True
    assert reason == "passed all thresholds"


def test_qualify_result_rejects_risky_candidate() -> None:
    qualifies, reason = qualify_result(
        cv_mean=0.85,
        cv_std=0.01,
        risk_adjusted_score=0.84,
        changed_count=20,
        total_0_to_1=10,
        pclass3_female_0_to_1=6,
        miss_mrs_0_to_1=6,
        guard_decision="DO_NOT_SUBMIT_WITHOUT_REVIEW",
        min_cv_mean=0.838,
        max_cv_std=0.018,
        min_risk_adjusted=0.82,
        max_total_changed=12,
        max_0_to_1=6,
        max_pclass3_female_0_to_1=3,
        max_miss_mrs_0_to_1=4,
    )

    assert qualifies is False
    assert "guard_decision" in reason


def test_feature_set_spec_holds_columns() -> None:
    spec = FeatureSetSpec(
        name="test",
        numeric_features=["Age"],
        categorical_features=["Sex"],
    )

    assert spec.name == "test"
    assert spec.numeric_features == ["Age"]
    assert spec.categorical_features == ["Sex"]



def test_sanitize_categorical_columns_converts_bool_flags_to_strings() -> None:
    train_df = pd.DataFrame(
        {
            "Sex": ["male", "female"],
            "IsSoloSurname": [True, False],
            "Survived": [0, 1],
        }
    )
    test_df = pd.DataFrame(
        {
            "Sex": ["female"],
            "IsSoloSurname": [True],
        }
    )
    feature_sets = [
        FeatureSetSpec(
            name="test",
            numeric_features=[],
            categorical_features=["Sex", "IsSoloSurname"],
        )
    ]

    train_out, test_out = sanitize_categorical_columns(train_df, test_df, feature_sets)

    assert train_out["IsSoloSurname"].tolist() == ["True", "False"]
    assert test_out["IsSoloSurname"].tolist() == ["True"]



def test_unique_columns_preserves_order_and_removes_duplicates() -> None:
    assert unique_columns(["Sex", "TitleGroup", "Sex", "Embarked"]) == [
        "Sex",
        "TitleGroup",
        "Embarked",
    ]


def test_feature_set_spec_removes_duplicate_columns() -> None:
    spec = FeatureSetSpec(
        name="dedupe",
        numeric_features=["Age", "Age", "Fare"],
        categorical_features=["Sex", "Sex", "TitleGroup"],
    )

    assert spec.numeric_features == ["Age", "Fare"]
    assert spec.categorical_features == ["Sex", "TitleGroup"]
