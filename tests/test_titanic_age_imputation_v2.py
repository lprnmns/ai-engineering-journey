import pandas as pd

from competitions.titanic.src.age_imputation_v2 import (
    add_age_bin,
    apply_group_age_imputation,
    fit_group_age_stats,
    lookup_group_age,
)


def build_sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Pclass": [1, 1, 3, 3],
            "Sex": ["female", "female", "male", "male"],
            "Title": ["Mrs", "Mrs", "Master", "Master"],
            "Age": [40.0, None, 5.0, None],
            "SibSp": [1, 1, 0, 0],
            "Parch": [0, 0, 1, 1],
            "Fare": [80.0, 90.0, 10.0, 12.0],
            "Embarked": ["C", "C", "S", "S"],
        }
    )


def test_fit_group_age_stats_uses_known_ages() -> None:
    df = build_sample_data()

    stats = fit_group_age_stats(df)

    assert stats.global_median == 22.5


def test_lookup_group_age_uses_group_median() -> None:
    df = build_sample_data()
    stats = fit_group_age_stats(df)

    row = pd.Series(
        {
            "Title": "Mrs",
            "Pclass": 1,
            "Sex": "female",
        }
    )

    assert lookup_group_age(row, stats) == 40.0


def test_apply_group_age_imputation_fills_missing_values() -> None:
    df = build_sample_data()
    stats = fit_group_age_stats(df)

    imputed = apply_group_age_imputation(df, stats)

    assert imputed["Age"].isna().sum() == 0
    assert imputed.loc[1, "Age"] == 40.0
    assert imputed.loc[3, "Age"] == 5.0


def test_add_age_bin_creates_age_categories() -> None:
    df = pd.DataFrame(
        {
            "Age": [5.0, 15.0, 30.0, 70.0],
        }
    )

    result = add_age_bin(df)

    assert list(result["AgeBin"]) == ["child", "teen", "adult", "senior"]
