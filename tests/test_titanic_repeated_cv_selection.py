import pandas as pd

from competitions.titanic.src.hgb_tuning import HgbConfig
from competitions.titanic.src.repeated_cv_selection import (
    calculate_mean,
    calculate_std,
    evaluate_config_repeated_cv,
    results_to_dataframe,
)


def test_calculate_mean_returns_average() -> None:
    assert calculate_mean([1.0, 2.0, 3.0]) == 2.0


def test_calculate_std_returns_population_std() -> None:
    value = calculate_std([1.0, 2.0, 3.0])

    assert round(value, 5) == 0.8165


def test_evaluate_config_repeated_cv_returns_scores() -> None:
    X = pd.DataFrame(
        {
            "Pclass": [1, 1, 2, 2, 3, 3, 1, 3],
            "Age": [20.0, 30.0, 40.0, None, 25.0, 35.0, 45.0, 22.0],
            "SibSp": [0, 1, 0, 1, 0, 1, 0, 1],
            "Parch": [0, 0, 1, 1, 0, 0, 1, 1],
            "Fare": [80.0, 90.0, 30.0, 35.0, 8.0, 9.0, 100.0, 7.0],
            "Sex": ["female", "female", "male", "male", "male", "female", "female", "male"],
            "Embarked": ["S", "C", "S", "Q", "S", None, "C", "Q"],
            "Title": ["Miss", "Mrs", "Mr", "Mr", "Mr", "Miss", "Mrs", "Master"],
        }
    )
    y = pd.Series([1, 1, 0, 0, 0, 1, 1, 0])

    config = HgbConfig(
        name="test_config",
        max_iter=10,
        learning_rate=0.1,
        max_leaf_nodes=7,
        min_samples_leaf=2,
        l2_regularization=0.0,
    )

    result = evaluate_config_repeated_cv(
        X,
        y,
        config,
        seeds=[1, 2],
        n_splits=2,
    )

    assert len(result.scores) == 4
    assert 0.0 <= result.repeated_cv_mean <= 1.0
    assert result.repeated_cv_std >= 0.0
    assert result.risk_adjusted_score == result.repeated_cv_mean - result.repeated_cv_std


def test_results_to_dataframe_contains_expected_columns() -> None:
    config = HgbConfig(
        name="test_config",
        max_iter=10,
        learning_rate=0.1,
        max_leaf_nodes=7,
        min_samples_leaf=2,
        l2_regularization=0.0,
    )

    from competitions.titanic.src.repeated_cv_selection import RepeatedCvResult

    result = RepeatedCvResult(
        config=config,
        scores=[0.7, 0.8],
        repeated_cv_mean=0.75,
        repeated_cv_std=0.05,
        repeated_cv_min=0.7,
        repeated_cv_max=0.8,
        risk_adjusted_score=0.70,
    )

    df = results_to_dataframe([result])

    assert "config_name" in df.columns
    assert "risk_adjusted_score" in df.columns
    assert df.loc[0, "config_name"] == "test_config"
