import pandas as pd

from competitions.titanic.src.hgb_tuning import (
    HgbConfig,
    build_configs,
    evaluate_config,
    results_to_dataframe,
)


def test_build_configs_contains_current_best_reference() -> None:
    config_names = [config.name for config in build_configs()]

    assert "current_best_reference" in config_names


def test_build_configs_contains_multiple_candidates() -> None:
    assert len(build_configs()) >= 5


def test_evaluate_config_returns_valid_cv_result() -> None:
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

    result = evaluate_config(X, y, config, n_splits=2)

    assert result.config.name == "test_config"
    assert len(result.cv_scores) == 2
    assert 0.0 <= result.cv_mean <= 1.0
    assert result.cv_std >= 0.0


def test_results_to_dataframe_contains_expected_columns() -> None:
    config = HgbConfig(
        name="test_config",
        max_iter=10,
        learning_rate=0.1,
        max_leaf_nodes=7,
        min_samples_leaf=2,
        l2_regularization=0.0,
    )
    X = pd.DataFrame(
        {
            "config_name": ["dummy"],
        }
    )

    from competitions.titanic.src.hgb_tuning import HgbTuningResult

    result = HgbTuningResult(
        config=config,
        cv_scores=[0.7, 0.8],
        cv_mean=0.75,
        cv_std=0.05,
        delta_vs_current_best=-0.01,
    )

    df = results_to_dataframe([result])

    assert "config_name" in df.columns
    assert "cv_mean" in df.columns
    assert df.loc[0, "config_name"] == "test_config"
