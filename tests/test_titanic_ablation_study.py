import pandas as pd

from competitions.titanic.src.ablation_study import (
    FeatureConfig,
    build_feature_configs,
    results_to_dataframe,
    run_cv_for_config,
)


def build_sample_featured_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "Pclass": [1, 1, 2, 2, 3, 3, 1, 3],
            "Age": [20.0, 30.0, 40.0, None, 25.0, 35.0, 45.0, 22.0],
            "SibSp": [0, 1, 0, 1, 0, 1, 0, 1],
            "Parch": [0, 0, 1, 1, 0, 0, 1, 1],
            "Fare": [80.0, 90.0, 30.0, 35.0, 8.0, 9.0, 100.0, 7.0],
            "Sex": ["female", "female", "male", "male", "male", "female", "female", "male"],
            "Embarked": ["S", "C", "S", "Q", "S", None, "C", "Q"],
            "FamilySize": [1, 2, 2, 3, 1, 2, 2, 3],
            "IsAlone": [1, 0, 0, 0, 1, 0, 0, 0],
            "HasCabin": [1, 1, 0, 0, 0, 0, 1, 0],
            "Title": ["Miss", "Mrs", "Mr", "Mr", "Mr", "Miss", "Mrs", "Master"],
        }
    )
    y = pd.Series([1, 1, 0, 0, 0, 1, 1, 0])

    return X, y


def test_build_feature_configs_includes_baseline_and_feature_v1() -> None:
    configs = build_feature_configs()
    names = [config.name for config in configs]

    assert "baseline" in names
    assert "feature_v1_all" in names
    assert "plus_title" in names


def test_run_cv_for_config_returns_scores() -> None:
    X, y = build_sample_featured_data()
    config = FeatureConfig(
        name="test_config",
        numeric_features=["Pclass", "Age", "Fare", "FamilySize"],
        categorical_features=["Sex", "Embarked", "Title"],
        hypothesis="test",
    )

    scores, mean_score, std_score = run_cv_for_config(
        X,
        y,
        config,
        n_splits=2,
    )

    assert len(scores) == 2
    assert 0.0 <= mean_score <= 1.0
    assert std_score >= 0.0


def test_results_to_dataframe_contains_expected_columns() -> None:
    from competitions.titanic.src.ablation_study import AblationResult

    result = AblationResult(
        name="baseline",
        numeric_features=["Pclass"],
        categorical_features=["Sex"],
        hypothesis="test",
        cv_scores=[0.7, 0.8],
        cv_mean=0.75,
        cv_std=0.05,
        delta_vs_baseline=0.0,
    )

    df = results_to_dataframe([result])

    assert list(df.columns) == [
        "name",
        "cv_mean",
        "cv_std",
        "delta_vs_baseline",
        "numeric_features",
        "categorical_features",
        "hypothesis",
        "fold_scores",
    ]
    assert df.loc[0, "name"] == "baseline"
