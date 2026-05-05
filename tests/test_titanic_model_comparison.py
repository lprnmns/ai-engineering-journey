import pandas as pd
import pytest

from competitions.titanic.src.model_comparison import (
    ModelComparisonResult,
    build_model_specs,
    build_submission,
    find_model_spec,
    results_to_dataframe,
    validate_submission,
)


def test_build_model_specs_contains_expected_models() -> None:
    model_names = [spec.name for spec in build_model_specs()]

    assert "logistic_regression" in model_names
    assert "random_forest" in model_names
    assert "hist_gradient_boosting" in model_names


def test_find_model_spec_returns_matching_model() -> None:
    spec = find_model_spec("logistic_regression")

    assert spec.name == "logistic_regression"


def test_find_model_spec_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model name"):
        find_model_spec("unknown_model")


def test_results_to_dataframe_contains_expected_columns() -> None:
    result = ModelComparisonResult(
        model_name="test_model",
        hypothesis="test hypothesis",
        cv_scores=[0.7, 0.8],
        cv_mean=0.75,
        cv_std=0.05,
    )

    df = results_to_dataframe([result])

    assert list(df.columns) == [
        "model_name",
        "cv_mean",
        "cv_std",
        "hypothesis",
        "fold_scores",
    ]
    assert df.loc[0, "model_name"] == "test_model"


def test_build_submission_uses_required_columns() -> None:
    test_raw = pd.DataFrame(
        {
            "PassengerId": [892, 893],
        }
    )
    predictions = [0, 1]

    submission = build_submission(test_raw, predictions)

    assert list(submission.columns) == ["PassengerId", "Survived"]
    assert submission.shape == (2, 2)


def test_validate_submission_accepts_valid_submission() -> None:
    submission = pd.DataFrame(
        {
            "PassengerId": [892, 893],
            "Survived": [0, 1],
        }
    )

    validate_submission(submission, expected_rows=2)
