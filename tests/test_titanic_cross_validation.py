import pandas as pd

from competitions.titanic.src.cross_validate import (
    CrossValidationResult,
    build_report,
    run_cross_validation,
)


def build_sample_dataset() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "feature_a": [0, 1, 0, 1, 0, 1, 0, 1],
            "feature_b": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    return X, y


def test_run_cross_validation_returns_expected_number_of_scores() -> None:
    X, y = build_sample_dataset()

    result = run_cross_validation(X, y, n_splits=2)

    assert len(result.scores) == 2
    assert result.n_splits == 2
    assert result.metric == "accuracy"
    assert 0.0 <= result.mean_score <= 1.0
    assert result.std_score >= 0.0


def test_build_report_contains_public_score_gap() -> None:
    result = CrossValidationResult(
        scores=[0.8, 0.7],
        mean_score=0.75,
        std_score=0.05,
        n_splits=2,
        metric="accuracy",
    )

    report = build_report(result, public_score=0.70)

    assert "# Titanic Cross-Validation Report" in report
    assert "CV mean accuracy: 0.75000" in report
    assert "Kaggle public score: 0.70000" in report
    assert "CV minus public score gap: 0.05000" in report
