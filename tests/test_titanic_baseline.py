import pandas as pd
import pytest

from competitions.titanic.src.train_baseline import (
    build_submission,
    validate_submission,
)


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
    assert list(submission["Survived"]) == [0, 1]


def test_validate_submission_accepts_valid_submission() -> None:
    submission = pd.DataFrame(
        {
            "PassengerId": [892, 893],
            "Survived": [0, 1],
        }
    )

    validate_submission(submission, expected_rows=2)


def test_validate_submission_rejects_wrong_columns() -> None:
    submission = pd.DataFrame(
        {
            "id": [892],
            "Survived": [1],
        }
    )

    with pytest.raises(ValueError, match="Submission columns"):
        validate_submission(submission, expected_rows=1)


def test_validate_submission_rejects_invalid_survived_values() -> None:
    submission = pd.DataFrame(
        {
            "PassengerId": [892],
            "Survived": [2],
        }
    )

    with pytest.raises(ValueError, match="invalid values"):
        validate_submission(submission, expected_rows=1)
