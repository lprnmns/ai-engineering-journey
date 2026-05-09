import pandas as pd

from competitions.titanic.src.submission_difference_analysis import (
    build_submission_differences,
    summarize_changed_segments,
    summarize_difference,
)


def test_build_submission_differences_finds_changed_rows() -> None:
    test_features = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Sex": ["male", "female", "male"],
            "Pclass": [3, 1, 2],
            "Title": ["Mr", "Mrs", "Mr"],
            "Embarked": ["S", "C", "S"],
        }
    )
    best = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Survived_best": [0, 1, 0],
        }
    )
    candidate = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Survived_candidate": [0, 0, 1],
        }
    )

    changed = build_submission_differences(
        test_features=test_features,
        best_submission=best,
        candidate_submission=candidate,
        candidate_name="candidate",
    )

    assert len(changed) == 2
    assert set(changed["change_direction"]) == {"1_to_0", "0_to_1"}


def test_summarize_difference_counts_change_directions() -> None:
    changed = pd.DataFrame(
        {
            "change_direction": ["1_to_0", "0_to_1", "0_to_1"],
        }
    )

    summary = summarize_difference(
        changed=changed,
        candidate_name="candidate",
        test_row_count=10,
    )

    assert summary.changed_count == 3
    assert summary.best_survived_to_candidate_died == 1
    assert summary.best_died_to_candidate_survived == 2
    assert summary.changed_rate == 0.3


def test_summarize_changed_segments_returns_segment_counts() -> None:
    changed = pd.DataFrame(
        {
            "candidate_name": ["a", "a", "a"],
            "change_direction": ["0_to_1", "0_to_1", "1_to_0"],
            "Sex": ["male", "female", "male"],
            "Pclass": [3, 1, 3],
            "Title": ["Mr", "Mrs", "Mr"],
            "Embarked": ["S", "C", "S"],
        }
    )

    result = summarize_changed_segments(changed)

    assert "segment" in result.columns
    assert "changed_rows" in result.columns
    assert result["changed_rows"].max() == 2
