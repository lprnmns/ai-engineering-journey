import pandas as pd

from competitions.titanic.src.risky_flip_guard import (
    build_flip_rows,
    decide_guard_result,
    summarize_guard,
    summaries_to_dataframe,
)


def build_test_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Pclass": [3, 3, 1, 2],
            "Sex": ["female", "female", "male", "female"],
            "Title": ["Miss", "Mrs", "Mr", "Miss"],
            "Embarked": ["S", "C", "C", "Q"],
        }
    )


def test_build_flip_rows_adds_risky_flags() -> None:
    best = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived_best": [0, 0, 1, 1],
        }
    )
    candidate = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived_candidate": [1, 1, 0, 1],
        }
    )

    changed = build_flip_rows(
        test_features=build_test_features(),
        best_submission=best,
        candidate_submission=candidate,
        candidate_name="candidate",
    )

    assert len(changed) == 3
    assert int(changed["is_0_to_1"].sum()) == 2
    assert int(changed["is_pclass3_female_0_to_1"].sum()) == 2
    assert int(changed["is_miss_mrs_0_to_1"].sum()) == 2


def test_decide_guard_result_blocks_risky_submission() -> None:
    decision, reason = decide_guard_result(
        changed_count=20,
        total_0_to_1=15,
        pclass3_female_0_to_1=8,
        miss_mrs_0_to_1=7,
        embarked_sc_0_to_1=9,
    )

    assert decision == "DO_NOT_SUBMIT_WITHOUT_REVIEW"
    assert "0_to_1" in reason


def test_decide_guard_result_detects_identical_submission() -> None:
    decision, reason = decide_guard_result(
        changed_count=0,
        total_0_to_1=0,
        pclass3_female_0_to_1=0,
        miss_mrs_0_to_1=0,
        embarked_sc_0_to_1=0,
    )

    assert decision == "NO_NEW_INFORMATION"
    assert "identical" in reason


def test_summarize_guard_returns_counts() -> None:
    best = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived_best": [0, 0, 1, 1],
        }
    )
    candidate = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived_candidate": [1, 1, 0, 1],
        }
    )

    changed = build_flip_rows(
        test_features=build_test_features(),
        best_submission=best,
        candidate_submission=candidate,
        candidate_name="candidate",
    )

    summary = summarize_guard(
        changed=changed,
        candidate_name="candidate",
        candidate_path=__import__("pathlib").Path("candidate.csv"),
        test_row_count=4,
    )

    assert summary.changed_count == 3
    assert summary.total_0_to_1 == 2
    assert summary.total_1_to_0 == 1


def test_summaries_to_dataframe_contains_decision() -> None:
    best = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived_best": [0, 0, 1, 1],
        }
    )
    candidate = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived_candidate": [1, 1, 0, 1],
        }
    )

    changed = build_flip_rows(
        test_features=build_test_features(),
        best_submission=best,
        candidate_submission=candidate,
        candidate_name="candidate",
    )

    summary = summarize_guard(
        changed=changed,
        candidate_name="candidate",
        candidate_path=__import__("pathlib").Path("candidate.csv"),
        test_row_count=4,
    )

    df = summaries_to_dataframe([summary])

    assert "decision" in df.columns
    assert df.loc[0, "candidate_name"] == "candidate"
