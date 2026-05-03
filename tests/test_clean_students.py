import pandas as pd

from labs.data.clean_students import clean_students


def test_clean_students_removes_missing_values() -> None:
    raw_df = pd.DataFrame(
        {
            "name": ["Ali", "Mehmet", "Deniz"],
            "age": [21.0, 20.0, None],
            "study_hours": [5.5, None, 4.0],
            "previous_score": [72, 60, 68],
            "passed": [True, False, True],
        }
    )

    clean_df = clean_students(raw_df)

    assert clean_df.isna().sum().sum() == 0


def test_clean_students_fills_age_with_median() -> None:
    raw_df = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "age": [20.0, 22.0, None],
            "study_hours": [1.0, 2.0, 3.0],
            "previous_score": [50, 60, 70],
            "passed": [False, True, True],
        }
    )

    clean_df = clean_students(raw_df)

    assert clean_df.loc[2, "age"] == 21.0


def test_clean_students_fills_study_hours_with_mean() -> None:
    raw_df = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "age": [20.0, 22.0, 24.0],
            "study_hours": [2.0, 4.0, None],
            "previous_score": [50, 60, 70],
            "passed": [False, True, True],
        }
    )

    clean_df = clean_students(raw_df)

    assert clean_df.loc[2, "study_hours"] == 3.0
