import pandas as pd

from labs.data.eda_report import build_eda_report


def test_build_eda_report_contains_dataset_overview() -> None:
    df = pd.DataFrame(
        {
            "name": ["Ali", "Ayse"],
            "age": [21.0, 23.0],
            "study_hours": [5.0, 3.0],
            "previous_score": [70, 50],
            "passed": [True, False],
        }
    )

    report = build_eda_report(df)

    assert "# W3 EDA Report" in report
    assert "Rows: 2" in report
    assert "Columns: 5" in report


def test_build_eda_report_contains_pass_rate() -> None:
    df = pd.DataFrame(
        {
            "name": ["Ali", "Ayse", "Can", "Ece"],
            "age": [21.0, 23.0, 21.0, 22.0],
            "study_hours": [5.0, 3.0, 8.0, 2.0],
            "previous_score": [70, 50, 90, 45],
            "passed": [True, False, True, False],
        }
    )

    report = build_eda_report(df)

    assert "Pass rate: 50.00%" in report
    assert "Failed rate: 50.00%" in report


def test_build_eda_report_mentions_modeling_readiness() -> None:
    df = pd.DataFrame(
        {
            "name": ["Ali", "Ayse"],
            "age": [21.0, 23.0],
            "study_hours": [5.0, 3.0],
            "previous_score": [70, 50],
            "passed": [True, False],
        }
    )

    report = build_eda_report(df)

    assert "features: age, study_hours, previous_score" in report
    assert "target: passed" in report
