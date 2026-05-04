from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from labs.data.clean_students import PROCESSED_DATA_PATH


REPORT_PATH = Path("docs/w3_eda_report.md")


def load_clean_students(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def build_eda_report(df: pd.DataFrame) -> str:
    row_count, column_count = df.shape

    missing_values = df.isna().sum()
    pass_rate = float(df["passed"].mean())

    avg_study_by_passed = df.groupby("passed")["study_hours"].mean()
    avg_score_by_passed = df.groupby("passed")["previous_score"].mean()

    top_student = df.sort_values("previous_score", ascending=False).iloc[0]

    report = f"""# W3 EDA Report — Students Dataset

## Dataset Overview

- Rows: {row_count}
- Columns: {column_count}
- Columns list: {", ".join(df.columns)}

## Missing Values

```text
{missing_values.to_string()}
```

## Target Distribution

The target column is `passed`.

- Pass rate: {pass_rate:.2%}
- Failed rate: {(1 - pass_rate):.2%}

## Average Study Hours by Target

```text
{avg_study_by_passed.to_string()}
```

## Average Previous Score by Target

```text
{avg_score_by_passed.to_string()}
```

## Top Student by Previous Score

- Name: {top_student["name"]}
- Previous score: {top_student["previous_score"]}
- Study hours: {top_student["study_hours"]}
- Passed: {top_student["passed"]}

## Initial Insights

1. Students who passed studied more hours on average.
2. Students who passed also had a higher previous score on average.
3. The cleaned dataset has no missing values.
4. The dataset is tiny, so these insights are not statistically strong.
5. This dataset is ready for a small baseline ML experiment.

## Modeling Readiness

This dataset can be used for a toy classification task:

```text
features: age, study_hours, previous_score
target: passed
```

The next step is to build a simple baseline model.
"""

    return report


def save_report(report: str, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    df = load_clean_students()
    report = build_eda_report(df)
    save_report(report)

    print(report)
    print(f"Saved EDA report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
