from __future__ import annotations

from pathlib import Path

import pandas as pd


TRAIN_PATH = Path("competitions/titanic/data/raw/train.csv")
REPORT_PATH = Path("competitions/titanic/reports/titanic_id_and_surname_analysis.md")
ID_BIN_CSV_PATH = Path("competitions/titanic/reports/titanic_passenger_id_bins.csv")
SURNAME_CSV_PATH = Path("competitions/titanic/reports/titanic_surname_survival.csv")


def load_train_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH)


def add_passenger_id_bins(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    result = df.copy()

    result["PassengerIdBin"] = pd.cut(
        result["PassengerId"],
        bins=n_bins,
        include_lowest=True,
    )

    return result


def build_passenger_id_bin_summary(df: pd.DataFrame) -> pd.DataFrame:
    with_bins = add_passenger_id_bins(df)

    summary = (
        with_bins.groupby("PassengerIdBin", observed=True)
        .agg(
            passenger_count=("PassengerId", "count"),
            passenger_id_min=("PassengerId", "min"),
            passenger_id_max=("PassengerId", "max"),
            survived_count=("Survived", "sum"),
            survival_rate=("Survived", "mean"),
        )
        .reset_index()
    )

    summary["PassengerIdRange"] = (
        summary["passenger_id_min"].astype(str)
        + "-"
        + summary["passenger_id_max"].astype(str)
    )

    return summary[
        [
            "PassengerIdRange",
            "passenger_count",
            "survived_count",
            "survival_rate",
        ]
    ]


def extract_surname(name: str) -> str:
    return name.split(",", maxsplit=1)[0].strip()


def add_surname(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Surname"] = result["Name"].astype(str).apply(extract_surname)

    return result


def build_surname_survival_summary(df: pd.DataFrame) -> pd.DataFrame:
    with_surname = add_surname(df)

    summary = (
        with_surname.groupby("Surname")
        .agg(
            passenger_count=("PassengerId", "count"),
            survived_count=("Survived", "sum"),
            survival_rate=("Survived", "mean"),
            passenger_ids=("PassengerId", lambda values: ", ".join(str(value) for value in values)),
            names=("Name", lambda values: " | ".join(str(value) for value in values)),
        )
        .reset_index()
        .sort_values(
            ["passenger_count", "survival_rate", "Surname"],
            ascending=[False, False, True],
        )
    )

    return summary


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for _, row in df.iterrows():
        values = [str(row[column]) for column in df.columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def build_report(
    id_bin_summary: pd.DataFrame,
    surname_summary: pd.DataFrame,
) -> str:
    top_family_groups = surname_summary[surname_summary["passenger_count"] >= 2].head(30)

    return f"""# Titanic PassengerId and Surname Analysis

## Goal

Check two possible hidden patterns:

1. PassengerId order groups
2. Surname / family groups

## PassengerId Bin Survival

{dataframe_to_markdown(id_bin_summary)}

## Surname Survival Groups

Only surnames with at least 2 passengers are shown below.

{dataframe_to_markdown(top_family_groups)}

## How To Read This

### PassengerId bins

If survival rate changes strongly across PassengerId ranges, there may be ordering/distribution effects.

But be careful:

```text
PassengerId itself should usually not be used as a direct model feature.
It can reveal dataset ordering, but it may not generalize.
```

### Surname groups

If the same surname appears multiple times, it may indicate family members.

This can be useful because families often share:

```text
Ticket
Cabin
Pclass
Fare
Survival conditions
```

Safe feature examples:

```text
SurnameGroupSize
Surname appears in train/test combined
Family group structure
```

Risky feature examples:

```text
Surname target mean survival
Using train survival labels to directly predict test family members
```
"""


def main() -> None:
    df = load_train_data()

    id_bin_summary = build_passenger_id_bin_summary(df)
    surname_summary = build_surname_survival_summary(df)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    id_bin_summary.to_csv(ID_BIN_CSV_PATH, index=False)
    surname_summary.to_csv(SURNAME_CSV_PATH, index=False)
    REPORT_PATH.write_text(
        build_report(id_bin_summary, surname_summary),
        encoding="utf-8",
    )

    print("=== PassengerId Bin Survival ===")
    print(id_bin_summary.to_string(index=False))
    print()

    print("=== Top Surname Groups ===")
    print(
        surname_summary[surname_summary["passenger_count"] >= 2]
        .head(30)
        .to_string(index=False)
    )
    print()

    print("Saved:")
    print("-", ID_BIN_CSV_PATH)
    print("-", SURNAME_CSV_PATH)
    print("-", REPORT_PATH)


if __name__ == "__main__":
    main()
