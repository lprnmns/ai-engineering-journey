from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


RAW_DATA_DIR = Path("competitions/titanic/data/raw")
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

REPORTS_DIR = Path("competitions/titanic/reports")

REPORT_PATH = REPORTS_DIR / "titanic_name_family_id_pattern_analysis.md"
SAFE_FAMILY_FEATURES_TRAIN_PATH = REPORTS_DIR / "titanic_safe_family_features_train.csv"
SAFE_FAMILY_FEATURES_TEST_PATH = REPORTS_DIR / "titanic_safe_family_features_test.csv"
SURNAME_FEATURE_ANALYSIS_PATH = REPORTS_DIR / "titanic_surname_feature_analysis.csv"
PASSENGER_ID_SEGMENT_ANALYSIS_PATH = REPORTS_DIR / "titanic_passenger_id_jump_segments.csv"
RAW_TITLE_ANALYSIS_PATH = REPORTS_DIR / "titanic_raw_title_analysis.csv"
TITLE_GROUP_ANALYSIS_PATH = REPORTS_DIR / "titanic_title_group_analysis.csv"
FIRST_GIVEN_TOKEN_ANALYSIS_PATH = REPORTS_DIR / "titanic_first_given_token_analysis.csv"


@dataclass(frozen=True)
class PassengerIdSegment:
    start_id: int
    end_id: int
    passenger_count: int
    survived_count: int
    survival_rate: float


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    return train, test


def extract_surname(name: str) -> str:
    return str(name).split(",", maxsplit=1)[0].strip()


def extract_raw_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", str(name))
    if match is None:
        return "Unknown"

    return match.group(1).strip()


def extract_text_after_title(name: str) -> str:
    match = re.search(r"\.\s*(.*)$", str(name))
    if match is None:
        return ""

    return match.group(1).strip()


def extract_parentheses_text(name: str) -> str:
    match = re.search(r"\((.*?)\)", str(name))
    if match is None:
        return ""

    return match.group(1).strip()


def extract_first_given_token(name: str) -> str:
    after_title = extract_text_after_title(name)
    cleaned = after_title.replace('"', " ").replace("(", " ").replace(")", " ")
    parts = [part.strip() for part in cleaned.split() if part.strip()]

    if not parts:
        return "Unknown"

    return parts[0]


def map_title_group(raw_title: str) -> str:
    normalized = raw_title.strip()

    replacements = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }

    normalized = replacements.get(normalized, normalized)

    if normalized == "Mr":
        return "CommonMale"

    if normalized in {"Mrs", "Miss"}:
        return "CommonFemale"

    if normalized == "Master":
        return "ChildMale"

    if normalized == "Rev":
        return "Clergy"

    if normalized == "Dr":
        return "Professional"

    if normalized in {"Col", "Major", "Capt"}:
        return "Military"

    if normalized in {"Lady", "Countess", "Dona"}:
        return "NobleFemale"

    if normalized in {"Sir", "Don", "Jonkheer"}:
        return "NobleMale"

    return "RareOther"


def add_name_parts(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["Surname"] = result["Name"].astype(str).apply(extract_surname)
    result["RawTitle"] = result["Name"].astype(str).apply(extract_raw_title)
    result["TitleGroup"] = result["RawTitle"].apply(map_title_group)
    result["TextAfterTitle"] = result["Name"].astype(str).apply(extract_text_after_title)
    result["FirstGivenToken"] = result["Name"].astype(str).apply(extract_first_given_token)
    result["ParenthesesText"] = result["Name"].astype(str).apply(extract_parentheses_text)
    result["HasParenthesesName"] = result["ParenthesesText"].ne("")
    result["HasQuotedNickname"] = result["Name"].astype(str).str.contains('"', regex=False)

    return result


def build_safe_family_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = add_name_parts(train)
    test_parts = add_name_parts(test)

    train_parts["_dataset"] = "train"
    test_parts["_dataset"] = "test"

    combined = pd.concat([train_parts, test_parts], ignore_index=True, sort=False)

    surname_counts = combined["Surname"].value_counts()
    surname_pclass_counts = combined.groupby(["Surname", "Pclass"]).size()
    surname_ticket_counts = combined.groupby(["Surname", "Ticket"]).size()

    combined["SurnameGroupSize"] = combined["Surname"].map(surname_counts).astype(int)
    combined["SurnamePclassGroupSize"] = [
        int(surname_pclass_counts.loc[(row["Surname"], row["Pclass"])])
        for _, row in combined.iterrows()
    ]
    combined["SurnameTicketGroupSize"] = [
        int(surname_ticket_counts.loc[(row["Surname"], row["Ticket"])])
        for _, row in combined.iterrows()
    ]

    combined["IsSoloSurname"] = combined["SurnameGroupSize"] == 1
    combined["IsSmallSurnameGroup"] = combined["SurnameGroupSize"].between(2, 3)
    combined["IsLargeSurnameGroup"] = combined["SurnameGroupSize"] >= 4
    combined["IsLargeSurnamePclassGroup"] = combined["SurnamePclassGroupSize"] >= 4

    feature_columns = [
        "PassengerId",
        "Surname",
        "RawTitle",
        "TitleGroup",
        "FirstGivenToken",
        "HasParenthesesName",
        "HasQuotedNickname",
        "SurnameGroupSize",
        "SurnamePclassGroupSize",
        "SurnameTicketGroupSize",
        "IsSoloSurname",
        "IsSmallSurnameGroup",
        "IsLargeSurnameGroup",
        "IsLargeSurnamePclassGroup",
    ]

    train_features = combined[combined["_dataset"] == "train"][feature_columns + ["Survived"]].copy()
    test_features = combined[combined["_dataset"] == "test"][feature_columns].copy()

    return train_features, test_features


def summarize_categorical_survival(
    df: pd.DataFrame,
    column: str,
    min_count: int = 1,
) -> pd.DataFrame:
    summary = (
        df.groupby(column, dropna=False)
        .agg(
            passenger_count=("PassengerId", "count"),
            survived_count=("Survived", "sum"),
            survival_rate=("Survived", "mean"),
        )
        .reset_index()
        .sort_values(
            ["passenger_count", "survival_rate", column],
            ascending=[False, False, True],
        )
    )

    return summary[summary["passenger_count"] >= min_count].copy()


def build_surname_feature_analysis(train_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for column in [
        "SurnameGroupSize",
        "SurnamePclassGroupSize",
        "SurnameTicketGroupSize",
        "IsSoloSurname",
        "IsSmallSurnameGroup",
        "IsLargeSurnameGroup",
        "IsLargeSurnamePclassGroup",
    ]:
        summary = summarize_categorical_survival(train_features, column)
        summary.insert(0, "feature", column)
        summary = summary.rename(columns={column: "value"})
        rows.append(summary)

    return pd.concat(rows, ignore_index=True, sort=False)


def build_passenger_id_base_bins(
    train: pd.DataFrame,
    n_bins: int = 20,
) -> pd.DataFrame:
    result = train.copy()
    result["PassengerIdBaseBin"] = pd.cut(
        result["PassengerId"],
        bins=n_bins,
        include_lowest=True,
    )

    summary = (
        result.groupby("PassengerIdBaseBin", observed=True)
        .agg(
            passenger_count=("PassengerId", "count"),
            start_id=("PassengerId", "min"),
            end_id=("PassengerId", "max"),
            survived_count=("Survived", "sum"),
            survival_rate=("Survived", "mean"),
        )
        .reset_index(drop=True)
    )

    summary["delta_from_previous"] = summary["survival_rate"].diff().abs().fillna(0.0)

    return summary


def detect_passenger_id_jump_boundaries(
    base_bins: pd.DataFrame,
    min_delta: float = 0.12,
) -> list[int]:
    boundaries: list[int] = []

    records = base_bins.to_dict("records")

    for index, row in enumerate(records):
        if index == 0:
            continue

        if float(row["delta_from_previous"]) >= min_delta:
            previous_end = int(records[index - 1]["end_id"])
            boundaries.append(previous_end)

    return boundaries


def assign_passenger_id_jump_segment(passenger_id: int, boundaries: list[int]) -> str:
    start = 1

    for boundary in boundaries:
        if passenger_id <= boundary:
            return f"{start}-{boundary}"
        start = boundary + 1

    return f"{start}-891"


def build_passenger_id_jump_segments(
    train: pd.DataFrame,
    n_bins: int = 20,
    min_delta: float = 0.12,
) -> pd.DataFrame:
    base_bins = build_passenger_id_base_bins(train, n_bins=n_bins)
    boundaries = detect_passenger_id_jump_boundaries(base_bins, min_delta=min_delta)

    result = train.copy()
    result["PassengerIdJumpSegment"] = result["PassengerId"].apply(
        lambda passenger_id: assign_passenger_id_jump_segment(int(passenger_id), boundaries)
    )

    segment_summary = (
        result.groupby("PassengerIdJumpSegment")
        .agg(
            passenger_count=("PassengerId", "count"),
            start_id=("PassengerId", "min"),
            end_id=("PassengerId", "max"),
            survived_count=("Survived", "sum"),
            survival_rate=("Survived", "mean"),
        )
        .reset_index()
        .sort_values("start_id")
    )

    segment_summary["method"] = f"20 base bins, boundary when adjacent survival delta >= {min_delta}"

    return segment_summary


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows).copy()

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
    surname_feature_analysis: pd.DataFrame,
    passenger_id_segments: pd.DataFrame,
    raw_title_analysis: pd.DataFrame,
    title_group_analysis: pd.DataFrame,
    first_given_token_analysis: pd.DataFrame,
) -> str:
    return f"""# Titanic Name, Family, and PassengerId Pattern Analysis

## Goal

Explore three possible pattern sources:

1. Safe surname/family features that do not use target labels directly.
2. PassengerId jump segments based on large survival-rate changes across ID ranges.
3. Name/title tokens, including raw title and first given token.

## Safe Family Feature Candidates

These features are safe because they are based on train+test structure/counts, not on target mean survival.

Examples:

```text
SurnameGroupSize
SurnamePclassGroupSize
SurnameTicketGroupSize
IsSoloSurname
IsLargeSurnameGroup
```

### Family Feature Survival Analysis

{dataframe_to_markdown(surname_feature_analysis, max_rows=40)}

## PassengerId Jump Segments

PassengerId may reveal dataset-order patterns, but it is risky as a model feature.

These segments are exploratory. They are created by first splitting PassengerId into 20 small bins, then placing a boundary where adjacent bin survival rates jump by at least 0.12.

{dataframe_to_markdown(passenger_id_segments)}

## Raw Title Analysis

RawTitle is the title/status text between comma and period in the Titanic name.

Example:

```text
"Carter, Rev. Ernest Courtenay" -> RawTitle = Rev
```

{dataframe_to_markdown(raw_title_analysis, max_rows=40)}

## Title Group Analysis

Raw titles are mapped into broader status groups.

{dataframe_to_markdown(title_group_analysis, max_rows=40)}

## First Given Token Analysis

This extracts the first token after the title.

Example:

```text
"Carter, Rev. Ernest Courtenay" -> FirstGivenToken = Ernest
```

This is exploratory and can be noisy. Use it for pattern discovery, not direct modeling unless validated carefully.

{dataframe_to_markdown(first_given_token_analysis, max_rows=50)}

## Strong Warning

Do not directly use survival_rate columns as model features.

That would use the answer key from train labels and can become target leakage.

Safer next modeling candidates:

```text
SurnameGroupSize
SurnamePclassGroupSize
SurnameTicketGroupSize
TitleGroup
HasParenthesesName
HasQuotedNickname
```

Riskier candidates:

```text
PassengerIdJumpSegment
FirstGivenToken
Any target mean survival by Surname/Title/Token
```
"""


def run_pipeline() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_raw_data()

    train_features, test_features = build_safe_family_features(train, test)
    train_features.to_csv(SAFE_FAMILY_FEATURES_TRAIN_PATH, index=False)
    test_features.to_csv(SAFE_FAMILY_FEATURES_TEST_PATH, index=False)

    surname_feature_analysis = build_surname_feature_analysis(train_features)
    surname_feature_analysis.to_csv(SURNAME_FEATURE_ANALYSIS_PATH, index=False)

    passenger_id_segments = build_passenger_id_jump_segments(train, n_bins=20, min_delta=0.12)
    passenger_id_segments.to_csv(PASSENGER_ID_SEGMENT_ANALYSIS_PATH, index=False)

    train_name_parts = add_name_parts(train)

    raw_title_analysis = summarize_categorical_survival(train_name_parts, "RawTitle")
    raw_title_analysis.to_csv(RAW_TITLE_ANALYSIS_PATH, index=False)

    title_group_analysis = summarize_categorical_survival(train_name_parts, "TitleGroup")
    title_group_analysis.to_csv(TITLE_GROUP_ANALYSIS_PATH, index=False)

    first_given_token_analysis = summarize_categorical_survival(
        train_name_parts,
        "FirstGivenToken",
        min_count=2,
    )
    first_given_token_analysis.to_csv(FIRST_GIVEN_TOKEN_ANALYSIS_PATH, index=False)

    REPORT_PATH.write_text(
        build_report(
            surname_feature_analysis=surname_feature_analysis,
            passenger_id_segments=passenger_id_segments,
            raw_title_analysis=raw_title_analysis,
            title_group_analysis=title_group_analysis,
            first_given_token_analysis=first_given_token_analysis,
        ),
        encoding="utf-8",
    )

    print("=== Safe Family Feature Analysis ===")
    print(surname_feature_analysis.head(40).to_string(index=False))
    print()

    print("=== PassengerId Jump Segments ===")
    print(passenger_id_segments.to_string(index=False))
    print()

    print("=== Raw Title Analysis ===")
    print(raw_title_analysis.to_string(index=False))
    print()

    print("=== Title Group Analysis ===")
    print(title_group_analysis.to_string(index=False))
    print()

    print("=== First Given Token Analysis ===")
    print(first_given_token_analysis.head(50).to_string(index=False))
    print()

    print("Saved:")
    print("-", SAFE_FAMILY_FEATURES_TRAIN_PATH)
    print("-", SAFE_FAMILY_FEATURES_TEST_PATH)
    print("-", SURNAME_FEATURE_ANALYSIS_PATH)
    print("-", PASSENGER_ID_SEGMENT_ANALYSIS_PATH)
    print("-", RAW_TITLE_ANALYSIS_PATH)
    print("-", TITLE_GROUP_ANALYSIS_PATH)
    print("-", FIRST_GIVEN_TOKEN_ANALYSIS_PATH)
    print("-", REPORT_PATH)


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
