from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from competitions.titanic.src.feature_v1 import add_feature_v1_columns


RAW_DATA_DIR = Path("competitions/titanic/data/raw")
TEST_PATH = RAW_DATA_DIR / "test.csv"

SUBMISSIONS_DIR = Path("competitions/titanic/submissions")
REPORTS_DIR = Path("competitions/titanic/reports")

CURRENT_BEST_SUBMISSION_PATH = SUBMISSIONS_DIR / "model_comparison_best_submission.csv"

CANDIDATE_SUBMISSIONS = {
    "deck_ticket_fare": SUBMISSIONS_DIR / "deck_ticket_fare_hgb_submission.csv",
    "hgb_tuned": SUBMISSIONS_DIR / "hgb_tuned_submission.csv",
    "rare_title_interactions": SUBMISSIONS_DIR / "rare_title_interactions_hgb_submission.csv",
}

DIFFERENCE_ROWS_PATH = REPORTS_DIR / "titanic_submission_difference_rows.csv"
DIFFERENCE_SEGMENTS_PATH = REPORTS_DIR / "titanic_submission_difference_segments.csv"
REPORT_PATH = REPORTS_DIR / "titanic_submission_difference_analysis.md"


@dataclass(frozen=True)
class SubmissionDifferenceSummary:
    candidate_name: str
    changed_count: int
    changed_rate: float
    best_survived_to_candidate_died: int
    best_died_to_candidate_survived: int


def load_test_features() -> pd.DataFrame:
    test_raw = pd.read_csv(TEST_PATH)
    featured = add_feature_v1_columns(test_raw)

    return featured


def load_submission(path: Path, prediction_column_name: str) -> pd.DataFrame:
    submission = pd.read_csv(path)

    return submission.rename(columns={"Survived": prediction_column_name})


def build_submission_differences(
    test_features: pd.DataFrame,
    best_submission: pd.DataFrame,
    candidate_submission: pd.DataFrame,
    candidate_name: str,
) -> pd.DataFrame:
    best_column = "Survived_best"
    candidate_column = f"Survived_{candidate_name}"

    merged = test_features.merge(
        best_submission,
        on="PassengerId",
        how="inner",
    ).merge(
        candidate_submission,
        on="PassengerId",
        how="inner",
    )

    changed = merged[merged[best_column] != merged[candidate_column]].copy()
    changed["candidate_name"] = candidate_name
    changed["change_direction"] = changed[best_column].astype(str) + "_to_" + changed[candidate_column].astype(str)

    return changed


def summarize_difference(
    changed: pd.DataFrame,
    candidate_name: str,
    test_row_count: int,
) -> SubmissionDifferenceSummary:
    best_survived_to_candidate_died = int((changed["change_direction"] == "1_to_0").sum())
    best_died_to_candidate_survived = int((changed["change_direction"] == "0_to_1").sum())

    return SubmissionDifferenceSummary(
        candidate_name=candidate_name,
        changed_count=len(changed),
        changed_rate=len(changed) / test_row_count,
        best_survived_to_candidate_died=best_survived_to_candidate_died,
        best_died_to_candidate_survived=best_died_to_candidate_survived,
    )


def summarize_changed_segments(
    all_changed: pd.DataFrame,
    segment_columns: list[str] | None = None,
) -> pd.DataFrame:
    if segment_columns is None:
        segment_columns = ["candidate_name", "change_direction", "Sex", "Pclass", "Title", "Embarked"]

    rows: list[dict[str, object]] = []

    for candidate_name, candidate_df in all_changed.groupby("candidate_name"):
        for column in segment_columns:
            if column == "candidate_name":
                continue

            grouped = candidate_df.groupby(column, dropna=False).size().reset_index(name="changed_rows")

            for _, row in grouped.iterrows():
                rows.append(
                    {
                        "candidate_name": str(candidate_name),
                        "segment": column,
                        "value": str(row[column]),
                        "changed_rows": int(row["changed_rows"]),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["candidate_name", "segment", "changed_rows"],
        ascending=[True, True, False],
    )


def build_summary_table(summaries: list[SubmissionDifferenceSummary]) -> str:
    lines = [
        "| Candidate | Changed rows | Changed rate | Best 1->Candidate 0 | Best 0->Candidate 1 |",
        "|---|---:|---:|---:|---:|",
    ]

    for summary in summaries:
        lines.append(
            f"| {summary.candidate_name} | {summary.changed_count} | "
            f"{summary.changed_rate:.2%} | {summary.best_survived_to_candidate_died} | "
            f"{summary.best_died_to_candidate_survived} |"
        )

    return "\n".join(lines)


def build_top_changed_segment_table(segment_summary: pd.DataFrame) -> str:
    lines = [
        "| Candidate | Segment | Value | Changed rows |",
        "|---|---|---|---:|",
    ]

    top_rows = (
        segment_summary
        .sort_values(["candidate_name", "changed_rows"], ascending=[True, False])
        .groupby("candidate_name")
        .head(8)
    )

    for _, row in top_rows.iterrows():
        lines.append(
            f"| {row['candidate_name']} | {row['segment']} | {row['value']} | {int(row['changed_rows'])} |"
        )

    return "\n".join(lines)


def build_report(
    summaries: list[SubmissionDifferenceSummary],
    segment_summary: pd.DataFrame,
) -> str:
    code_fence = chr(96) * 3

    return f"""# Titanic Submission Difference Analysis

## Goal

Understand why recent submissions damaged public leaderboard score.

We compare each weaker submission against the current best submission:

{code_fence}text
current best = model_comparison_best_submission.csv
public LB = 0.77272
{code_fence}

The key question:

{code_fence}text
Which passengers did the weaker models change?
What do those changed passengers have in common?
{code_fence}

## Candidate Difference Summary

{build_summary_table(summaries)}

## Top Changed Segments

{build_top_changed_segment_table(segment_summary)}

## Interpretation

If a candidate changes many predictions and public LB drops, those changes are probably moving in the wrong direction.

This analysis does not reveal true Kaggle labels.

It helps identify dangerous feature/model changes that are repeatedly flipping similar passenger groups.

## Next Decision Rule

Do not trust a new experiment only because CV improves.

Before submitting, compare changed rows against current best:

{code_fence}text
small meaningful change + stable CV = possible submission
large unstable change + CV/public mismatch history = high risk
{code_fence}

## Output Files

- `{DIFFERENCE_ROWS_PATH}`
- `{DIFFERENCE_SEGMENTS_PATH}`
"""


def run_pipeline() -> tuple[list[SubmissionDifferenceSummary], pd.DataFrame, pd.DataFrame]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    test_features = load_test_features()
    best_submission = load_submission(CURRENT_BEST_SUBMISSION_PATH, "Survived_best")

    changed_frames = []
    summaries = []

    for candidate_name, candidate_path in CANDIDATE_SUBMISSIONS.items():
        if not candidate_path.exists():
            continue

        candidate_submission = load_submission(candidate_path, f"Survived_{candidate_name}")
        changed = build_submission_differences(
            test_features=test_features,
            best_submission=best_submission,
            candidate_submission=candidate_submission,
            candidate_name=candidate_name,
        )

        changed_frames.append(changed)
        summaries.append(
            summarize_difference(
                changed=changed,
                candidate_name=candidate_name,
                test_row_count=len(test_features),
            )
        )

    if changed_frames:
        all_changed = pd.concat(changed_frames, ignore_index=True, sort=False)
    else:
        all_changed = pd.DataFrame()

    all_changed.to_csv(DIFFERENCE_ROWS_PATH, index=False)

    if all_changed.empty:
        segment_summary = pd.DataFrame(columns=["candidate_name", "segment", "value", "changed_rows"])
    else:
        segment_summary = summarize_changed_segments(all_changed)

    segment_summary.to_csv(DIFFERENCE_SEGMENTS_PATH, index=False)
    REPORT_PATH.write_text(build_report(summaries, segment_summary), encoding="utf-8")

    return summaries, all_changed, segment_summary


def main() -> None:
    summaries, all_changed, segment_summary = run_pipeline()

    print("=== Titanic Submission Difference Analysis ===")
    for summary in summaries:
        print(
            f"{summary.candidate_name}: "
            f"changed={summary.changed_count} "
            f"rate={summary.changed_rate:.2%} "
            f"1_to_0={summary.best_survived_to_candidate_died} "
            f"0_to_1={summary.best_died_to_candidate_survived}"
        )

    print()
    print(f"Total changed rows stored: {len(all_changed)}")
    print(f"Segment rows stored: {len(segment_summary)}")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Saved changed rows to: {DIFFERENCE_ROWS_PATH}")
    print(f"Saved segment summary to: {DIFFERENCE_SEGMENTS_PATH}")


if __name__ == "__main__":
    main()
