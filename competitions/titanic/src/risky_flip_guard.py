from __future__ import annotations

import argparse
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

DEFAULT_CANDIDATE_SUBMISSIONS = {
    "deck_ticket_fare": SUBMISSIONS_DIR / "deck_ticket_fare_hgb_submission.csv",
    "hgb_tuned": SUBMISSIONS_DIR / "hgb_tuned_submission.csv",
    "rare_title_interactions": SUBMISSIONS_DIR / "rare_title_interactions_hgb_submission.csv",
    "repeated_cv_best": SUBMISSIONS_DIR / "repeated_cv_best_submission.csv",
}

GUARD_SUMMARY_PATH = REPORTS_DIR / "titanic_risky_flip_guard_summary.csv"
GUARD_ROWS_PATH = REPORTS_DIR / "titanic_risky_flip_guard_rows.csv"
GUARD_REPORT_PATH = REPORTS_DIR / "titanic_risky_flip_guard_report.md"

MAX_TOTAL_CHANGED_FOR_SAFE_SUBMIT = 12
MAX_RISKY_0_TO_1_FOR_SAFE_SUBMIT = 6
MAX_PCLASS3_FEMALE_0_TO_1_FOR_SAFE_SUBMIT = 3
MAX_MISS_MRS_0_TO_1_FOR_SAFE_SUBMIT = 4
MAX_EMBARKED_SC_0_TO_1_FOR_SAFE_SUBMIT = 6


@dataclass(frozen=True)
class GuardSummary:
    candidate_name: str
    candidate_path: str
    changed_count: int
    changed_rate: float
    total_0_to_1: int
    total_1_to_0: int
    pclass3_female_0_to_1: int
    miss_mrs_0_to_1: int
    embarked_sc_0_to_1: int
    pclass3_0_to_1: int
    decision: str
    reason: str


def load_test_features(path: Path = TEST_PATH) -> pd.DataFrame:
    test_raw = pd.read_csv(path)
    return add_feature_v1_columns(test_raw)


def load_submission(path: Path, prediction_column_name: str) -> pd.DataFrame:
    submission = pd.read_csv(path)
    return submission.rename(columns={"Survived": prediction_column_name})


def build_flip_rows(
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

    changed["is_0_to_1"] = changed["change_direction"] == "0_to_1"
    changed["is_1_to_0"] = changed["change_direction"] == "1_to_0"
    changed["is_pclass3_female_0_to_1"] = (
        changed["is_0_to_1"]
        & (changed["Pclass"] == 3)
        & (changed["Sex"].astype(str) == "female")
    )
    changed["is_miss_mrs_0_to_1"] = (
        changed["is_0_to_1"]
        & changed["Title"].astype(str).isin(["Miss", "Mrs"])
    )
    changed["is_embarked_sc_0_to_1"] = (
        changed["is_0_to_1"]
        & changed["Embarked"].astype(str).isin(["S", "C"])
    )
    changed["is_pclass3_0_to_1"] = (
        changed["is_0_to_1"]
        & (changed["Pclass"] == 3)
    )

    return changed


def decide_guard_result(
    changed_count: int,
    total_0_to_1: int,
    pclass3_female_0_to_1: int,
    miss_mrs_0_to_1: int,
    embarked_sc_0_to_1: int,
) -> tuple[str, str]:
    if changed_count == 0:
        return (
            "NO_NEW_INFORMATION",
            "Candidate is identical to current best submission.",
        )

    blocking_reasons = []

    if changed_count > MAX_TOTAL_CHANGED_FOR_SAFE_SUBMIT:
        blocking_reasons.append(
            f"changed_count {changed_count} exceeds {MAX_TOTAL_CHANGED_FOR_SAFE_SUBMIT}"
        )

    if total_0_to_1 > MAX_RISKY_0_TO_1_FOR_SAFE_SUBMIT:
        blocking_reasons.append(
            f"0_to_1 flips {total_0_to_1} exceed {MAX_RISKY_0_TO_1_FOR_SAFE_SUBMIT}"
        )

    if pclass3_female_0_to_1 > MAX_PCLASS3_FEMALE_0_TO_1_FOR_SAFE_SUBMIT:
        blocking_reasons.append(
            "Pclass=3 female 0_to_1 flips "
            f"{pclass3_female_0_to_1} exceed {MAX_PCLASS3_FEMALE_0_TO_1_FOR_SAFE_SUBMIT}"
        )

    if miss_mrs_0_to_1 > MAX_MISS_MRS_0_TO_1_FOR_SAFE_SUBMIT:
        blocking_reasons.append(
            f"Miss/Mrs 0_to_1 flips {miss_mrs_0_to_1} exceed {MAX_MISS_MRS_0_TO_1_FOR_SAFE_SUBMIT}"
        )

    if embarked_sc_0_to_1 > MAX_EMBARKED_SC_0_TO_1_FOR_SAFE_SUBMIT:
        blocking_reasons.append(
            f"Embarked S/C 0_to_1 flips {embarked_sc_0_to_1} exceed {MAX_EMBARKED_SC_0_TO_1_FOR_SAFE_SUBMIT}"
        )

    if blocking_reasons:
        return (
            "DO_NOT_SUBMIT_WITHOUT_REVIEW",
            "; ".join(blocking_reasons),
        )

    return (
        "REVIEW_OK",
        "Flip profile is within current guard thresholds. Manual review still recommended.",
    )


def summarize_guard(
    changed: pd.DataFrame,
    candidate_name: str,
    candidate_path: Path,
    test_row_count: int,
) -> GuardSummary:
    changed_count = len(changed)
    total_0_to_1 = int(changed["is_0_to_1"].sum()) if changed_count else 0
    total_1_to_0 = int(changed["is_1_to_0"].sum()) if changed_count else 0
    pclass3_female_0_to_1 = int(changed["is_pclass3_female_0_to_1"].sum()) if changed_count else 0
    miss_mrs_0_to_1 = int(changed["is_miss_mrs_0_to_1"].sum()) if changed_count else 0
    embarked_sc_0_to_1 = int(changed["is_embarked_sc_0_to_1"].sum()) if changed_count else 0
    pclass3_0_to_1 = int(changed["is_pclass3_0_to_1"].sum()) if changed_count else 0

    decision, reason = decide_guard_result(
        changed_count=changed_count,
        total_0_to_1=total_0_to_1,
        pclass3_female_0_to_1=pclass3_female_0_to_1,
        miss_mrs_0_to_1=miss_mrs_0_to_1,
        embarked_sc_0_to_1=embarked_sc_0_to_1,
    )

    return GuardSummary(
        candidate_name=candidate_name,
        candidate_path=str(candidate_path),
        changed_count=changed_count,
        changed_rate=changed_count / test_row_count,
        total_0_to_1=total_0_to_1,
        total_1_to_0=total_1_to_0,
        pclass3_female_0_to_1=pclass3_female_0_to_1,
        miss_mrs_0_to_1=miss_mrs_0_to_1,
        embarked_sc_0_to_1=embarked_sc_0_to_1,
        pclass3_0_to_1=pclass3_0_to_1,
        decision=decision,
        reason=reason,
    )


def summaries_to_dataframe(summaries: list[GuardSummary]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_name": summary.candidate_name,
                "candidate_path": summary.candidate_path,
                "changed_count": summary.changed_count,
                "changed_rate": summary.changed_rate,
                "total_0_to_1": summary.total_0_to_1,
                "total_1_to_0": summary.total_1_to_0,
                "pclass3_female_0_to_1": summary.pclass3_female_0_to_1,
                "miss_mrs_0_to_1": summary.miss_mrs_0_to_1,
                "embarked_sc_0_to_1": summary.embarked_sc_0_to_1,
                "pclass3_0_to_1": summary.pclass3_0_to_1,
                "decision": summary.decision,
                "reason": summary.reason,
            }
            for summary in summaries
        ]
    )


def build_guard_table(summaries: list[GuardSummary]) -> str:
    lines = [
        "| Candidate | Changed | 0->1 | 1->0 | P3 female 0->1 | Miss/Mrs 0->1 | Embarked S/C 0->1 | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for summary in summaries:
        lines.append(
            f"| {summary.candidate_name} | {summary.changed_count} | {summary.total_0_to_1} | "
            f"{summary.total_1_to_0} | {summary.pclass3_female_0_to_1} | "
            f"{summary.miss_mrs_0_to_1} | {summary.embarked_sc_0_to_1} | {summary.decision} |"
        )

    return "\n".join(lines)


def build_top_risky_rows_table(all_changed: pd.DataFrame) -> str:
    if all_changed.empty:
        return "No changed rows."

    risky = all_changed[
        all_changed["is_pclass3_female_0_to_1"]
        | all_changed["is_miss_mrs_0_to_1"]
        | all_changed["is_embarked_sc_0_to_1"]
    ].copy()

    if risky.empty:
        return "No risky rows under current rule set."

    columns = [
        "candidate_name",
        "PassengerId",
        "Pclass",
        "Sex",
        "Title",
        "Embarked",
        "change_direction",
    ]

    lines = [
        "| Candidate | PassengerId | Pclass | Sex | Title | Embarked | Change |",
        "|---|---:|---:|---|---|---|---|",
    ]

    for _, row in risky[columns].head(25).iterrows():
        lines.append(
            f"| {row['candidate_name']} | {int(row['PassengerId'])} | {int(row['Pclass'])} | "
            f"{row['Sex']} | {row['Title']} | {row['Embarked']} | {row['change_direction']} |"
        )

    return "\n".join(lines)


def build_report(summaries: list[GuardSummary], all_changed: pd.DataFrame) -> str:
    code_fence = chr(96) * 3

    return f"""# Titanic Risky Flip Guard Report

## Goal

Create a pre-submission guardrail against risky prediction flips.

Current best public submission:

{code_fence}text
model_comparison_best_submission.csv
public LB = 0.77272
{code_fence}

Recent failed submissions showed a pattern:

{code_fence}text
Too many current-best 0 predictions were flipped to 1.
Many risky flips involved Pclass=3, female, Miss/Mrs, and Embarked S/C groups.
{code_fence}

## Guard Thresholds

{code_fence}text
MAX_TOTAL_CHANGED_FOR_SAFE_SUBMIT = {MAX_TOTAL_CHANGED_FOR_SAFE_SUBMIT}
MAX_RISKY_0_TO_1_FOR_SAFE_SUBMIT = {MAX_RISKY_0_TO_1_FOR_SAFE_SUBMIT}
MAX_PCLASS3_FEMALE_0_TO_1_FOR_SAFE_SUBMIT = {MAX_PCLASS3_FEMALE_0_TO_1_FOR_SAFE_SUBMIT}
MAX_MISS_MRS_0_TO_1_FOR_SAFE_SUBMIT = {MAX_MISS_MRS_0_TO_1_FOR_SAFE_SUBMIT}
MAX_EMBARKED_SC_0_TO_1_FOR_SAFE_SUBMIT = {MAX_EMBARKED_SC_0_TO_1_FOR_SAFE_SUBMIT}
{code_fence}

## Guard Summary

{build_guard_table(summaries)}

## Top Risky Rows

{build_top_risky_rows_table(all_changed)}

## Interpretation

This guard does not know the real Kaggle labels.

It is not a replacement for validation.

It is a submission safety check based on observed public leaderboard failures.

## Decision Rule

{code_fence}text
If decision = DO_NOT_SUBMIT_WITHOUT_REVIEW:
    Do not submit just because CV improved.

If decision = REVIEW_OK:
    Submission may be reasonable, but still inspect changed rows.

If decision = NO_NEW_INFORMATION:
    Submission is identical to current best and should not be sent.
{code_fence}

## Output Files

- `{GUARD_SUMMARY_PATH}`
- `{GUARD_ROWS_PATH}`
"""


def parse_candidate_args(candidate_args: list[str]) -> dict[str, Path]:
    candidates: dict[str, Path] = {}

    for raw_value in candidate_args:
        if "=" not in raw_value:
            raise ValueError(
                "Candidate arguments must use name=path format, "
                f"got: {raw_value}"
            )

        name, raw_path = raw_value.split("=", 1)
        candidates[name] = Path(raw_path)

    return candidates


def run_guard(candidate_submissions: dict[str, Path]) -> tuple[list[GuardSummary], pd.DataFrame]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    test_features = load_test_features()
    best_submission = load_submission(CURRENT_BEST_SUBMISSION_PATH, "Survived_best")

    summaries: list[GuardSummary] = []
    changed_frames: list[pd.DataFrame] = []

    for candidate_name, candidate_path in candidate_submissions.items():
        if not candidate_path.exists():
            continue

        candidate_submission = load_submission(candidate_path, f"Survived_{candidate_name}")
        changed = build_flip_rows(
            test_features=test_features,
            best_submission=best_submission,
            candidate_submission=candidate_submission,
            candidate_name=candidate_name,
        )

        summaries.append(
            summarize_guard(
                changed=changed,
                candidate_name=candidate_name,
                candidate_path=candidate_path,
                test_row_count=len(test_features),
            )
        )
        changed_frames.append(changed)

    if changed_frames:
        all_changed = pd.concat(changed_frames, ignore_index=True, sort=False)
    else:
        all_changed = pd.DataFrame()

    summaries_to_dataframe(summaries).to_csv(GUARD_SUMMARY_PATH, index=False)
    all_changed.to_csv(GUARD_ROWS_PATH, index=False)
    GUARD_REPORT_PATH.write_text(build_report(summaries, all_changed), encoding="utf-8")

    return summaries, all_changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate submission in name=path format. Can be passed multiple times.",
    )
    args = parser.parse_args()

    if args.candidate:
        candidate_submissions = parse_candidate_args([str(value) for value in args.candidate])
    else:
        candidate_submissions = DEFAULT_CANDIDATE_SUBMISSIONS

    summaries, all_changed = run_guard(candidate_submissions)

    print("=== Titanic Risky Flip Guard ===")
    for summary in summaries:
        print(
            f"{summary.candidate_name}: "
            f"changed={summary.changed_count} "
            f"0_to_1={summary.total_0_to_1} "
            f"p3_female_0_to_1={summary.pclass3_female_0_to_1} "
            f"miss_mrs_0_to_1={summary.miss_mrs_0_to_1} "
            f"decision={summary.decision}"
        )

    print()
    print(f"Total changed rows stored: {len(all_changed)}")
    print(f"Saved guard summary to: {GUARD_SUMMARY_PATH}")
    print(f"Saved guard rows to: {GUARD_ROWS_PATH}")
    print(f"Saved guard report to: {GUARD_REPORT_PATH}")


if __name__ == "__main__":
    main()
