from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


EXPERIMENTS_PATH = Path("competitions/titanic/experiments/experiments.csv")
GAP_REPORT_PATH = Path("competitions/titanic/reports/titanic_080_gap_analysis.md")
PRIORITY_BACKLOG_PATH = Path("competitions/titanic/reports/titanic_priority_experiments.csv")

TARGET_PUBLIC_SCORE = 0.80
TITANIC_TEST_ROWS = 418


@dataclass(frozen=True)
class ExperimentSummary:
    experiment_id: str
    features: str
    model: str
    local_score: str
    public_lb_score: float
    notes: str


@dataclass(frozen=True)
class ScoreGap:
    current_score: float
    target_score: float
    test_rows: int
    estimated_current_correct: int
    estimated_target_correct: int
    estimated_extra_correct_needed: int


def parse_public_score(value: str) -> float:
    cleaned = value.strip()
    if not cleaned:
        return 0.0

    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def load_experiments(path: Path = EXPERIMENTS_PATH) -> list[ExperimentSummary]:
    if not path.exists():
        return []

    experiments: list[ExperimentSummary] = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for raw_row in reader:
            row = {key: str(value) for key, value in raw_row.items()}

            experiments.append(
                ExperimentSummary(
                    experiment_id=row.get("experiment_id", ""),
                    features=row.get("features", ""),
                    model=row.get("model", ""),
                    local_score=row.get("local_score", ""),
                    public_lb_score=parse_public_score(row.get("public_lb_score", "")),
                    notes=row.get("notes", ""),
                )
            )

    return experiments


def find_best_public_experiment(
    experiments: list[ExperimentSummary],
) -> ExperimentSummary:
    if not experiments:
        raise ValueError("No experiments found.")

    return max(experiments, key=lambda experiment: experiment.public_lb_score)


def calculate_score_gap(
    current_score: float,
    target_score: float = TARGET_PUBLIC_SCORE,
    test_rows: int = TITANIC_TEST_ROWS,
) -> ScoreGap:
    estimated_current_correct = round(current_score * test_rows)
    estimated_target_correct = math.ceil(target_score * test_rows)
    estimated_extra_correct_needed = estimated_target_correct - estimated_current_correct

    return ScoreGap(
        current_score=current_score,
        target_score=target_score,
        test_rows=test_rows,
        estimated_current_correct=estimated_current_correct,
        estimated_target_correct=estimated_target_correct,
        estimated_extra_correct_needed=max(0, estimated_extra_correct_needed),
    )


def build_priority_experiments() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "experiment_id": "titanic_exp_004_age_imputation_v2",
            "hypothesis": "Age global median is too crude. Title + Pclass + Sex grouped age imputation may improve generalization.",
            "change": "Impute Age by Title/Pclass/Sex group median; add AgeBin.",
            "risk": "Small groups can overfit or create noisy bins.",
            "submit_rule": "Submit only if CV improves and predictions differ meaningfully from exp_003.",
        },
        {
            "priority": "2",
            "experiment_id": "titanic_exp_005_deck_ticket_fare_person",
            "hypothesis": "Deck, TicketGroupSize, and FarePerPerson may capture status/group travel better than raw Cabin/Fare.",
            "change": "Add Deck from Cabin, TicketGroupSize from combined train+test, FarePerPerson = Fare / TicketGroupSize.",
            "risk": "Cabin is sparse; Ticket may add noise.",
            "submit_rule": "Submit if CV improves or model produces a complementary prediction pattern.",
        },
        {
            "priority": "3",
            "experiment_id": "titanic_exp_006_hgb_tuning",
            "hypothesis": "HistGradientBoosting is the current best model; tuning may improve public score.",
            "change": "Tune max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization.",
            "risk": "Can overfit local CV because Titanic is small.",
            "submit_rule": "Submit only if repeated CV remains stable.",
        },
        {
            "priority": "4",
            "experiment_id": "titanic_exp_007_catboost_lightgbm",
            "hypothesis": "Gradient boosting libraries may handle tabular patterns better than sklearn defaults.",
            "change": "Try CatBoost/LightGBM if dependencies are acceptable.",
            "risk": "Extra library complexity; small data may not benefit much.",
            "submit_rule": "Submit if CV improves and prediction distribution is not extreme.",
        },
        {
            "priority": "5",
            "experiment_id": "titanic_exp_008_simple_ensemble",
            "hypothesis": "Different models may make different errors; voting or averaging can improve stability.",
            "change": "Blend LogisticRegression, RandomForest, HistGradientBoosting predictions.",
            "risk": "Ensemble may overfit public LB if abused.",
            "submit_rule": "Submit once after local agreement/error analysis.",
        },
    ]


def write_priority_backlog(
    rows: list[dict[str, str]],
    path: Path = PRIORITY_BACKLOG_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_experiment_table(experiments: list[ExperimentSummary]) -> str:
    lines = [
        "| Experiment | Model | Public LB | Local score |",
        "|---|---|---:|---:|",
    ]

    for experiment in experiments:
        lines.append(
            f"| {experiment.experiment_id} | {experiment.model} | "
            f"{experiment.public_lb_score:.5f} | {experiment.local_score} |"
        )

    return "\n".join(lines)


def build_priority_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Priority | Experiment | Hypothesis |",
        "|---:|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row['priority']} | {row['experiment_id']} | {row['hypothesis']} |"
        )

    return "\n".join(lines)


def build_gap_report(
    experiments: list[ExperimentSummary],
    gap: ScoreGap,
    priority_rows: list[dict[str, str]],
) -> str:
    best = find_best_public_experiment(experiments)

    code_fence = chr(96) * 3

    return f"""# Titanic 0.80 Gap Analysis

## Goal

Understand what is missing between the current best honest score and the 0.80+ target.

## Current Best Experiment

- Experiment: `{best.experiment_id}`
- Model: `{best.model}`
- Public LB: `{best.public_lb_score:.5f}`
- Features: `{best.features}`

## Score Gap

{code_fence}text
Test rows: {gap.test_rows}
Current public score: {gap.current_score:.5f}
Target public score: {gap.target_score:.5f}
Estimated current correct predictions: {gap.estimated_current_correct}
Estimated target correct predictions: {gap.estimated_target_correct}
Estimated extra correct predictions needed: {gap.estimated_extra_correct_needed}
{code_fence}

## Existing Experiments

{build_experiment_table(experiments)}

## Diagnosis

The gap is not huge, but it is meaningful.

We probably need around `{gap.estimated_extra_correct_needed}` additional correct predictions on the 418-row Kaggle test set.

This means we should not randomly throw models at the problem.

We need targeted experiments:

1. Better missing value handling
2. Stronger Titanic-specific features
3. More stable validation
4. Careful model tuning
5. Small number of meaningful submissions

## Priority Experiment Backlog

{build_priority_table(priority_rows)}

## Competition Thinking

The rule is:

{code_fence}text
hypothesis
→ implementation
→ CV
→ public LB if promising
→ experiment log
→ decision
{code_fence}

We do not trust public leaderboard alone.

We do not trust CV alone.

We compare both.

## Next Immediate Experiment

Start with:

`titanic_exp_004_age_imputation_v2`

Reason:

Age is currently imputed too simply.

A more targeted age strategy may correct a small number of important passengers, especially children and title-based groups.
"""


def run_pipeline() -> None:
    experiments = load_experiments()
    best = find_best_public_experiment(experiments)
    gap = calculate_score_gap(best.public_lb_score)

    priority_rows = build_priority_experiments()

    write_priority_backlog(priority_rows)
    GAP_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GAP_REPORT_PATH.write_text(
        build_gap_report(experiments, gap, priority_rows),
        encoding="utf-8",
    )


def main() -> None:
    run_pipeline()
    print(f"Saved gap report to: {GAP_REPORT_PATH}")
    print(f"Saved priority backlog to: {PRIORITY_BACKLOG_PATH}")


if __name__ == "__main__":
    main()
