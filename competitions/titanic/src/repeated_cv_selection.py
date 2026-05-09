from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from competitions.titanic.src.hgb_tuning import (
    HgbConfig,
    build_configs,
    build_pipeline,
    load_train_test,
)
from competitions.titanic.src.model_comparison import (
    build_submission,
    validate_submission,
)


REPORT_PATH = Path("competitions/titanic/reports/titanic_repeated_cv_selection_report.md")
RESULTS_PATH = Path("competitions/titanic/reports/titanic_repeated_cv_selection_results.csv")
SUBMISSION_PATH = Path("competitions/titanic/submissions/repeated_cv_best_submission.csv")

CURRENT_BEST_PUBLIC_LB = 0.77272
CURRENT_BEST_EXPERIMENT = "titanic_exp_003_model_comparison_hgb"

SEEDS = [42, 7, 123, 2026, 99]


@dataclass(frozen=True)
class RepeatedCvResult:
    config: HgbConfig
    scores: list[float]
    repeated_cv_mean: float
    repeated_cv_std: float
    repeated_cv_min: float
    repeated_cv_max: float
    risk_adjusted_score: float


def calculate_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def calculate_std(values: list[float]) -> float:
    mean_value = calculate_mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)

    return math.sqrt(variance)


def evaluate_config_repeated_cv(
    X: pd.DataFrame,
    y: pd.Series,
    config: HgbConfig,
    seeds: list[int] = SEEDS,
    n_splits: int = 5,
) -> RepeatedCvResult:
    all_scores: list[float] = []

    for seed in seeds:
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        pipeline = build_pipeline(config)

        raw_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="accuracy",
        )

        all_scores.extend(float(score) for score in raw_scores)

    mean_score = calculate_mean(all_scores)
    std_score = calculate_std(all_scores)
    risk_adjusted_score = mean_score - std_score

    return RepeatedCvResult(
        config=config,
        scores=all_scores,
        repeated_cv_mean=mean_score,
        repeated_cv_std=std_score,
        repeated_cv_min=min(all_scores),
        repeated_cv_max=max(all_scores),
        risk_adjusted_score=risk_adjusted_score,
    )


def run_repeated_cv_selection() -> list[RepeatedCvResult]:
    X, y, _, _ = load_train_test()

    results = [
        evaluate_config_repeated_cv(X, y, config)
        for config in build_configs()
    ]

    return sorted(
        results,
        key=lambda result: result.risk_adjusted_score,
        reverse=True,
    )


def results_to_dataframe(results: list[RepeatedCvResult]) -> pd.DataFrame:
    rows = []

    for result in results:
        config = result.config
        rows.append(
            {
                "config_name": config.name,
                "repeated_cv_mean": result.repeated_cv_mean,
                "repeated_cv_std": result.repeated_cv_std,
                "risk_adjusted_score": result.risk_adjusted_score,
                "repeated_cv_min": result.repeated_cv_min,
                "repeated_cv_max": result.repeated_cv_max,
                "max_iter": config.max_iter,
                "learning_rate": config.learning_rate,
                "max_leaf_nodes": config.max_leaf_nodes,
                "min_samples_leaf": config.min_samples_leaf,
                "l2_regularization": config.l2_regularization,
                "all_scores": ", ".join(f"{score:.5f}" for score in result.scores),
            }
        )

    return pd.DataFrame(rows)


def save_results_csv(results: list[RepeatedCvResult]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_to_dataframe(results).to_csv(RESULTS_PATH, index=False)


def train_best_and_save_submission(best_config: HgbConfig) -> None:
    X, y, X_test, test_raw = load_train_test()

    pipeline = build_pipeline(best_config)
    pipeline.fit(X, y)

    predictions = [int(value) for value in pipeline.predict(X_test)]

    submission = build_submission(
        test_raw=test_raw,
        predictions=predictions,
    )
    validate_submission(submission, expected_rows=len(test_raw))

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)


def build_report(results: list[RepeatedCvResult]) -> str:
    table_lines = [
        "| Rank | Config | Mean | Std | Risk-adjusted | Min | Max |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]

    for rank, result in enumerate(results, start=1):
        table_lines.append(
            f"| {rank} | {result.config.name} | "
            f"{result.repeated_cv_mean:.5f} | "
            f"{result.repeated_cv_std:.5f} | "
            f"{result.risk_adjusted_score:.5f} | "
            f"{result.repeated_cv_min:.5f} | "
            f"{result.repeated_cv_max:.5f} |"
        )

    best = results[0]
    code_fence = chr(96) * 3

    return f"""# Titanic Repeated CV Selection Report

## Goal

Select a more reliable HGB configuration using repeated cross-validation.

The previous tuning experiment selected by single-seed CV mean and failed on public leaderboard.

## Current Best Public Reference

- Experiment: `{CURRENT_BEST_EXPERIMENT}`
- Public LB: `{CURRENT_BEST_PUBLIC_LB:.5f}`

## Validation Setup

{code_fence}text
seeds = {SEEDS}
folds per seed = 5
total validation scores per config = {len(SEEDS) * 5}
selection score = repeated_cv_mean - repeated_cv_std
{code_fence}

## Results

{chr(10).join(table_lines)}

## Best Risk-Adjusted Config

- Name: `{best.config.name}`
- Repeated CV mean: `{best.repeated_cv_mean:.5f}`
- Repeated CV std: `{best.repeated_cv_std:.5f}`
- Risk-adjusted score: `{best.risk_adjusted_score:.5f}`

## Submission

- Path: `{SUBMISSION_PATH}`

## Competition Interpretation

This experiment tests whether selecting a more stable configuration is safer than selecting the highest single CV mean.

A config with slightly lower mean but much lower std may be better for public/private leaderboard generalization.
"""


def save_report(results: list[RepeatedCvResult]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(results), encoding="utf-8")


def run_pipeline() -> list[RepeatedCvResult]:
    results = run_repeated_cv_selection()
    save_results_csv(results)
    save_report(results)
    train_best_and_save_submission(results[0].config)

    return results


def main() -> None:
    results = run_pipeline()

    print("=== Titanic Repeated CV Selection ===")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result.config.name} | "
            f"mean={result.repeated_cv_mean:.5f} | "
            f"std={result.repeated_cv_std:.5f} | "
            f"risk_adjusted={result.risk_adjusted_score:.5f}"
        )

    print()
    print(f"Best config: {results[0].config.name}")
    print(f"Saved results to: {RESULTS_PATH}")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Saved submission to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
