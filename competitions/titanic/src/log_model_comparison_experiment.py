from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from competitions.titanic.src.experiment_logging import (
    ExperimentRecord,
    append_experiment_record,
)


MODEL_RESULTS_PATH = Path("competitions/titanic/reports/titanic_model_comparison_results.csv")


def build_model_comparison_experiment_record(public_score: str) -> ExperimentRecord:
    results = pd.read_csv(MODEL_RESULTS_PATH)
    best_row = results.sort_values("cv_mean", ascending=False).iloc[0]

    model_name = str(best_row["model_name"])
    cv_mean = float(best_row["cv_mean"])
    cv_std = float(best_row["cv_std"])

    return ExperimentRecord(
        experiment_id="titanic_exp_003_model_comparison_hgb",
        date="2026-05-05",
        features="plus_title: Pclass, Age, SibSp, Parch, Fare, Sex, Embarked, Title",
        model=model_name,
        validation_strategy="5-fold StratifiedKFold accuracy",
        local_score=f"{cv_mean:.5f} +/- {cv_std:.5f}",
        public_lb_score=public_score,
        notes="Model comparison on plus_title feature set. HistGradientBoosting achieved the best local CV and improved public LB over feature_v1 logistic regression.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-score",
        required=True,
        help="Kaggle public leaderboard score, for example 0.77272",
    )
    args = parser.parse_args()

    record = build_model_comparison_experiment_record(public_score=str(args.public_score))
    was_added = append_experiment_record(record)

    if was_added:
        print(f"Logged experiment: {record.experiment_id}")
    else:
        print(f"Experiment already exists: {record.experiment_id}")


if __name__ == "__main__":
    main()
