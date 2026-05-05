from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from competitions.titanic.src.experiment_logging import (
    ExperimentRecord,
    append_experiment_record,
)
from competitions.titanic.src.feature_v1 import load_raw_data, add_feature_v1_columns, split_features_target, run_cross_validation


def build_feature_v1_experiment_record(public_score: str) -> ExperimentRecord:
    train_raw, _ = load_raw_data()
    train_featured = add_feature_v1_columns(train_raw)
    X, y = split_features_target(train_featured)

    _, cv_mean, cv_std = run_cross_validation(X, y)

    return ExperimentRecord(
        experiment_id="titanic_exp_002_feature_v1_logreg",
        date="2026-05-05",
        features="baseline + FamilySize, IsAlone, HasCabin, Title",
        model="LogisticRegression",
        validation_strategy="5-fold StratifiedKFold accuracy",
        local_score=f"{cv_mean:.5f} +/- {cv_std:.5f}",
        public_lb_score=public_score,
        notes="Feature engineering v1: family, cabin availability, and title extracted from name.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-score",
        required=True,
        help="Kaggle public leaderboard score, for example 0.77511",
    )
    args = parser.parse_args()

    record = build_feature_v1_experiment_record(public_score=str(args.public_score))
    was_added = append_experiment_record(record)

    if was_added:
        print(f"Logged experiment: {record.experiment_id}")
    else:
        print(f"Experiment already exists: {record.experiment_id}")


if __name__ == "__main__":
    main()
