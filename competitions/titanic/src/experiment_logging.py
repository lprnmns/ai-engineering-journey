from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from competitions.titanic.src.cross_validate import load_training_data, run_cross_validation


EXPERIMENT_LOG_PATH = Path("competitions/titanic/experiments/experiments.csv")

FIELD_NAMES = [
    "experiment_id",
    "date",
    "features",
    "model",
    "validation_strategy",
    "local_score",
    "public_lb_score",
    "notes",
]


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    date: str
    features: str
    model: str
    validation_strategy: str
    local_score: str
    public_lb_score: str
    notes: str


def load_existing_experiment_ids(path: Path = EXPERIMENT_LOG_PATH) -> set[str]:
    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return {
            str(row["experiment_id"])
            for row in reader
            if row.get("experiment_id")
        }


def append_experiment_record(
    record: ExperimentRecord,
    path: Path = EXPERIMENT_LOG_PATH,
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = load_existing_experiment_ids(path)
    if record.experiment_id in existing_ids:
        return False

    should_write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELD_NAMES)

        if should_write_header:
            writer.writeheader()

        writer.writerow(asdict(record))

    return True


def build_baseline_experiment_record() -> ExperimentRecord:
    X, y = load_training_data()
    cv_result = run_cross_validation(X, y)

    local_score = f"{cv_result.mean_score:.5f} +/- {cv_result.std_score:.5f}"

    return ExperimentRecord(
        experiment_id="titanic_exp_001_logreg_baseline",
        date="2026-05-05",
        features="Pclass, Age, SibSp, Parch, Fare, Sex, Embarked",
        model="LogisticRegression",
        validation_strategy="5-fold StratifiedKFold accuracy",
        local_score=local_score,
        public_lb_score="0.76555",
        notes="First valid Titanic baseline submission. Public score is lower than single split validation, so CV tracking is required.",
    )


def main() -> None:
    record = build_baseline_experiment_record()
    was_added = append_experiment_record(record)

    if was_added:
        print(f"Logged experiment: {record.experiment_id}")
    else:
        print(f"Experiment already exists: {record.experiment_id}")

    print(f"Experiment log: {EXPERIMENT_LOG_PATH}")


if __name__ == "__main__":
    main()
