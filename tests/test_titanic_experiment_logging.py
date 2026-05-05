from pathlib import Path

from competitions.titanic.src.experiment_logging import (
    ExperimentRecord,
    append_experiment_record,
    load_existing_experiment_ids,
)


def build_record(experiment_id: str = "exp_test") -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=experiment_id,
        date="2026-05-05",
        features="test features",
        model="test model",
        validation_strategy="test validation",
        local_score="0.80000",
        public_lb_score="0.76000",
        notes="test notes",
    )


def test_append_experiment_record_writes_record(tmp_path: Path) -> None:
    path = tmp_path / "experiments.csv"
    record = build_record()

    was_added = append_experiment_record(record, path=path)

    assert was_added is True
    assert "exp_test" in path.read_text(encoding="utf-8")


def test_append_experiment_record_skips_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "experiments.csv"
    record = build_record()

    first_add = append_experiment_record(record, path=path)
    second_add = append_experiment_record(record, path=path)

    assert first_add is True
    assert second_add is False


def test_load_existing_experiment_ids(tmp_path: Path) -> None:
    path = tmp_path / "experiments.csv"
    append_experiment_record(build_record("exp_a"), path=path)
    append_experiment_record(build_record("exp_b"), path=path)

    experiment_ids = load_existing_experiment_ids(path)

    assert experiment_ids == {"exp_a", "exp_b"}
