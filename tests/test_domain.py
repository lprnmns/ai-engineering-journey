from datetime import date

import pytest

from src.journey.domain import Artifact, ArtifactType, DailyLog, Milestone


def test_artifact_summary_includes_type_title_and_url() -> None:
    artifact = Artifact(
        title="Initial commit",
        url="https://github.com/lprnmns/ai-engineering-journey",
        artifact_type=ArtifactType.COMMIT,
    )

    assert artifact.summary() == (
        "[commit] Initial commit: "
        "https://github.com/lprnmns/ai-engineering-journey"
    )


def test_artifact_rejects_empty_title() -> None:
    with pytest.raises(ValueError, match="Artifact title cannot be empty"):
        Artifact(
            title=" ",
            url="https://github.com/lprnmns/ai-engineering-journey",
            artifact_type=ArtifactType.COMMIT,
        )


def test_milestone_can_be_marked_completed() -> None:
    milestone = Milestone(
        code="M1W2D3",
        title="pytest basics",
        goal="Add unit tests for domain objects.",
    )

    assert milestone.status_label() == "in_progress"

    milestone.mark_completed()

    assert milestone.status_label() == "completed"


def test_daily_log_is_shippable_when_it_has_task_and_artifact() -> None:
    log = DailyLog(
        title="W2D3 — pytest basics",
        log_date=date(2026, 5, 3),
        hours_spent=1.0,
    )

    assert log.is_shippable() is False

    log.add_completed_task("Added domain tests.")
    log.add_artifact(
        Artifact(
            title="pytest domain tests",
            url="https://github.com/lprnmns/ai-engineering-journey",
            artifact_type=ArtifactType.PR,
        )
    )

    assert log.is_shippable() is True


def test_daily_log_rejects_negative_hours() -> None:
    with pytest.raises(ValueError, match="Hours spent cannot be negative"):
        DailyLog(
            title="Invalid log",
            log_date=date(2026, 5, 3),
            hours_spent=-1.0,
        )


def test_daily_log_summary_counts_tasks_and_artifacts() -> None:
    log = DailyLog(
        title="W2D3 — pytest basics",
        log_date=date(2026, 5, 3),
        hours_spent=2.0,
    )

    log.add_completed_task("Added tests.")
    log.add_completed_task("Ran pytest.")
    log.add_artifact(
        Artifact(
            title="Test PR",
            url="https://github.com/lprnmns/ai-engineering-journey",
            artifact_type=ArtifactType.PR,
        )
    )

    assert log.summary() == (
        "W2D3 — pytest basics | "
        "date=2026-05-03 | "
        "tasks=2 | "
        "artifacts=1 | "
        "hours=2.0"
    )


def test_daily_log_rejects_empty_learned_item() -> None:
    log = DailyLog(
        title="W2D4 — refactor",
        log_date=date(2026, 5, 3),
        hours_spent=1.0,
    )

    with pytest.raises(ValueError, match="Learned item cannot be empty"):
        log.add_learned_item(" ")


def test_daily_log_rejects_empty_completed_task() -> None:
    log = DailyLog(
        title="W2D4 — refactor",
        log_date=date(2026, 5, 3),
        hours_spent=1.0,
    )

    with pytest.raises(ValueError, match="Completed task cannot be empty"):
        log.add_completed_task(" ")
