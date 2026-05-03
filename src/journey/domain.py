from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from src.journey.validation import ensure_non_empty, ensure_non_negative


class ArtifactType(str, Enum):
    COMMIT = "commit"
    PR = "pull_request"
    NOTEBOOK = "notebook"
    DEMO = "demo"
    DOC = "document"


@dataclass
class Artifact:
    title: str
    url: str
    artifact_type: ArtifactType

    def __post_init__(self) -> None:
        ensure_non_empty(self.title, "Artifact title")
        ensure_non_empty(self.url, "Artifact URL")

    def summary(self) -> str:
        return f"[{self.artifact_type.value}] {self.title}: {self.url}"


@dataclass
class Milestone:
    code: str
    title: str
    goal: str
    is_completed: bool = False

    def __post_init__(self) -> None:
        ensure_non_empty(self.code, "Milestone code")
        ensure_non_empty(self.title, "Milestone title")
        ensure_non_empty(self.goal, "Milestone goal")

    def mark_completed(self) -> None:
        self.is_completed = True

    def status_label(self) -> str:
        if self.is_completed:
            return "completed"

        return "in_progress"


@dataclass
class DailyLog:
    title: str
    log_date: date
    watched_or_learned: list[str] = field(default_factory=list)
    completed_tasks: list[str] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    hours_spent: float = 0.0

    def __post_init__(self) -> None:
        ensure_non_empty(self.title, "Daily log title")
        ensure_non_negative(self.hours_spent, "Hours spent")

    def add_learned_item(self, item: str) -> None:
        ensure_non_empty(item, "Learned item")
        self.watched_or_learned.append(item)

    def add_completed_task(self, task: str) -> None:
        ensure_non_empty(task, "Completed task")
        self.completed_tasks.append(task)

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact)

    def is_shippable(self) -> bool:
        return bool(self.completed_tasks) and bool(self.artifacts)

    def summary(self) -> str:
        artifact_count = len(self.artifacts)
        task_count = len(self.completed_tasks)

        return (
            f"{self.title} | "
            f"date={self.log_date.isoformat()} | "
            f"tasks={task_count} | "
            f"artifacts={artifact_count} | "
            f"hours={self.hours_spent}"
        )
