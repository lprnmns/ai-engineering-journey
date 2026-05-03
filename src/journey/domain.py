from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import List


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
        if not self.title.strip():
            raise ValueError("Artifact title cannot be empty.")

        if not self.url.strip():
            raise ValueError("Artifact URL cannot be empty.")

    def summary(self) -> str:
        return f"[{self.artifact_type.value}] {self.title}: {self.url}"


@dataclass
class Milestone:
    code: str
    title: str
    goal: str
    is_completed: bool = False

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("Milestone code cannot be empty.")

        if not self.title.strip():
            raise ValueError("Milestone title cannot be empty.")

        if not self.goal.strip():
            raise ValueError("Milestone goal cannot be empty.")

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
    watched_or_learned: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    hours_spent: float = 0.0

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("Daily log title cannot be empty.")

        if self.hours_spent < 0:
            raise ValueError("Hours spent cannot be negative.")

    def add_learned_item(self, item: str) -> None:
        if not item.strip():
            raise ValueError("Learned item cannot be empty.")

        self.watched_or_learned.append(item)

    def add_completed_task(self, task: str) -> None:
        if not task.strip():
            raise ValueError("Completed task cannot be empty.")

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
