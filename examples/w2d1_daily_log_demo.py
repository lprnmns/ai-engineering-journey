from datetime import date
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.journey.domain import Artifact, ArtifactType, DailyLog, Milestone


def main() -> None:
    milestone = Milestone(
        code="M1W2D1",
        title="OOP domain classes",
        goal="Model learning progress with Python classes.",
    )

    log = DailyLog(
        title="W2D1 — OOP domain classes",
        log_date=date.today(),
        hours_spent=1.5,
    )

    log.add_learned_item("Why OOP is about modeling concepts, not just syntax.")
    log.add_learned_item("How dataclasses reduce boilerplate.")
    log.add_learned_item("How validation can live inside domain objects.")

    log.add_completed_task("Created Artifact class.")
    log.add_completed_task("Created Milestone class.")
    log.add_completed_task("Created DailyLog class.")

    log.add_artifact(
        Artifact(
            title="OOP domain classes commit",
            url="https://github.com/lprnmns/ai-engineering-journey",
            artifact_type=ArtifactType.COMMIT,
        )
    )

    milestone.mark_completed()

    print(milestone)
    print("Milestone status:", milestone.status_label())
    print(log.summary())
    print("Is shippable?", log.is_shippable())

    print("\nArtifacts:")
    for artifact in log.artifacts:
        print("-", artifact.summary())


if __name__ == "__main__":
    main()
