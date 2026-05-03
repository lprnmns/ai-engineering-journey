from datetime import date
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.journey.domain import Artifact, ArtifactType, DailyLog


def build_valid_log() -> DailyLog:
    log = DailyLog(
        title="W2D2 — Type checking",
        log_date=date.today(),
        hours_spent=1.0,
    )

    log.add_learned_item("Type hints document intent.")
    log.add_completed_task("Configured mypy in strict mode.")
    log.add_artifact(
        Artifact(
            title="Type checking setup",
            url="https://github.com/lprnmns/ai-engineering-journey",
            artifact_type=ArtifactType.COMMIT,
        )
    )

    return log


def main() -> None:
    log = build_valid_log()
    print(log.summary())

    # Uncomment this line to see mypy catch the type error:
    # bad_log = DailyLog(title="Bad log", log_date=date.today(), hours_spent="two hours")


if __name__ == "__main__":
    main()
