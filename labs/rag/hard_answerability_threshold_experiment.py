from __future__ import annotations

from labs.rag.answerability_threshold_experiment import (
    ThresholdExperimentResult,
    find_best_threshold,
    run_threshold_experiment,
)
from labs.rag.hard_answerability_benchmark import HARD_ANSWERABILITY_CASES


def run_hard_threshold_experiment(
    min_scores: list[float] | None = None,
    min_margins: list[float] | None = None,
    top_k: int = 3,
) -> list[ThresholdExperimentResult]:
    if min_scores is None:
        min_scores = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    if min_margins is None:
        min_margins = [0.00, 0.05, 0.10, 0.20]

    return run_threshold_experiment(
        min_scores=min_scores,
        min_margins=min_margins,
        cases=HARD_ANSWERABILITY_CASES,
        top_k=top_k,
    )


def format_hard_threshold_experiment(
    results: list[ThresholdExperimentResult],
) -> str:
    lines = [
        "=== Hard RAG Answerability Threshold Experiment ===",
        "",
        "| min_score | min_margin | accuracy | TP | TN | FP | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        metrics = result.metrics

        lines.append(
            f"| {result.min_score:.2f} | "
            f"{result.min_margin:.2f} | "
            f"{metrics.accuracy:.3f} | "
            f"{metrics.true_positive} | "
            f"{metrics.true_negative} | "
            f"{metrics.false_positive} | "
            f"{metrics.false_negative} |"
        )

    best = find_best_threshold(results)

    if best is not None:
        lines.extend(
            [
                "",
                "Best hard-threshold candidate:",
                (
                    f"- min_score={best.min_score:.2f}, "
                    f"min_margin={best.min_margin:.2f}, "
                    f"accuracy={best.metrics.accuracy:.3f}, "
                    f"TP={best.metrics.true_positive}, "
                    f"TN={best.metrics.true_negative}, "
                    f"FP={best.metrics.false_positive}, "
                    f"FN={best.metrics.false_negative}"
                ),
            ]
        )

    return "\n".join(lines)


def main() -> None:
    results = run_hard_threshold_experiment()

    print(format_hard_threshold_experiment(results))


if __name__ == "__main__":
    main()
