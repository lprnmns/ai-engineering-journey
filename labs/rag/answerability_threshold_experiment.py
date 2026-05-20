from __future__ import annotations

from dataclasses import dataclass

from labs.rag.answerability_evaluation import (
    AnswerabilityCase,
    AnswerabilityMetrics,
    DEFAULT_ANSWERABILITY_CASES,
    evaluate_answerability,
)


@dataclass(frozen=True)
class ThresholdExperimentResult:
    min_score: float
    min_margin: float
    metrics: AnswerabilityMetrics


def run_threshold_experiment(
    min_scores: list[float] | None = None,
    min_margins: list[float] | None = None,
    cases: list[AnswerabilityCase] | None = None,
    top_k: int = 3,
) -> list[ThresholdExperimentResult]:
    if min_scores is None:
        min_scores = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]

    if min_margins is None:
        min_margins = [0.0, 0.05, 0.10, 0.20]

    if cases is None:
        cases = DEFAULT_ANSWERABILITY_CASES

    results: list[ThresholdExperimentResult] = []

    for min_score in min_scores:
        for min_margin in min_margins:
            report = evaluate_answerability(
                cases=cases,
                top_k=top_k,
                min_score=min_score,
                min_margin=min_margin,
            )

            results.append(
                ThresholdExperimentResult(
                    min_score=min_score,
                    min_margin=min_margin,
                    metrics=report.metrics,
                )
            )

    return results


def find_best_threshold(
    results: list[ThresholdExperimentResult],
) -> ThresholdExperimentResult | None:
    if not results:
        return None

    return sorted(
        results,
        key=lambda result: (
            result.metrics.accuracy,
            -result.metrics.false_positive,
            -result.metrics.false_negative,
        ),
        reverse=True,
    )[0]


def format_threshold_experiment(
    results: list[ThresholdExperimentResult],
) -> str:
    lines = [
        "=== RAG Answerability Threshold Experiment ===",
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
                "Best threshold candidate:",
                (
                    f"- min_score={best.min_score:.2f}, "
                    f"min_margin={best.min_margin:.2f}, "
                    f"accuracy={best.metrics.accuracy:.3f}, "
                    f"FP={best.metrics.false_positive}, "
                    f"FN={best.metrics.false_negative}"
                ),
            ]
        )

    return "\n".join(lines)


def main() -> None:
    results = run_threshold_experiment()

    print(format_threshold_experiment(results))


if __name__ == "__main__":
    main()
