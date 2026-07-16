from __future__ import annotations

from dataclasses import dataclass

from labs.rag.answerability_evaluation import (
    AnswerabilityCase,
    AnswerabilityEvaluation,
    AnswerabilityMetrics,
    AnswerabilityReport,
    calculate_answerability_metrics,
)
from labs.rag.dense_vector_store import build_dense_vector_store
from labs.rag.evaluation import ChunkRetriever
from labs.rag.hard_answerability_benchmark import HARD_ANSWERABILITY_CASES
from labs.rag.no_answer_detection import decide_answerability


@dataclass(frozen=True)
class DenseThresholdExperimentResult:
    min_score: float
    min_margin: float
    metrics: AnswerabilityMetrics


def evaluate_dense_answerability_case(
    case: AnswerabilityCase,
    store: ChunkRetriever,
    top_k: int = 3,
    min_score: float = 0.05,
    min_margin: float = 0.0,
) -> AnswerabilityEvaluation:
    results = store.search(query=case.query, top_k=top_k)
    decision = decide_answerability(
        query=case.query,
        results=results,
        min_score=min_score,
        min_margin=min_margin,
    )

    predicted = decision.is_answerable

    return AnswerabilityEvaluation(
        query=case.query,
        should_be_answerable=case.should_be_answerable,
        predicted_answerable=predicted,
        is_correct=predicted == case.should_be_answerable,
        reason=decision.reason,
        max_score=decision.max_score,
    )


def evaluate_dense_answerability(
    cases: list[AnswerabilityCase] | None = None,
    store: ChunkRetriever | None = None,
    top_k: int = 3,
    min_score: float = 0.05,
    min_margin: float = 0.0,
) -> AnswerabilityReport:
    if cases is None:
        cases = HARD_ANSWERABILITY_CASES

    if store is None:
        store = build_dense_vector_store()

    evaluations = [
        evaluate_dense_answerability_case(
            case=case,
            store=store,
            top_k=top_k,
            min_score=min_score,
            min_margin=min_margin,
        )
        for case in cases
    ]

    return AnswerabilityReport(
        metrics=calculate_answerability_metrics(evaluations),
        evaluations=evaluations,
    )


def run_dense_threshold_experiment(
    min_scores: list[float] | None = None,
    min_margins: list[float] | None = None,
    cases: list[AnswerabilityCase] | None = None,
    top_k: int = 3,
) -> list[DenseThresholdExperimentResult]:
    if min_scores is None:
        min_scores = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

    if min_margins is None:
        min_margins = [0.0, 0.02, 0.05, 0.10, 0.20]

    if cases is None:
        cases = HARD_ANSWERABILITY_CASES

    store = build_dense_vector_store()
    results: list[DenseThresholdExperimentResult] = []

    for min_score in min_scores:
        for min_margin in min_margins:
            report = evaluate_dense_answerability(
                cases=cases,
                store=store,
                top_k=top_k,
                min_score=min_score,
                min_margin=min_margin,
            )
            results.append(
                DenseThresholdExperimentResult(
                    min_score=min_score,
                    min_margin=min_margin,
                    metrics=report.metrics,
                )
            )

    return results


def find_best_dense_threshold(
    results: list[DenseThresholdExperimentResult],
) -> DenseThresholdExperimentResult | None:
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


def format_dense_threshold_experiment(
    results: list[DenseThresholdExperimentResult],
) -> str:
    lines = [
        "=== Dense Answerability Threshold Experiment ===",
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

    best = find_best_dense_threshold(results)
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
    results = run_dense_threshold_experiment()
    print(format_dense_threshold_experiment(results))


if __name__ == "__main__":
    main()
