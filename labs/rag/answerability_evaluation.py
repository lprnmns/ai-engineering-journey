from __future__ import annotations

from dataclasses import dataclass

from labs.rag.no_answer_detection import GuardedRagAnswer, answer_query_with_guard


@dataclass(frozen=True)
class AnswerabilityCase:
    query: str
    should_be_answerable: bool


@dataclass(frozen=True)
class AnswerabilityEvaluation:
    query: str
    should_be_answerable: bool
    predicted_answerable: bool
    is_correct: bool
    reason: str
    max_score: float


@dataclass(frozen=True)
class AnswerabilityMetrics:
    total_cases: int
    accuracy: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


@dataclass(frozen=True)
class AnswerabilityReport:
    metrics: AnswerabilityMetrics
    evaluations: list[AnswerabilityEvaluation]


DEFAULT_ANSWERABILITY_CASES: list[AnswerabilityCase] = [
    AnswerabilityCase(
        query="Python sanal ortam nasıl kurulur?",
        should_be_answerable=True,
    ),
    AnswerabilityCase(
        query="Pull request ile main branch'e nasıl değişiklik alınır?",
        should_be_answerable=True,
    ),
    AnswerabilityCase(
        query="RAG sisteminde ilgili doküman parçaları nasıl bulunur?",
        should_be_answerable=True,
    ),
    AnswerabilityCase(
        query="Fenerbahçe maç skoru nedir?",
        should_be_answerable=False,
    ),
    AnswerabilityCase(
        query="Bugün Konya hava durumu nasıl?",
        should_be_answerable=False,
    ),
    AnswerabilityCase(
        query="Bitcoin fiyatı şu anda kaç dolar?",
        should_be_answerable=False,
    ),
]


def evaluate_answerability_case(
    case: AnswerabilityCase,
    top_k: int = 3,
    min_score: float = 0.05,
    min_margin: float = 0.0,
) -> AnswerabilityEvaluation:
    output: GuardedRagAnswer = answer_query_with_guard(
        query=case.query,
        top_k=top_k,
        min_score=min_score,
        min_margin=min_margin,
    )

    predicted = output.decision.is_answerable

    return AnswerabilityEvaluation(
        query=case.query,
        should_be_answerable=case.should_be_answerable,
        predicted_answerable=predicted,
        is_correct=predicted == case.should_be_answerable,
        reason=output.decision.reason,
        max_score=output.decision.max_score,
    )


def calculate_answerability_metrics(
    evaluations: list[AnswerabilityEvaluation],
) -> AnswerabilityMetrics:
    if not evaluations:
        return AnswerabilityMetrics(
            total_cases=0,
            accuracy=0.0,
            true_positive=0,
            true_negative=0,
            false_positive=0,
            false_negative=0,
        )

    true_positive = sum(
        item.should_be_answerable and item.predicted_answerable
        for item in evaluations
    )
    true_negative = sum(
        (not item.should_be_answerable) and (not item.predicted_answerable)
        for item in evaluations
    )
    false_positive = sum(
        (not item.should_be_answerable) and item.predicted_answerable
        for item in evaluations
    )
    false_negative = sum(
        item.should_be_answerable and (not item.predicted_answerable)
        for item in evaluations
    )

    total = len(evaluations)

    return AnswerabilityMetrics(
        total_cases=total,
        accuracy=sum(item.is_correct for item in evaluations) / total,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def evaluate_answerability(
    cases: list[AnswerabilityCase] | None = None,
    top_k: int = 3,
    min_score: float = 0.05,
    min_margin: float = 0.0,
) -> AnswerabilityReport:
    if cases is None:
        cases = DEFAULT_ANSWERABILITY_CASES

    evaluations = [
        evaluate_answerability_case(
            case=case,
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


def format_answerability_report(report: AnswerabilityReport) -> str:
    metrics = report.metrics

    lines = [
        "=== RAG Answerability Evaluation ===",
        f"Total cases: {metrics.total_cases}",
        f"Accuracy: {metrics.accuracy:.3f}",
        f"True positive: {metrics.true_positive}",
        f"True negative: {metrics.true_negative}",
        f"False positive: {metrics.false_positive}",
        f"False negative: {metrics.false_negative}",
        "",
        "| Query | Expected | Predicted | Correct | Reason | Max score |",
        "|---|---:|---:|---:|---|---:|",
    ]

    for evaluation in report.evaluations:
        lines.append(
            f"| {evaluation.query} | "
            f"{evaluation.should_be_answerable} | "
            f"{evaluation.predicted_answerable} | "
            f"{evaluation.is_correct} | "
            f"{evaluation.reason} | "
            f"{evaluation.max_score:.4f} |"
        )

    return "\n".join(lines)


def main() -> None:
    report = evaluate_answerability()

    print(format_answerability_report(report))


if __name__ == "__main__":
    main()
