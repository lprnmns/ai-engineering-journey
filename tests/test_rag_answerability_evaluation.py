from labs.rag.answerability_evaluation import (
    AnswerabilityCase,
    AnswerabilityEvaluation,
    AnswerabilityMetrics,
    calculate_answerability_metrics,
    evaluate_answerability,
    evaluate_answerability_case,
    format_answerability_report,
)


def test_evaluate_answerability_case_accepts_known_query() -> None:
    evaluation = evaluate_answerability_case(
        AnswerabilityCase(
            query="RAG sisteminde ilgili doküman parçaları nasıl bulunur?",
            should_be_answerable=True,
        )
    )

    assert evaluation.predicted_answerable is True
    assert evaluation.is_correct is True


def test_evaluate_answerability_case_rejects_unknown_query() -> None:
    evaluation = evaluate_answerability_case(
        AnswerabilityCase(
            query="Fenerbahçe maç skoru nedir?",
            should_be_answerable=False,
        )
    )

    assert evaluation.predicted_answerable is False
    assert evaluation.is_correct is True


def test_calculate_answerability_metrics_handles_empty_input() -> None:
    metrics = calculate_answerability_metrics([])

    assert metrics == AnswerabilityMetrics(
        total_cases=0,
        accuracy=0.0,
        true_positive=0,
        true_negative=0,
        false_positive=0,
        false_negative=0,
    )


def test_calculate_answerability_metrics_counts_confusion_matrix() -> None:
    evaluations = [
        AnswerabilityEvaluation(
            query="known ok",
            should_be_answerable=True,
            predicted_answerable=True,
            is_correct=True,
            reason="answerable",
            max_score=0.8,
        ),
        AnswerabilityEvaluation(
            query="unknown ok",
            should_be_answerable=False,
            predicted_answerable=False,
            is_correct=True,
            reason="low_score",
            max_score=0.0,
        ),
        AnswerabilityEvaluation(
            query="false positive",
            should_be_answerable=False,
            predicted_answerable=True,
            is_correct=False,
            reason="answerable",
            max_score=0.2,
        ),
        AnswerabilityEvaluation(
            query="false negative",
            should_be_answerable=True,
            predicted_answerable=False,
            is_correct=False,
            reason="low_score",
            max_score=0.01,
        ),
    ]

    metrics = calculate_answerability_metrics(evaluations)

    assert metrics.total_cases == 4
    assert metrics.accuracy == 0.5
    assert metrics.true_positive == 1
    assert metrics.true_negative == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1


def test_evaluate_answerability_returns_report() -> None:
    report = evaluate_answerability(
        cases=[
            AnswerabilityCase(
                query="Python sanal ortam nasıl kurulur?",
                should_be_answerable=True,
            ),
            AnswerabilityCase(
                query="Bitcoin fiyatı şu anda kaç dolar?",
                should_be_answerable=False,
            ),
        ]
    )

    assert report.metrics.total_cases == 2
    assert 0.0 <= report.metrics.accuracy <= 1.0


def test_format_answerability_report_contains_metrics_and_rows() -> None:
    report = evaluate_answerability(
        cases=[
            AnswerabilityCase(
                query="Python sanal ortam nasıl kurulur?",
                should_be_answerable=True,
            )
        ]
    )

    text = format_answerability_report(report)

    assert "RAG Answerability Evaluation" in text
    assert "Accuracy" in text
    assert "True positive" in text
    assert "Python sanal ortam" in text
