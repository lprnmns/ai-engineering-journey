from labs.rag.hard_answerability_benchmark import (
    HARD_ANSWERABILITY_CASES,
    evaluate_hard_answerability,
)
from labs.rag.answerability_evaluation import format_answerability_report


def test_hard_answerability_cases_include_known_and_unknown_queries() -> None:
    answerable_count = sum(case.should_be_answerable for case in HARD_ANSWERABILITY_CASES)
    unknown_count = sum(not case.should_be_answerable for case in HARD_ANSWERABILITY_CASES)

    assert answerable_count > 0
    assert unknown_count > 0


def test_hard_answerability_cases_include_near_miss_queries() -> None:
    queries = [case.query for case in HARD_ANSWERABILITY_CASES]

    assert any("Docker" in query for query in queries)
    assert any("conflict" in query for query in queries)
    assert any("fine-tuning" in query for query in queries)


def test_evaluate_hard_answerability_returns_report() -> None:
    report = evaluate_hard_answerability()

    assert report.metrics.total_cases == len(HARD_ANSWERABILITY_CASES)
    assert 0.0 <= report.metrics.accuracy <= 1.0
    assert report.metrics.true_positive >= 0
    assert report.metrics.true_negative >= 0
    assert report.metrics.false_positive >= 0
    assert report.metrics.false_negative >= 0


def test_format_hard_answerability_report_contains_near_miss_query() -> None:
    report = evaluate_hard_answerability()
    text = format_answerability_report(report)

    assert "RAG Answerability Evaluation" in text
    assert "Docker" in text
    assert "fine-tuning" in text
