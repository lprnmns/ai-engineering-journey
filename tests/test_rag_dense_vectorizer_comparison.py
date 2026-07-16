import pytest

from labs.rag.dense_vectorizer_comparison import (
    DenseComparisonReport,
    compare_query_evaluations,
    format_paraphrase_comparison,
)
from labs.rag.evaluation import EvaluationReport, QueryEvaluation, RetrievalMetrics


def make_evaluation(query: str, rank: int | None) -> QueryEvaluation:
    return QueryEvaluation(
        query=query,
        expected_doc_id="doc_001",
        expected_chunk_id=None,
        retrieved_doc_ids=["doc_001"] if rank is not None else [],
        retrieved_chunk_ids=["chunk_001"] if rank is not None else [],
        rank=rank,
        hit_at_1=rank == 1,
        hit_at_k=rank is not None,
        reciprocal_rank=0.0 if rank is None else 1.0 / rank,
    )


def test_compare_query_evaluations_marks_dense_as_better() -> None:
    comparisons = compare_query_evaluations(
        [make_evaluation("query", rank=2)],
        [make_evaluation("query", rank=1)],
    )

    assert comparisons[0].result == "dense_better"
    assert comparisons[0].tfidf_rank == 2
    assert comparisons[0].dense_rank == 1


def test_compare_query_evaluations_rejects_mismatched_queries() -> None:
    with pytest.raises(ValueError):
        compare_query_evaluations(
            [make_evaluation("first query", rank=1)],
            [make_evaluation("second query", rank=1)],
        )


def test_format_paraphrase_comparison_includes_both_retrievers() -> None:
    evaluation = make_evaluation("query", rank=1)
    report = EvaluationReport(
        metrics=RetrievalMetrics(
            total_queries=1,
            hit_at_1=1.0,
            hit_at_k=1.0,
            mean_reciprocal_rank=1.0,
        ),
        query_evaluations=[evaluation],
    )
    comparison_report = DenseComparisonReport(
        tfidf_report=report,
        dense_report=report,
        query_comparisons=compare_query_evaluations([evaluation], [evaluation]),
    )

    text = format_paraphrase_comparison(comparison_report)

    assert "tfidf" in text
    assert "dense" in text
    assert "same_rank" in text
