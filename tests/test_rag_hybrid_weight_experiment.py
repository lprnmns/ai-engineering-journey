import pytest

from labs.rag.evaluation import GoldenQuery, RetrievalMetrics
from labs.rag.hybrid_retrieval import build_hybrid_retriever
from labs.rag.hybrid_weight_experiment import (
    calculate_hybrid_metrics,
    evaluate_hybrid_query,
    evaluate_hybrid_weight,
    find_expected_rank,
    format_hybrid_weight_experiment,
    run_hybrid_weight_experiment,
)


def test_find_expected_rank_returns_one_based_rank() -> None:
    rank = find_expected_rank(
        retrieved_doc_ids=["doc_a", "doc_b", "doc_c"],
        expected_doc_id="doc_b",
    )

    assert rank == 2


def test_find_expected_rank_returns_none_when_missing() -> None:
    rank = find_expected_rank(
        retrieved_doc_ids=["doc_a", "doc_b"],
        expected_doc_id="doc_missing",
    )

    assert rank is None


def test_evaluate_hybrid_query_returns_hit_for_matching_doc() -> None:
    retriever = build_hybrid_retriever()
    golden = GoldenQuery(
        query="Python sanal ortam nasıl kurulur?",
        expected_doc_id="doc_001",
    )

    evaluation = evaluate_hybrid_query(
        retriever=retriever,
        golden_query=golden,
        top_k=3,
    )

    assert evaluation.hit_at_1 is True
    assert evaluation.hit_at_k is True
    assert evaluation.rank == 1


def test_evaluate_hybrid_query_rejects_invalid_top_k() -> None:
    retriever = build_hybrid_retriever()

    with pytest.raises(ValueError):
        evaluate_hybrid_query(
            retriever=retriever,
            golden_query=GoldenQuery(query="test", expected_doc_id="doc_001"),
            top_k=0,
        )


def test_calculate_hybrid_metrics_handles_empty_input() -> None:
    metrics = calculate_hybrid_metrics([])

    assert metrics == RetrievalMetrics(
        total_queries=0,
        hit_at_1=0.0,
        hit_at_k=0.0,
        mean_reciprocal_rank=0.0,
    )


def test_evaluate_hybrid_weight_returns_metrics() -> None:
    result = evaluate_hybrid_weight(vector_weight=0.7)

    assert result.vector_weight == 0.7
    assert result.metrics.total_queries > 0
    assert 0.0 <= result.metrics.hit_at_1 <= 1.0
    assert 0.0 <= result.metrics.hit_at_k <= 1.0
    assert 0.0 <= result.metrics.mean_reciprocal_rank <= 1.0


def test_run_hybrid_weight_experiment_returns_all_weights() -> None:
    results = run_hybrid_weight_experiment(weights=[0.0, 0.5, 1.0])

    assert [result.vector_weight for result in results] == [0.0, 0.5, 1.0]


def test_format_hybrid_weight_experiment_contains_table() -> None:
    results = run_hybrid_weight_experiment(weights=[0.0, 1.0])

    text = format_hybrid_weight_experiment(results)

    assert "Hybrid Retrieval Weight Experiment" in text
    assert "Vector weight" in text
    assert "Keyword weight" in text
    assert "MRR" in text
