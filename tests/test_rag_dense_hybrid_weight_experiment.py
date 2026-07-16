import pytest

from labs.rag.dense_hybrid_weight_experiment import (
    DenseHybridQueryEvaluation,
    calculate_dense_hybrid_metrics,
    find_expected_rank,
    format_dense_hybrid_weight_experiment,
)
from labs.rag.evaluation import RetrievalMetrics


def test_find_expected_rank_returns_one_based_rank() -> None:
    assert find_expected_rank(["doc_a", "doc_b"], "doc_b") == 2


def test_find_expected_rank_returns_none_for_missing_document() -> None:
    assert find_expected_rank(["doc_a", "doc_b"], "doc_missing") is None


def test_calculate_dense_hybrid_metrics_handles_empty_input() -> None:
    assert calculate_dense_hybrid_metrics([]) == RetrievalMetrics(
        total_queries=0,
        hit_at_1=0.0,
        hit_at_k=0.0,
        mean_reciprocal_rank=0.0,
    )


def test_calculate_dense_hybrid_metrics_averages_ranks() -> None:
    metrics = calculate_dense_hybrid_metrics(
        [
            DenseHybridQueryEvaluation(
                query="first",
                expected_doc_id="doc_a",
                retrieved_doc_ids=["doc_a"],
                rank=1,
                hit_at_1=True,
                hit_at_k=True,
                reciprocal_rank=1.0,
            ),
            DenseHybridQueryEvaluation(
                query="second",
                expected_doc_id="doc_b",
                retrieved_doc_ids=["doc_a", "doc_b"],
                rank=2,
                hit_at_1=False,
                hit_at_k=True,
                reciprocal_rank=0.5,
            ),
        ]
    )

    assert metrics.hit_at_1 == pytest.approx(0.5)
    assert metrics.hit_at_k == pytest.approx(1.0)
    assert metrics.mean_reciprocal_rank == pytest.approx(0.75)


def test_format_dense_hybrid_weight_experiment_contains_table() -> None:
    text = format_dense_hybrid_weight_experiment([])

    assert "Dense Hybrid Weight Experiment" in text
    assert "Dense weight" in text
    assert "Keyword weight" in text
