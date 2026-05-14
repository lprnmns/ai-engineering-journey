import pytest

from labs.rag.chunking import ChunkSearchResult
from labs.rag.evaluation import (
    GoldenQuery,
    calculate_metrics,
    evaluate_query,
    evaluate_retrieval,
    find_expected_rank,
    format_evaluation_report,
    result_matches_expected,
)
from labs.rag.vector_store import InMemoryVectorStore
from labs.rag.chunking import Chunk


def make_result(
    doc_id: str = "doc_001",
    chunk_id: str = "doc_001_chunk_001",
) -> ChunkSearchResult:
    return ChunkSearchResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        title="Test",
        text="Test text",
        source="test/source",
        chunk_index=1,
        score=0.9,
    )


def make_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add_chunks(
        [
            Chunk(
                chunk_id="python_chunk",
                doc_id="doc_python",
                title="Python",
                text="Python sanal ortam oluşturmak için venv kullanılır.",
                source="source/python",
                chunk_index=1,
            ),
            Chunk(
                chunk_id="git_chunk",
                doc_id="doc_git",
                title="Git",
                text="Pull request ile değişiklikler main branch içine alınır.",
                source="source/git",
                chunk_index=1,
            ),
        ]
    )

    return store


def test_result_matches_expected_doc_id() -> None:
    golden = GoldenQuery(
        query="test query",
        expected_doc_id="doc_001",
    )

    assert result_matches_expected(make_result(doc_id="doc_001"), golden) is True
    assert result_matches_expected(make_result(doc_id="doc_002"), golden) is False


def test_result_matches_expected_chunk_id_when_given() -> None:
    golden = GoldenQuery(
        query="test query",
        expected_doc_id="doc_001",
        expected_chunk_id="target_chunk",
    )

    assert result_matches_expected(make_result(chunk_id="target_chunk"), golden) is True
    assert result_matches_expected(make_result(chunk_id="other_chunk"), golden) is False


def test_find_expected_rank_returns_one_based_rank() -> None:
    golden = GoldenQuery(query="test", expected_doc_id="doc_002")
    results = [
        make_result(doc_id="doc_001"),
        make_result(doc_id="doc_002"),
    ]

    assert find_expected_rank(results, golden) == 2


def test_find_expected_rank_returns_none_when_missing() -> None:
    golden = GoldenQuery(query="test", expected_doc_id="doc_missing")

    assert find_expected_rank([make_result(doc_id="doc_001")], golden) is None


def test_evaluate_query_returns_hit_metrics() -> None:
    store = make_store()
    golden = GoldenQuery(
        query="Python sanal ortam nasıl kurulur?",
        expected_doc_id="doc_python",
    )

    evaluation = evaluate_query(store=store, golden_query=golden, top_k=2)

    assert evaluation.hit_at_1 is True
    assert evaluation.hit_at_k is True
    assert evaluation.rank == 1
    assert evaluation.reciprocal_rank == 1.0


def test_evaluate_query_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError):
        evaluate_query(
            store=make_store(),
            golden_query=GoldenQuery(query="test", expected_doc_id="doc_python"),
            top_k=0,
        )


def test_calculate_metrics_handles_empty_input() -> None:
    metrics = calculate_metrics([])

    assert metrics.total_queries == 0
    assert metrics.hit_at_1 == 0.0
    assert metrics.hit_at_k == 0.0
    assert metrics.mean_reciprocal_rank == 0.0


def test_evaluate_retrieval_returns_report() -> None:
    store = make_store()
    golden_queries = [
        GoldenQuery(query="Python sanal ortam", expected_doc_id="doc_python"),
        GoldenQuery(query="Pull request main branch", expected_doc_id="doc_git"),
    ]

    report = evaluate_retrieval(
        store=store,
        golden_queries=golden_queries,
        top_k=2,
    )

    assert report.metrics.total_queries == 2
    assert report.metrics.hit_at_k == 1.0


def test_format_evaluation_report_contains_metrics() -> None:
    report = evaluate_retrieval(
        store=make_store(),
        golden_queries=[
            GoldenQuery(query="Python sanal ortam", expected_doc_id="doc_python"),
        ],
        top_k=2,
    )

    text = format_evaluation_report(report)

    assert "Hit@1" in text
    assert "Hit@K" in text
    assert "MRR" in text
    assert "Python sanal ortam" in text
