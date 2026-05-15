from labs.rag.hard_retrieval_benchmark import (
    HARD_DOCUMENTS,
    HARD_GOLDEN_QUERIES,
    evaluate_hybrid_retriever,
    format_hard_benchmark,
    run_hard_retrieval_benchmark,
)


def test_hard_documents_include_similar_topics() -> None:
    doc_ids = {document.doc_id for document in HARD_DOCUMENTS}

    assert "hard_http_404" in doc_ids
    assert "hard_http_500" in doc_ids
    assert "hard_git_branch" in doc_ids
    assert "hard_git_pull_request" in doc_ids


def test_hard_golden_queries_are_available() -> None:
    assert len(HARD_GOLDEN_QUERIES) >= 5


def test_run_hard_retrieval_benchmark_returns_multiple_retrievers() -> None:
    results = run_hard_retrieval_benchmark(top_k=3)

    names = {result.name for result in results}

    assert "term_frequency" in names
    assert "tfidf" in names
    assert "hybrid_balanced" in names


def test_hard_benchmark_metrics_are_valid() -> None:
    results = run_hard_retrieval_benchmark(top_k=3)

    for result in results:
        assert result.metrics.total_queries == len(HARD_GOLDEN_QUERIES)
        assert 0.0 <= result.metrics.hit_at_1 <= 1.0
        assert 0.0 <= result.metrics.hit_at_k <= 1.0
        assert 0.0 <= result.metrics.mean_reciprocal_rank <= 1.0


def test_evaluate_hybrid_retriever_returns_metrics() -> None:
    result = evaluate_hybrid_retriever(
        name="hybrid_test",
        vector_weight=0.5,
        top_k=3,
    )

    assert result.name == "hybrid_test"
    assert result.metrics.total_queries == len(HARD_GOLDEN_QUERIES)


def test_format_hard_benchmark_contains_table_and_details() -> None:
    results = run_hard_retrieval_benchmark(top_k=3)

    text = format_hard_benchmark(results)

    assert "Hard Retrieval Benchmark" in text
    assert "Hit@1" in text
    assert "MRR" in text
    assert "Per-query Details" in text
    assert "term_frequency" in text
    assert "tfidf" in text
