from labs.rag.vectorizer_comparison import (
    build_tfidf_vector_store,
    chunk_text_for_vectorizer,
    format_comparison,
    run_vectorizer_comparison,
)
from labs.rag.chunking import Chunk


def test_chunk_text_for_vectorizer_combines_title_and_text() -> None:
    chunk = Chunk(
        chunk_id="chunk_001",
        doc_id="doc_001",
        title="Python",
        text="Sanal ortam venv ile oluşturulur.",
        source="source/python",
        chunk_index=1,
    )

    text = chunk_text_for_vectorizer(chunk)

    assert "Python" in text
    assert "Sanal ortam" in text


def test_build_tfidf_vector_store_indexes_sample_documents() -> None:
    store = build_tfidf_vector_store()

    stats = store.stats()

    assert stats.chunk_count > 0
    assert stats.document_count == 5


def test_tfidf_vector_store_can_retrieve_rag_document() -> None:
    store = build_tfidf_vector_store()

    results = store.search(
        "RAG sisteminde ilgili doküman parçaları nasıl bulunur?",
        top_k=1,
    )

    assert results[0].doc_id == "doc_005"


def test_run_vectorizer_comparison_returns_two_results() -> None:
    results = run_vectorizer_comparison(top_k=3)

    names = {result.name for result in results}

    assert names == {"term_frequency", "tfidf"}


def test_vectorizer_comparison_reports_metrics() -> None:
    results = run_vectorizer_comparison(top_k=3)

    for result in results:
        assert result.report.metrics.total_queries > 0
        assert 0.0 <= result.report.metrics.hit_at_1 <= 1.0
        assert 0.0 <= result.report.metrics.hit_at_k <= 1.0
        assert 0.0 <= result.report.metrics.mean_reciprocal_rank <= 1.0


def test_format_comparison_contains_both_vectorizer_names() -> None:
    results = run_vectorizer_comparison(top_k=3)

    text = format_comparison(results)

    assert "term_frequency" in text
    assert "tfidf" in text
    assert "Hit@1" in text
    assert "MRR" in text
