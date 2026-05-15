import pytest

from labs.rag.chunking import Chunk
from labs.rag.hybrid_retrieval import (
    HybridRetriever,
    build_hybrid_retriever,
    combine_scores,
    keyword_overlap_score,
)


def test_keyword_overlap_score_counts_query_terms_found_in_text() -> None:
    score = keyword_overlap_score(
        query="python sanal ortam",
        text="Python projelerinde sanal ortam venv ile kurulur.",
    )

    assert score == 1.0


def test_keyword_overlap_score_returns_zero_for_empty_query() -> None:
    assert keyword_overlap_score("", "python text") == 0.0


def test_combine_scores_uses_vector_weight() -> None:
    combined = combine_scores(
        vector_score=0.8,
        keyword_score=0.2,
        vector_weight=0.75,
    )

    assert combined == pytest.approx(0.65)


def test_combine_scores_rejects_invalid_weight() -> None:
    with pytest.raises(ValueError):
        combine_scores(0.5, 0.5, vector_weight=-0.1)

    with pytest.raises(ValueError):
        combine_scores(0.5, 0.5, vector_weight=1.1)


def test_hybrid_retriever_returns_python_chunk_for_python_query() -> None:
    chunks = [
        Chunk(
            chunk_id="chunk_python",
            doc_id="doc_python",
            title="Python",
            text="Python sanal ortam oluşturmak için venv kullanılır.",
            source="source/python",
            chunk_index=1,
        ),
        Chunk(
            chunk_id="chunk_git",
            doc_id="doc_git",
            title="Git",
            text="Git branch ile kod değişiklikleri yönetilir.",
            source="source/git",
            chunk_index=1,
        ),
    ]

    retriever = HybridRetriever(chunks=chunks)

    results = retriever.search("sanal ortam nasıl kurulur?", top_k=1)

    assert results[0].chunk_id == "chunk_python"
    assert results[0].combined_score > 0.0


def test_hybrid_retriever_rejects_invalid_top_k() -> None:
    retriever = HybridRetriever(chunks=[])

    with pytest.raises(ValueError):
        retriever.search("test", top_k=0)


def test_hybrid_retriever_rejects_invalid_vector_weight() -> None:
    with pytest.raises(ValueError):
        HybridRetriever(chunks=[], vector_weight=1.5)


def test_build_hybrid_retriever_works_with_sample_documents() -> None:
    retriever = build_hybrid_retriever()

    results = retriever.search(
        "RAG sisteminde ilgili doküman parçaları nasıl bulunur?",
        top_k=1,
    )

    assert results[0].doc_id == "doc_005"


def test_hybrid_result_contains_separate_scores() -> None:
    retriever = build_hybrid_retriever()

    results = retriever.search(
        "Pull request ile main branch'e nasıl değişiklik alınır?",
        top_k=1,
    )

    result = results[0]

    assert result.doc_id == "doc_002"
    assert result.vector_score >= 0.0
    assert result.keyword_score >= 0.0
    assert result.combined_score >= 0.0
