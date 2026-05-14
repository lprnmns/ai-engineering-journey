import pytest

from labs.rag.answerer import (
    build_extractive_answer,
    build_source_references,
    deduplicate_texts,
    answer_query,
)
from labs.rag.chunking import ChunkSearchResult


def make_result(
    text: str = "RAG sistemlerinde ilgili doküman parçaları bulunur.",
    score: float = 0.8,
    chunk_id: str = "doc_005_chunk_001",
) -> ChunkSearchResult:
    return ChunkSearchResult(
        chunk_id=chunk_id,
        doc_id="doc_005",
        title="RAG Retrieval",
        text=text,
        source="toolbox/rag",
        chunk_index=1,
        score=score,
    )


def test_build_source_references_keeps_metadata() -> None:
    references = build_source_references([make_result()])

    assert len(references) == 1
    assert references[0].source == "toolbox/rag"
    assert references[0].doc_id == "doc_005"
    assert references[0].chunk_id == "doc_005_chunk_001"
    assert references[0].score == 0.8


def test_deduplicate_texts_removes_duplicate_chunk_texts() -> None:
    results = [
        make_result(text="Aynı bilgi.", chunk_id="chunk_001"),
        make_result(text="Aynı bilgi.", chunk_id="chunk_002"),
        make_result(text="Farklı bilgi.", chunk_id="chunk_003"),
    ]

    texts = deduplicate_texts(results)

    assert texts == ["Aynı bilgi.", "Farklı bilgi."]


def test_build_extractive_answer_returns_answer_when_score_is_high_enough() -> None:
    output = build_extractive_answer(
        query="RAG nasıl bilgi bulur?",
        results=[make_result()],
        min_score=0.05,
    )

    assert output.is_answered is True
    assert "Context'e göre" in output.answer
    assert "doküman parçaları" in output.answer
    assert output.used_chunk_count == 1
    assert output.max_score == 0.8


def test_build_extractive_answer_returns_not_enough_context_for_empty_results() -> None:
    output = build_extractive_answer(
        query="Bilinmeyen soru",
        results=[],
    )

    assert output.is_answered is False
    assert output.sources == []
    assert "yeterli context bulunamadı" in output.answer


def test_build_extractive_answer_returns_not_enough_context_for_low_score() -> None:
    output = build_extractive_answer(
        query="Bilinmeyen soru",
        results=[make_result(score=0.0)],
        min_score=0.05,
    )

    assert output.is_answered is False
    assert "context yeterli değil" in output.answer
    assert output.max_score == 0.0


def test_build_extractive_answer_rejects_negative_min_score() -> None:
    with pytest.raises(ValueError):
        build_extractive_answer(
            query="test",
            results=[make_result()],
            min_score=-0.1,
        )


def test_answer_query_works_with_sample_vector_store() -> None:
    output = answer_query(
        query="RAG sisteminde ilgili doküman parçaları nasıl bulunur?",
        top_k=2,
    )

    assert output.is_answered is True
    assert output.used_chunk_count == 2
    assert "doküman parçaları" in output.answer
    assert output.sources[0].doc_id == "doc_005"
