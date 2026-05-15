import pytest

from labs.rag.chunking import ChunkSearchResult
from labs.rag.no_answer_detection import (
    answer_query_with_guard,
    decide_answerability,
    format_guarded_answer,
    get_score_at,
)


def make_result(
    score: float,
    chunk_id: str = "chunk_001",
    doc_id: str = "doc_001",
) -> ChunkSearchResult:
    return ChunkSearchResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        title="Test",
        text="Test text",
        source="test/source",
        chunk_index=1,
        score=score,
    )


def test_get_score_at_returns_zero_when_index_is_missing() -> None:
    assert get_score_at([], 0) == 0.0
    assert get_score_at([make_result(0.7)], 1) == 0.0


def test_decide_answerability_returns_no_results_for_empty_results() -> None:
    decision = decide_answerability(
        query="unknown",
        results=[],
    )

    assert decision.is_answerable is False
    assert decision.reason == "no_results"


def test_decide_answerability_rejects_low_score() -> None:
    decision = decide_answerability(
        query="unknown",
        results=[make_result(0.01)],
        min_score=0.05,
    )

    assert decision.is_answerable is False
    assert decision.reason == "low_score"


def test_decide_answerability_accepts_high_score() -> None:
    decision = decide_answerability(
        query="known",
        results=[make_result(0.8), make_result(0.2)],
        min_score=0.05,
    )

    assert decision.is_answerable is True
    assert decision.reason == "answerable"
    assert decision.max_score == 0.8
    assert decision.second_score == 0.2
    assert decision.score_margin == pytest.approx(0.6)


def test_decide_answerability_can_reject_ambiguous_results() -> None:
    decision = decide_answerability(
        query="ambiguous",
        results=[make_result(0.51), make_result(0.50)],
        min_score=0.05,
        min_margin=0.05,
    )

    assert decision.is_answerable is False
    assert decision.reason == "ambiguous_top_results"


def test_decide_answerability_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        decide_answerability("test", [], min_score=-0.1)

    with pytest.raises(ValueError):
        decide_answerability("test", [], min_margin=-0.1)


def test_answer_query_with_guard_answers_known_query() -> None:
    output = answer_query_with_guard(
        query="RAG sisteminde ilgili doküman parçaları nasıl bulunur?",
        top_k=3,
        min_score=0.05,
    )

    assert output.decision.is_answerable is True
    assert output.answer.is_answered is True
    assert output.answer.sources


def test_answer_query_with_guard_rejects_unknown_query() -> None:
    output = answer_query_with_guard(
        query="Fenerbahçe maç skoru nedir?",
        top_k=3,
        min_score=0.05,
    )

    assert output.decision.is_answerable is False
    assert output.answer.is_answered is False
    assert "yeterli context yok" in output.answer.answer


def test_format_guarded_answer_contains_decision_and_answer() -> None:
    output = answer_query_with_guard(
        query="Fenerbahçe maç skoru nedir?",
        top_k=3,
        min_score=0.05,
    )

    text = format_guarded_answer(output)

    assert "Answerable" in text
    assert "Reason" in text
    assert "Answer:" in text
