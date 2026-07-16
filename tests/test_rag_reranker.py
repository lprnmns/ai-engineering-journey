import pytest

from labs.rag.chunking import ChunkSearchResult
from labs.rag.reranker import CrossEncoderReranker


class FakePairScoringModel:
    def __init__(self) -> None:
        self.received_pairs: list[tuple[str, str]] | None = None

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        show_progress_bar: bool,
    ) -> list[float]:
        self.received_pairs = pairs
        assert show_progress_bar is False
        return [0.2, 0.9]


def make_candidate(
    doc_id: str,
    title: str,
    text: str,
    score: float,
) -> ChunkSearchResult:
    return ChunkSearchResult(
        chunk_id=f"{doc_id}_chunk_001",
        doc_id=doc_id,
        title=title,
        text=text,
        source="test/source",
        chunk_index=1,
        score=score,
    )


def test_reranker_reorders_dense_candidates() -> None:
    model = FakePairScoringModel()
    reranker = CrossEncoderReranker(model=model)
    candidates = [
        make_candidate("doc_venv", "Virtual Environment", "Sanal ortam oluşturma", 0.8),
        make_candidate("doc_pip", "Package Installation", "pip install paket yükler", 0.7),
    ]

    results = reranker.rerank(
        query="Python paketi terminalden nasıl yüklenir?",
        candidates=candidates,
        top_k=2,
    )

    assert [result.doc_id for result in results] == ["doc_pip", "doc_venv"]
    assert results[0].retrieval_score == 0.7
    assert results[0].reranker_score == 0.9
    assert model.received_pairs == [
        ("Python paketi terminalden nasıl yüklenir?", "Virtual Environment Sanal ortam oluşturma"),
        ("Python paketi terminalden nasıl yüklenir?", "Package Installation pip install paket yükler"),
    ]


def test_reranker_returns_empty_list_for_empty_candidates() -> None:
    reranker = CrossEncoderReranker(model=FakePairScoringModel())

    assert reranker.rerank("query", [], top_k=3) == []


def test_reranker_rejects_invalid_top_k() -> None:
    reranker = CrossEncoderReranker(model=FakePairScoringModel())

    with pytest.raises(ValueError):
        reranker.rerank("query", [], top_k=0)
