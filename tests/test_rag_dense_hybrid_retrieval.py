import pytest

from labs.rag.chunking import Chunk
from labs.rag.dense_hybrid_retrieval import DenseHybridRetriever


class FakeDenseVectorizer:
    def vectorize(self, text: str) -> list[float]:
        normalized = text.lower()

        if "404" in normalized:
            return [1.0, 1.0]

        if "500" in normalized:
            return [1.0, 1.0]

        if "izole" in normalized or "sanal" in normalized or "venv" in normalized:
            return [1.0, 0.0]

        if "git" in normalized or "branch" in normalized:
            return [0.0, 1.0]

        raise ValueError(f"unexpected text: {text}")


def test_dense_hybrid_uses_keyword_signal_to_break_dense_tie() -> None:
    retriever = DenseHybridRetriever(
        chunks=[
            Chunk(
                chunk_id="chunk_404",
                doc_id="doc_404",
                title="HTTP 404",
                text="İstenen sayfa bulunamadı.",
                source="source/http",
                chunk_index=1,
            ),
            Chunk(
                chunk_id="chunk_500",
                doc_id="doc_500",
                title="HTTP 500",
                text="Sunucuda beklenmeyen hata oluştu.",
                source="source/http",
                chunk_index=1,
            ),
        ],
        vectorizer=FakeDenseVectorizer(),
        dense_weight=0.5,
    )

    results = retriever.search("HTTP 404 hatası ne demek?", top_k=1)

    assert results[0].chunk_id == "chunk_404"
    assert results[0].dense_score == pytest.approx(1.0)
    assert results[0].keyword_score > 0.0


def test_dense_hybrid_retrieves_semantic_paraphrase() -> None:
    retriever = DenseHybridRetriever(
        chunks=[
            Chunk(
                chunk_id="chunk_python",
                doc_id="doc_python",
                title="Python",
                text="Sanal ortam venv ile oluşturulur.",
                source="source/python",
                chunk_index=1,
            ),
            Chunk(
                chunk_id="chunk_git",
                doc_id="doc_git",
                title="Git",
                text="Git branch ile değişiklikler yönetilir.",
                source="source/git",
                chunk_index=1,
            ),
        ],
        vectorizer=FakeDenseVectorizer(),
    )

    results = retriever.search("izole çalışma ortamı nasıl hazırlanır?", top_k=1)

    assert results[0].chunk_id == "chunk_python"
    assert results[0].dense_score == pytest.approx(1.0)


def test_dense_hybrid_rejects_invalid_dense_weight() -> None:
    with pytest.raises(ValueError):
        DenseHybridRetriever(chunks=[], vectorizer=FakeDenseVectorizer(), dense_weight=1.1)
