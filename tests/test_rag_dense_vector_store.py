import pytest

from labs.rag.chunking import Chunk
from labs.rag.dense_vector_store import DenseVectorStore


class FakeDenseVectorizer:
    def vectorize(self, text: str) -> list[float]:
        normalized = text.lower()

        if "izole" in normalized or "sanal" in normalized or "venv" in normalized:
            return [1.0, 0.0]

        if "git" in normalized or "branch" in normalized:
            return [0.0, 1.0]

        raise ValueError(f"unexpected text: {text}")


def test_dense_store_retrieves_semantically_mapped_chunk() -> None:
    store = DenseVectorStore(vectorizer=FakeDenseVectorizer())
    store.add_chunks(
        [
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
        ]
    )

    results = store.search("izole çalışma ortamı nasıl hazırlanır?", top_k=1)

    assert results[0].chunk_id == "chunk_python"
    assert results[0].score == pytest.approx(1.0)


def test_dense_store_rejects_invalid_top_k() -> None:
    store = DenseVectorStore(vectorizer=FakeDenseVectorizer())

    with pytest.raises(ValueError):
        store.search("İzole çalışma ortamı", top_k=0)


def test_dense_store_stats_count_chunks_documents_and_sources() -> None:
    store = DenseVectorStore(vectorizer=FakeDenseVectorizer())
    store.add_chunk(
        Chunk(
            chunk_id="chunk_python",
            doc_id="doc_python",
            title="Python",
            text="Sanal ortam venv ile oluşturulur.",
            source="source/python",
            chunk_index=1,
        )
    )

    stats = store.stats()

    assert stats.chunk_count == 1
    assert stats.document_count == 1
    assert stats.unique_sources == 1
