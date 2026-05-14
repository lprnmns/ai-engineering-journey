import pytest

from labs.rag.chunking import Chunk
from labs.rag.sample_docs import Document
from labs.rag.vector_store import InMemoryVectorStore, build_vector_store


def test_add_chunk_increases_store_length() -> None:
    store = InMemoryVectorStore()

    chunk = Chunk(
        chunk_id="chunk_001",
        doc_id="doc_001",
        title="Python",
        text="Python sanal ortam oluşturmak için venv kullanılır.",
        source="source/python",
        chunk_index=1,
    )

    store.add_chunk(chunk)

    assert len(store) == 1


def test_search_returns_most_relevant_stored_chunk() -> None:
    store = InMemoryVectorStore()

    python_chunk = Chunk(
        chunk_id="chunk_python",
        doc_id="doc_python",
        title="Python",
        text="Python sanal ortam oluşturmak için venv kullanılır.",
        source="source/python",
        chunk_index=1,
    )
    git_chunk = Chunk(
        chunk_id="chunk_git",
        doc_id="doc_git",
        title="Git",
        text="Git branch ile kod değişiklikleri yönetilir.",
        source="source/git",
        chunk_index=1,
    )

    store.add_chunks([python_chunk, git_chunk])

    results = store.search("sanal ortam nasıl kurulur?", top_k=1)

    assert results[0].chunk_id == "chunk_python"
    assert results[0].score > 0.0


def test_search_rejects_invalid_top_k() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError):
        store.search("test", top_k=0)


def test_stats_counts_chunks_documents_and_sources() -> None:
    store = InMemoryVectorStore()

    store.add_chunks(
        [
            Chunk(
                chunk_id="chunk_001",
                doc_id="doc_001",
                title="Python",
                text="Python venv kullanır.",
                source="source/python",
                chunk_index=1,
            ),
            Chunk(
                chunk_id="chunk_002",
                doc_id="doc_001",
                title="Python",
                text="pip paket kurar.",
                source="source/python",
                chunk_index=2,
            ),
            Chunk(
                chunk_id="chunk_003",
                doc_id="doc_002",
                title="Git",
                text="Git branch kullanır.",
                source="source/git",
                chunk_index=1,
            ),
        ]
    )

    stats = store.stats()

    assert stats.chunk_count == 3
    assert stats.document_count == 2
    assert stats.unique_sources == 2


def test_build_vector_store_chunks_documents_and_indexes_them() -> None:
    documents = [
        Document(
            doc_id="doc_001",
            title="Python",
            text="Python sanal ortam oluşturur. pip paket kurar.",
            source="source/python",
        ),
        Document(
            doc_id="doc_002",
            title="Git",
            text="Git branch açar. Pull request main branch'e alınır.",
            source="source/git",
        ),
    ]

    store = build_vector_store(
        documents=documents,
        sentences_per_chunk=1,
        overlap=0,
    )

    stats = store.stats()

    assert stats.chunk_count == 4
    assert stats.document_count == 2


def test_build_vector_store_can_search_sample_documents() -> None:
    store = build_vector_store()

    results = store.search(
        "RAG sisteminde ilgili doküman parçaları nasıl bulunur?",
        top_k=1,
    )

    assert results[0].doc_id == "doc_005"
