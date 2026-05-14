import pytest

from labs.rag.chunking import (
    Chunk,
    chunk_document,
    chunk_documents,
    chunk_sentences,
    search_chunks,
    split_into_sentences,
)
from labs.rag.sample_docs import Document


def test_split_into_sentences_splits_basic_text() -> None:
    text = "Birinci cümle. İkinci cümle! Üçüncü cümle?"

    sentences = split_into_sentences(text)

    assert sentences == ["Birinci cümle.", "İkinci cümle!", "Üçüncü cümle?"]


def test_split_into_sentences_returns_empty_for_blank_text() -> None:
    assert split_into_sentences("   ") == []


def test_chunk_sentences_uses_overlap() -> None:
    sentences = [
        "Sentence one.",
        "Sentence two.",
        "Sentence three.",
        "Sentence four.",
    ]

    chunks = chunk_sentences(
        sentences=sentences,
        sentences_per_chunk=2,
        overlap=1,
    )

    assert chunks == [
        "Sentence one. Sentence two.",
        "Sentence two. Sentence three.",
        "Sentence three. Sentence four.",
    ]


def test_chunk_sentences_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        chunk_sentences(["A."], sentences_per_chunk=0)

    with pytest.raises(ValueError):
        chunk_sentences(["A."], sentences_per_chunk=2, overlap=-1)

    with pytest.raises(ValueError):
        chunk_sentences(["A."], sentences_per_chunk=2, overlap=2)


def test_chunk_document_creates_chunk_metadata() -> None:
    document = Document(
        doc_id="doc_test",
        title="Test Document",
        text="İlk bilgi. İkinci bilgi. Üçüncü bilgi.",
        source="test/source",
    )

    chunks = chunk_document(
        document=document,
        sentences_per_chunk=2,
        overlap=1,
    )

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "doc_test_chunk_001"
    assert chunks[0].doc_id == "doc_test"
    assert chunks[0].title == "Test Document"
    assert chunks[0].source == "test/source"
    assert chunks[0].chunk_index == 1


def test_chunk_documents_combines_multiple_documents() -> None:
    documents = [
        Document(
            doc_id="doc_1",
            title="Doc 1",
            text="Birinci doküman.",
            source="source/1",
        ),
        Document(
            doc_id="doc_2",
            title="Doc 2",
            text="İkinci doküman.",
            source="source/2",
        ),
    ]

    chunks = chunk_documents(documents, sentences_per_chunk=1, overlap=0)

    assert len(chunks) == 2
    assert {chunk.doc_id for chunk in chunks} == {"doc_1", "doc_2"}


def test_search_chunks_returns_most_relevant_chunk() -> None:
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

    results = search_chunks(
        query="sanal ortam nasıl oluşturulur?",
        chunks=chunks,
        top_k=1,
    )

    assert results[0].chunk_id == "chunk_python"


def test_search_chunks_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError):
        search_chunks("test", chunks=[], top_k=0)
