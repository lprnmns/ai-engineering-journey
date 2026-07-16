from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from labs.rag.chunking import Chunk, ChunkSearchResult, chunk_documents
from labs.rag.dense_vectorizer import DenseVectorizer
from labs.rag.sample_docs import Document, SAMPLE_DOCUMENTS
from labs.rag.similarity import DenseVector, cosine_similarity
from labs.rag.vector_store import VectorStoreStats


class DenseTextVectorizer(Protocol):
    def vectorize(self, text: str) -> DenseVector: ...


@dataclass(frozen=True)
class StoredDenseChunk:
    chunk: Chunk
    vector: DenseVector


class DenseVectorStore:
    """In-memory chunk store that retrieves with dense sentence embeddings."""

    def __init__(self, vectorizer: DenseTextVectorizer | None = None) -> None:
        self._items: list[StoredDenseChunk] = []
        self._vectorizer = vectorizer or DenseVectorizer()

    def add_chunk(self, chunk: Chunk) -> None:
        vector = self._vectorizer.vectorize(f"{chunk.title} {chunk.text}")
        self._items.append(StoredDenseChunk(chunk=chunk, vector=vector))

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.add_chunk(chunk)

    def search(self, query: str, top_k: int = 3) -> list[ChunkSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self._vectorizer.vectorize(query)
        results = [
            ChunkSearchResult(
                chunk_id=item.chunk.chunk_id,
                doc_id=item.chunk.doc_id,
                title=item.chunk.title,
                text=item.chunk.text,
                source=item.chunk.source,
                chunk_index=item.chunk.chunk_index,
                score=cosine_similarity(query_vector, item.vector),
            )
            for item in self._items
        ]

        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def stats(self) -> VectorStoreStats:
        doc_ids = {item.chunk.doc_id for item in self._items}
        sources = {item.chunk.source for item in self._items}

        return VectorStoreStats(
            chunk_count=len(self._items),
            document_count=len(doc_ids),
            unique_sources=len(sources),
        )

    def __len__(self) -> int:
        return len(self._items)


def build_dense_vector_store(
    documents: list[Document] | None = None,
    sentences_per_chunk: int = 1,
    overlap: int = 0,
    vectorizer: DenseTextVectorizer | None = None,
) -> DenseVectorStore:
    if documents is None:
        documents = SAMPLE_DOCUMENTS

    chunks = chunk_documents(
        documents=documents,
        sentences_per_chunk=sentences_per_chunk,
        overlap=overlap,
    )

    store = DenseVectorStore(vectorizer=vectorizer)
    store.add_chunks(chunks)

    return store
