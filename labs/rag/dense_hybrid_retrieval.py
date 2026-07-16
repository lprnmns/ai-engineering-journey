from __future__ import annotations

from dataclasses import dataclass

from labs.rag.chunking import Chunk, chunk_documents
from labs.rag.dense_vector_store import DenseTextVectorizer
from labs.rag.dense_vectorizer import DenseVectorizer
from labs.rag.hybrid_retrieval import combine_scores, keyword_overlap_score
from labs.rag.sample_docs import Document, SAMPLE_DOCUMENTS
from labs.rag.similarity import DenseVector, cosine_similarity


@dataclass(frozen=True)
class StoredDenseHybridChunk:
    chunk: Chunk
    vector: DenseVector


@dataclass(frozen=True)
class DenseHybridSearchResult:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    chunk_index: int
    dense_score: float
    keyword_score: float
    combined_score: float


class DenseHybridRetriever:
    """Combine dense semantic similarity with exact keyword overlap."""

    def __init__(
        self,
        chunks: list[Chunk],
        vectorizer: DenseTextVectorizer | None = None,
        dense_weight: float = 0.7,
    ) -> None:
        if dense_weight < 0.0 or dense_weight > 1.0:
            raise ValueError("dense_weight must be between 0 and 1")

        self._vectorizer = vectorizer or DenseVectorizer()
        self._dense_weight = dense_weight
        self._items = [
            StoredDenseHybridChunk(
                chunk=chunk,
                vector=self._vectorizer.vectorize(f"{chunk.title} {chunk.text}"),
            )
            for chunk in chunks
        ]

    def search(self, query: str, top_k: int = 3) -> list[DenseHybridSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self._vectorizer.vectorize(query)
        results: list[DenseHybridSearchResult] = []

        for item in self._items:
            chunk = item.chunk
            chunk_text = f"{chunk.title} {chunk.text}"
            dense_score = cosine_similarity(query_vector, item.vector)
            keyword_score = keyword_overlap_score(query, chunk_text)
            combined_score = combine_scores(
                vector_score=dense_score,
                keyword_score=keyword_score,
                vector_weight=self._dense_weight,
            )

            results.append(
                DenseHybridSearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    text=chunk.text,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    dense_score=dense_score,
                    keyword_score=keyword_score,
                    combined_score=combined_score,
                )
            )

        return sorted(
            results,
            key=lambda result: result.combined_score,
            reverse=True,
        )[:top_k]


def build_dense_hybrid_retriever(
    documents: list[Document] | None = None,
    sentences_per_chunk: int = 1,
    overlap: int = 0,
    vectorizer: DenseTextVectorizer | None = None,
    dense_weight: float = 0.7,
) -> DenseHybridRetriever:
    if documents is None:
        documents = SAMPLE_DOCUMENTS

    chunks = chunk_documents(
        documents=documents,
        sentences_per_chunk=sentences_per_chunk,
        overlap=overlap,
    )

    return DenseHybridRetriever(
        chunks=chunks,
        vectorizer=vectorizer,
        dense_weight=dense_weight,
    )
