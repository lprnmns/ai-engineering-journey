from __future__ import annotations

from dataclasses import dataclass

from labs.rag.chunking import Chunk, chunk_documents
from labs.rag.mini_semantic_search import cosine_similarity, tokenize
from labs.rag.sample_docs import Document, SAMPLE_DOCUMENTS
from labs.rag.vectorizer import TermFrequencyVectorizer, Vector, Vectorizer


@dataclass(frozen=True)
class StoredHybridChunk:
    chunk: Chunk
    vector: Vector


@dataclass(frozen=True)
class HybridSearchResult:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    chunk_index: int
    vector_score: float
    keyword_score: float
    combined_score: float


def keyword_overlap_score(query: str, text: str) -> float:
    query_terms = set(tokenize(query))

    if not query_terms:
        return 0.0

    text_terms = set(tokenize(text))
    overlap = query_terms.intersection(text_terms)

    return len(overlap) / len(query_terms)


def combine_scores(
    vector_score: float,
    keyword_score: float,
    vector_weight: float = 0.7,
) -> float:
    if vector_weight < 0.0 or vector_weight > 1.0:
        raise ValueError("vector_weight must be between 0 and 1")

    keyword_weight = 1.0 - vector_weight

    return (vector_weight * vector_score) + (keyword_weight * keyword_score)


class HybridRetriever:
    """Educational hybrid retriever.

    It combines:
    - vector similarity
    - exact keyword overlap

    This is useful because vector search captures similarity,
    while keyword search protects exact important terms like error codes,
    package names, function names, or product IDs.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        vectorizer: Vectorizer | None = None,
        vector_weight: float = 0.7,
    ) -> None:
        if vector_weight < 0.0 or vector_weight > 1.0:
            raise ValueError("vector_weight must be between 0 and 1")

        self._vectorizer = vectorizer or TermFrequencyVectorizer()
        self._vector_weight = vector_weight
        self._items = [
            StoredHybridChunk(
                chunk=chunk,
                vector=self._vectorizer.vectorize(f"{chunk.title} {chunk.text}"),
            )
            for chunk in chunks
        ]

    def search(self, query: str, top_k: int = 3) -> list[HybridSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self._vectorizer.vectorize(query)
        results: list[HybridSearchResult] = []

        for item in self._items:
            chunk = item.chunk
            chunk_text = f"{chunk.title} {chunk.text}"

            vector_score = cosine_similarity(query_vector, item.vector)
            keyword_score = keyword_overlap_score(query, chunk_text)
            combined_score = combine_scores(
                vector_score=vector_score,
                keyword_score=keyword_score,
                vector_weight=self._vector_weight,
            )

            results.append(
                HybridSearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    text=chunk.text,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    combined_score=combined_score,
                )
            )

        return sorted(
            results,
            key=lambda result: result.combined_score,
            reverse=True,
        )[:top_k]


def build_hybrid_retriever(
    documents: list[Document] | None = None,
    sentences_per_chunk: int = 1,
    overlap: int = 0,
    vectorizer: Vectorizer | None = None,
    vector_weight: float = 0.7,
) -> HybridRetriever:
    if documents is None:
        documents = SAMPLE_DOCUMENTS

    chunks = chunk_documents(
        documents=documents,
        sentences_per_chunk=sentences_per_chunk,
        overlap=overlap,
    )

    return HybridRetriever(
        chunks=chunks,
        vectorizer=vectorizer,
        vector_weight=vector_weight,
    )


def format_hybrid_results(results: list[HybridSearchResult]) -> str:
    lines: list[str] = []

    for index, result in enumerate(results, start=1):
        lines.append(
            f"{index}. {result.chunk_id} | {result.title} | "
            f"combined={result.combined_score:.4f} | "
            f"vector={result.vector_score:.4f} | "
            f"keyword={result.keyword_score:.4f}"
        )
        lines.append(f"   {result.text}")

    return "\n".join(lines)


def main() -> None:
    retriever = build_hybrid_retriever(vector_weight=0.7)
    query = "Pull request ile main branch'e nasıl değişiklik alınır?"

    results = retriever.search(query=query, top_k=3)

    print("=== Hybrid Retrieval ===")
    print(f"Query: {query}")
    print()
    print(format_hybrid_results(results))


if __name__ == "__main__":
    main()
