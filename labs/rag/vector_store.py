from __future__ import annotations

from dataclasses import dataclass

from labs.rag.chunking import Chunk, ChunkSearchResult, chunk_documents
from labs.rag.mini_semantic_search import cosine_similarity, text_to_vector
from labs.rag.sample_docs import Document, SAMPLE_DOCUMENTS


Vector = dict[str, float]


@dataclass(frozen=True)
class StoredChunk:
    chunk: Chunk
    vector: Vector


@dataclass(frozen=True)
class VectorStoreStats:
    chunk_count: int
    document_count: int
    unique_sources: int


class InMemoryVectorStore:
    """Small educational vector store for chunk-level retrieval.

    This class stores chunks and their vectors in memory.

    Real vector databases do the same idea at larger scale:
    chunk + metadata + vector.
    """

    def __init__(self) -> None:
        self._items: list[StoredChunk] = []

    def add_chunk(self, chunk: Chunk) -> None:
        vector = text_to_vector(f"{chunk.title} {chunk.text}")

        self._items.append(
            StoredChunk(
                chunk=chunk,
                vector=vector,
            )
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.add_chunk(chunk)

    def search(self, query: str, top_k: int = 3) -> list[ChunkSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = text_to_vector(query)
        results: list[ChunkSearchResult] = []

        for item in self._items:
            score = cosine_similarity(query_vector, item.vector)
            chunk = item.chunk

            results.append(
                ChunkSearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    text=chunk.text,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    score=score,
                )
            )

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


def build_vector_store(
    documents: list[Document] | None = None,
    sentences_per_chunk: int = 1,
    overlap: int = 0,
) -> InMemoryVectorStore:
    if documents is None:
        documents = SAMPLE_DOCUMENTS

    chunks = chunk_documents(
        documents=documents,
        sentences_per_chunk=sentences_per_chunk,
        overlap=overlap,
    )

    store = InMemoryVectorStore()
    store.add_chunks(chunks)

    return store


def format_store_results(results: list[ChunkSearchResult]) -> str:
    lines: list[str] = []

    for index, result in enumerate(results, start=1):
        lines.append(
            f"{index}. {result.chunk_id} | {result.title} | "
            f"score={result.score:.4f} | source={result.source}"
        )
        lines.append(f"   {result.text}")

    return "\n".join(lines)


def main() -> None:
    store = build_vector_store()
    stats = store.stats()

    query = "RAG sisteminde ilgili doküman parçaları nasıl bulunur?"
    results = store.search(query, top_k=3)

    print("=== Mini Vector Store ===")
    print(
        f"Indexed chunks: {stats.chunk_count} | "
        f"documents: {stats.document_count} | "
        f"sources: {stats.unique_sources}"
    )
    print()
    print(f"Query: {query}")
    print()
    print(format_store_results(results))


if __name__ == "__main__":
    main()
