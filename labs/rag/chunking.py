from __future__ import annotations

import re
from dataclasses import dataclass

from labs.rag.mini_semantic_search import cosine_similarity, text_to_vector
from labs.rag.sample_docs import Document, SAMPLE_DOCUMENTS


SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    chunk_index: int


@dataclass(frozen=True)
class ChunkSearchResult:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    chunk_index: int
    score: float


def split_into_sentences(text: str) -> list[str]:
    cleaned_text = " ".join(text.strip().split())

    if not cleaned_text:
        return []

    return [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(cleaned_text)
        if sentence.strip()
    ]


def chunk_sentences(
    sentences: list[str],
    sentences_per_chunk: int = 2,
    overlap: int = 1,
) -> list[str]:
    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap must be zero or greater")

    if overlap >= sentences_per_chunk:
        raise ValueError("overlap must be smaller than sentences_per_chunk")

    chunks: list[str] = []
    start = 0
    step = sentences_per_chunk - overlap

    while start < len(sentences):
        end = start + sentences_per_chunk
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)

        if end >= len(sentences):
            break

        start += step

    return chunks


def chunk_document(
    document: Document,
    sentences_per_chunk: int = 2,
    overlap: int = 1,
) -> list[Chunk]:
    sentences = split_into_sentences(document.text)
    chunk_texts = chunk_sentences(
        sentences=sentences,
        sentences_per_chunk=sentences_per_chunk,
        overlap=overlap,
    )

    return [
        Chunk(
            chunk_id=f"{document.doc_id}_chunk_{index:03d}",
            doc_id=document.doc_id,
            title=document.title,
            text=chunk_text,
            source=document.source,
            chunk_index=index,
        )
        for index, chunk_text in enumerate(chunk_texts, start=1)
    ]


def chunk_documents(
    documents: list[Document],
    sentences_per_chunk: int = 2,
    overlap: int = 1,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                sentences_per_chunk=sentences_per_chunk,
                overlap=overlap,
            )
        )

    return chunks


def score_chunk(query: str, chunk: Chunk) -> ChunkSearchResult:
    query_vector = text_to_vector(query)
    chunk_vector = text_to_vector(f"{chunk.title} {chunk.text}")

    score = cosine_similarity(query_vector, chunk_vector)

    return ChunkSearchResult(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        text=chunk.text,
        source=chunk.source,
        chunk_index=chunk.chunk_index,
        score=score,
    )


def search_chunks(
    query: str,
    chunks: list[Chunk] | None = None,
    top_k: int = 3,
) -> list[ChunkSearchResult]:
    if chunks is None:
        chunks = chunk_documents(SAMPLE_DOCUMENTS)

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    results = [score_chunk(query, chunk) for chunk in chunks]

    return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


def format_chunk_results(results: list[ChunkSearchResult]) -> str:
    lines: list[str] = []

    for index, result in enumerate(results, start=1):
        lines.append(
            f"{index}. {result.chunk_id} | {result.title} | "
            f"score={result.score:.4f} | source={result.source}"
        )
        lines.append(f"   {result.text}")

    return "\n".join(lines)


def main() -> None:
    query = "Pull request ile main branch'e nasıl değişiklik alınır?"
    chunks = chunk_documents(SAMPLE_DOCUMENTS, sentences_per_chunk=1, overlap=0)
    results = search_chunks(query=query, chunks=chunks, top_k=3)

    print("=== Chunk-Level Semantic Search ===")
    print(f"Query: {query}")
    print()
    print(format_chunk_results(results))


if __name__ == "__main__":
    main()
