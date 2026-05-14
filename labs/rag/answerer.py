from __future__ import annotations

from dataclasses import dataclass

from labs.rag.chunking import ChunkSearchResult
from labs.rag.context_builder import build_context
from labs.rag.vector_store import build_vector_store


@dataclass(frozen=True)
class SourceReference:
    source: str
    doc_id: str
    chunk_id: str
    score: float


@dataclass(frozen=True)
class RagAnswer:
    query: str
    answer: str
    context: str
    sources: list[SourceReference]
    used_chunk_count: int
    max_score: float
    is_answered: bool


def build_source_references(results: list[ChunkSearchResult]) -> list[SourceReference]:
    return [
        SourceReference(
            source=result.source,
            doc_id=result.doc_id,
            chunk_id=result.chunk_id,
            score=result.score,
        )
        for result in results
    ]


def deduplicate_texts(results: list[ChunkSearchResult]) -> list[str]:
    seen: set[str] = set()
    texts: list[str] = []

    for result in results:
        normalized = " ".join(result.text.split())

        if normalized not in seen:
            texts.append(normalized)
            seen.add(normalized)

    return texts


def build_extractive_answer(
    query: str,
    results: list[ChunkSearchResult],
    min_score: float = 0.05,
) -> RagAnswer:
    if min_score < 0:
        raise ValueError("min_score must be zero or greater")

    if not results:
        return RagAnswer(
            query=query,
            answer="Bu soruyu cevaplamak için yeterli context bulunamadı.",
            context="",
            sources=[],
            used_chunk_count=0,
            max_score=0.0,
            is_answered=False,
        )

    max_score = max(result.score for result in results)

    if max_score < min_score:
        return RagAnswer(
            query=query,
            answer="Bu soruyu cevaplamak için context yeterli değil.",
            context=build_context(results),
            sources=build_source_references(results),
            used_chunk_count=len(results),
            max_score=max_score,
            is_answered=False,
        )

    texts = deduplicate_texts(results)

    answer = "Context'e göre: " + " ".join(texts)

    return RagAnswer(
        query=query,
        answer=answer,
        context=build_context(results),
        sources=build_source_references(results),
        used_chunk_count=len(results),
        max_score=max_score,
        is_answered=True,
    )


def answer_query(
    query: str,
    top_k: int = 2,
    min_score: float = 0.05,
) -> RagAnswer:
    store = build_vector_store()
    results = store.search(query=query, top_k=top_k)

    return build_extractive_answer(
        query=query,
        results=results,
        min_score=min_score,
    )


def format_answer(answer: RagAnswer) -> str:
    lines = [
        "=== Mini RAG Answer ===",
        f"Query: {answer.query}",
        f"Answered: {answer.is_answered}",
        f"Max score: {answer.max_score:.4f}",
        "",
        "Answer:",
        answer.answer,
        "",
        "Sources:",
    ]

    if not answer.sources:
        lines.append("- No sources")
    else:
        for index, source in enumerate(answer.sources, start=1):
            lines.append(
                f"- [{index}] {source.source} | {source.doc_id} | "
                f"{source.chunk_id} | score={source.score:.4f}"
            )

    lines.extend(
        [
            "",
            "Context:",
            answer.context,
        ]
    )

    return "\n".join(lines)


def main() -> None:
    query = "RAG sisteminde ilgili doküman parçaları nasıl bulunur?"
    answer = answer_query(query=query, top_k=2)

    print(format_answer(answer))


if __name__ == "__main__":
    main()
