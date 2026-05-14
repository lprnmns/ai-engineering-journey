from __future__ import annotations

from dataclasses import dataclass

from labs.rag.chunking import ChunkSearchResult
from labs.rag.vector_store import build_vector_store


@dataclass(frozen=True)
class ContextBuildResult:
    query: str
    context: str
    prompt: str
    used_chunk_count: int


def format_context_block(index: int, result: ChunkSearchResult) -> str:
    return (
        f"[{index}] Source: {result.source} | "
        f"Document: {result.doc_id} | "
        f"Chunk: {result.chunk_id} | "
        f"Score: {result.score:.4f}\n"
        f"{result.text}"
    )


def build_context(
    results: list[ChunkSearchResult],
    max_chars: int | None = None,
) -> str:
    if max_chars is not None and max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    blocks = [
        format_context_block(index=index, result=result)
        for index, result in enumerate(results, start=1)
    ]

    context = "\n\n".join(blocks)

    if max_chars is not None and len(context) > max_chars:
        return context[:max_chars].rstrip() + "\n[TRUNCATED]"

    return context


def build_rag_prompt(query: str, context: str) -> str:
    return f"""You are a careful AI assistant.

Answer the user's question using only the context below.

If the context is not enough, say that the context is not enough.
Do not invent facts outside the context.

Question:
{query}

Context:
{context}

Answer:
"""


def build_context_and_prompt(
    query: str,
    results: list[ChunkSearchResult],
    max_context_chars: int | None = 2_000,
) -> ContextBuildResult:
    context = build_context(results, max_chars=max_context_chars)
    prompt = build_rag_prompt(query=query, context=context)

    return ContextBuildResult(
        query=query,
        context=context,
        prompt=prompt,
        used_chunk_count=len(results),
    )


def main() -> None:
    query = "RAG sisteminde ilgili doküman parçaları nasıl bulunur?"

    store = build_vector_store()
    results = store.search(query=query, top_k=2)

    context_result = build_context_and_prompt(
        query=query,
        results=results,
        max_context_chars=2_000,
    )

    print("=== RAG Context Builder ===")
    print(f"Query: {context_result.query}")
    print()
    print("=== Context ===")
    print(context_result.context)
    print()
    print("=== Prompt Sent To LLM ===")
    print(context_result.prompt)


if __name__ == "__main__":
    main()
