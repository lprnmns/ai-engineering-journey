from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from labs.rag.sample_docs import Document, SAMPLE_DOCUMENTS


TOKEN_PATTERN = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+")


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    title: str
    text: str
    source: str
    score: float


def tokenize(text: str) -> list[str]:
    """Lowercase text and split it into simple word tokens."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def text_to_vector(text: str) -> dict[str, float]:
    """Convert text into a simple term-frequency vector."""
    tokens = tokenize(text)
    counts = Counter(tokens)

    return {token: float(count) for token, count in counts.items()}


def dot_product(left: dict[str, float], right: dict[str, float]) -> float:
    common_keys = set(left).intersection(right)

    return sum(left[key] * right[key] for key in common_keys)


def vector_norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    left_norm = vector_norm(left)
    right_norm = vector_norm(right)

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product(left, right) / (left_norm * right_norm)


def score_document(query: str, document: Document) -> SearchResult:
    query_vector = text_to_vector(query)
    document_vector = text_to_vector(f"{document.title} {document.text}")

    score = cosine_similarity(query_vector, document_vector)

    return SearchResult(
        doc_id=document.doc_id,
        title=document.title,
        text=document.text,
        source=document.source,
        score=score,
    )


def search(
    query: str,
    documents: list[Document] | None = None,
    top_k: int = 3,
) -> list[SearchResult]:
    if documents is None:
        documents = SAMPLE_DOCUMENTS

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    results = [score_document(query, document) for document in documents]

    return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


def format_results(results: list[SearchResult]) -> str:
    lines: list[str] = []

    for index, result in enumerate(results, start=1):
        lines.append(
            f"{index}. {result.title} | score={result.score:.4f} | source={result.source}"
        )
        lines.append(f"   {result.text}")

    return "\n".join(lines)


def main() -> None:
    query = "Python sanal ortam nasıl kurulur?"
    results = search(query, top_k=3)

    print("=== Mini Semantic Search ===")
    print(f"Query: {query}")
    print()
    print(format_results(results))


if __name__ == "__main__":
    main()
