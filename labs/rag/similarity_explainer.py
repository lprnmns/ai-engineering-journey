from __future__ import annotations

import math
from dataclasses import dataclass

from labs.rag.mini_semantic_search import dot_product, text_to_vector, vector_norm
from labs.rag.vectorizer import TermFrequencyVectorizer, Vector, Vectorizer


@dataclass(frozen=True)
class TermContribution:
    term: str
    query_weight: float
    chunk_weight: float
    contribution: float


@dataclass(frozen=True)
class SimilarityExplanation:
    query: str
    chunk_text: str
    query_vector: Vector
    chunk_vector: Vector
    contributions: list[TermContribution]
    dot_product: float
    query_norm: float
    chunk_norm: float
    cosine_similarity: float


def build_term_contributions(
    query_vector: Vector,
    chunk_vector: Vector,
) -> list[TermContribution]:
    common_terms = sorted(set(query_vector).intersection(chunk_vector))

    return [
        TermContribution(
            term=term,
            query_weight=query_vector[term],
            chunk_weight=chunk_vector[term],
            contribution=query_vector[term] * chunk_vector[term],
        )
        for term in common_terms
    ]


def safe_cosine_similarity(
    query_vector: Vector,
    chunk_vector: Vector,
) -> float:
    query_norm = vector_norm(query_vector)
    chunk_norm = vector_norm(chunk_vector)

    if query_norm == 0.0 or chunk_norm == 0.0:
        return 0.0

    return dot_product(query_vector, chunk_vector) / (query_norm * chunk_norm)


def explain_similarity(
    query: str,
    chunk_text: str,
    vectorizer: Vectorizer | None = None,
) -> SimilarityExplanation:
    if vectorizer is None:
        vectorizer = TermFrequencyVectorizer()

    query_vector = vectorizer.vectorize(query)
    chunk_vector = vectorizer.vectorize(chunk_text)

    contributions = build_term_contributions(
        query_vector=query_vector,
        chunk_vector=chunk_vector,
    )

    dot = sum(item.contribution for item in contributions)
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
    chunk_norm = math.sqrt(sum(value * value for value in chunk_vector.values()))

    cosine = 0.0
    if query_norm != 0.0 and chunk_norm != 0.0:
        cosine = dot / (query_norm * chunk_norm)

    return SimilarityExplanation(
        query=query,
        chunk_text=chunk_text,
        query_vector=query_vector,
        chunk_vector=chunk_vector,
        contributions=contributions,
        dot_product=dot,
        query_norm=query_norm,
        chunk_norm=chunk_norm,
        cosine_similarity=cosine,
    )


def format_vector(vector: Vector) -> str:
    if not vector:
        return "{}"

    parts = [
        f"{term}: {weight:.4f}"
        for term, weight in sorted(vector.items())
    ]

    return "{ " + ", ".join(parts) + " }"


def format_similarity_explanation(explanation: SimilarityExplanation) -> str:
    lines = [
        "=== Similarity Explainer ===",
        "",
        "Query:",
        explanation.query,
        "",
        "Chunk:",
        explanation.chunk_text,
        "",
        "Query vector:",
        format_vector(explanation.query_vector),
        "",
        "Chunk vector:",
        format_vector(explanation.chunk_vector),
        "",
        "Term contributions:",
    ]

    if not explanation.contributions:
        lines.append("- No overlapping terms")
    else:
        for item in explanation.contributions:
            lines.append(
                f"- {item.term}: "
                f"{item.query_weight:.4f} * {item.chunk_weight:.4f} "
                f"= {item.contribution:.4f}"
            )

    lines.extend(
        [
            "",
            f"Dot product: {explanation.dot_product:.4f}",
            f"Query norm: {explanation.query_norm:.4f}",
            f"Chunk norm: {explanation.chunk_norm:.4f}",
            f"Cosine similarity: {explanation.cosine_similarity:.4f}",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    query = "Python sanal ortam nasıl kurulur?"
    chunk_text = (
        "Python projelerinde sanal ortam oluşturmak için "
        "python -m venv .venv komutu kullanılır."
    )

    explanation = explain_similarity(query=query, chunk_text=chunk_text)

    print(format_similarity_explanation(explanation))


if __name__ == "__main__":
    main()
