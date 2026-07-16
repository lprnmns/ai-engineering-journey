from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from labs.rag.mini_semantic_search import text_to_vector

Vector = dict[str, float]


class Vectorizer(Protocol):
    """Interface for converting text into vectors.

    Today we use a term-frequency vectorizer.

    Later we can replace it with:
    - TF-IDF
    - BM25
    - OpenAI embeddings
    - Gemini embeddings
    - sentence-transformers
    """

    def vectorize(self, text: str) -> Vector:
        """Convert text into a vector."""
        ...


@dataclass(frozen=True)
class TermFrequencyVectorizer:
    """Simple educational vectorizer based on word counts."""

    def vectorize(self, text: str) -> Vector:
        return text_to_vector(text)
