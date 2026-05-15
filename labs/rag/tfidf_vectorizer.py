from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from labs.rag.mini_semantic_search import tokenize
from labs.rag.vectorizer import Vector


@dataclass
class TfidfVectorizer:
    """Educational TF-IDF vectorizer.

    TF-IDF means:

    - TF: Term Frequency
    - IDF: Inverse Document Frequency

    A word gets higher weight when:
    - it appears in the current text
    - it is not common across all documents
    """

    idf: dict[str, float] = field(default_factory=dict)
    vocabulary: set[str] = field(default_factory=set)
    is_fitted: bool = False

    def fit(self, texts: list[str]) -> None:
        if not texts:
            raise ValueError("texts must not be empty")

        document_count = len(texts)
        document_frequency: Counter[str] = Counter()

        for text in texts:
            unique_tokens = set(tokenize(text))
            document_frequency.update(unique_tokens)

        self.vocabulary = set(document_frequency)

        self.idf = {
            token: math.log((1 + document_count) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }

        self.is_fitted = True

    def vectorize(self, text: str) -> Vector:
        if not self.is_fitted:
            raise RuntimeError("TfidfVectorizer must be fitted before vectorize is called")

        tokens = tokenize(text)
        term_counts = Counter(tokens)

        if not tokens:
            return {}

        total_terms = len(tokens)
        vector: Vector = {}

        for token, count in term_counts.items():
            if token not in self.vocabulary:
                continue

            tf = count / total_terms
            vector[token] = tf * self.idf[token]

        return vector
