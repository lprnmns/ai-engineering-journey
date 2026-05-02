#!/usr/bin/env python3

import logging
import time
from math import sqrt
from typing import Sequence


Vector = Sequence[float]


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger(__name__)


def dot(a: Vector, b: Vector) -> float:
    """Return the dot product of two vectors."""
    if len(a) != len(b):
        logger.error("Vector length mismatch: len(a)=%s, len(b)=%s", len(a), len(b))
        raise ValueError("Vectors must have the same length.")

    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    """Return the Euclidean norm of a vector."""
    return sqrt(dot(a, a))


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Return cosine similarity between two vectors."""
    norm_a = norm(a)
    norm_b = norm(b)

    if norm_a == 0 or norm_b == 0:
        logger.error("Zero vector detected: norm_a=%s, norm_b=%s", norm_a, norm_b)
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    similarity = dot(a, b) / (norm_a * norm_b)

    logger.debug("cosine_similarity=%s", similarity)

    return similarity


def _almost_equal(x: float, y: float, tolerance: float = 1e-9) -> bool:
    return abs(x - y) < tolerance


def benchmark_cosine_similarity(iterations: int = 100_000) -> float:
    """Run a tiny benchmark for cosine similarity."""
    query = [0.2, 0.8, 0.1]
    doc = [0.3, 0.7, 0.2]

    start = time.perf_counter()

    for _ in range(iterations):
        cosine_similarity(query, doc)

    end = time.perf_counter()

    duration = end - start
    logger.info("benchmark iterations=%s duration=%.6fs", iterations, duration)

    return duration


def run_self_tests() -> None:
    assert dot([1, 2, 3], [4, 5, 6]) == 32
    assert norm([3, 4]) == 5
    assert _almost_equal(cosine_similarity([1, 0], [1, 0]), 1.0)
    assert _almost_equal(cosine_similarity([1, 0], [0, 1]), 0.0)
    assert _almost_equal(cosine_similarity([1, 1], [1, 1]), 1.0)

    try:
        dot([1, 2], [1])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for vectors with different lengths.")

    try:
        cosine_similarity([0, 0], [1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for zero vector.")

    print("All vector tests passed.")


if __name__ == "__main__":
    run_self_tests()

    query = [0.2, 0.8, 0.1]
    doc_a = [0.3, 0.7, 0.2]
    doc_b = [0.9, 0.1, 0.0]

    print("query vs doc_a:", cosine_similarity(query, doc_a))
    print("query vs doc_b:", cosine_similarity(query, doc_b))

    benchmark_cosine_similarity()
