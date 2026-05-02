#!/usr/bin/env python3

from math import sqrt
from typing import Sequence


Vector = Sequence[float]


def dot(a: Vector, b: Vector) -> float:
    """Return the dot product of two vectors."""
    if len(a) != len(b):
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
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    return dot(a, b) / (norm_a * norm_b)


def _almost_equal(x: float, y: float, tolerance: float = 1e-9) -> bool:
    return abs(x - y) < tolerance


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
