from __future__ import annotations

import math

SparseVector = dict[str, float]
DenseVector = list[float]


def cosine_similarity(
    left: SparseVector | DenseVector,
    right: SparseVector | DenseVector,
) -> float:
    """Calculate cosine similarity for sparse or dense vectors.

    Sparse vectors use token keys. Dense vectors use the same numeric position
    in each list as a dimension from an embedding model.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        return _sparse_cosine_similarity(left, right)

    if isinstance(left, dict) or isinstance(right, dict):
        raise TypeError("cannot compare a sparse vector with a dense vector")

    return _dense_cosine_similarity(left, right)


def _sparse_cosine_similarity(left: SparseVector, right: SparseVector) -> float:
    dot_product = sum(left[key] * right[key] for key in set(left).intersection(right))
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def _dense_cosine_similarity(left: DenseVector, right: DenseVector) -> float:
    if len(left) != len(right):
        raise ValueError("dense vectors must have the same dimension")

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (left_norm * right_norm)
