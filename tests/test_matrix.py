import pytest

from labs.lin_alg.matrix import matmul, shape, transpose


def test_shape_returns_rows_and_columns() -> None:
    assert shape([[1.0, 2.0], [3.0, 4.0]]) == (2, 2)


def test_shape_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError, match="Matrix cannot be empty"):
        shape([])


def test_shape_rejects_jagged_matrix() -> None:
    with pytest.raises(ValueError, match="All matrix rows must have the same length"):
        shape([[1.0, 2.0], [3.0]])


def test_transpose_flips_rows_and_columns() -> None:
    assert transpose([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) == [
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0],
    ]


def test_matmul_multiplies_compatible_matrices() -> None:
    left = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]

    right = [
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ]

    assert matmul(left, right) == [
        [58.0, 64.0],
        [139.0, 154.0],
    ]


def test_matmul_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="Matrix dimensions are incompatible"):
        matmul([[1.0, 2.0]], [[1.0, 2.0]])
