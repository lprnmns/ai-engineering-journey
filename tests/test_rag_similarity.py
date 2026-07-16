import pytest

from labs.rag.similarity import cosine_similarity


def test_sparse_vectors_with_shared_tokens_are_similar() -> None:
    score = cosine_similarity(
        {"python": 1.0, "venv": 1.0},
        {"python": 1.0, "git": 1.0},
    )

    assert score == pytest.approx(0.5)


def test_dense_vectors_with_same_direction_are_identical() -> None:
    score = cosine_similarity([2.0, 2.0], [1.0, 1.0])

    assert score == pytest.approx(1.0)


def test_dense_vectors_with_different_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_sparse_and_dense_vectors_cannot_be_compared() -> None:
    with pytest.raises(TypeError):
        cosine_similarity({"python": 1.0}, [1.0])
