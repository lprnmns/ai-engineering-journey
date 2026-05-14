import pytest

from labs.rag.mini_semantic_search import (
    cosine_similarity,
    dot_product,
    search,
    text_to_vector,
    tokenize,
    vector_norm,
)


def test_tokenize_lowercases_and_extracts_words() -> None:
    tokens = tokenize("Python'da Sanal Ortam!")

    assert "python" in tokens
    assert "sanal" in tokens
    assert "ortam" in tokens


def test_text_to_vector_counts_terms() -> None:
    vector = text_to_vector("python python git")

    assert vector["python"] == 2.0
    assert vector["git"] == 1.0


def test_dot_product_uses_common_terms() -> None:
    left = {"python": 2.0, "git": 1.0}
    right = {"python": 3.0, "rag": 5.0}

    assert dot_product(left, right) == 6.0


def test_vector_norm_returns_length() -> None:
    vector = {"x": 3.0, "y": 4.0}

    assert vector_norm(vector) == 5.0


def test_cosine_similarity_is_one_for_same_vector() -> None:
    vector = {"python": 1.0, "venv": 1.0}

    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_search_returns_python_doc_for_venv_query() -> None:
    results = search("Python sanal ortam nasıl kurulur?", top_k=1)

    assert results[0].doc_id == "doc_001"
    assert results[0].score > 0.0


def test_search_returns_rag_doc_for_retrieval_query() -> None:
    results = search("RAG sisteminde ilgili doküman nasıl bulunur?", top_k=1)

    assert results[0].doc_id == "doc_005"


def test_search_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError):
        search("test query", top_k=0)
