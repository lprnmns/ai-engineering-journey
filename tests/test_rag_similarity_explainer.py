import pytest

from labs.rag.similarity_explainer import (
    build_term_contributions,
    explain_similarity,
    format_similarity_explanation,
    format_vector,
    safe_cosine_similarity,
)


def test_build_term_contributions_uses_only_common_terms() -> None:
    query_vector = {"python": 1.0, "git": 1.0}
    chunk_vector = {"python": 2.0, "rag": 1.0}

    contributions = build_term_contributions(query_vector, chunk_vector)

    assert len(contributions) == 1
    assert contributions[0].term == "python"
    assert contributions[0].contribution == 2.0


def test_safe_cosine_similarity_returns_zero_for_empty_vectors() -> None:
    assert safe_cosine_similarity({}, {"python": 1.0}) == 0.0
    assert safe_cosine_similarity({"python": 1.0}, {}) == 0.0


def test_safe_cosine_similarity_returns_one_for_same_vector() -> None:
    vector = {"python": 1.0, "venv": 1.0}

    assert safe_cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_explain_similarity_returns_contributions_and_score() -> None:
    explanation = explain_similarity(
        query="python sanal ortam",
        chunk_text="python sanal ortam venv",
    )

    terms = {item.term for item in explanation.contributions}

    assert terms == {"python", "sanal", "ortam"}
    assert explanation.dot_product == 3.0
    assert explanation.cosine_similarity > 0.0


def test_format_vector_handles_empty_vector() -> None:
    assert format_vector({}) == "{}"


def test_format_similarity_explanation_contains_core_sections() -> None:
    explanation = explain_similarity(
        query="python sanal ortam",
        chunk_text="python sanal ortam venv",
    )

    text = format_similarity_explanation(explanation)

    assert "Query vector" in text
    assert "Chunk vector" in text
    assert "Term contributions" in text
    assert "Cosine similarity" in text
