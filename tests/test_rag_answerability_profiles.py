import pytest

from labs.rag.answerability_profiles import (
    ANSWERABILITY_PROFILES,
    answer_query_with_profile,
    format_profile_comparison,
    get_answerability_profile,
)


def test_profiles_are_available() -> None:
    assert "loose" in ANSWERABILITY_PROFILES
    assert "balanced" in ANSWERABILITY_PROFILES
    assert "conservative" in ANSWERABILITY_PROFILES


def test_get_answerability_profile_returns_profile() -> None:
    profile = get_answerability_profile("balanced")

    assert profile.name == "balanced"
    assert profile.min_score == 0.40
    assert profile.min_margin == 0.0


def test_get_answerability_profile_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError):
        get_answerability_profile("unknown")


def test_balanced_profile_answers_known_query() -> None:
    output = answer_query_with_profile(
        query="RAG sisteminde ilgili doküman parçaları nasıl bulunur?",
        profile_name="balanced",
    )

    assert output.decision.is_answerable is True
    assert output.answer.is_answered is True


def test_balanced_profile_rejects_near_miss_query() -> None:
    output = answer_query_with_profile(
        query="RAG sisteminde embedding fine-tuning nasıl yapılır?",
        profile_name="balanced",
    )

    assert output.decision.is_answerable is False
    assert output.answer.is_answered is False


def test_loose_profile_is_more_willing_than_balanced_on_near_miss_query() -> None:
    query = "RAG sisteminde embedding fine-tuning nasıl yapılır?"

    loose = answer_query_with_profile(query=query, profile_name="loose")
    balanced = answer_query_with_profile(query=query, profile_name="balanced")

    assert loose.decision.max_score == balanced.decision.max_score
    assert loose.decision.is_answerable is True
    assert balanced.decision.is_answerable is False


def test_format_profile_comparison_contains_all_profiles() -> None:
    text = format_profile_comparison(
        query="RAG sisteminde embedding fine-tuning nasıl yapılır?"
    )

    assert "Answerability Profile Comparison" in text
    assert "loose" in text
    assert "balanced" in text
    assert "conservative" in text
