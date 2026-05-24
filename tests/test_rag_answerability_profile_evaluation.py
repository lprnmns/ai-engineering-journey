from labs.rag.answerability_profile_evaluation import (
    choose_profile_with_lowest_false_positive,
    evaluate_all_profiles,
    evaluate_profile,
    format_profile_evaluation,
)
from labs.rag.answerability_profiles import get_answerability_profile


def test_evaluate_profile_returns_metrics() -> None:
    result = evaluate_profile(get_answerability_profile("balanced"))

    assert result.profile.name == "balanced"
    assert result.report.metrics.total_cases > 0
    assert 0.0 <= result.report.metrics.accuracy <= 1.0


def test_evaluate_all_profiles_returns_all_profiles() -> None:
    results = evaluate_all_profiles()

    names = {result.profile.name for result in results}

    assert names == {"loose", "balanced", "conservative"}


def test_balanced_profile_has_no_more_false_positives_than_loose() -> None:
    results = {
        result.profile.name: result
        for result in evaluate_all_profiles()
    }

    assert (
        results["balanced"].report.metrics.false_positive
        <= results["loose"].report.metrics.false_positive
    )


def test_conservative_profile_has_no_more_false_positives_than_loose() -> None:
    results = {
        result.profile.name: result
        for result in evaluate_all_profiles()
    }

    assert (
        results["conservative"].report.metrics.false_positive
        <= results["loose"].report.metrics.false_positive
    )


def test_choose_profile_with_lowest_false_positive_handles_empty_input() -> None:
    assert choose_profile_with_lowest_false_positive([]) is None


def test_choose_profile_with_lowest_false_positive_returns_profile() -> None:
    results = evaluate_all_profiles()

    selected = choose_profile_with_lowest_false_positive(results)

    assert selected is not None
    assert selected.profile.name in {"balanced", "conservative"}


def test_format_profile_evaluation_contains_table_and_descriptions() -> None:
    text = format_profile_evaluation(evaluate_all_profiles())

    assert "Answerability Profile Evaluation" in text
    assert "loose" in text
    assert "balanced" in text
    assert "conservative" in text
    assert "Lowest false-positive profile" in text
    assert "Per-profile descriptions" in text
