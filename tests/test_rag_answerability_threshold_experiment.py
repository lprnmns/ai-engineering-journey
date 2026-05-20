from labs.rag.answerability_evaluation import AnswerabilityCase
from labs.rag.answerability_threshold_experiment import (
    find_best_threshold,
    format_threshold_experiment,
    run_threshold_experiment,
)


def test_run_threshold_experiment_returns_grid_results() -> None:
    results = run_threshold_experiment(
        min_scores=[0.0, 0.05],
        min_margins=[0.0, 0.10],
    )

    assert len(results) == 4


def test_threshold_experiment_metrics_are_valid() -> None:
    results = run_threshold_experiment(
        min_scores=[0.05],
        min_margins=[0.0],
    )

    metrics = results[0].metrics

    assert metrics.total_cases > 0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.true_positive >= 0
    assert metrics.true_negative >= 0
    assert metrics.false_positive >= 0
    assert metrics.false_negative >= 0


def test_find_best_threshold_returns_none_for_empty_results() -> None:
    assert find_best_threshold([]) is None


def test_find_best_threshold_prefers_high_accuracy() -> None:
    results = run_threshold_experiment(
        min_scores=[0.0, 0.05],
        min_margins=[0.0],
        cases=[
            AnswerabilityCase(
                query="Python sanal ortam nasıl kurulur?",
                should_be_answerable=True,
            ),
            AnswerabilityCase(
                query="Bitcoin fiyatı şu anda kaç dolar?",
                should_be_answerable=False,
            ),
        ],
    )

    best = find_best_threshold(results)

    assert best is not None
    assert 0.0 <= best.metrics.accuracy <= 1.0


def test_format_threshold_experiment_contains_table_and_best_candidate() -> None:
    results = run_threshold_experiment(
        min_scores=[0.05],
        min_margins=[0.0],
    )

    text = format_threshold_experiment(results)

    assert "RAG Answerability Threshold Experiment" in text
    assert "min_score" in text
    assert "min_margin" in text
    assert "Best threshold candidate" in text
