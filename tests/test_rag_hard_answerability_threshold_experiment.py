from labs.rag.hard_answerability_threshold_experiment import (
    format_hard_threshold_experiment,
    run_hard_threshold_experiment,
)


def test_run_hard_threshold_experiment_returns_grid_results() -> None:
    results = run_hard_threshold_experiment(
        min_scores=[0.05, 0.40],
        min_margins=[0.0, 0.1],
    )

    assert len(results) == 4


def test_hard_threshold_experiment_metrics_are_valid() -> None:
    results = run_hard_threshold_experiment(
        min_scores=[0.40],
        min_margins=[0.0],
    )

    metrics = results[0].metrics

    assert metrics.total_cases > 0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.true_positive >= 0
    assert metrics.true_negative >= 0
    assert metrics.false_positive >= 0
    assert metrics.false_negative >= 0


def test_hard_threshold_experiment_can_reduce_false_positives_with_higher_min_score() -> None:
    loose = run_hard_threshold_experiment(
        min_scores=[0.05],
        min_margins=[0.0],
    )[0]

    stricter = run_hard_threshold_experiment(
        min_scores=[0.40],
        min_margins=[0.0],
    )[0]

    assert stricter.metrics.false_positive <= loose.metrics.false_positive


def test_format_hard_threshold_experiment_contains_table_and_best_candidate() -> None:
    results = run_hard_threshold_experiment(
        min_scores=[0.05, 0.40],
        min_margins=[0.0],
    )

    text = format_hard_threshold_experiment(results)

    assert "Hard RAG Answerability Threshold Experiment" in text
    assert "min_score" in text
    assert "min_margin" in text
    assert "Best hard-threshold candidate" in text
