from competitions.titanic.src.gap_analysis import (
    build_priority_experiments,
    calculate_score_gap,
    find_best_public_experiment,
    parse_public_score,
    ExperimentSummary,
)


def test_parse_public_score_handles_empty_value() -> None:
    assert parse_public_score("") == 0.0


def test_parse_public_score_parses_float_text() -> None:
    assert parse_public_score("0.77272") == 0.77272


def test_calculate_score_gap_estimates_extra_correct_predictions() -> None:
    gap = calculate_score_gap(
        current_score=0.77272,
        target_score=0.80,
        test_rows=418,
    )

    assert gap.estimated_current_correct == 323
    assert gap.estimated_target_correct == 335
    assert gap.estimated_extra_correct_needed == 12


def test_find_best_public_experiment_returns_highest_score() -> None:
    experiments = [
        ExperimentSummary(
            experiment_id="exp_a",
            features="a",
            model="model_a",
            local_score="0.80",
            public_lb_score=0.76,
            notes="a",
        ),
        ExperimentSummary(
            experiment_id="exp_b",
            features="b",
            model="model_b",
            local_score="0.82",
            public_lb_score=0.77,
            notes="b",
        ),
    ]

    best = find_best_public_experiment(experiments)

    assert best.experiment_id == "exp_b"


def test_build_priority_experiments_starts_with_age_imputation() -> None:
    experiments = build_priority_experiments()

    assert experiments[0]["experiment_id"] == "titanic_exp_004_age_imputation_v2"
