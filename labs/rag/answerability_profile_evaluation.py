from __future__ import annotations

from dataclasses import dataclass

from labs.rag.answerability_evaluation import (
    AnswerabilityMetrics,
    AnswerabilityReport,
    calculate_answerability_metrics,
    evaluate_answerability_case,
)
from labs.rag.answerability_profiles import ANSWERABILITY_PROFILES, AnswerabilityProfile
from labs.rag.hard_answerability_benchmark import HARD_ANSWERABILITY_CASES


@dataclass(frozen=True)
class ProfileEvaluationResult:
    profile: AnswerabilityProfile
    report: AnswerabilityReport


def evaluate_profile(
    profile: AnswerabilityProfile,
    top_k: int = 3,
) -> ProfileEvaluationResult:
    evaluations = [
        evaluate_answerability_case(
            case=case,
            top_k=top_k,
            min_score=profile.min_score,
            min_margin=profile.min_margin,
        )
        for case in HARD_ANSWERABILITY_CASES
    ]

    return ProfileEvaluationResult(
        profile=profile,
        report=AnswerabilityReport(
            metrics=calculate_answerability_metrics(evaluations),
            evaluations=evaluations,
        ),
    )


def evaluate_all_profiles(top_k: int = 3) -> list[ProfileEvaluationResult]:
    return [
        evaluate_profile(profile=profile, top_k=top_k)
        for profile in ANSWERABILITY_PROFILES.values()
    ]


def choose_profile_with_lowest_false_positive(
    results: list[ProfileEvaluationResult],
) -> ProfileEvaluationResult | None:
    if not results:
        return None

    return sorted(
        results,
        key=lambda result: (
            result.report.metrics.false_positive,
            result.report.metrics.false_negative,
            -result.report.metrics.accuracy,
        ),
    )[0]


def format_profile_evaluation(results: list[ProfileEvaluationResult]) -> str:
    lines = [
        "=== Answerability Profile Evaluation on Hard Benchmark ===",
        "",
        "| Profile | min_score | min_margin | accuracy | TP | TN | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        profile = result.profile
        metrics: AnswerabilityMetrics = result.report.metrics

        lines.append(
            f"| {profile.name} | "
            f"{profile.min_score:.2f} | "
            f"{profile.min_margin:.2f} | "
            f"{metrics.accuracy:.3f} | "
            f"{metrics.true_positive} | "
            f"{metrics.true_negative} | "
            f"{metrics.false_positive} | "
            f"{metrics.false_negative} |"
        )

    safest = choose_profile_with_lowest_false_positive(results)

    if safest is not None:
        metrics = safest.report.metrics
        lines.extend(
            [
                "",
                "Lowest false-positive profile:",
                (
                    f"- {safest.profile.name}: "
                    f"accuracy={metrics.accuracy:.3f}, "
                    f"FP={metrics.false_positive}, "
                    f"FN={metrics.false_negative}"
                ),
            ]
        )

    lines.append("")
    lines.append("Per-profile descriptions:")

    for result in results:
        lines.append(f"- {result.profile.name}: {result.profile.description}")

    return "\n".join(lines)


def main() -> None:
    results = evaluate_all_profiles()

    print(format_profile_evaluation(results))


if __name__ == "__main__":
    main()
