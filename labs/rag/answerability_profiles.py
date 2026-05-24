from __future__ import annotations

from dataclasses import dataclass

from labs.rag.no_answer_detection import GuardedRagAnswer, answer_query_with_guard


@dataclass(frozen=True)
class AnswerabilityProfile:
    name: str
    min_score: float
    min_margin: float
    description: str


LOOSE_PROFILE = AnswerabilityProfile(
    name="loose",
    min_score=0.05,
    min_margin=0.0,
    description="More willing to answer. Higher false-positive risk on near-miss queries.",
)

BALANCED_PROFILE = AnswerabilityProfile(
    name="balanced",
    min_score=0.40,
    min_margin=0.0,
    description="Calibrated on the current hard benchmark. Reduces near-miss false positives.",
)

CONSERVATIVE_PROFILE = AnswerabilityProfile(
    name="conservative",
    min_score=0.50,
    min_margin=0.05,
    description="More cautious. Lower false-positive risk, but may reject some answerable queries.",
)


ANSWERABILITY_PROFILES: dict[str, AnswerabilityProfile] = {
    profile.name: profile
    for profile in [
        LOOSE_PROFILE,
        BALANCED_PROFILE,
        CONSERVATIVE_PROFILE,
    ]
}


def get_answerability_profile(name: str) -> AnswerabilityProfile:
    try:
        return ANSWERABILITY_PROFILES[name]
    except KeyError as error:
        available = ", ".join(sorted(ANSWERABILITY_PROFILES))
        raise ValueError(f"Unknown profile: {name}. Available profiles: {available}") from error


def answer_query_with_profile(
    query: str,
    profile_name: str = "balanced",
    top_k: int = 3,
) -> GuardedRagAnswer:
    profile = get_answerability_profile(profile_name)

    return answer_query_with_guard(
        query=query,
        top_k=top_k,
        min_score=profile.min_score,
        min_margin=profile.min_margin,
    )


def format_profile_comparison(query: str, top_k: int = 3) -> str:
    lines = [
        "=== Answerability Profile Comparison ===",
        f"Query: {query}",
        "",
        "| Profile | min_score | min_margin | Answerable | Reason | Max score |",
        "|---|---:|---:|---:|---|---:|",
    ]

    for profile in ANSWERABILITY_PROFILES.values():
        output = answer_query_with_profile(
            query=query,
            profile_name=profile.name,
            top_k=top_k,
        )

        lines.append(
            f"| {profile.name} | "
            f"{profile.min_score:.2f} | "
            f"{profile.min_margin:.2f} | "
            f"{output.decision.is_answerable} | "
            f"{output.decision.reason} | "
            f"{output.decision.max_score:.4f} |"
        )

    return "\n".join(lines)


def main() -> None:
    known_query = "RAG sisteminde ilgili doküman parçaları nasıl bulunur?"
    near_miss_query = "RAG sisteminde embedding fine-tuning nasıl yapılır?"

    print(format_profile_comparison(known_query))
    print()
    print(format_profile_comparison(near_miss_query))


if __name__ == "__main__":
    main()
