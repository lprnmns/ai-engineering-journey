from __future__ import annotations

from dataclasses import dataclass

from labs.rag.answerer import RagAnswer, build_extractive_answer
from labs.rag.chunking import ChunkSearchResult
from labs.rag.vector_store import build_vector_store


@dataclass(frozen=True)
class RetrievalDecision:
    query: str
    is_answerable: bool
    reason: str
    max_score: float
    second_score: float
    score_margin: float
    retrieved_chunk_count: int


@dataclass(frozen=True)
class GuardedRagAnswer:
    decision: RetrievalDecision
    answer: RagAnswer


def get_score_at(results: list[ChunkSearchResult], index: int) -> float:
    if index >= len(results):
        return 0.0

    return results[index].score


def filter_results_by_score(
    results: list[ChunkSearchResult],
    min_score: float,
) -> list[ChunkSearchResult]:
    if min_score < 0.0:
        raise ValueError("min_score must be zero or greater")

    return [
        result
        for result in results
        if result.score >= min_score
    ]


def decide_answerability(
    query: str,
    results: list[ChunkSearchResult],
    min_score: float = 0.05,
    min_margin: float = 0.0,
) -> RetrievalDecision:
    if min_score < 0.0:
        raise ValueError("min_score must be zero or greater")

    if min_margin < 0.0:
        raise ValueError("min_margin must be zero or greater")

    if not results:
        return RetrievalDecision(
            query=query,
            is_answerable=False,
            reason="no_results",
            max_score=0.0,
            second_score=0.0,
            score_margin=0.0,
            retrieved_chunk_count=0,
        )

    sorted_results = sorted(results, key=lambda result: result.score, reverse=True)

    max_score = get_score_at(sorted_results, 0)
    second_score = get_score_at(sorted_results, 1)
    margin = max_score - second_score

    if max_score < min_score:
        return RetrievalDecision(
            query=query,
            is_answerable=False,
            reason="low_score",
            max_score=max_score,
            second_score=second_score,
            score_margin=margin,
            retrieved_chunk_count=len(results),
        )

    if min_margin > 0.0 and len(sorted_results) > 1 and margin < min_margin:
        return RetrievalDecision(
            query=query,
            is_answerable=False,
            reason="ambiguous_top_results",
            max_score=max_score,
            second_score=second_score,
            score_margin=margin,
            retrieved_chunk_count=len(results),
        )

    return RetrievalDecision(
        query=query,
        is_answerable=True,
        reason="answerable",
        max_score=max_score,
        second_score=second_score,
        score_margin=margin,
        retrieved_chunk_count=len(results),
    )


def answer_query_with_guard(
    query: str,
    top_k: int = 3,
    min_score: float = 0.05,
    min_margin: float = 0.0,
) -> GuardedRagAnswer:
    store = build_vector_store()
    results = store.search(query=query, top_k=top_k)

    decision = decide_answerability(
        query=query,
        results=results,
        min_score=min_score,
        min_margin=min_margin,
    )

    if not decision.is_answerable:
        return GuardedRagAnswer(
            decision=decision,
            answer=RagAnswer(
                query=query,
                answer="Bu soruyu cevaplamak için elimde yeterli context yok.",
                context="",
                sources=[],
                used_chunk_count=0,
                max_score=decision.max_score,
                is_answered=False,
            ),
        )

    filtered_results = filter_results_by_score(
        results=results,
        min_score=min_score,
    )

    return GuardedRagAnswer(
        decision=decision,
        answer=build_extractive_answer(
            query=query,
            results=filtered_results,
            min_score=min_score,
        ),
    )


def format_guarded_answer(output: GuardedRagAnswer) -> str:
    decision = output.decision
    answer = output.answer

    lines = [
        "=== Guarded RAG Answer ===",
        f"Query: {decision.query}",
        f"Answerable: {decision.is_answerable}",
        f"Reason: {decision.reason}",
        f"Max score: {decision.max_score:.4f}",
        f"Second score: {decision.second_score:.4f}",
        f"Score margin: {decision.score_margin:.4f}",
        "",
        "Answer:",
        answer.answer,
    ]

    if answer.sources:
        lines.append("")
        lines.append("Sources:")

        for index, source in enumerate(answer.sources, start=1):
            lines.append(
                f"- [{index}] {source.source} | {source.doc_id} | "
                f"{source.chunk_id} | score={source.score:.4f}"
            )

    return "\n".join(lines)


def main() -> None:
    known_query = "RAG sisteminde ilgili doküman parçaları nasıl bulunur?"
    unknown_query = "Fenerbahçe maç skoru nedir?"

    print(format_guarded_answer(answer_query_with_guard(known_query)))
    print()
    print(format_guarded_answer(answer_query_with_guard(unknown_query)))


if __name__ == "__main__":
    main()
