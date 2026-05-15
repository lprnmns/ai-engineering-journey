from __future__ import annotations

from dataclasses import dataclass

from labs.rag.evaluation import DEFAULT_GOLDEN_QUERIES, GoldenQuery, RetrievalMetrics
from labs.rag.hybrid_retrieval import HybridRetriever, build_hybrid_retriever


@dataclass(frozen=True)
class HybridQueryEvaluation:
    query: str
    expected_doc_id: str
    retrieved_doc_ids: list[str]
    retrieved_chunk_ids: list[str]
    rank: int | None
    hit_at_1: bool
    hit_at_k: bool
    reciprocal_rank: float


@dataclass(frozen=True)
class HybridWeightResult:
    vector_weight: float
    metrics: RetrievalMetrics
    query_evaluations: list[HybridQueryEvaluation]


def find_expected_rank(
    retrieved_doc_ids: list[str],
    expected_doc_id: str,
) -> int | None:
    for index, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == expected_doc_id:
            return index

    return None


def evaluate_hybrid_query(
    retriever: HybridRetriever,
    golden_query: GoldenQuery,
    top_k: int = 3,
) -> HybridQueryEvaluation:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    results = retriever.search(query=golden_query.query, top_k=top_k)
    retrieved_doc_ids = [result.doc_id for result in results]
    retrieved_chunk_ids = [result.chunk_id for result in results]
    rank = find_expected_rank(
        retrieved_doc_ids=retrieved_doc_ids,
        expected_doc_id=golden_query.expected_doc_id,
    )

    reciprocal_rank = 0.0 if rank is None else 1.0 / rank

    return HybridQueryEvaluation(
        query=golden_query.query,
        expected_doc_id=golden_query.expected_doc_id,
        retrieved_doc_ids=retrieved_doc_ids,
        retrieved_chunk_ids=retrieved_chunk_ids,
        rank=rank,
        hit_at_1=rank == 1,
        hit_at_k=rank is not None,
        reciprocal_rank=reciprocal_rank,
    )


def calculate_hybrid_metrics(
    evaluations: list[HybridQueryEvaluation],
) -> RetrievalMetrics:
    if not evaluations:
        return RetrievalMetrics(
            total_queries=0,
            hit_at_1=0.0,
            hit_at_k=0.0,
            mean_reciprocal_rank=0.0,
        )

    total = len(evaluations)

    return RetrievalMetrics(
        total_queries=total,
        hit_at_1=sum(item.hit_at_1 for item in evaluations) / total,
        hit_at_k=sum(item.hit_at_k for item in evaluations) / total,
        mean_reciprocal_rank=sum(item.reciprocal_rank for item in evaluations) / total,
    )


def evaluate_hybrid_weight(
    vector_weight: float,
    golden_queries: list[GoldenQuery] | None = None,
    top_k: int = 3,
) -> HybridWeightResult:
    if golden_queries is None:
        golden_queries = DEFAULT_GOLDEN_QUERIES

    retriever = build_hybrid_retriever(vector_weight=vector_weight)

    evaluations = [
        evaluate_hybrid_query(
            retriever=retriever,
            golden_query=golden_query,
            top_k=top_k,
        )
        for golden_query in golden_queries
    ]

    return HybridWeightResult(
        vector_weight=vector_weight,
        metrics=calculate_hybrid_metrics(evaluations),
        query_evaluations=evaluations,
    )


def run_hybrid_weight_experiment(
    weights: list[float] | None = None,
    top_k: int = 3,
) -> list[HybridWeightResult]:
    if weights is None:
        weights = [0.0, 0.3, 0.5, 0.7, 1.0]

    return [
        evaluate_hybrid_weight(
            vector_weight=weight,
            top_k=top_k,
        )
        for weight in weights
    ]


def format_hybrid_weight_experiment(results: list[HybridWeightResult]) -> str:
    lines = [
        "=== Hybrid Retrieval Weight Experiment ===",
        "",
        "| Vector weight | Keyword weight | Total queries | Hit@1 | Hit@K | MRR |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        keyword_weight = 1.0 - result.vector_weight
        metrics = result.metrics

        lines.append(
            f"| {result.vector_weight:.1f} | "
            f"{keyword_weight:.1f} | "
            f"{metrics.total_queries} | "
            f"{metrics.hit_at_1:.3f} | "
            f"{metrics.hit_at_k:.3f} | "
            f"{metrics.mean_reciprocal_rank:.3f} |"
        )

    return "\n".join(lines)


def main() -> None:
    results = run_hybrid_weight_experiment()

    print(format_hybrid_weight_experiment(results))


if __name__ == "__main__":
    main()
