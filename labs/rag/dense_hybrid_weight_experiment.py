from __future__ import annotations

from dataclasses import dataclass

from labs.rag.dense_hybrid_retrieval import DenseHybridRetriever, build_dense_hybrid_retriever
from labs.rag.dense_retrieval_benchmark import PARAPHRASE_GOLDEN_QUERIES
from labs.rag.evaluation import GoldenQuery, RetrievalMetrics
from labs.rag.hard_retrieval_benchmark import HARD_DOCUMENTS


@dataclass(frozen=True)
class DenseHybridQueryEvaluation:
    query: str
    expected_doc_id: str
    retrieved_doc_ids: list[str]
    rank: int | None
    hit_at_1: bool
    hit_at_k: bool
    reciprocal_rank: float


@dataclass(frozen=True)
class DenseHybridWeightResult:
    dense_weight: float
    metrics: RetrievalMetrics
    query_evaluations: list[DenseHybridQueryEvaluation]


def find_expected_rank(retrieved_doc_ids: list[str], expected_doc_id: str) -> int | None:
    for index, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == expected_doc_id:
            return index

    return None


def evaluate_dense_hybrid_query(
    retriever: DenseHybridRetriever,
    golden_query: GoldenQuery,
    top_k: int = 3,
) -> DenseHybridQueryEvaluation:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    results = retriever.search(query=golden_query.query, top_k=top_k)
    retrieved_doc_ids = [result.doc_id for result in results]
    rank = find_expected_rank(retrieved_doc_ids, golden_query.expected_doc_id)

    return DenseHybridQueryEvaluation(
        query=golden_query.query,
        expected_doc_id=golden_query.expected_doc_id,
        retrieved_doc_ids=retrieved_doc_ids,
        rank=rank,
        hit_at_1=rank == 1,
        hit_at_k=rank is not None,
        reciprocal_rank=0.0 if rank is None else 1.0 / rank,
    )


def calculate_dense_hybrid_metrics(
    evaluations: list[DenseHybridQueryEvaluation],
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


def evaluate_dense_hybrid_weight(
    dense_weight: float,
    golden_queries: list[GoldenQuery] | None = None,
    top_k: int = 3,
) -> DenseHybridWeightResult:
    if golden_queries is None:
        golden_queries = PARAPHRASE_GOLDEN_QUERIES

    retriever = build_dense_hybrid_retriever(
        documents=HARD_DOCUMENTS,
        dense_weight=dense_weight,
    )
    evaluations = [
        evaluate_dense_hybrid_query(
            retriever=retriever,
            golden_query=golden_query,
            top_k=top_k,
        )
        for golden_query in golden_queries
    ]

    return DenseHybridWeightResult(
        dense_weight=dense_weight,
        metrics=calculate_dense_hybrid_metrics(evaluations),
        query_evaluations=evaluations,
    )


def run_dense_hybrid_weight_experiment(
    weights: list[float] | None = None,
    top_k: int = 3,
) -> list[DenseHybridWeightResult]:
    if weights is None:
        weights = [0.0, 0.3, 0.5, 0.7, 1.0]

    return [
        evaluate_dense_hybrid_weight(
            dense_weight=weight,
            top_k=top_k,
        )
        for weight in weights
    ]


def format_dense_hybrid_weight_experiment(results: list[DenseHybridWeightResult]) -> str:
    lines = [
        "=== Dense Hybrid Weight Experiment ===",
        "",
        "| Dense weight | Keyword weight | Total queries | Hit@1 | Hit@K | MRR |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        metrics = result.metrics
        keyword_weight = 1.0 - result.dense_weight
        lines.append(
            f"| {result.dense_weight:.1f} | "
            f"{keyword_weight:.1f} | "
            f"{metrics.total_queries} | "
            f"{metrics.hit_at_1:.3f} | "
            f"{metrics.hit_at_k:.3f} | "
            f"{metrics.mean_reciprocal_rank:.3f} |"
        )

    return "\n".join(lines)


def main() -> None:
    results = run_dense_hybrid_weight_experiment()
    print(format_dense_hybrid_weight_experiment(results))


if __name__ == "__main__":
    main()
