from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from labs.rag.chunking import ChunkSearchResult
from labs.rag.vector_store import build_vector_store


class ChunkRetriever(Protocol):
    def search(self, query: str, top_k: int = 3) -> list[ChunkSearchResult]: ...


@dataclass(frozen=True)
class GoldenQuery:
    query: str
    expected_doc_id: str
    expected_chunk_id: str | None = None


@dataclass(frozen=True)
class QueryEvaluation:
    query: str
    expected_doc_id: str
    expected_chunk_id: str | None
    retrieved_doc_ids: list[str]
    retrieved_chunk_ids: list[str]
    rank: int | None
    hit_at_1: bool
    hit_at_k: bool
    reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalMetrics:
    total_queries: int
    hit_at_1: float
    hit_at_k: float
    mean_reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationReport:
    metrics: RetrievalMetrics
    query_evaluations: list[QueryEvaluation]


DEFAULT_GOLDEN_QUERIES: list[GoldenQuery] = [
    GoldenQuery(
        query="Python sanal ortam nasıl kurulur?",
        expected_doc_id="doc_001",
    ),
    GoldenQuery(
        query="Pull request ile main branch'e nasıl değişiklik alınır?",
        expected_doc_id="doc_002",
    ),
    GoldenQuery(
        query="Cosine similarity iki vektör arasında neyi ölçer?",
        expected_doc_id="doc_003",
    ),
    GoldenQuery(
        query="Pandas ile CSV dosyası nasıl okunur?",
        expected_doc_id="doc_004",
    ),
    GoldenQuery(
        query="RAG sisteminde ilgili doküman parçaları nasıl bulunur?",
        expected_doc_id="doc_005",
    ),
]


def result_matches_expected(
    result: ChunkSearchResult,
    golden_query: GoldenQuery,
) -> bool:
    if golden_query.expected_chunk_id is not None:
        return result.chunk_id == golden_query.expected_chunk_id

    return result.doc_id == golden_query.expected_doc_id


def find_expected_rank(
    results: list[ChunkSearchResult],
    golden_query: GoldenQuery,
) -> int | None:
    for index, result in enumerate(results, start=1):
        if result_matches_expected(result, golden_query):
            return index

    return None


def evaluate_query(
    store: ChunkRetriever,
    golden_query: GoldenQuery,
    top_k: int = 3,
) -> QueryEvaluation:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    results = store.search(query=golden_query.query, top_k=top_k)
    rank = find_expected_rank(results, golden_query)

    reciprocal_rank = 0.0 if rank is None else 1.0 / rank

    return QueryEvaluation(
        query=golden_query.query,
        expected_doc_id=golden_query.expected_doc_id,
        expected_chunk_id=golden_query.expected_chunk_id,
        retrieved_doc_ids=[result.doc_id for result in results],
        retrieved_chunk_ids=[result.chunk_id for result in results],
        rank=rank,
        hit_at_1=rank == 1,
        hit_at_k=rank is not None,
        reciprocal_rank=reciprocal_rank,
    )


def calculate_metrics(evaluations: list[QueryEvaluation]) -> RetrievalMetrics:
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


def evaluate_retrieval(
    store: ChunkRetriever | None = None,
    golden_queries: list[GoldenQuery] | None = None,
    top_k: int = 3,
) -> EvaluationReport:
    if store is None:
        store = build_vector_store()

    if golden_queries is None:
        golden_queries = DEFAULT_GOLDEN_QUERIES

    evaluations = [
        evaluate_query(
            store=store,
            golden_query=golden_query,
            top_k=top_k,
        )
        for golden_query in golden_queries
    ]

    return EvaluationReport(
        metrics=calculate_metrics(evaluations),
        query_evaluations=evaluations,
    )


def format_evaluation_report(report: EvaluationReport) -> str:
    lines = [
        "=== RAG Retrieval Evaluation ===",
        f"Total queries: {report.metrics.total_queries}",
        f"Hit@1: {report.metrics.hit_at_1:.3f}",
        f"Hit@K: {report.metrics.hit_at_k:.3f}",
        f"MRR: {report.metrics.mean_reciprocal_rank:.3f}",
        "",
        "Per-query results:",
    ]

    for index, evaluation in enumerate(report.query_evaluations, start=1):
        lines.extend(
            [
                f"{index}. Query: {evaluation.query}",
                f"   Expected doc: {evaluation.expected_doc_id}",
                f"   Retrieved docs: {', '.join(evaluation.retrieved_doc_ids)}",
                f"   Retrieved chunks: {', '.join(evaluation.retrieved_chunk_ids)}",
                f"   Rank: {evaluation.rank}",
                f"   Hit@1: {evaluation.hit_at_1}",
                f"   Hit@K: {evaluation.hit_at_k}",
                f"   Reciprocal rank: {evaluation.reciprocal_rank:.3f}",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    report = evaluate_retrieval(top_k=3)

    print(format_evaluation_report(report))


if __name__ == "__main__":
    main()
