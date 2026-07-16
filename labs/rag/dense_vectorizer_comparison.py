from __future__ import annotations

from dataclasses import dataclass

from labs.rag.dense_retrieval_benchmark import PARAPHRASE_GOLDEN_QUERIES
from labs.rag.dense_vector_store import DenseVectorStore, build_dense_vector_store
from labs.rag.evaluation import ChunkRetriever, EvaluationReport, QueryEvaluation, evaluate_retrieval
from labs.rag.hard_retrieval_benchmark import HARD_DOCUMENTS
from labs.rag.vectorizer_comparison import build_tfidf_vector_store


@dataclass(frozen=True)
class QueryComparison:
    query: str
    expected_doc_id: str
    tfidf_rank: int | None
    dense_rank: int | None
    result: str


@dataclass(frozen=True)
class DenseComparisonReport:
    tfidf_report: EvaluationReport
    dense_report: EvaluationReport
    query_comparisons: list[QueryComparison]


def rank_value(rank: int | None) -> float:
    return float("inf") if rank is None else float(rank)


def compare_query_evaluations(
    tfidf_evaluations: list[QueryEvaluation],
    dense_evaluations: list[QueryEvaluation],
) -> list[QueryComparison]:
    if len(tfidf_evaluations) != len(dense_evaluations):
        raise ValueError("both retrievers must evaluate the same number of queries")

    comparisons: list[QueryComparison] = []

    for tfidf, dense in zip(tfidf_evaluations, dense_evaluations):
        if tfidf.query != dense.query or tfidf.expected_doc_id != dense.expected_doc_id:
            raise ValueError("retriever evaluations must use the same golden queries")

        if rank_value(dense.rank) < rank_value(tfidf.rank):
            result = "dense_better"
        elif rank_value(tfidf.rank) < rank_value(dense.rank):
            result = "tfidf_better"
        else:
            result = "same_rank"

        comparisons.append(
            QueryComparison(
                query=tfidf.query,
                expected_doc_id=tfidf.expected_doc_id,
                tfidf_rank=tfidf.rank,
                dense_rank=dense.rank,
                result=result,
            )
        )

    return comparisons


def run_paraphrase_comparison(
    tfidf_store: ChunkRetriever | None = None,
    dense_store: DenseVectorStore | None = None,
    top_k: int = 3,
) -> DenseComparisonReport:
    if tfidf_store is None:
        tfidf_store = build_tfidf_vector_store(documents=HARD_DOCUMENTS)

    if dense_store is None:
        dense_store = build_dense_vector_store(documents=HARD_DOCUMENTS)

    tfidf_report = evaluate_retrieval(
        store=tfidf_store,
        golden_queries=PARAPHRASE_GOLDEN_QUERIES,
        top_k=top_k,
    )
    dense_report = evaluate_retrieval(
        store=dense_store,
        golden_queries=PARAPHRASE_GOLDEN_QUERIES,
        top_k=top_k,
    )

    return DenseComparisonReport(
        tfidf_report=tfidf_report,
        dense_report=dense_report,
        query_comparisons=compare_query_evaluations(
            tfidf_report.query_evaluations,
            dense_report.query_evaluations,
        ),
    )


def format_paraphrase_comparison(report: DenseComparisonReport) -> str:
    tfidf_metrics = report.tfidf_report.metrics
    dense_metrics = report.dense_report.metrics
    lines = [
        "=== TF-IDF vs Dense Paraphrase Benchmark ===",
        "",
        "| Retriever | Hit@1 | Hit@K | MRR |",
        "|---|---:|---:|---:|",
        f"| tfidf | {tfidf_metrics.hit_at_1:.3f} | {tfidf_metrics.hit_at_k:.3f} | {tfidf_metrics.mean_reciprocal_rank:.3f} |",
        f"| dense | {dense_metrics.hit_at_1:.3f} | {dense_metrics.hit_at_k:.3f} | {dense_metrics.mean_reciprocal_rank:.3f} |",
        "",
        "| Query | Expected document | TF-IDF rank | Dense rank | Result |",
        "|---|---|---:|---:|---|",
    ]

    for comparison in report.query_comparisons:
        tfidf_rank = "missing" if comparison.tfidf_rank is None else str(comparison.tfidf_rank)
        dense_rank = "missing" if comparison.dense_rank is None else str(comparison.dense_rank)
        lines.append(
            f"| {comparison.query} | {comparison.expected_doc_id} | "
            f"{tfidf_rank} | {dense_rank} | {comparison.result} |"
        )

    return "\n".join(lines)


def main() -> None:
    report = run_paraphrase_comparison(top_k=3)
    print(format_paraphrase_comparison(report))


if __name__ == "__main__":
    main()
