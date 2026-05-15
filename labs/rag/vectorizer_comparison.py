from __future__ import annotations

from dataclasses import dataclass

from labs.rag.chunking import Chunk, chunk_documents
from labs.rag.evaluation import (
    EvaluationReport,
    evaluate_retrieval,
    format_evaluation_report,
)
from labs.rag.sample_docs import Document, SAMPLE_DOCUMENTS
from labs.rag.tfidf_vectorizer import TfidfVectorizer
from labs.rag.vector_store import InMemoryVectorStore, build_vector_store
from labs.rag.vectorizer import TermFrequencyVectorizer


@dataclass(frozen=True)
class VectorizerExperimentResult:
    name: str
    report: EvaluationReport


def chunk_text_for_vectorizer(chunk: Chunk) -> str:
    return f"{chunk.title} {chunk.text}"


def build_tfidf_vector_store(
    documents: list[Document] | None = None,
    sentences_per_chunk: int = 1,
    overlap: int = 0,
) -> InMemoryVectorStore:
    if documents is None:
        documents = SAMPLE_DOCUMENTS

    chunks = chunk_documents(
        documents=documents,
        sentences_per_chunk=sentences_per_chunk,
        overlap=overlap,
    )

    vectorizer = TfidfVectorizer()
    vectorizer.fit([chunk_text_for_vectorizer(chunk) for chunk in chunks])

    store = InMemoryVectorStore(vectorizer=vectorizer)
    store.add_chunks(chunks)

    return store


def run_vectorizer_comparison(top_k: int = 3) -> list[VectorizerExperimentResult]:
    term_frequency_store = build_vector_store(
        vectorizer=TermFrequencyVectorizer(),
    )
    tfidf_store = build_tfidf_vector_store()

    return [
        VectorizerExperimentResult(
            name="term_frequency",
            report=evaluate_retrieval(store=term_frequency_store, top_k=top_k),
        ),
        VectorizerExperimentResult(
            name="tfidf",
            report=evaluate_retrieval(store=tfidf_store, top_k=top_k),
        ),
    ]


def format_comparison(results: list[VectorizerExperimentResult]) -> str:
    lines = [
        "=== RAG Vectorizer Comparison ===",
        "",
        "| Vectorizer | Total queries | Hit@1 | Hit@K | MRR |",
        "|---|---:|---:|---:|---:|",
    ]

    for result in results:
        metrics = result.report.metrics
        lines.append(
            f"| {result.name} | "
            f"{metrics.total_queries} | "
            f"{metrics.hit_at_1:.3f} | "
            f"{metrics.hit_at_k:.3f} | "
            f"{metrics.mean_reciprocal_rank:.3f} |"
        )

    lines.append("")
    lines.append("=== Detailed Reports ===")

    for result in results:
        lines.append("")
        lines.append(f"## {result.name}")
        lines.append(format_evaluation_report(result.report))

    return "\n".join(lines)


def main() -> None:
    results = run_vectorizer_comparison(top_k=3)

    print(format_comparison(results))


if __name__ == "__main__":
    main()
