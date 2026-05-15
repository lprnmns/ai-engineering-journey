from __future__ import annotations

from dataclasses import dataclass

from labs.rag.evaluation import (
    EvaluationReport,
    GoldenQuery,
    QueryEvaluation,
    RetrievalMetrics,
    evaluate_retrieval,
    format_evaluation_report,
)
from labs.rag.hybrid_retrieval import build_hybrid_retriever
from labs.rag.hybrid_weight_experiment import (
    HybridQueryEvaluation,
    calculate_hybrid_metrics,
    evaluate_hybrid_query,
)
from labs.rag.sample_docs import Document
from labs.rag.vector_store import InMemoryVectorStore, build_vector_store
from labs.rag.vectorizer import TermFrequencyVectorizer
from labs.rag.vectorizer_comparison import build_tfidf_vector_store


HARD_DOCUMENTS: list[Document] = [
    Document(
        doc_id="hard_python_venv",
        title="Python Virtual Environment",
        text=(
            "Python projelerinde sanal ortam oluşturmak için python -m venv .venv "
            "komutu kullanılır. Sanal ortam proje bağımlılıklarını izole eder."
        ),
        source="hard/python",
    ),
    Document(
        doc_id="hard_python_pip",
        title="Python Package Installation",
        text=(
            "Python paket yönetiminde pip install komutu kullanılır. "
            "Paketler requirements dosyasına yazılarak tekrar kurulabilir."
        ),
        source="hard/python",
    ),
    Document(
        doc_id="hard_git_branch",
        title="Git Branch",
        text=(
            "Git branch, ana kodu bozmadan yeni özellik geliştirmek için kullanılır. "
            "Branch üzerinde commit atılır ve çalışma tamamlanınca değişiklikler paylaşılır."
        ),
        source="hard/git",
    ),
    Document(
        doc_id="hard_git_pull_request",
        title="Git Pull Request",
        text=(
            "Pull request, değişikliklerin main branch içine alınmadan önce incelenmesini sağlar. "
            "Kod gözden geçirilir, yorum yapılır ve onaydan sonra merge edilir."
        ),
        source="hard/git",
    ),
    Document(
        doc_id="hard_http_404",
        title="HTTP 404 Not Found",
        text=(
            "HTTP 404 Not Found hatası, istenen sayfanın veya kaynağın bulunamadığını gösterir. "
            "Yanlış URL, silinmiş sayfa veya eksik route bu hataya neden olabilir."
        ),
        source="hard/http",
    ),
    Document(
        doc_id="hard_http_500",
        title="HTTP 500 Internal Server Error",
        text=(
            "HTTP 500 Internal Server Error, sunucu tarafında beklenmeyen bir hata olduğunu gösterir. "
            "Kod hatası, servis problemi veya yanlış yapılandırma bu hataya neden olabilir."
        ),
        source="hard/http",
    ),
    Document(
        doc_id="hard_rag_context",
        title="RAG Context Building",
        text=(
            "RAG sisteminde retrieval sonrası bulunan chunklar context haline getirilir. "
            "Bu context dil modeline verilir ve cevap sadece bu bilgiye dayanarak üretilir."
        ),
        source="hard/rag",
    ),
    Document(
        doc_id="hard_rag_embedding",
        title="RAG Embedding",
        text=(
            "Embedding modeli metni sayısal dense vector temsiline çevirir. "
            "Benzer anlamlı metinler vector uzayında birbirine yakın konumlanır."
        ),
        source="hard/rag",
    ),
]


HARD_GOLDEN_QUERIES: list[GoldenQuery] = [
    GoldenQuery(
        query="Python sanal ortam nasıl oluşturulur?",
        expected_doc_id="hard_python_venv",
    ),
    GoldenQuery(
        query="pip install paketleri ne için kullanılır?",
        expected_doc_id="hard_python_pip",
    ),
    GoldenQuery(
        query="Pull request main branch'e almadan önce ne işe yarar?",
        expected_doc_id="hard_git_pull_request",
    ),
    GoldenQuery(
        query="Git branch ana kodu bozmadan geliştirme sağlar mı?",
        expected_doc_id="hard_git_branch",
    ),
    GoldenQuery(
        query="HTTP 404 not found hatası ne demek?",
        expected_doc_id="hard_http_404",
    ),
    GoldenQuery(
        query="HTTP 500 internal server error neden olur?",
        expected_doc_id="hard_http_500",
    ),
    GoldenQuery(
        query="RAG sisteminde context dil modeline nasıl verilir?",
        expected_doc_id="hard_rag_context",
    ),
    GoldenQuery(
        query="Embedding modeli metni nasıl vektöre çevirir?",
        expected_doc_id="hard_rag_embedding",
    ),
]


@dataclass(frozen=True)
class HardBenchmarkResult:
    name: str
    metrics: RetrievalMetrics
    retrieved_rows: list[str]


def format_standard_rows(evaluations: list[QueryEvaluation]) -> list[str]:
    rows: list[str] = []

    for evaluation in evaluations:
        rows.append(
            f"- Query: {evaluation.query}\n"
            f"  Expected: {evaluation.expected_doc_id}\n"
            f"  Retrieved: {', '.join(evaluation.retrieved_doc_ids)}\n"
            f"  Rank: {evaluation.rank}"
        )

    return rows


def format_hybrid_rows(evaluations: list[HybridQueryEvaluation]) -> list[str]:
    rows: list[str] = []

    for evaluation in evaluations:
        rows.append(
            f"- Query: {evaluation.query}\n"
            f"  Expected: {evaluation.expected_doc_id}\n"
            f"  Retrieved: {', '.join(evaluation.retrieved_doc_ids)}\n"
            f"  Rank: {evaluation.rank}"
        )

    return rows


def evaluate_standard_store(
    name: str,
    store: InMemoryVectorStore,
    top_k: int = 3,
) -> HardBenchmarkResult:
    report: EvaluationReport = evaluate_retrieval(
        store=store,
        golden_queries=HARD_GOLDEN_QUERIES,
        top_k=top_k,
    )

    return HardBenchmarkResult(
        name=name,
        metrics=report.metrics,
        retrieved_rows=format_standard_rows(report.query_evaluations),
    )


def evaluate_hybrid_retriever(
    name: str,
    vector_weight: float,
    top_k: int = 3,
) -> HardBenchmarkResult:
    retriever = build_hybrid_retriever(
        documents=HARD_DOCUMENTS,
        vector_weight=vector_weight,
    )

    evaluations = [
        evaluate_hybrid_query(
            retriever=retriever,
            golden_query=golden_query,
            top_k=top_k,
        )
        for golden_query in HARD_GOLDEN_QUERIES
    ]

    return HardBenchmarkResult(
        name=name,
        metrics=calculate_hybrid_metrics(evaluations),
        retrieved_rows=format_hybrid_rows(evaluations),
    )


def run_hard_retrieval_benchmark(top_k: int = 3) -> list[HardBenchmarkResult]:
    term_frequency_store = build_vector_store(
        documents=HARD_DOCUMENTS,
        vectorizer=TermFrequencyVectorizer(),
    )
    tfidf_store = build_tfidf_vector_store(
        documents=HARD_DOCUMENTS,
    )

    return [
        evaluate_standard_store(
            name="term_frequency",
            store=term_frequency_store,
            top_k=top_k,
        ),
        evaluate_standard_store(
            name="tfidf",
            store=tfidf_store,
            top_k=top_k,
        ),
        evaluate_hybrid_retriever(
            name="hybrid_keyword_only",
            vector_weight=0.0,
            top_k=top_k,
        ),
        evaluate_hybrid_retriever(
            name="hybrid_balanced",
            vector_weight=0.5,
            top_k=top_k,
        ),
        evaluate_hybrid_retriever(
            name="hybrid_vector_heavy",
            vector_weight=0.7,
            top_k=top_k,
        ),
    ]


def format_hard_benchmark(results: list[HardBenchmarkResult]) -> str:
    lines = [
        "=== Hard Retrieval Benchmark ===",
        "",
        "| Retriever | Total queries | Hit@1 | Hit@K | MRR |",
        "|---|---:|---:|---:|---:|",
    ]

    for result in results:
        metrics = result.metrics
        lines.append(
            f"| {result.name} | "
            f"{metrics.total_queries} | "
            f"{metrics.hit_at_1:.3f} | "
            f"{metrics.hit_at_k:.3f} | "
            f"{metrics.mean_reciprocal_rank:.3f} |"
        )

    lines.append("")
    lines.append("=== Per-query Details ===")

    for result in results:
        lines.append("")
        lines.append(f"## {result.name}")
        lines.extend(result.retrieved_rows)

    return "\n".join(lines)


def main() -> None:
    results = run_hard_retrieval_benchmark(top_k=3)

    print(format_hard_benchmark(results))


if __name__ == "__main__":
    main()
