from __future__ import annotations

from labs.rag.dense_vector_store import DenseVectorStore, build_dense_vector_store
from labs.rag.evaluation import EvaluationReport, GoldenQuery, evaluate_retrieval, format_evaluation_report
from labs.rag.hard_retrieval_benchmark import HARD_DOCUMENTS


PARAPHRASE_GOLDEN_QUERIES: list[GoldenQuery] = [
    GoldenQuery(
        query="Her proje için paketleri birbirinden ayıran ortam nasıl hazırlanır?",
        expected_doc_id="hard_python_venv",
    ),
    GoldenQuery(
        query="Bir Python kütüphanesini terminalden yüklemek için hangi araç kullanılır?",
        expected_doc_id="hard_python_pip",
    ),
    GoldenQuery(
        query="Kod değişiklikleri incelenip onaylanmadan önce kullanılan süreç nedir?",
        expected_doc_id="hard_git_pull_request",
    ),
    GoldenQuery(
        query="Ana kodu etkilemeden yeni bir özellik üzerinde nasıl çalışılır?",
        expected_doc_id="hard_git_branch",
    ),
    GoldenQuery(
        query="İstediğim web sayfası mevcut değilse hangi HTTP durumu oluşur?",
        expected_doc_id="hard_http_404",
    ),
    GoldenQuery(
        query="Sunucu içinde beklenmeyen bir problem olduğunda hangi hata görülür?",
        expected_doc_id="hard_http_500",
    ),
    GoldenQuery(
        query="RAG ile bulunan bilgi parçaları modele hangi amaçla iletilir?",
        expected_doc_id="hard_rag_context",
    ),
    GoldenQuery(
        query="Bir metni anlamını koruyarak sayısal temsile dönüştüren model nedir?",
        expected_doc_id="hard_rag_embedding",
    ),
]


def evaluate_dense_paraphrases(
    store: DenseVectorStore | None = None,
    top_k: int = 3,
) -> EvaluationReport:
    if store is None:
        store = build_dense_vector_store(documents=HARD_DOCUMENTS)

    return evaluate_retrieval(
        store=store,
        golden_queries=PARAPHRASE_GOLDEN_QUERIES,
        top_k=top_k,
    )


def main() -> None:
    report = evaluate_dense_paraphrases(top_k=3)
    print(format_evaluation_report(report))


if __name__ == "__main__":
    main()
