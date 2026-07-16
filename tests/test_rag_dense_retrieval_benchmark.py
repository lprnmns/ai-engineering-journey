from labs.rag.dense_retrieval_benchmark import PARAPHRASE_GOLDEN_QUERIES


def test_paraphrase_benchmark_covers_all_hard_documents() -> None:
    expected_doc_ids = {query.expected_doc_id for query in PARAPHRASE_GOLDEN_QUERIES}

    assert expected_doc_ids == {
        "hard_python_venv",
        "hard_python_pip",
        "hard_git_branch",
        "hard_git_pull_request",
        "hard_http_404",
        "hard_http_500",
        "hard_rag_context",
        "hard_rag_embedding",
    }


def test_paraphrase_benchmark_has_one_query_per_hard_document() -> None:
    assert len(PARAPHRASE_GOLDEN_QUERIES) == 8
