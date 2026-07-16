from labs.rag.answerability_evaluation import AnswerabilityCase
from labs.rag.chunking import ChunkSearchResult
from labs.rag.dense_answerability_experiment import (
    evaluate_dense_answerability,
    find_best_dense_threshold,
    format_dense_threshold_experiment,
)


class FakeRetriever:
    def search(self, query: str, top_k: int = 3) -> list[ChunkSearchResult]:
        score = 0.8 if query == "known" else 0.2
        return [
            ChunkSearchResult(
                chunk_id="chunk_001",
                doc_id="doc_001",
                title="Test",
                text="Test text",
                source="test/source",
                chunk_index=1,
                score=score,
            )
        ]


def test_dense_answerability_uses_retrieval_scores() -> None:
    report = evaluate_dense_answerability(
        cases=[
            AnswerabilityCase(query="known", should_be_answerable=True),
            AnswerabilityCase(query="unknown", should_be_answerable=False),
        ],
        store=FakeRetriever(),
        min_score=0.5,
    )

    assert report.metrics.true_positive == 1
    assert report.metrics.true_negative == 1
    assert report.metrics.accuracy == 1.0


def test_best_dense_threshold_returns_none_for_empty_results() -> None:
    assert find_best_dense_threshold([]) is None


def test_format_dense_threshold_experiment_contains_title() -> None:
    text = format_dense_threshold_experiment([])

    assert "Dense Answerability Threshold Experiment" in text
