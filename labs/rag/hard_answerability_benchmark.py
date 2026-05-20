from __future__ import annotations

from labs.rag.answerability_evaluation import (
    AnswerabilityCase,
    AnswerabilityReport,
    evaluate_answerability,
    format_answerability_report,
)


HARD_ANSWERABILITY_CASES: list[AnswerabilityCase] = [
    # Known / answerable queries
    AnswerabilityCase(
        query="Python sanal ortam nasıl kurulur?",
        should_be_answerable=True,
    ),
    AnswerabilityCase(
        query="Pull request ile main branch'e nasıl değişiklik alınır?",
        should_be_answerable=True,
    ),
    AnswerabilityCase(
        query="RAG sisteminde ilgili doküman parçaları nasıl bulunur?",
        should_be_answerable=True,
    ),
    AnswerabilityCase(
        query="Cosine similarity iki vektör arasında neyi ölçer?",
        should_be_answerable=True,
    ),
    AnswerabilityCase(
        query="Pandas ile CSV dosyası nasıl okunur?",
        should_be_answerable=True,
    ),

    # Easy unknown queries
    AnswerabilityCase(
        query="Fenerbahçe maç skoru nedir?",
        should_be_answerable=False,
    ),
    AnswerabilityCase(
        query="Bitcoin fiyatı şu anda kaç dolar?",
        should_be_answerable=False,
    ),

    # Hard unknown / near-miss queries
    AnswerabilityCase(
        query="Python sanal ortam Docker ile nasıl deploy edilir?",
        should_be_answerable=False,
    ),
    AnswerabilityCase(
        query="Git pull request conflict nasıl çözülür?",
        should_be_answerable=False,
    ),
    AnswerabilityCase(
        query="RAG sisteminde embedding fine-tuning nasıl yapılır?",
        should_be_answerable=False,
    ),
    AnswerabilityCase(
        query="Pandas ile Excel dosyası nasıl okunur?",
        should_be_answerable=False,
    ),
    AnswerabilityCase(
        query="Cosine similarity yerine Euclidean distance ne zaman kullanılır?",
        should_be_answerable=False,
    ),
]


def evaluate_hard_answerability(
    min_score: float = 0.05,
    min_margin: float = 0.0,
    top_k: int = 3,
) -> AnswerabilityReport:
    return evaluate_answerability(
        cases=HARD_ANSWERABILITY_CASES,
        top_k=top_k,
        min_score=min_score,
        min_margin=min_margin,
    )


def main() -> None:
    report = evaluate_hard_answerability()

    print(format_answerability_report(report))


if __name__ == "__main__":
    main()
