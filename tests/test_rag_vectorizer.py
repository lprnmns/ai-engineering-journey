from labs.rag.vectorizer import TermFrequencyVectorizer


def test_term_frequency_vectorizer_counts_terms() -> None:
    vectorizer = TermFrequencyVectorizer()

    vector = vectorizer.vectorize("python python git")

    assert vector["python"] == 2.0
    assert vector["git"] == 1.0


def test_term_frequency_vectorizer_lowercases_tokens() -> None:
    vectorizer = TermFrequencyVectorizer()

    vector = vectorizer.vectorize("Python PYTHON")

    assert vector["python"] == 2.0
