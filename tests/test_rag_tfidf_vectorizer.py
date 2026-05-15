import pytest

from labs.rag.tfidf_vectorizer import TfidfVectorizer


def test_tfidf_vectorizer_requires_texts_to_fit() -> None:
    vectorizer = TfidfVectorizer()

    with pytest.raises(ValueError):
        vectorizer.fit([])


def test_tfidf_vectorizer_requires_fit_before_vectorize() -> None:
    vectorizer = TfidfVectorizer()

    with pytest.raises(RuntimeError):
        vectorizer.vectorize("python venv")


def test_tfidf_vectorizer_builds_vocabulary() -> None:
    vectorizer = TfidfVectorizer()
    vectorizer.fit([
        "python sanal ortam",
        "git branch workflow",
    ])

    assert "python" in vectorizer.vocabulary
    assert "git" in vectorizer.vocabulary
    assert vectorizer.is_fitted is True


def test_tfidf_vectorizer_vectorizes_known_terms() -> None:
    vectorizer = TfidfVectorizer()
    vectorizer.fit([
        "python sanal ortam",
        "git branch workflow",
    ])

    vector = vectorizer.vectorize("python ortam")

    assert "python" in vector
    assert "ortam" in vector
    assert vector["python"] > 0.0


def test_tfidf_vectorizer_ignores_unknown_terms() -> None:
    vectorizer = TfidfVectorizer()
    vectorizer.fit(["python sanal ortam"])

    vector = vectorizer.vectorize("unknown_token")

    assert vector == {}


def test_rare_term_gets_higher_idf_than_common_term() -> None:
    vectorizer = TfidfVectorizer()
    vectorizer.fit([
        "python common",
        "git common",
        "rag common",
    ])

    assert vectorizer.idf["python"] > vectorizer.idf["common"]
