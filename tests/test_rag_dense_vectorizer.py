from labs.rag.dense_vectorizer import DenseVectorizer


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.received_text: str | None = None
        self.normalized: bool | None = None
        self.show_progress: bool | None = None

    def encode(
        self,
        sentences: str,
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[float]:
        self.received_text = sentences
        self.normalized = normalize_embeddings
        self.show_progress = show_progress_bar

        return [0.25, -0.5, 0.75]


def test_dense_vectorizer_delegates_to_embedding_model() -> None:
    model = FakeEmbeddingModel()
    vectorizer = DenseVectorizer(model=model)

    vector = vectorizer.vectorize("Python sanal ortam nasıl kurulur?")

    assert vector == [0.25, -0.5, 0.75]
    assert model.received_text == "Python sanal ortam nasıl kurulur?"
    assert model.normalized is True
    assert model.show_progress is False
