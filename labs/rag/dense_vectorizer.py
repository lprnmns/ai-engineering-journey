from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, cast

from labs.rag.similarity import DenseVector

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingModel(Protocol):
    def encode(
        self,
        sentences: str,
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Sequence[float]: ...


def load_embedding_model(model_name: str) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    return cast(EmbeddingModel, SentenceTransformer(model_name))


@dataclass
class DenseVectorizer:
    """Vectorize text with a pre-trained multilingual sentence embedding model."""

    model_name: str = DEFAULT_MODEL_NAME
    model: EmbeddingModel | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = load_embedding_model(self.model_name)

    def vectorize(self, text: str) -> DenseVector:
        if self.model is None:
            raise RuntimeError("embedding model must be loaded before vectorizing")

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return [float(value) for value in embedding]
