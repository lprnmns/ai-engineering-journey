from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, cast

from labs.rag.chunking import ChunkSearchResult

DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class PairScoringModel(Protocol):
    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        show_progress_bar: bool,
    ) -> Sequence[float]: ...


def load_pair_scoring_model(model_name: str) -> PairScoringModel:
    from sentence_transformers import CrossEncoder

    return cast(PairScoringModel, CrossEncoder(model_name, max_length=512))


@dataclass(frozen=True)
class RerankedChunkResult:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    chunk_index: int
    retrieval_score: float
    reranker_score: float


@dataclass
class CrossEncoderReranker:
    """Re-score retrieval candidates by reading each query/chunk pair together."""

    model_name: str = DEFAULT_RERANKER_MODEL
    model: PairScoringModel | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = load_pair_scoring_model(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: list[ChunkSearchResult],
        top_k: int = 3,
    ) -> list[RerankedChunkResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not candidates:
            return []

        if self.model is None:
            raise RuntimeError("pair scoring model must be loaded before reranking")

        pairs = [
            (query, f"{candidate.title} {candidate.text}")
            for candidate in candidates
        ]
        scores = self.model.predict(pairs, show_progress_bar=False)

        if len(scores) != len(candidates):
            raise RuntimeError("reranker must return one score for each candidate")

        reranked = [
            RerankedChunkResult(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                title=candidate.title,
                text=candidate.text,
                source=candidate.source,
                chunk_index=candidate.chunk_index,
                retrieval_score=candidate.score,
                reranker_score=float(score),
            )
            for candidate, score in zip(candidates, scores)
        ]

        return sorted(
            reranked,
            key=lambda result: result.reranker_score,
            reverse=True,
        )[:top_k]
