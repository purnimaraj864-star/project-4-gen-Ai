"""ChromaDB retrieval with relevance scores."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import Settings, settings
from .ingest import get_vectorstore


@dataclass(frozen=True)
class RetrievedChunk:
    document: Document
    distance: float  # L2 distance for MiniLM embeddings; smaller = more relevant

    @property
    def page(self) -> int:
        return int(self.document.metadata["page"])

    @property
    def is_relevant(self) -> bool:
        return self.distance <= settings.retrieval_score_threshold


def retrieve(
    question: str,
    cfg: Settings | None = None,
    k: int | None = None,
    store: Chroma | None = None,
) -> list[RetrievedChunk]:
    """Return the k nearest chunks to the question, ordered by relevance."""
    cfg = cfg or settings
    store = store or get_vectorstore(cfg)
    hits = store.similarity_search_with_score(question, k=k or cfg.top_k)
    return [RetrievedChunk(document=doc, distance=float(score)) for doc, score in hits]


def evidence_chunks(
    chunks: list[RetrievedChunk], cfg: Settings | None = None
) -> list[RetrievedChunk]:
    """Keep only chunks whose relevance distance is below the configured threshold."""
    cfg = cfg or settings
    return [c for c in chunks if c.distance <= cfg.retrieval_score_threshold]
