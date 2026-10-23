"""Unit tests: retrieval, relevance thresholding, context building."""

from __future__ import annotations

from obsw_history_bot.chain import build_context
from obsw_history_bot.config import Settings
from obsw_history_bot.retriever import evidence_chunks, retrieve


def test_retrieve_returns_chunks_with_scores(test_cfg: Settings, sample_store):
    chunks = retrieve("When were prices last updated?", test_cfg, store=sample_store)
    assert len(chunks) == test_cfg.top_k
    assert all(c.distance >= 0 for c in chunks)
    assert all(c.document.metadata["page"] for c in chunks)


def test_relevant_question_beats_irrelevant(test_cfg: Settings, sample_store):
    hits = retrieve("Physics textbook price", test_cfg, store=sample_store)
    best_page = hits[0].page
    assert best_page in {40, 41}


def test_evidence_chunks_respects_threshold(test_cfg: Settings):
    from obsw_history_bot.retriever import RetrievedChunk
    from langchain_core.documents import Document

    doc = Document(page_content="x", metadata={"page": 1})
    near = RetrievedChunk(doc, 0.5)
    far = RetrievedChunk(doc, 1.9)
    assert evidence_chunks([near, far], test_cfg) == [near]


def test_build_context_groups_by_page_sorted(test_cfg: Settings, sample_store):
    chunks = retrieve("Physics price incorporated 1948 updated", test_cfg, store=sample_store)
    context = build_context(chunks)
    assert "[PDF page" in context
    labels = [line for line in context.splitlines() if line.startswith("[PDF page")]
    pages = [int(label.split()[2].rstrip("]")) for label in labels]
    assert pages == sorted(pages)
    assert len(pages) == len(set(pages)), "one header per page, no duplicates"
