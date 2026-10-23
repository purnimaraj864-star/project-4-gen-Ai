"""Unit tests: chunking, page metadata, ChromaDB persistence, idempotent ingestion."""

from __future__ import annotations

from langchain_chroma import Chroma

from obsw_history_bot.config import Settings
from obsw_history_bot.ingest import chunk_id, chunk_pages, ingest_pdf
from tests.conftest import CATALOGUE_PAGES


def test_chunks_carry_exact_page_metadata():
    docs = chunk_pages(CATALOGUE_PAGES, source="catalogue.pdf", chunk_size=120, chunk_overlap=20)
    assert docs, "expected at least one chunk"
    for doc in docs:
        assert doc.metadata["source"] == "catalogue.pdf"
        assert doc.metadata["page"] in {p.page_number for p in CATALOGUE_PAGES}
        assert doc.metadata["chunk_id"]


def test_chunk_ids_are_deterministic():
    a = chunk_id("s.pdf", 3, 0, "hello world")
    b = chunk_id("s.pdf", 3, 0, "hello world")
    c = chunk_id("s.pdf", 3, 0, "different text")
    assert a == b
    assert a != c


def test_long_page_splits_into_multiple_chunks():
    docs = chunk_pages(CATALOGUE_PAGES, source="s.pdf", chunk_size=60, chunk_overlap=10)
    pages_40 = [d for d in docs if d.metadata["page"] == 40]
    assert len(pages_40) >= 2


def test_ingest_persists_to_disk(test_cfg: Settings, toy_embeddings):
    stats = ingest_pdf(test_cfg, embeddings=toy_embeddings)
    assert stats.pages_indexed == 4
    assert stats.chunks_added == stats.chunks_produced > 0

    reopened = Chroma(
        collection_name=test_cfg.collection_name,
        embedding_function=toy_embeddings,
        persist_directory=str(test_cfg.chroma_dir),
    )
    assert reopened._collection.count() == stats.chunks_added  # noqa: SLF001


def test_repeated_ingestion_adds_no_duplicates(test_cfg: Settings, toy_embeddings):
    first = ingest_pdf(test_cfg, embeddings=toy_embeddings)
    second = ingest_pdf(test_cfg, embeddings=toy_embeddings)
    assert first.chunks_added > 0
    assert second.chunks_added == 0
    assert second.chunks_skipped == first.chunks_added

    store = Chroma(
        collection_name=test_cfg.collection_name,
        embedding_function=toy_embeddings,
        persist_directory=str(test_cfg.chroma_dir),
    )
    assert store._collection.count() == first.chunks_added  # noqa: SLF001


def test_reset_rebuilds_collection(test_cfg: Settings, toy_embeddings):
    ingest_pdf(test_cfg, embeddings=toy_embeddings)
    rebuilt = ingest_pdf(test_cfg, embeddings=toy_embeddings, reset=True)
    assert rebuilt.chunks_skipped == 0
    assert rebuilt.chunks_added == rebuilt.chunks_produced
