"""Shared fixtures: synthetic catalogue pages, offline embeddings, tmp settings."""

from __future__ import annotations

import hashlib
import math
import sys
import textwrap
from pathlib import Path

import pymupdf
import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from obsw_history_bot.config import Settings  # noqa: E402
from obsw_history_bot.ingest import chunk_pages  # noqa: E402
from obsw_history_bot.pdf_loader import PdfPage  # noqa: E402

# A miniature stand-in for the real catalogue, with known facts on known pages.
# Pages 3-39 are intentionally blank in the generated PDF.
CATALOGUE_PAGES = [
    PdfPage(1, "Orient BlackSwan History Catalogue 2024-25. Prices in this "
               "catalogue were last updated on 15 March 2024."),
    PdfPage(2, "Orient BlackSwan Private Limited was incorporated in 1948. "
               "Registered office: Himayatnagar, Hyderabad."),
    PdfPage(40, "Physics for Class 11. A Physics textbook for senior secondary "
                "students. Price: Rs 425."),
    PdfPage(41, "Physics for Class 12. Price: Rs 450."),
]


class ToyEmbeddings(Embeddings):
    """Deterministic offline embeddings: L2-normalised bag of hashed words.

    Texts sharing words get small L2 distances, so retrieval behaves like the
    real thing without downloading a model.
    """

    DIMS = 512

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.DIMS
        for word in text.lower().split():
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIMS
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture()
def toy_embeddings() -> ToyEmbeddings:
    return ToyEmbeddings()


@pytest.fixture()
def catalogue_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "History_2024-25.pdf"
    by_number = {p.page_number: p.text for p in CATALOGUE_PAGES}
    doc = pymupdf.open()
    for number in range(1, max(by_number) + 1):
        sheet = doc.new_page()
        if number in by_number:
            for line_no, line in enumerate(textwrap.wrap(by_number[number], width=60)):
                sheet.insert_text((72, 72 + line_no * 16), line)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture()
def test_cfg(tmp_path: Path, catalogue_pdf: Path) -> Settings:
    return Settings(
        pdf_path=catalogue_pdf,
        chroma_dir=tmp_path / "chroma",
        collection_name="test_catalogue",
        chunk_size=120,
        chunk_overlap=20,
        top_k=4,
        retrieval_score_threshold=1.5,
        evidence_overlap_threshold=0.35,
    )


@pytest.fixture()
def sample_store(test_cfg: Settings, toy_embeddings: ToyEmbeddings) -> Chroma:
    docs = chunk_pages(
        CATALOGUE_PAGES, source="History_2024-25.pdf",
        chunk_size=test_cfg.chunk_size, chunk_overlap=test_cfg.chunk_overlap,
    )
    store = Chroma(
        collection_name=test_cfg.collection_name,
        embedding_function=toy_embeddings,
        persist_directory=str(test_cfg.chroma_dir),
    )
    store.add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs])
    return store
