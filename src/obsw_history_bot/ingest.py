"""Chunking, embedding and persistent, idempotent ChromaDB ingestion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, settings
from .pdf_loader import PdfPage, load_pdf_pages


@dataclass(frozen=True)
class IngestStats:
    chunks_produced: int
    chunks_added: int
    chunks_skipped: int
    pages_indexed: int


def chunk_id(source: str, page_number: int, ordinal: int, text: str) -> str:
    """Deterministic content hash so re-ingestion never creates duplicates."""
    digest = hashlib.sha256(f"{source}|{page_number}|{ordinal}|{text}".encode()).hexdigest()
    return f"p{page_number}-{digest[:16]}"


def chunk_pages(
    pages: list[PdfPage],
    *,
    source: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split each page independently so every chunk maps to exactly one PDF page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []
    for page in pages:
        for ordinal, piece in enumerate(splitter.split_text(page.text)):
            text = piece.strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page.page_number,
                        "source": source,
                        "chunk_id": chunk_id(
                            source, page.page_number, ordinal, text
                        ),
                    },
                )
            )
    return documents


def get_embeddings(model_name: str | None = None) -> Embeddings:
    return HuggingFaceEmbeddings(model_name=model_name or settings.embedding_model)


def get_vectorstore(
    cfg: Settings | None = None, embeddings: Embeddings | None = None
) -> Chroma:
    cfg = cfg or settings
    return Chroma(
        collection_name=cfg.collection_name,
        embedding_function=embeddings or get_embeddings(cfg.embedding_model),
        persist_directory=str(cfg.chroma_dir),
    )


def ingest_pdf(
    cfg: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
    reset: bool = False,
) -> IngestStats:
    """Build the knowledge base. Safe to run repeatedly: existing chunk IDs are skipped."""
    cfg = cfg or settings
    pages = load_pdf_pages(cfg.pdf_path)
    docs = chunk_pages(
        pages,
        source=Path(cfg.pdf_path).name,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    store = get_vectorstore(cfg, embeddings)
    if reset:
        store.reset_collection()

    ids = [d.metadata["chunk_id"] for d in docs]
    existing: set[str] = set()
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        existing.update(store.get(ids=batch, limit=len(batch), include=[])["ids"])
    fresh = [d for d in docs if d.metadata["chunk_id"] not in existing]
    if fresh:
        store.add_documents(fresh, ids=[d.metadata["chunk_id"] for d in fresh])

    return IngestStats(
        chunks_produced=len(docs),
        chunks_added=len(fresh),
        chunks_skipped=len(docs) - len(fresh),
        pages_indexed=len(pages),
    )
