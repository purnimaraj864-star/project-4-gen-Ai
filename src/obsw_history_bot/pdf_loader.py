"""PDF text extraction with faithful 1-based PDF page numbers (PyMuPDF)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF


@dataclass(frozen=True)
class PdfPage:
    page_number: int  # 1-based index into the PDF file
    text: str


def load_pdf_pages(pdf_path: str | Path) -> list[PdfPage]:
    """Return one PdfPage per non-empty page of the PDF.

    Raises:
        FileNotFoundError: if the PDF does not exist.
        ValueError: if the file exists but has no extractable text.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[PdfPage] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            text = (page.get_text("text") or "").strip()
            if text:
                pages.append(PdfPage(page_number=index, text=text))

    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path}")
    return pages
