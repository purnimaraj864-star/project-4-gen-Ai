"""Unit tests: PDF loading and page metadata."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from obsw_history_bot.pdf_loader import load_pdf_pages


def test_loads_pages_with_1_based_numbers(catalogue_pdf: Path):
    pages = load_pdf_pages(catalogue_pdf)
    assert [p.page_number for p in pages] == [1, 2, 40, 41]
    assert all(p.text for p in pages)
    assert "incorporated in 1948" in pages[1].text


def test_missing_pdf_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_pdf_pages(tmp_path / "nope.pdf")


def test_empty_pdf_raises_value_error(tmp_path: Path):
    path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    with pytest.raises(ValueError):
        load_pdf_pages(path)
