"""CLI: build the ChromaDB knowledge base from the catalogue PDF.

Usage:
    python -m scripts.ingest            # add only missing chunks (idempotent)
    python -m scripts.ingest --reset    # drop and rebuild the collection
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from obsw_history_bot.config import settings  # noqa: E402
from obsw_history_bot.ingest import ingest_pdf  # noqa: E402


def main() -> None:
    reset = "--reset" in sys.argv
    if not settings.pdf_path.is_file():
        print(f"PDF not found at {settings.pdf_path}. Set PDF_PATH or add the file.")
        raise SystemExit(1)
    stats = ingest_pdf(reset=reset)
    print(
        f"Ingestion complete: {stats.chunks_added} chunks added, "
        f"{stats.chunks_skipped} already present, {stats.pages_indexed} PDF pages indexed."
    )


if __name__ == "__main__":
    main()
