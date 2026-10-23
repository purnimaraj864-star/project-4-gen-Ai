"""Central configuration loaded from environment variables (.env supported)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None and raw.strip() else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Values come from env vars; see .env.example."""

    pdf_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "PDF_PATH",
                str(PROJECT_ROOT / "History_2024-25.pdf")
                if (PROJECT_ROOT / "History_2024-25.pdf").is_file()
                else str(PROJECT_ROOT / "data" / "History_2024-25.pdf"),
            )
        )
    )
    chroma_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
        )
    )
    collection_name: str = os.getenv("COLLECTION_NAME", "obsw_history_catalogue")

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # LLM — any OpenAI-compatible endpoint works (OpenAI, Groq, ...).
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = _get_float("LLM_TEMPERATURE", 0.0)

    chunk_size: int = _get_int("CHUNK_SIZE", 900)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 150)
    top_k: int = _get_int("TOP_K", 6)

    # Maximum L2 distance (MiniLM) for a chunk to count as evidence.
    retrieval_score_threshold: float = _get_float("RETRIEVAL_SCORE_THRESHOLD", 1.2)

    # Minimum fraction of the answer's content words that must appear on the
    # cited pages for the answer to be considered grounded.
    evidence_overlap_threshold: float = _get_float("EVIDENCE_OVERLAP_THRESHOLD", 0.35)

    not_found_text: str = "Not found"


settings = Settings()
