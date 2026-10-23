"""Grounding checks: citation sanity and answer-vs-evidence word overlap."""

from __future__ import annotations

import re

from .config import Settings, settings

_WORD = re.compile(r"[a-z0-9]+")

# Common stop words plus catalogue boilerplate that carries no evidence value.
_STOP = set(
    "a an the and or of to in on for with is are was were be been this that these those "
    "what which who whom when where why how its it as at by from i you he she they we do "
    "does did not no yes page catalogue orient blackswan history india indian new delhi "
    "rupees rs please contact your sales representative".split()
)


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOP}


def validate_citations(pages: list[int], retrieved_pages: list[int]) -> bool:
    """Every cited page must come from the retrieved evidence set."""
    return bool(pages) and all(p in retrieved_pages for p in pages)


def evidence_overlap(answer: str, page_texts: list[str], cfg: Settings | None = None) -> float:
    """Fraction of the answer's content words present in the cited pages' text.

    0.0 means the answer shares no content words with the cited pages, i.e. the
    citation does not support the answer. 1.0 means full support.
    """
    cfg = cfg or settings
    answer_words = content_words(answer)
    if not answer_words:
        return 0.0
    evidence_words: set[str] = set()
    for text in page_texts:
        evidence_words |= content_words(text)
    hits = sum(1 for w in answer_words if w in evidence_words)
    return hits / len(answer_words)


def is_grounded(
    answer: str, pages: list[int], retrieved_pages: list[int], page_texts: list[str],
    cfg: Settings | None = None,
) -> bool:
    cfg = cfg or settings
    return (
        validate_citations(pages, retrieved_pages)
        and evidence_overlap(answer, page_texts, cfg) >= cfg.evidence_overlap_threshold
    )
