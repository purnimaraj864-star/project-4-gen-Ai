"""Unit tests: citation validation and answer-vs-evidence overlap."""

from __future__ import annotations

from obsw_history_bot.config import Settings
from obsw_history_bot.validation import (
    content_words,
    evidence_overlap,
    is_grounded,
    validate_citations,
)

EVIDENCE = "Physics for Class 11. A Physics textbook. Price: Rs 425."


def test_content_words_drops_stopwords_and_boilerplate():
    words = content_words("The price of the Physics textbook")
    assert "price" in words and "physics" in words
    assert "the" not in words and "of" not in words


def test_validate_citations_rejects_unknown_and_empty():
    assert validate_citations([40, 41], [40, 41, 2])
    assert not validate_citations([99], [40, 41])
    assert not validate_citations([], [40])


def test_overlap_high_when_answer_matches_evidence():
    assert evidence_overlap("The Physics textbook costs Rs 425.", [EVIDENCE]) > 0.5


def test_overlap_zero_for_unrelated_answer():
    assert evidence_overlap("The minister lives in New Delhi.", [EVIDENCE]) == 0.0


def test_is_grounded_combines_checks(test_cfg: Settings):
    assert is_grounded("The Physics textbook costs Rs 425.", [40], [40], [EVIDENCE], test_cfg)
    assert not is_grounded("The Physics textbook costs Rs 425.", [99], [40], [EVIDENCE], test_cfg)
    assert not is_grounded("The minister lives in New Delhi.", [40], [40], [EVIDENCE], test_cfg)
