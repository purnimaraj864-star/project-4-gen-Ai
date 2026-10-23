"""Unit tests (mocked LLM): the RAG chain's grounding and Not-found behaviour."""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from obsw_history_bot.chain import (
    AnswerResult,
    MissingApiKeyError,
    _parse_llm_json,
    answer_question,
    build_llm,
)
from obsw_history_bot.config import Settings

NOT_FOUND = Settings.__dataclass_fields__["not_found_text"].default


def fake_llm(payload: dict) -> FakeListChatModel:
    return FakeListChatModel(responses=[json.dumps(payload)], sleep=0)


def ask(question: str, payload: dict, test_cfg: Settings, sample_store) -> AnswerResult:
    return answer_question(question, test_cfg, llm=fake_llm(payload), store=sample_store)


def test_grounded_answer_with_citation(test_cfg: Settings, sample_store):
    result = ask(
        "When were the prices last updated?",
        {"answer": "Prices were last updated on 15 March 2024.", "pages": [1]},
        test_cfg, sample_store,
    )
    assert result.found
    assert result.answer == "Prices were last updated on 15 March 2024."
    assert result.pages == [1]
    assert result.reason == "ok"


def test_llm_declining_to_answer_becomes_not_found(test_cfg: Settings, sample_store):
    result = ask(
        "When were the prices last updated?",
        {"answer": None, "pages": []},
        test_cfg, sample_store,
    )
    assert not result.found
    assert result.answer == NOT_FOUND
    assert result.reason == "llm_unsupported"


def test_garbage_llm_output_becomes_not_found(test_cfg: Settings, sample_store):
    llm = FakeListChatModel(responses=["I cannot say for sure, sorry!"], sleep=0)
    result = answer_question(
        "When were the prices last updated?", test_cfg, llm=llm, store=sample_store
    )
    assert not result.found
    assert result.answer == NOT_FOUND
    assert result.reason == "invalid_llm_output"


def test_citation_outside_evidence_rejected(test_cfg: Settings, sample_store):
    result = ask(
        "When were the prices last updated?",
        {"answer": "Prices were last updated on 15 March 2024.", "pages": [77]},
        test_cfg, sample_store,
    )
    assert not result.found
    assert result.reason == "citation_failed"


def test_answer_without_evidence_overlap_rejected(test_cfg: Settings, sample_store):
    result = ask(
        "When were the prices last updated?",
        {"answer": "The minister announced new education policy.", "pages": [1]},
        test_cfg, sample_store,
    )
    assert not result.found
    assert result.reason == "citation_failed"


def test_no_relevant_evidence_short_circuits(test_cfg: Settings, sample_store):
    strict = Settings(**{**test_cfg.__dict__, "retrieval_score_threshold": 0.0})
    result = answer_question(
        "quantum chromodynamics", strict, llm=fake_llm({"answer": "x", "pages": [1]}),
        store=sample_store,
    )
    assert not result.found
    assert result.reason == "no_relevant_evidence"


def test_parse_llm_json_variants():
    assert _parse_llm_json('{"answer": "a", "pages": [3, 1, 3]}') == ("a", [1, 3])
    assert _parse_llm_json('```json\n{"answer": null, "pages": []}\n```') == (None, [])
    assert _parse_llm_json("no json here") is None
    assert _parse_llm_json('{"answer": 5, "pages": []}') is None
    assert _parse_llm_json('{"answer": "a", "pages": "forty"}') is None


def test_missing_api_key_raises_actionable_error(test_cfg: Settings, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = Settings(**{**test_cfg.__dict__, "openai_api_key": None})
    with pytest.raises(MissingApiKeyError, match="OPENAI_API_KEY"):
        build_llm(cfg)
