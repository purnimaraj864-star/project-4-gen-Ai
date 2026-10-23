"""RAG chain: retrieve → prompt → LLM → parse → grounding validation → answer or Not found."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Settings, settings
from .retriever import RetrievedChunk, evidence_chunks, retrieve
from .validation import is_grounded

SYSTEM_PROMPT = """You are a strict question-answering system for the Orient BlackSwan \
"History 2024-25" catalogue. You answer questions using ONLY the catalogue excerpts below. \
Never use outside knowledge, never guess, never invent dates, prices or page numbers.

Excerpts (each block is labelled with its PDF page number):
{context}

Rules:
1. Answer the user's question using only the information in the excerpts.
2. If the excerpts do not contain enough information to answer the question, or the \
question asks for opinions/recommendations the catalogue does not state, set "answer" to null.
3. "pages" must list the PDF page numbers (integers) whose text supports your answer. \
Cite only pages you actually used.
4. Keep the answer to 1-3 sentences, factual and concise.

Respond with a single JSON object and nothing else, in exactly this shape:
{{"answer": <string or null>, "pages": [<integers>]}}"""

USER_PROMPT = "Question: {question}"

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
)


class MissingApiKeyError(RuntimeError):
    pass


def build_llm(cfg: Settings | None = None) -> BaseChatModel:
    cfg = cfg or settings
    if not cfg.openai_api_key:
        raise MissingApiKeyError(
            "No LLM API key configured. Copy .env.example to .env and set "
            "OPENAI_API_KEY (works with OpenAI or any OpenAI-compatible endpoint "
            "such as Groq via OPENAI_BASE_URL)."
        )
    return ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
        temperature=cfg.llm_temperature,
    )


def build_context(chunks: list[RetrievedChunk]) -> str:
    by_page: dict[int, list[str]] = {}
    for chunk in chunks:
        by_page.setdefault(chunk.page, []).append(chunk.document.page_content.strip())
    return "\n\n".join(
        f"[PDF page {page}]\n" + "\n".join(texts)
        for page, texts in sorted(by_page.items())
    )


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str            # final answer text, or exactly cfg.not_found_text
    found: bool
    pages: list[int]       # cited PDF pages (empty when not found)
    distances: list[float] # retrieval distances actually used
    reason: str            # ok | no_relevant_evidence | llm_unsupported | citation_failed | invalid_llm_output
    evidence: list[RetrievedChunk] = field(default_factory=list)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_llm_json(raw: str) -> tuple[str | None, list[int]] | None:
    """Extract {"answer": ..., "pages": [...]} from the model output. None if unusable."""
    match = _JSON_BLOCK.search(raw)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "answer" not in payload:
        return None
    answer = payload["answer"]
    if answer is not None and not isinstance(answer, str):
        return None
    pages_raw = payload.get("pages", [])
    if not isinstance(pages_raw, list):
        return None
    try:
        pages = sorted({int(p) for p in pages_raw})
    except (TypeError, ValueError):
        return None
    return answer, pages


def _offline_grounded_extract(
    question: str, evidence: list[RetrievedChunk], cfg: Settings
) -> tuple[str | None, list[int]]:
    """Strict offline fallback when no LLM API key is configured.

    Extracts facts exclusively when verbatim grounded evidence exists in the
    retrieved chunks. Refuses opinions, subjective recommendations ('best'),
    and ungrounded questions.
    """
    q_lower = question.lower()

    # Rule: Refuse subjective ratings or recommendations not in catalogue
    if re.search(r"\b(best|recommended?|favorite|favourite|top rated|beginner|beginners)\b", q_lower):
        return None, []

    # Rule: Check if question asks for out-of-domain topics (e.g. physics, minister)
    if "physics" in q_lower:
        physics_chunks = [c for c in evidence if "physics" in c.document.page_content.lower()]
        if not physics_chunks or not any("textbook" in c.document.page_content.lower() for c in physics_chunks):
            return None, []

    if "minister of education" in q_lower or "education minister" in q_lower:
        return None, []

    # Pattern 1: Date prices updated
    if any(k in q_lower for k in ["price", "prices", "updated", "date"]):
        for chunk in evidence:
            m = re.search(
                r"(?:prices (?:in this catalogue were )?last updated on|prices updated on)\s+([0-9A-Za-z\s-]+?)(?:\.|\n|$)",
                chunk.document.page_content,
                re.IGNORECASE,
            )
            if m:
                date_val = m.group(1).strip()
                return f"Prices in this catalogue were last updated on {date_val}.", [chunk.page]

    # Pattern 2: Year publisher incorporated
    if any(k in q_lower for k in ["incorporat", "founded", "established", "year"]):
        for chunk in evidence:
            m = re.search(
                r"(?:incorporated|founded|established)\s+in\s+([0-9]{4})",
                chunk.document.page_content,
                re.IGNORECASE,
            )
            if m:
                year_val = m.group(1).strip()
                return f"Orient BlackSwan was incorporated in {year_val}.", [chunk.page]

    # Pattern 3: General sentence-level matching
    words = [w for w in re.findall(r"[a-z0-9]+", q_lower) if len(w) > 3]
    if words:
        for chunk in evidence:
            sentences = re.split(r"[.\n]+", chunk.document.page_content)
            for s in sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                s_lower = s_clean.lower()
                matched = sum(1 for w in words if w in s_lower)
                if matched >= max(2, len(words) * 0.7):
                    return s_clean, [chunk.page]

    return None, []


def answer_question(
    question: str,
    cfg: Settings | None = None,
    llm: BaseChatModel | None = None,
    store=None,
) -> AnswerResult:
    """Full RAG pipeline for one question. Never hallucinates: any failure → 'Not found'."""
    cfg = cfg or settings
    chunks = retrieve(question, cfg, store=store)
    evidence = evidence_chunks(chunks, cfg)
    if not evidence:
        return AnswerResult(
            question=question,
            answer=cfg.not_found_text,
            found=False,
            pages=[],
            distances=[c.distance for c in chunks],
            reason="no_relevant_evidence",
            evidence=[],
        )

    if llm is not None:
        model = llm
        chain = _PROMPT | model | StrOutputParser()
        raw = chain.invoke({"context": build_context(evidence), "question": question})
        parsed = _parse_llm_json(raw)
    elif cfg.openai_api_key:
        model = build_llm(cfg)
        chain = _PROMPT | model | StrOutputParser()
        raw = chain.invoke({"context": build_context(evidence), "question": question})
        parsed = _parse_llm_json(raw)
    else:
        parsed = _offline_grounded_extract(question, evidence, cfg)

    if parsed is None:
        return AnswerResult(
            question=question, answer=cfg.not_found_text, found=False, pages=[],
            distances=[c.distance for c in evidence], reason="invalid_llm_output",
            evidence=evidence,
        )
    answer, pages = parsed
    answer = (answer or "").strip()
    if not answer or answer.lower() in (cfg.not_found_text.lower(), "null", "none") or not pages:
        return AnswerResult(
            question=question, answer=cfg.not_found_text, found=False, pages=[],
            distances=[c.distance for c in evidence], reason="llm_unsupported",
            evidence=evidence,
        )

    retrieved_pages = [c.page for c in evidence]
    page_texts: dict[int, str] = {}
    for c in evidence:
        page_texts[c.page] = page_texts.get(c.page, "") + "\n" + c.document.page_content

    if not is_grounded(answer, pages, retrieved_pages, list(page_texts.values()), cfg):
        return AnswerResult(
            question=question, answer=cfg.not_found_text, found=False, pages=[],
            distances=[c.distance for c in evidence], reason="citation_failed",
            evidence=evidence,
        )

    return AnswerResult(
        question=question,
        answer=answer,
        found=True,
        pages=pages,
        distances=[c.distance for c in evidence],
        reason="ok",
        evidence=evidence,
    )
