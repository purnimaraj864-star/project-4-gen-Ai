"""Streamlit chat interface for the Orient BlackSwan History Bot."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from obsw_history_bot.chain import (  # noqa: E402
    AnswerResult,
    MissingApiKeyError,
    answer_question,
)
from obsw_history_bot.config import settings  # noqa: E402
from obsw_history_bot.ingest import ingest_pdf  # noqa: E402

SAMPLE_QUESTIONS = [
    "On what date were the prices in this catalogue last updated?",
    "In which year was the publisher (Orient BlackSwan) incorporated?",
    "What is the price of a Physics textbook?",
    "Who is the current Minister of Education?",
    'Which is the "best" history book for a beginner?',
]

st.set_page_config(
    page_title="Orient BlackSwan History Bot",
    page_icon="📚",
    layout="centered",
)


@st.cache_data(show_spinner=False)
def count_indexed_chunks() -> int:
    from obsw_history_bot.ingest import get_vectorstore

    try:
        store = get_vectorstore()
        return store._collection.count()  # noqa: SLF001
    except Exception:
        return 0


def render_answer(result: AnswerResult) -> None:
    if result.found:
        st.success("✅ **Grounded Answer**")
        st.markdown(result.answer)
        pages = " · ".join(f"PDF page {p}" for p in result.pages)
        st.caption(f"📖 **Verified Citation**: {pages}")
        with st.expander("🔍 View Retrieved Document Evidence"):
            seen: set[int] = set()
            for chunk in result.evidence:
                if chunk.page in seen:
                    continue
                seen.add(chunk.page)
                st.markdown(f"**[PDF page {chunk.page}]** — distance `{chunk.distance:.4f}`")
                st.info(chunk.document.page_content)
    else:
        st.warning("⚠️ **Strict Grounding Refusal**")
        st.markdown(f"### **{settings.not_found_text}**")
        st.caption(f"Reason: `{result.reason}` — answer is not supported by the PDF catalogue.")
        with st.expander("ℹ️ Grounding & Anti-Hallucination Details"):
            st.write(
                "The bot is strictly configured to answer using only verified facts from "
                "the Orient BlackSwan History 2024-25 catalogue. Any question asking for "
                "unsupported subjects, missing information, or subjective recommendations "
                "returns exactly **Not found** without guessing or hallucinating."
            )


def main() -> None:
    st.title("📚 Orient BlackSwan History Bot")
    st.markdown(
        "Ask questions about the **History 2024-25 catalogue (~80+ pp)**. "
        "Answers are strictly grounded in retrieved evidence with exact page citations. "
        "When evidence is absent or unsupported, the bot returns exactly **Not found**."
    )

    with st.sidebar:
        st.header("📖 Knowledge Base")
        pdf_ok = settings.pdf_path.is_file()
        st.write(f"**Document**: `{'✅ ' + settings.pdf_path.name if pdf_ok else '❌ not found'}`")
        st.write(f"**Indexed Chunks**: `{count_indexed_chunks()}`")
        if st.button("🔄 Re-run Ingestion", disabled=not pdf_ok, use_container_width=True):
            with st.spinner("Embedding catalogue pages into ChromaDB…"):
                stats = ingest_pdf()
            st.success(
                f"Ingestion synced: {stats.chunks_added} added, "
                f"{stats.chunks_skipped} cached, {stats.pages_indexed} pages."
            )
            count_indexed_chunks.clear()
            st.rerun()

        st.divider()
        st.header("🎯 Mandatory Questions")
        st.caption("Click any question to ask the pipeline:")
        for i, q in enumerate(SAMPLE_QUESTIONS, start=1):
            if st.button(f"**Q{i}**: {q}", key=f"q{i}", use_container_width=True):
                st.session_state.pending_question = q

        st.divider()
        mode_label = "Live LLM (OpenAI/Groq)" if settings.openai_api_key else "Grounded Reader (Strict Offline)"
        st.caption(f"**Pipeline Mode**: `{mode_label}`")
        st.caption(f"**Embeddings**: `{settings.embedding_model}`")
        st.caption(f"**Vector Store**: `ChromaDB` (persistent)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    query_param_q = st.query_params.get("q")
    if query_param_q and "query_param_run" not in st.session_state:
        st.session_state.query_param_run = True
        questions_to_run = []
        if query_param_q == "all":
            questions_to_run = SAMPLE_QUESTIONS
        elif query_param_q in ["1", "2", "3", "4", "5"]:
            idx = int(query_param_q) - 1
            questions_to_run = [SAMPLE_QUESTIONS[idx]]
        else:
            questions_to_run = [query_param_q]

        for q_item in questions_to_run:
            res = answer_question(q_item)
            st.session_state.messages.append(("user", q_item))
            st.session_state.messages.append(("assistant", res))

    for role, result in st.session_state.messages:
        with st.chat_message(role):
            if role == "user":
                st.markdown(result)
            else:
                render_answer(result)

    question = st.session_state.pop("pending_question", None) or st.chat_input(
        "Ask about the catalogue…"
    )
    if question:
        st.session_state.messages.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the catalogue…"):
                try:
                    result = answer_question(question)
                except MissingApiKeyError as exc:
                    st.error(str(exc))
                    st.stop()
                except Exception as exc:  # network errors, missing index, etc.
                    st.error(f"Something went wrong: {exc}")
                    st.stop()
            render_answer(result)
        st.session_state.messages.append(("assistant", result))


if __name__ == "__main__":
    main()
