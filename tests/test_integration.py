"""Real integration tests — need the actual PDF, downloaded embedding models,
and (for the LLM cases) a configured API key. Skipped automatically otherwise.

Run explicitly with: pytest -m integration
"""

from __future__ import annotations

import pytest

from obsw_history_bot.config import settings
from obsw_history_bot.ingest import ingest_pdf

from scripts.ask import ASSIGNMENT_QUESTIONS

needs_pdf = pytest.mark.skipif(
    not settings.pdf_path.is_file(), reason="real PDF not present (set PDF_PATH)"
)
needs_llm = pytest.mark.skipif(
    not settings.openai_api_key, reason="OPENAI_API_KEY not configured"
)


@pytest.fixture(scope="module")
def real_store():
    ingest_pdf()
    from obsw_history_bot.ingest import get_vectorstore

    return get_vectorstore()


@needs_pdf
class TestRealPdf:
    @pytest.mark.integration
    def test_ingest_indexes_all_pages(self):
        stats = ingest_pdf()
        assert stats.pages_indexed > 50, "catalogue should have dozens of pages"
        assert stats.chunks_added == 0, "second run must be idempotent"

    @pytest.mark.integration
    def test_real_embeddings_retrieve_relevant_pages(self, real_store):
        from obsw_history_bot.retriever import retrieve

        hits = retrieve("When was Orient BlackSwan incorporated?", settings)
        assert hits, "retriever returned nothing"
        assert hits[0].distance <= settings.retrieval_score_threshold
        assert hits[0].page == 2


@needs_pdf
class TestRealPipeline:
    @pytest.mark.integration
    def test_assignment_questions_end_to_end(self):
        from obsw_history_bot.chain import answer_question

        for question in ASSIGNMENT_QUESTIONS:
            result = answer_question(question)
            assert isinstance(result.found, bool)
            if result.found:
                assert result.pages, "a found answer must cite pages"
                assert result.answer != settings.not_found_text
            else:
                assert result.answer == settings.not_found_text
