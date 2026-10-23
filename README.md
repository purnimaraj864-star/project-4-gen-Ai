# 📚 Orient BlackSwan History Bot — Production-Grade RAG Pipeline

A strict, production-ready Retrieval-Augmented Generation (RAG) system built with **Python, LangChain, ChromaDB, HuggingFace embeddings (`all-MiniLM-L6-v2`), and LLM / Grounded Extractor**. The chatbot answers queries regarding the **Orient BlackSwan History 2024–25 catalogue (~80+ pages)**, returns exact 1-based PDF page references for all supported facts, and guarantees **zero hallucination** by returning exactly `Not found` whenever evidence is absent, out-of-domain, or ambiguous.

---

## 🌟 Key Features

1. **Strict Zero-Hallucination & Evidence Grounding**:
   - Every answer is validated against the retrieved document chunks.
   - Enforces dual-barrier validation: relevance distance threshold ($L_2 \le 1.2$) and content-word evidence overlap ($\ge 35\%$).
   - Returns strictly `Not found` if evidence is missing, subjective, or out-of-domain.
2. **Faithful 1-Based PDF Page Tracking**:
   - Uses PyMuPDF (`pymupdf`) to extract text and retain exact physical PDF page numbers matching PDF readers.
3. **Deterministic, Idempotent ChromaDB Ingestion**:
   - Generates SHA-256 chunk identifiers (`p{page}-{hash}`).
   - Re-running ingestion skips existing chunks without duplicating records.
4. **Flexible LLM & Offline Architecture**:
   - Supports Google Gemini, OpenAI, Groq, or local OpenAI-compatible endpoints via `.env`.
   - Built-in strict offline deterministic grounded reader fallback when no API key or network is present.
5. **Interactive Streamlit Web Interface**:
   - Chat interface with quick-action buttons for the 5 mandatory assignment questions.
   - Expandable source inspector displaying exact text chunks, citations, and retrieval distance scores.
6. **Automated Test Coverage**:
   - 29 unit and integration tests covering PDF loading, chunking, metadata preservation, vector store persistence, citation validity, hallucination prevention, and end-to-end question answering.

---

## 🏗️ Technology Stack & RAG Workflow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Knowledge Base Creation (Ingestion)                                     │
│                                                                                  │
│ History_2024-25.pdf                                                              │
│       │                                                                          │
│       ▼ (PyMuPDF Text Extraction + 1-Based Page Indexing)                        │
│ PdfPage(page_number, text)                                                       │
│       │                                                                          │
│       ▼ (RecursiveCharacterTextSplitter: 900 chars, 150 overlap)                 │
│ Document Chunks with Deterministic IDs (p{page}-{sha256[:16]})                   │
│       │                                                                          │
│       ▼ (HuggingFace sentence-transformers/all-MiniLM-L6-v2)                     │
│ ChromaDB Vector Store (Persistent storage: chroma_db/)                          │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│ Phase 2 & 3: Query Processing & Grounded Answer Generation                       │
│                                                                                  │
│ User Question                                                                    │
│       │                                                                          │
│       ▼ (Embedding & Similarity Search)                                          │
│ Top-K Chunks from ChromaDB                                                       │
│       │                                                                          │
│       ▼ Distance Threshold Check (L2 <= 1.2)                                     │
│ Relevant Evidence Chunks ─────────────────────────► If Empty ───► "Not found"    │
│       │                                                                          │
│       ▼ Strict LangChain Prompt Template + LLM / Grounded Extractor              │
│ Candidate Answer + Cited Pages                                                   │
│       │                                                                          │
│       ▼ Validation Check (Valid citations + Evidence Overlap >= 35%)             │
│ Grounded? ────► YES ───► Final Answer + Citations (e.g., "PDF page 2")          │
│       │                                                                          │
│       └───────► NO  ───► "Not found"                                            │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
project 4/
├── app/
│   └── streamlit_app.py        # Streamlit web chat UI with sidebar & inspectors
├── data/
│   └── History_2024-25.pdf     # Orient BlackSwan History catalogue source PDF
├── scripts/
│   ├── __init__.py
│   ├── ask.py                  # CLI query tool (supports single query or --all)
│   └── ingest.py               # Ingestion CLI (idempotent, supports --reset)
├── src/
│   └── obsw_history_bot/
│       ├── __init__.py
│       ├── chain.py            # RAG pipeline, prompt template, parsing, fallback
│       ├── config.py           # Dataclass settings loaded from .env
│       ├── ingest.py           # Text splitter, embeddings, ChromaDB persistence
│       ├── pdf_loader.py       # PyMuPDF loader preserving 1-based PDF pages
│       ├── retriever.py        # Vector retrieval & distance threshold filtering
│       └── validation.py       # Grounding & citation overlap verification
├── tests/
│   ├── conftest.py             # Shared fixtures (synthetic catalogue & mock store)
│   ├── test_chain.py           # Chain & grounding unit tests
│   ├── test_ingest.py          # Chunking, idempotency & persistence tests
│   ├── test_integration.py     # Real PDF and real embeddings integration tests
│   ├── test_pdf_loader.py      # Page extraction and 1-based index tests
│   ├── test_retrieval.py       # Retrieval and thresholding tests
│   └── test_validation.py      # Citation sanity and word overlap tests
├── .env.example                # Example environment variables template
├── .gitignore                  # Git ignore rules (.env, chroma_db, venv)
├── pytest.ini                  # Pytest configuration with pythonpath and markers
├── requirements.txt            # Pinned, compatible dependencies
└── README.md                   # Project documentation and submission report
```

---

## ⚙️ Installation & Configuration

### 1. Prerequisites
- Python 3.11+
- Virtual environment (`venv` or `conda`)

### 2. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd "project 4"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables (.env)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Configure your credentials in `.env`:
```ini
# Google Gemini API Key:
GEMINI_API_KEY=your_gemini_api_key_here

# OR OpenAI / Groq (OpenAI-compatible):
# OPENAI_API_KEY=your_key_here
# OPENAI_BASE_URL=https://api.groq.com/openai/v1
# LLM_MODEL=llama-3.1-8b-instant

# Knowledge Base Paths
PDF_PATH=History_2024-25.pdf
CHROMA_DIR=chroma_db
COLLECTION_NAME=obsw_history_catalogue
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Tuning Parameters
CHUNK_SIZE=900
CHUNK_OVERLAP=150
TOP_K=6
RETRIEVAL_SCORE_THRESHOLD=1.2
EVIDENCE_OVERLAP_THRESHOLD=0.35
```

---

## 🚀 Ingestion & Execution

### 1. Ingest Catalogue into ChromaDB
```bash
# Initial or incremental ingestion (skips existing chunks)
python -m scripts.ingest

# Full rebuild (drops and re-indexes collection)
python -m scripts.ingest --reset
```
*Output:*
```text
Ingestion complete: 739 chunks added, 0 already present, 90 PDF pages indexed.
```

### 2. Run the CLI Query Tool
```bash
# Run all 5 mandatory questions
python -m scripts.ask --all

# Ask an individual question
python -m scripts.ask "In which year was Orient BlackSwan incorporated?"
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app/streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🎯 Mandatory Questions — Verified Results

All five questions were evaluated through the RAG pipeline against `History_2024-25.pdf`:

| # | Question | Verified Answer | Source Citation | Status / Grounding Reason |
|---|---|---|---|---|
| **1** | On what date were the prices in this catalogue last updated? | **Prices in this catalogue were last updated on 20-01-2025.** | **PDF Page 2** (also appears on subsequent pages such as 22, 83–91) | Grounded fact (`reason: ok`, $L_2 = 0.808$, 75% overlap) |
| **2** | In which year was the publisher (Orient BlackSwan) incorporated? | **Orient BlackSwan was incorporated in 1948.** | **PDF Page 2** | Grounded fact (`reason: ok`, $L_2 = 0.281$, 100% overlap) |
| **3** | What is the price of a Physics textbook? | **Not found** | *None* | Grounded refusal (`reason: llm_unsupported`). The catalogue is exclusively a History catalogue; no physics textbooks exist in the document. |
| **4** | Who is the current Minister of Education? | **Not found** | *None* | Grounded refusal (`reason: no_relevant_evidence`). All retrieval distances exceeded threshold ($L_2 \ge 1.275$). |
| **5** | Which is the "best" history book for a beginner? | **Not found** | *None* | Grounded refusal (`reason: llm_unsupported`). The catalogue lists academic books without subjective rankings or beginner recommendations. |

---

## 🧪 Automated Testing & Results

The test suite consists of **29 automated tests** dividing cleanly into:
- **Unit Tests (26)**: Fast, offline tests with synthetic data verifying PDF parsing, text chunking, deterministic hashing, persistence, retrieval scoring, citation validation, and anti-hallucination refusals.
- **Integration Tests (3)**: Real end-to-end tests verifying PyMuPDF extraction on `History_2024-25.pdf`, real ChromaDB retrieval with `sentence-transformers`, and complete answer generation across all 5 mandatory questions.

### Execute Tests
```bash
# Run all tests
pytest -v

# Run only unit tests
pytest -v -m "not integration"

# Run only integration tests
pytest -v -m integration
```

### Test Suite Execution Output
```text
============================= test session starts ==============================
platform darwin -- Python 3.13.9, pytest-9.1.1, pluggy-1.7.0
rootdir: /Users/purnimaraj/project 4
configfile: pytest.ini
testpaths: tests
collected 29 items

tests/test_chain.py::test_grounded_answer_with_citation PASSED           [  3%]
tests/test_chain.py::test_llm_declining_to_answer_becomes_not_found PASSED [  6%]
tests/test_chain.py::test_garbage_llm_output_becomes_not_found PASSED    [ 10%]
tests/test_chain.py::test_citation_outside_evidence_rejected PASSED      [ 13%]
tests/test_chain.py::test_answer_without_evidence_overlap_rejected PASSED [ 17%]
tests/test_chain.py::test_no_relevant_evidence_short_circuits PASSED     [ 20%]
tests/test_chain.py::test_parse_llm_json_variants PASSED                 [ 24%]
tests/test_chain.py::test_missing_api_key_raises_actionable_error PASSED [ 27%]
tests/test_ingest.py::test_chunks_carry_exact_page_metadata PASSED       [ 31%]
tests/test_ingest.py::test_chunk_ids_are_deterministic PASSED            [ 34%]
tests/test_ingest.py::test_long_page_splits_into_multiple_chunks PASSED  [ 37%]
tests/test_ingest.py::test_ingest_persists_to_disk PASSED                [ 41%]
tests/test_ingest.py::test_repeated_ingestion_adds_no_duplicates PASSED  [ 44%]
tests/test_ingest.py::test_reset_rebuilds_collection PASSED              [ 48%]
tests/test_integration.py::TestRealPdf::test_ingest_indexes_all_pages PASSED [ 51%]
tests/test_integration.py::TestRealPdf::test_real_embeddings_retrieve_relevant_pages PASSED [ 55%]
tests/test_integration.py::TestRealPipeline::test_assignment_questions_end_to_end PASSED [ 58%]
tests/test_pdf_loader.py::test_loads_pages_with_1_based_numbers PASSED   [ 62%]
tests/test_pdf_loader.py::test_missing_pdf_raises_file_not_found PASSED  [ 65%]
tests/test_pdf_loader.py::test_empty_pdf_raises_value_error PASSED       [ 68%]
tests/test_retrieval.py::test_retrieve_returns_chunks_with_scores PASSED [ 72%]
tests/test_retrieval.py::test_relevant_question_beats_irrelevant PASSED  [ 75%]
tests/test_retrieval.py::test_evidence_chunks_respects_threshold PASSED  [ 79%]
tests/test_retrieval.py::test_build_context_groups_by_page_sorted PASSED [ 82%]
tests/test_validation.py::test_content_words_drops_stopwords_and_boilerplate PASSED [ 86%]
tests/test_validation.py::test_validate_citations_rejects_unknown_and_empty PASSED [ 89%]
tests/test_validation.py::test_overlap_high_when_answer_matches_evidence PASSED [ 93%]
tests/test_validation.py::test_overlap_zero_for_unrelated_answer PASSED  [ 96%]
tests/test_validation.py::test_is_grounded_combines_checks PASSED        [100%]

======================= 29 passed, 7 warnings in 25.31s ========================
```

---

## 📸 Screenshots of Running Application

The Streamlit UI runs locally at `http://localhost:8501`. 

### Manual Verification & Screenshot Steps
1. Launch the app: `PYTHONPATH=src streamlit run app/streamlit_app.py --server.port 8501`
2. Open your web browser to `http://localhost:8501`.
3. Click **Q1** in the sidebar: observe the verified answer `Prices in this catalogue were last updated on 20-01-2025` citing `PDF page 2`.
4. Click **Q2** in the sidebar: observe `Orient BlackSwan was incorporated in 1948` citing `PDF page 2`.
5. Click **Q3**, **Q4**, and **Q5**: observe the bold response `Not found` with grounding refusal explanation.
6. Expand **"🔍 View Retrieved Document Evidence"** under any response to inspect the exact chunk contents and distance metrics.

---

## 🔒 Security & Git Hygiene

- `.env` contains API keys and is strictly excluded via `.gitignore`.
- `.env.example` provides a sanitized configuration template without secrets.
- ChromaDB persistence directory (`chroma_db/`) and virtual environments (`.venv/`) are excluded from version control.
# project-4-gen-Ai
