"""CLI: ask questions through the RAG pipeline.

Usage:
    python -m scripts.ask "What is the price of a Physics textbook?"
    python -m scripts.ask --all     # run the five assignment questions
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from obsw_history_bot.chain import answer_question  # noqa: E402

ASSIGNMENT_QUESTIONS = [
    "On what date were the prices in this catalogue last updated?",
    "In which year was the publisher (Orient BlackSwan) incorporated?",
    "What is the price of a Physics textbook?",
    "Who is the current Minister of Education?",
    'Which is the "best" history book for a beginner?',
]


def show(question: str) -> None:
    result = answer_question(question)
    pages = ", ".join(f"page {p}" for p in result.pages) or "-"
    print(f"\nQ: {question}")
    print(f"A: {result.answer}")
    print(f"   cited: {pages} | reason: {result.reason} | distances: "
          f"{['%.3f' % d for d in result.distances]}")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--all"]
    questions = ASSIGNMENT_QUESTIONS if "--all" in sys.argv or not args else [" ".join(args)]
    for question in questions:
        show(question)


if __name__ == "__main__":
    main()
