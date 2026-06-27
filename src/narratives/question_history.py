"""Helpers for participant-visible strategic question history."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def participant_visible_question_entry(
    *,
    central_tension: dict[str, Any] | None,
    broader_strategic_question: dict[str, Any] | None,
    fallback_mapped_tension: str = "",
) -> dict[str, Any] | None:
    """Normalize the participant-visible wider debate question for continuity."""
    if not isinstance(broader_strategic_question, dict):
        return None
    question = str(broader_strategic_question.get("question") or "").strip()
    if not question:
        return None
    tension = central_tension if isinstance(central_tension, dict) else {}
    mapped_tension = str(
        broader_strategic_question.get("mapped_tension")
        or tension.get("summary")
        or fallback_mapped_tension
        or ""
    ).strip()
    return {
        "central_tension": deepcopy(tension),
        "broader_strategic_question": deepcopy(broader_strategic_question),
        "question": question,
        "mapped_tension": mapped_tension,
    }


def merge_participant_visible_question_history(
    previous_history: Any,
    current_entry: dict[str, Any] | None,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Append the current visible question, dedupe by question, and cap recency."""
    entries: list[dict[str, Any]] = []
    if isinstance(previous_history, list):
        for item in previous_history:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            if not question:
                nested_question = item.get("broader_strategic_question")
                if isinstance(nested_question, dict):
                    question = str(nested_question.get("question") or "").strip()
            if not question:
                continue
            normalized = deepcopy(item)
            normalized["question"] = question
            entries.append(normalized)

    if current_entry:
        current_question = str(current_entry.get("question") or "").strip()
        entries = [
            item
            for item in entries
            if str(item.get("question") or "").strip() != current_question
        ]
        entries.append(deepcopy(current_entry))

    return entries[-limit:]
