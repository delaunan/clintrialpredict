"""Conservative structured/text alignment checks for narrative review.

The checks here are intentionally small. They do not correct fields or score
quality; they only identify obvious situations that should be clarified before
running a full Quality Review.
"""

from __future__ import annotations

import re
from typing import Any


CLARIFICATION_NEEDED = "clarification_needed"


def _text_blob(packet: dict[str, Any]) -> str:
    text_context = packet.get("text_context") or {}
    parts = [str(value) for value in text_context.values() if value is not None]
    return " ".join(parts).lower()


def _feature_value(packet: dict[str, Any], field_id: str) -> str:
    structured = packet.get("structured_features") or {}
    display = packet.get("structured_feature_display_values") or {}
    value = display.get(field_id, structured.get(field_id, ""))
    return str(value or "").strip().lower()


def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _issue(
    *,
    issue_id: str,
    field_id: str,
    structured_value: str,
    text_signal: str,
    severity: str = CLARIFICATION_NEEDED,
) -> dict[str, Any]:
    return {
        "issue_id": issue_id,
        "field_id": field_id,
        "structured_value": structured_value,
        "text_signal": text_signal,
        "severity": severity,
    }


def detect_alignment_issues(packet: dict[str, Any]) -> list[dict[str, Any]]:
    """Return conservative material text/structured mismatch candidates."""
    iteration = packet.get("iteration_context") or {}
    try:
        iteration_number = int(iteration.get("iteration_number") or 0)
    except (TypeError, ValueError):
        iteration_number = 0
    if iteration_number <= 0:
        return []

    text = _text_blob(packet)
    if not text:
        return []

    issues: list[dict[str, Any]] = []

    placebo_value = _feature_value(packet, "has_placebo_ml")
    placebo_yes = placebo_value in {"1", "yes", "true", "placebo control"}
    placebo_no_text = _has_any(text, (r"\bno placebo\b", r"\bwithout placebo\b", r"\bplacebo-free\b"))
    if placebo_yes and placebo_no_text:
        issues.append(
            _issue(
                issue_id="placebo_text_structured_mismatch",
                field_id="has_placebo_ml",
                structured_value=placebo_value,
                text_signal="text appears to describe no placebo use",
            )
        )

    endpoint_structure = _feature_value(packet, "endpoint_structure_ml")
    multiple_endpoint_value = any(
        token in endpoint_structure
        for token in ("multiple", "co-primary", "coprimary", "multi", "composite")
    )
    single_endpoint_text = _has_any(text, (r"\bsingle primary endpoint\b", r"\bone primary endpoint\b"))
    if multiple_endpoint_value and single_endpoint_text:
        issues.append(
            _issue(
                issue_id="endpoint_structure_text_mismatch",
                field_id="endpoint_structure_ml",
                structured_value=endpoint_structure,
                text_signal="text appears to describe a single primary endpoint",
            )
        )

    return issues


def resolved_alignment_issue_ids(packet: dict[str, Any]) -> set[str]:
    """Return issue ids that have a user clarification attached."""
    context = packet.get("clarification_context") or {}
    clarifications = context.get("user_clarifications") or []
    resolved: set[str] = set()
    for item in clarifications:
        if not isinstance(item, dict):
            continue
        issue_id = str(item.get("issue_id") or "").strip()
        explanation = str(item.get("explanation") or "").strip()
        if issue_id and explanation:
            resolved.add(issue_id)
    return resolved


def unresolved_clarification_issues(packet: dict[str, Any]) -> list[dict[str, Any]]:
    """Return material issues that still require user clarification."""
    resolved = resolved_alignment_issue_ids(packet)
    return [
        issue
        for issue in detect_alignment_issues(packet)
        if issue.get("severity") == CLARIFICATION_NEEDED and issue.get("issue_id") not in resolved
    ]
