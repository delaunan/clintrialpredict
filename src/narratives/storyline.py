"""App-owned Trial Score storyline state helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def build_storyline_state(validated_review: dict[str, Any] | None) -> dict[str, Any]:
    """Build compact continuity state from a validated review."""
    validated_review = validated_review or {}
    strategic_review = validated_review.get("strategic_review") or {}
    reality_check = validated_review.get("reality_check") or {}
    central_tension = validated_review.get("central_tension_candidate") or {}
    continuity_update = validated_review.get("continuity_update") or {}
    continuity = validated_review.get("continuity") or {}
    active_tension = (
        central_tension.get("summary")
        or strategic_review.get("current_tension")
        or validated_review.get("main_tension")
        or continuity_update.get("active_tension")
        or ""
    )
    return {
        "active_tension": str(active_tension).strip(),
        "active_tension_status": str(strategic_review.get("tension_status") or "not_applicable"),
        "last_effect_label": str(reality_check.get("effect") or strategic_review.get("effect_label") or ""),
        "last_move_classification": _string_list(strategic_review.get("move_classification")),
        "protected_gains": _string_list(continuity.get("prior_concerns_resolved")),
        "regression_watch": _string_list(continuity.get("prior_concerns_worsened")),
        "active_carryover": _string_list(continuity.get("prior_concerns_unchanged")),
        "new_concerns": _string_list(continuity.get("new_concerns")),
        "next_consideration": str(
            continuity_update.get("watch_next")
            or strategic_review.get("next_consideration")
            or ""
        ).strip(),
        "storyline_update": str(
            continuity_update.get("what_changed")
            or continuity.get("storyline_update")
            or ""
        ).strip(),
    }


def merge_storyline_state(trace_or_context: dict[str, Any] | None) -> dict[str, Any]:
    """Return stored storyline state, deriving it from validated review when needed."""
    trace_or_context = trace_or_context or {}
    stored = trace_or_context.get("storyline_state")
    if isinstance(stored, dict) and stored:
        return deepcopy(stored)
    return build_storyline_state(trace_or_context.get("validated_review") or {})
