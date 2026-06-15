"""Pure data helpers for scenario-review plots."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd

from src.narratives.scoring import DESIGN_SUBCATEGORY_LABELS


def trace_allows_design_confidence_display(trace: dict[str, Any] | None) -> bool:
    if not trace:
        return False
    if trace.get("hidden_baseline") or trace.get("participant_visible") is False:
        return False
    return trace.get("design_confidence") is not None


def design_subcategory_impacts(trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not trace_allows_design_confidence_display(trace):
        return []
    assessment = (trace or {}).get("design_confidence_assessment") or {}
    validated_subcategories = (
        (trace or {}).get("design_confidence_subcategories")
        or ((trace or {}).get("validated_review") or {}).get("design_confidence_subcategories")
        or {}
    )

    def short_treemap_rationale(subcategory_name: str, subcategory: dict[str, Any]) -> str:
        fallback = validated_subcategories.get(str(subcategory_name)) if isinstance(validated_subcategories, dict) else {}
        raw = (
            str(subcategory.get("short_rationale") or "").strip()
            or str((fallback or {}).get("short_rationale") or "").strip()
            or str(subcategory.get("rationale") or "").strip()
            or str((fallback or {}).get("rationale") or "").strip()
        )
        raw = " ".join(raw.split())
        if len(raw) > 92:
            raw = raw[:89].rstrip(" .,;:") + "..."
        return raw

    rows = []
    for pillar in (assessment.get("pillars") or {}).values():
        pillar_label = str(pillar.get("label") or "Design Confidence")
        for subcategory_name, subcategory in (pillar.get("design_subcategories") or {}).items():
            impact = pd.to_numeric(subcategory.get("points"), errors="coerce")
            if pd.isna(impact):
                continue
            details = []
            short_rationale = short_treemap_rationale(str(subcategory_name), subcategory)
            if short_rationale:
                details.append(html.escape(short_rationale))
            rows.append({
                "Pillar": pillar_label,
                "Subcategory": DESIGN_SUBCATEGORY_LABELS.get(
                    str(subcategory_name),
                    str(subcategory_name).replace("_", " ").title(),
                ),
                "Impact": float(impact),
                "FeatureDetails": details,
            })
    return rows
