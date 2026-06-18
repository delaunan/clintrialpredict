"""Pure data helpers for scenario-review plots."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd


def trace_allows_reality_check_display(trace: dict[str, Any] | None) -> bool:
    if not trace:
        return False
    if trace.get("hidden_baseline") or trace.get("participant_visible") is False:
        return False
    return trace.get("reality_check_points") is not None


def design_subcategory_impacts(trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not trace_allows_reality_check_display(trace):
        return []
    reality_assessment = (trace or {}).get("reality_check_assessment") or {}
    reality_points = pd.to_numeric(
        (trace or {}).get("reality_check_points", reality_assessment.get("points")),
        errors="coerce",
    )
    if reality_assessment and pd.notna(reality_points):
        allocation_rows = (trace or {}).get("reality_check_allocation_points") or []
        rows = []
        for allocation in allocation_rows:
            impact = pd.to_numeric(allocation.get("points"), errors="coerce")
            if pd.isna(impact):
                continue
            rows.append({
                "Pillar": allocation.get("pillar") or "Reality Check",
                "Subcategory": allocation.get("subpillar") or "Reality Check",
                "Impact": float(impact),
                "ShowImpactValue": True,
                "TreemapValue": abs(float(impact)) or 1.0,
                "FeatureDetails": [html.escape(str(allocation.get("rationale") or ""))],
            })
        if rows:
            return rows
        if (
            (trace or {}).get("reality_check_points") is not None
            or (trace or {}).get("operational_fit_points") is not None
        ):
            return []

    return []
