"""Pure data helpers for scenario-review plots."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd


def _reality_check_display_detail(allocation: dict[str, Any]) -> str:
    target_id = str(allocation.get("allocation_target_id") or "")
    points = pd.to_numeric(allocation.get("points"), errors="coerce")
    positive = bool(pd.notna(points) and float(points) > 0)
    labels = {
        "therapeutic_context.therapeutic_area_profile": ("Disease fit improved", "Disease-context concern"),
        "therapeutic_context.development_phase_and_goal": ("Phase fit improved", "Phase fit concern"),
        "scientific_challenge.biological_profile": ("Biology fit improved", "Biology concern"),
        "scientific_challenge.protocol_architecture": ("Evidence design strengthened", "Evidence design concern"),
        "patient_profile.clinical_severity": ("Patient fit improved", "Patient burden"),
        "patient_profile.population_scope": ("Population fit improved", "Population concern"),
        "execution_framework.trial_complexity_footprint": ("Execution realism improved", "Execution burden"),
        "execution_framework.methodological_setup": ("Methodology strengthened", "Methodology concern"),
        "execution_framework.operational_fit": ("Operational support improved", "Operational support gap"),
    }
    if target_id in labels:
        return labels[target_id][0 if positive else 1]
    fallback = "Realism support" if positive else "Realism concern"
    return str(allocation.get("short_explanation") or fallback)


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
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for allocation in allocation_rows:
            impact = pd.to_numeric(allocation.get("points"), errors="coerce")
            if pd.isna(impact):
                continue
            pillar = str(allocation.get("pillar") or "Reality Check")
            subcategory = str(allocation.get("subpillar") or "Reality Check")
            key = (pillar, subcategory)
            row = grouped.setdefault(
                key,
                {
                    "Pillar": pillar,
                    "Subcategory": subcategory,
                    "Impact": 0.0,
                    "ShowImpactValue": True,
                    "TreemapValue": 0.0,
                    "FeatureDetails": [],
                },
            )
            row["Impact"] = float(row["Impact"]) + float(impact)
            detail = html.escape(_reality_check_display_detail(allocation).strip())
            if detail and detail not in row["FeatureDetails"]:
                row["FeatureDetails"].append(detail)
        rows = []
        for row in grouped.values():
            row["Impact"] = round(float(row["Impact"]), 4)
            row["TreemapValue"] = abs(float(row["Impact"])) or 1.0
            rows.append(row)
        if rows:
            return rows
        if (
            (trace or {}).get("reality_check_points") is not None
            or (trace or {}).get("operational_fit_points") is not None
        ):
            return []

    return []
