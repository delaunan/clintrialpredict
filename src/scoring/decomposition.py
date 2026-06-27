"""Shared Completion Score decomposition helpers.

The API and frontend both need to assemble Completion Score chart artifacts
from SHAP values, taxonomy metadata, thresholds, and display values. This module
owns that assembly so audit-mode API responses, local audit views, and local
simulator baselines stay structurally aligned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


DEFAULT_DISABLED_COLS = (
    "includes_us_ml",
    "is_fda_regulated_drug_ml",
    "gbd_cause_id_ml",
    "gbd_cause_id_2_ml",
    "gbd_cause_id_4_ml",
    "gbd_hierarchy_level_ml",
    "is_duration_unknown_ml",
    "target",
    "masking_ml",
    "therapeutic_area_ml",
    "strategic_ambition_ml",
    "intervention_model_ml",
)


def load_audit_decomposition_artifacts(
    *,
    model_path: str | Path,
    shap_path: str | Path,
    thresholds_path: str | Path,
) -> dict[str, Any]:
    """Load artifacts needed to assemble prerecorded audit decomposition."""
    model = joblib.load(model_path)
    shap_dict = joblib.load(shap_path)
    with Path(thresholds_path).open("r", encoding="utf-8") as f:
        thresholds = json.load(f)
    feature_names = list(model.named_steps["prep"].get_feature_names_out())
    return {
        "shap_dict": shap_dict,
        "thresholds": thresholds,
        "feature_names": feature_names,
    }


def _feature_prefix(field_id: str, field_meta: Mapping[str, Any], disabled_cols: set[str]) -> str:
    if field_id in disabled_cols:
        return ""
    encoding = field_meta.get("encoding")
    if encoding == "ordinal":
        return "ordinal__"
    if encoding == "target":
        return "target__"
    if encoding == "numeric":
        if "arms" in field_id:
            return "num_arms__"
        if "duration" in field_id:
            return "num_duration__"
    return ""


def _display_value(data: Mapping[str, Any], field_id: str) -> str:
    ui_col = field_id.replace("_ml", "_ui")
    if "gbd_cause_id_3" in field_id:
        ui_col = "gbd_indication_name_3"

    value = data.get(ui_col, data.get(field_id, "N/A"))
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    if isinstance(value, (float, int, np.integer, np.floating)):
        return f"{float(value):.1f}"
    return str(value) if str(value).strip() else "N/A"


def _pillar_order_from_registry(registry: Mapping[str, Any]) -> list[str]:
    pillars: list[str] = []
    for field_meta in registry.values():
        pillar = (field_meta.get("ui") or {}).get("pillar")
        if pillar and pillar not in pillars:
            pillars.append(pillar)
    return pillars


def build_completion_decomposition(
    *,
    data: Mapping[str, Any],
    shap_vals: Sequence[float],
    registry: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    feature_names: Sequence[str],
    mode: str,
    therapeutic_area: str | None = None,
    live_probability: float | None = None,
    disabled_cols: Sequence[str] = DEFAULT_DISABLED_COLS,
    pillar_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build gauge, impact-bar, and treemap inputs from one SHAP vector."""
    ta = str(therapeutic_area or data.get("therapeutic_area", "UNCLASSIFIED") or "UNCLASSIFIED").upper()
    threshold_logit = thresholds.get("ta_threshold_logits", {}).get(
        ta,
        thresholds.get("global_threshold_logit", 0.0),
    )
    gain_factor = thresholds.get("gain_factor", 25.0)
    intercept = thresholds.get("base_value", 0.0)

    disabled = {str(col) for col in disabled_cols}
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    sub_sums_raw: dict[tuple[str, str], float] = {}
    sub_features: dict[tuple[str, str], list[tuple[Any, str]]] = {}
    feature_level_impacts: list[dict[str, Any]] = []
    mapped_indices: set[int] = set()
    calibration_target: tuple[str, str] | None = None

    for field_id, field_meta in registry.items():
        ui = field_meta.get("ui", {})
        pillar = ui.get("pillar")
        subgroup = ui.get("subgroup")
        label = ui.get("label", field_id)
        if not pillar or not subgroup:
            continue
        if field_id == "therapeutic_area_ml":
            calibration_target = (pillar, subgroup)
        if pillar == "Metadata":
            continue

        impact = 0.0
        matched_model_feature = False
        prefixed_field = f"{_feature_prefix(field_id, field_meta, disabled)}{field_id}"
        for full_name, idx in feature_to_idx.items():
            if full_name == prefixed_field or full_name.startswith(f"{prefixed_field}_"):
                matched_model_feature = True
                impact += -float(shap_vals[idx]) * gain_factor
                mapped_indices.add(idx)

        rounded_feature_impact = round(impact, 1)
        if rounded_feature_impact == -0.0:
            rounded_feature_impact = 0.0
        if matched_model_feature and rounded_feature_impact != 0.0:
            feature_level_impacts.append({
                "Feature": field_id,
                "Label": label,
                "Value": _display_value(data, field_id),
                "Pillar": pillar,
                "Subcategory": subgroup,
                "Impact": rounded_feature_impact,
            })

        key = (pillar, subgroup)
        sub_sums_raw[key] = sub_sums_raw.get(key, 0.0) + impact
        sub_features.setdefault(key, []).append((
            ui.get("priority", 99),
            f"{label}: <b>{_display_value(data, field_id)}</b>",
        ))

    unmapped_indices = set(range(len(shap_vals))) - mapped_indices
    if unmapped_indices:
        unmapped_impact = sum(-float(shap_vals[idx]) * gain_factor for idx in unmapped_indices)
        key = ("Therapeutic Context", "Other Model Signals")
        sub_sums_raw[key] = sub_sums_raw.get(key, 0.0) + unmapped_impact
        sub_features.setdefault(key, []).append((
            999,
            f"Unmapped internal factors: <b>{len(unmapped_indices)}</b>",
        ))

    anchor_pillar, anchor_subcat = calibration_target or ("Therapeutic Context", "Therapeutic Area Profile")
    calibration_offset_pts = (threshold_logit - intercept) * gain_factor
    sub_sums_raw[(anchor_pillar, anchor_subcat)] = (
        sub_sums_raw.get((anchor_pillar, anchor_subcat), 0.0)
        + calibration_offset_pts
    )

    ordered_pillars = list(pillar_order or _pillar_order_from_registry(registry))
    pillar_totals = {pillar: 0.0 for pillar in ordered_pillars}
    final_subcats = []
    for (pillar, subgroup), raw_impact in sub_sums_raw.items():
        rounded_impact = round(raw_impact, 1)
        if rounded_impact == -0.0:
            rounded_impact = 0.0
        pillar_totals[pillar] = round(pillar_totals.get(pillar, 0.0) + rounded_impact, 1)

        narrative = ""
        for candidate_meta in registry.values():
            candidate_ui = candidate_meta.get("ui", {})
            if candidate_ui.get("pillar") == pillar and candidate_ui.get("subgroup") == subgroup:
                narrative = candidate_ui.get("pos_impact" if rounded_impact >= 0 else "neg_impact", "")
                break

        final_subcats.append({
            "Pillar": pillar,
            "Subcategory": subgroup,
            "Impact": rounded_impact,
            "Narrative": narrative,
            "FeatureDetails": [
                item[1]
                for item in sorted(sub_features.get((pillar, subgroup), []), key=lambda item: item[0])
            ],
        })

    total_impact_points = round(
        sum(value for pillar, value in pillar_totals.items() if pillar != "Metadata"),
        1,
    )
    final_score = round(float(np.clip(50.0 + total_impact_points, 1.0, 99.0)), 1)
    residual = round((final_score - 50.0) - total_impact_points, 1)

    if residual != 0:
        pillar_totals[anchor_pillar] = round(pillar_totals.get(anchor_pillar, 0.0) + residual, 1)
        for subcat in final_subcats:
            if subcat["Pillar"] == anchor_pillar and subcat["Subcategory"] == anchor_subcat:
                subcat["Impact"] = round(subcat["Impact"] + residual, 1)
                if subcat["Impact"] == -0.0:
                    subcat["Impact"] = 0.0
                break

    for pillar, value in list(pillar_totals.items()):
        if value == -0.0:
            pillar_totals[pillar] = 0.0

    return {
        "score": final_score,
        "threshold": 50.0,
        "pillar_impacts": [
            {"Pillar": pillar, "Impact": value}
            for pillar, value in pillar_totals.items()
            if pillar != "Metadata"
        ],
        "feature_impacts": final_subcats,
        "subcat_impacts": final_subcats,
        "feature_level_impacts": feature_level_impacts,
        "mode": mode,
        "probability": live_probability,
    }
