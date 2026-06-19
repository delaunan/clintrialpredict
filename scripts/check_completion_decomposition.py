#!/usr/bin/env python3
"""Focused checks for Completion Score decomposition outputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scoring.decomposition import build_completion_decomposition


def main() -> int:
    registry = {
        "phase_ml": {
            "encoding": "target",
            "ui": {
                "pillar": "Therapeutic Context",
                "subgroup": "Development Phase",
                "label": "Phase",
            },
        },
        "allocation_ml": {
            "encoding": "target",
            "ui": {
                "pillar": "Execution Framework",
                "subgroup": "Methodological Setup",
                "label": "Allocation",
            },
        },
        "therapeutic_area_ml": {
            "ui": {
                "pillar": "Therapeutic Context",
                "subgroup": "Therapeutic Area Profile",
                "label": "Therapeutic Area",
            },
        },
        "non_model_field_ml": {
            "encoding": "target",
            "ui": {
                "pillar": "Patient Profile",
                "subgroup": "Population Scope",
                "label": "Non-model Field",
            },
        },
    }
    result = build_completion_decomposition(
        data={
            "phase_ml": "PHASE2",
            "phase_ui": "Phase 2",
            "allocation_ml": "RANDOMIZED",
            "allocation_ui": "Randomized",
            "therapeutic_area": "INFECTIOUS_DISEASES",
        },
        shap_vals=[-0.20, 0.12, 0.50],
        registry=registry,
        thresholds={
            "global_threshold_logit": 0.0,
            "ta_threshold_logits": {"INFECTIOUS_DISEASES": 1.0},
            "gain_factor": 10.0,
            "base_value": 0.0,
        },
        feature_names=[
            "target__phase_ml",
            "target__allocation_ml",
            "internal_unmapped_signal",
        ],
        mode="unit_check",
        therapeutic_area="INFECTIOUS_DISEASES",
        pillar_order=["Therapeutic Context", "Execution Framework", "Patient Profile"],
    )

    feature_rows = result.get("feature_level_impacts") or []
    by_feature = {row.get("Feature"): row for row in feature_rows}
    errors: list[str] = []
    if set(by_feature) != {"phase_ml", "allocation_ml"}:
        errors.append("feature_level_impacts should include only fields matched to XGBoost feature columns")
    if by_feature.get("phase_ml", {}).get("Impact") != 2.0:
        errors.append("feature_level_impacts should use direct signed SHAP contribution for phase_ml")
    if by_feature.get("allocation_ml", {}).get("Impact") != -1.2:
        errors.append("feature_level_impacts should use direct signed SHAP contribution for allocation_ml")
    if "therapeutic_area_ml" in by_feature:
        errors.append("feature_level_impacts should not include therapeutic-area threshold correction")
    if "non_model_field_ml" in by_feature:
        errors.append("feature_level_impacts should not include registry fields absent from XGBoost features")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("Completion decomposition feature-level checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
