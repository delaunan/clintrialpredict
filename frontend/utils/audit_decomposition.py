"""Frontend access to prerecorded audit Completion Score decomposition."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from src.scoring.decomposition import (
    build_completion_decomposition,
    load_audit_decomposition_artifacts as load_shared_audit_decomposition_artifacts,
)


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "model_prod_01.joblib"
SHAP_PATH = PROJECT_ROOT / "models" / "shap_values_01.joblib"
THRESHOLDS_PATH = PROJECT_ROOT / "models" / "thresholds_01.json"


@st.cache_resource
def load_frontend_audit_decomposition_artifacts() -> dict[str, Any]:
    return load_shared_audit_decomposition_artifacts(
        model_path=MODEL_PATH,
        shap_path=SHAP_PATH,
        thresholds_path=THRESHOLDS_PATH,
    )


def build_prerecorded_audit_decomposition_result(
    row: Any,
    taxonomy: Mapping[str, Any],
    *,
    mode: str = "audit",
) -> dict[str, Any] | None:
    """Build a prerecorded audit result locally from saved SHAP artifacts."""
    nct_id = str(row.get("nct_id", "") or "").strip()
    if not nct_id:
        return None

    try:
        artifacts = load_frontend_audit_decomposition_artifacts()
        shap_vals = artifacts["shap_dict"].get(nct_id)
        if shap_vals is None:
            return None

        data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        return build_completion_decomposition(
            data=data,
            shap_vals=shap_vals,
            registry=taxonomy,
            thresholds=artifacts["thresholds"],
            feature_names=artifacts["feature_names"],
            mode=mode,
            therapeutic_area=str(row.get("therapeutic_area", "UNCLASSIFIED") or "UNCLASSIFIED"),
            live_probability=None,
        )
    except Exception:
        logger.exception("Prerecorded audit decomposition could not be built")
        return None
