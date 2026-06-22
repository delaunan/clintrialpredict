"""Deterministic input-packet builder for narrative review.

The builder owns data assembly only. It does not call an LLM, validate LLM
output, calculate Trial Score, or mutate Streamlit session state.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.narratives.contract_fixtures import PROMPT_VERSION, RUBRIC_VERSION
from src.narratives.question_history import (
    merge_participant_visible_question_history,
    participant_visible_question_entry,
)
from src.narratives.storyline import merge_storyline_state
from src.narratives.trial_score_contract import REALITY_CHECK_CARRYOVER_MATERIALITY_THRESHOLD

MODE_EXISTING_STUDY = "existing_study"
FIELD_DICTIONARY_VERSION = "taxonomy_01_narrative_v1"
MODEL_FEATURE_EVIDENCE_LIMIT = 3

TRIAL_IDENTITY_KEYS = (
    "nct_id",
    "trial_label",
    "lead_sponsor_canonical",
    "start_year",
)

TEXT_CONTEXT_KEYS = (
    "title",
    "summary_ui",
    "conditions_ui",
    "primary_outcomes_ui",
    "interventions_ui",
)

STRUCTURED_FEATURE_KEYS = (
    "therapeutic_area_ml",
    "gbd_cause_id_3_ml",
    "is_rare_disease_ml",
    "phase_ml",
    "strategic_ambition_ml",
    "target_precedent_ml",
    "target_pathway_class_ml",
    "therapeutic_modality_ml",
    "innovation_tier_ml",
    "intervention_model_ml",
    "primary_purpose_ml",
    "adaptive_design_ml",
    "endpoint_rigor_ml",
    "endpoint_structure_ml",
    "biomarker_stratification_ml",
    "patient_severity_ml",
    "line_of_therapy_ml",
    "gender_ml",
    "healthy_volunteers_ml",
    "adult_ml",
    "child_ml",
    "older_adult_ml",
    "masking_ml",
    "allocation_ml",
    "has_dmc_ml",
    "has_placebo_ml",
    "comparator_benchmark_ml",
    "administration_complexity_ml",
    "number_of_arms_ml",
    "sponsor_tier_ml",
    "primary_duration_months_ml",
)

DIRECT_XGBOOST_SHAP_FIELDS = tuple(
    key
    for key in STRUCTURED_FEATURE_KEYS
    if key
    not in {
        "therapeutic_area_ml",
        "strategic_ambition_ml",
        "intervention_model_ml",
        "masking_ml",
    }
)

ACTIVE_OPERATIONAL_ASSUMPTION_KEYS = (
    "planned_enrollment",
    "planned_sites",
    "planned_duration_months",
)
OPERATIONAL_ASSUMPTION_DISPLAY_LABELS = {
    "planned_enrollment": "Planned Enrollment",
    "planned_sites": "Planned Sites",
    "planned_duration_months": "Planned Total Timeline",
}

REFERENCE_PACK_DIR = Path(__file__).resolve().parents[2] / "frontend" / "data" / "docs" / "narrative_reference_packs"
REFERENCE_PACK_MANIFEST = REFERENCE_PACK_DIR / "pack_manifest_v1.json"
THERAPEUTIC_AREA_PACK_DIR = REFERENCE_PACK_DIR
DEFAULT_REFERENCE_PACK_IDS = (
    "core_clinical_development_v1",
    "strategic_context_2026_v1",
    "ich_e8_quality_by_design_v1",
)
OPERATIONAL_REFERENCE_FIELDS = {
    "administration_complexity_ml",
    "has_dmc_ml",
    "number_of_arms_ml",
    "sponsor_tier_ml",
    "primary_duration_months_ml",
    "operational_assumptions.planned_enrollment",
    "operational_assumptions.planned_sites",
    "operational_assumptions.planned_duration_months",
}
ENDPOINT_STATISTICAL_REFERENCE_FIELDS = {
    "adaptive_design_ml",
    "allocation_ml",
    "biomarker_stratification_ml",
    "comparator_benchmark_ml",
    "endpoint_rigor_ml",
    "endpoint_structure_ml",
    "has_placebo_ml",
    "intervention_model_ml",
    "masking_ml",
    "number_of_arms_ml",
    "primary_duration_months_ml",
    "text_context.primary_outcomes_ui",
}
COMPACT_FIELD_MEANINGS = {
    "therapeutic_area_ml": "Clinical domain: disease context, endpoint norms, operational benchmarks, calibration limits.",
    "gbd_cause_id_3_ml": "Disease category: clinical context, patient relevance, feasibility, similar-trial comparisons.",
    "is_rare_disease_ml": "Rare-condition flag: feasible population size, recruitment difficulty, evidence expectations.",
    "phase_ml": "Development phase: evidence ambition, endpoint maturity, comparator strength, population scope.",
    "strategic_ambition_ml": "Development objective: learning, signal detection, or confirmatory evidence standard.",
    "target_precedent_ml": "Target precedent: biological risk and evidentiary burden.",
    "target_pathway_class_ml": "Pathway/mechanism class: plausibility, novelty, modality risk, endpoint fit.",
    "therapeutic_modality_ml": "Product modality: mechanism, delivery burden, safety oversight, site capability.",
    "innovation_tier_ml": "Innovation level: uncertainty, precedent, evidence burden, safeguard needs.",
    "intervention_model_ml": "Arm structure: comparison credibility, bias risk, burden, operational complexity.",
    "primary_purpose_ml": "Study purpose: treatment, prevention, supportive-care, or other decision question.",
    "adaptive_design_ml": "Adaptive/static design: flexibility, governance burden, inference complexity.",
    "endpoint_rigor_ml": "Endpoint rigor: clinical meaningfulness, bias risk, maturity, decision interpretability.",
    "endpoint_structure_ml": "Endpoint structure: primary-question clarity, multiplicity, component relevance.",
    "biomarker_stratification_ml": "Biomarker strategy: enrichment, treatment-effect clarity, recruitment feasibility.",
    "patient_severity_ml": "Patient severity: risk tolerance, endpoint relevance, ethical threshold.",
    "line_of_therapy_ml": "Treatment line: unmet need, comparator expectations, patient-selection fit.",
    "gender_ml": "Gender scope: target-population fit, generalizability, justified restrictions.",
    "healthy_volunteers_ml": "Healthy-volunteer flag: phase fit, safety tolerance, endpoint relevance.",
    "adult_ml": "Adult eligibility: population fit, generalizability, ethical threshold.",
    "child_ml": "Pediatric eligibility: safeguards, dosing uncertainty, endpoint relevance.",
    "older_adult_ml": "Older-adult eligibility: representativeness, comorbidity relevance, safety monitoring.",
    "masking_ml": "Masking: bias control, endpoint subjectivity, operational feasibility.",
    "allocation_ml": "Allocation: comparison credibility, selection bias, causal inference support.",
    "has_dmc_ml": "DMC/oversight: safety governance, risk proportionality, vulnerable-population protection.",
    "has_placebo_ml": "Placebo control: assay sensitivity, ethics, comparator credibility.",
    "comparator_benchmark_ml": "Comparator strategy: treatment-effect interpretability versus current care or control.",
    "administration_complexity_ml": "Administration complexity: site capability, participant burden, oversight needs.",
    "number_of_arms_ml": "Arm count: evidentiary breadth, multiplicity, recruitment and site burden.",
    "sponsor_tier_ml": "Sponsor scale proxy: execution capability and trial-footprint context only.",
    "primary_duration_months_ml": "Max Endpoint Duration: primary endpoint assessment time horizon, endpoint maturity, attrition risk.",
    "title": "Trial title: concise identity and high-level objective context.",
    "summary_ui": "Study summary: design rationale, intent, and structured-field coherence.",
    "conditions_ui": "Condition text: indication and population coherence.",
    "interventions_ui": "Intervention text: modality, mechanism, delivery complexity, comparator coherence.",
    "primary_outcomes_ui": "Primary outcome text: endpoint coherence, structure, timing, interpretability.",
}
def json_safe(value: Any) -> Any:
    """Return a deterministic JSON-serializable copy of common app values."""
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return deepcopy(value)


def stable_packet_hash(packet: dict[str, Any]) -> str:
    """Hash a packet for future cache/replay lookup."""
    payload = json.dumps(json_safe(packet), sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _scenario_state_payload(
    *,
    field_dictionary_version: Any,
    mode: Any,
    trial_identity: dict[str, Any],
    text_context: dict[str, Any],
    structured_features: dict[str, Any],
    operational_assumptions: dict[str, Any],
    completion_score: Any,
    pillar_impacts: list[Any],
    direct_xgboost_shap_fields: list[Any],
) -> dict[str, Any]:
    return {
        "field_dictionary_version": field_dictionary_version,
        "mode": mode,
        "trial_identity": trial_identity or {},
        "structured_features": structured_features or {},
        "text_context": text_context or {},
        "operational_assumptions": operational_assumptions or {},
        "model_interpretation": {
            "completion_score": completion_score,
            "pillar_impacts": pillar_impacts or [],
            "direct_xgboost_shap_fields": direct_xgboost_shap_fields or [],
        },
    }


def scenario_state_hash_from_packet(packet: dict[str, Any]) -> str:
    """Hash the current scenario state without storyline or iteration context."""
    model = packet.get("model_interpretation") or {}
    state_payload = _scenario_state_payload(
        field_dictionary_version=packet.get("field_dictionary_version"),
        mode=packet.get("mode"),
        trial_identity=packet.get("trial_identity") or {},
        text_context=packet.get("text_context") or {},
        structured_features=packet.get("structured_features") or {},
        operational_assumptions=packet.get("operational_assumptions") or {},
        completion_score=model.get("completion_score"),
        pillar_impacts=model.get("pillar_impacts") or [],
        direct_xgboost_shap_fields=model.get("direct_xgboost_shap_fields") or [],
    )
    return stable_packet_hash(state_payload)


def _baseline_scenario_state_hash(
    packet: dict[str, Any],
    baseline_snapshot: dict[str, Any] | None,
    *,
    baseline_text_context: dict[str, Any] | None = None,
    baseline_trial_identity: dict[str, Any] | None = None,
) -> str | None:
    if not baseline_snapshot:
        return None
    baseline_values = _snapshot_values(baseline_snapshot)
    state_payload = _scenario_state_payload(
        field_dictionary_version=packet.get("field_dictionary_version"),
        mode=packet.get("mode"),
        trial_identity=_select_keys(baseline_trial_identity or {}, TRIAL_IDENTITY_KEYS),
        text_context=_select_keys(baseline_text_context or {}, TEXT_CONTEXT_KEYS),
        structured_features=_select_keys(baseline_values, STRUCTURED_FEATURE_KEYS),
        operational_assumptions=_select_keys(
            baseline_snapshot.get("operational_assumptions") or {},
            ACTIVE_OPERATIONAL_ASSUMPTION_KEYS,
        ),
        completion_score=_completion_score(baseline_snapshot),
        pillar_impacts=_pillar_impacts(baseline_snapshot),
        direct_xgboost_shap_fields=list(DIRECT_XGBOOST_SHAP_FIELDS),
    )
    return stable_packet_hash(state_payload)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def _round_number(value: Any, digits: int = 3) -> float | None:
    numeric = _number(value)
    if numeric is None:
        return None
    return round(numeric, digits)


def _select_keys(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: json_safe(source.get(key)) for key in keys if _first_present(source.get(key)) is not None}


@lru_cache(maxsize=1)
def _taxonomy_field_meanings() -> dict[str, str]:
    taxonomy_path = Path(__file__).resolve().parents[2] / "models" / "taxonomy_01.json"
    try:
        taxonomy = json.loads(taxonomy_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    fields = taxonomy.get("FIELDS") or {}
    meanings: dict[str, str] = {}
    for field_id, field in fields.items():
        meaning = ((field or {}).get("ui") or {}).get("meaning")
        if isinstance(meaning, str) and meaning.strip():
            meanings[str(field_id)] = meaning.strip()
    return meanings


def _select_field_meanings(keys: tuple[str, ...]) -> dict[str, str]:
    meanings = _taxonomy_field_meanings()
    return {
        key: COMPACT_FIELD_MEANINGS.get(key, meanings[key])
        for key in keys
        if key in meanings or key in COMPACT_FIELD_MEANINGS
    }


def _extract_prompt_safe_summary(text: str) -> str:
    marker = "## Prompt-Safe Summary"
    if marker not in text:
        return ""
    summary = text.split(marker, 1)[1].strip()
    if "\n## " in summary:
        summary = summary.split("\n## ", 1)[0].strip()
    return " ".join(summary.split())


def _safe_therapeutic_area_filename(canonical_value: Any) -> str:
    raw = str(canonical_value or "").strip()
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in raw)
    return f"{safe or 'UNKNOWN'}.md"


def _therapeutic_area_context(structured_features: dict[str, Any]) -> dict[str, Any]:
    canonical_value = str(structured_features.get("therapeutic_area_ml") or "").strip()
    expected_filename = _safe_therapeutic_area_filename(canonical_value)
    expected_path = THERAPEUTIC_AREA_PACK_DIR / expected_filename
    context = {
        "canonical_value": canonical_value,
        "expected_filename": expected_filename,
        "pack_found": False,
        "pack_id": "",
        "prompt_safe_summary": "",
        "missing_pack_instruction": (
            "No therapeutic-area pack was found. Use cautious broad clinical-development knowledge only; "
            "do not invent specific disease, regulatory, efficacy, safety, prevalence, or cost facts."
        ),
    }
    try:
        if expected_path.parent.resolve() != THERAPEUTIC_AREA_PACK_DIR.resolve():
            return context
        text = expected_path.read_text()
    except OSError:
        return context
    summary = _extract_prompt_safe_summary(text) or " ".join(text.split())
    if not summary:
        return context
    context.update({
        "pack_found": True,
        "pack_id": f"therapeutic_area:{canonical_value}",
        "prompt_safe_summary": summary[:2500],
        "missing_pack_instruction": "",
    })
    return context


@lru_cache(maxsize=1)
def _reference_pack_catalog() -> dict[str, dict[str, Any]]:
    try:
        manifest = json.loads(REFERENCE_PACK_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    catalog: dict[str, dict[str, Any]] = {}
    for pack in manifest.get("packs", []):
        if not isinstance(pack, dict):
            continue
        pack_id = str(pack.get("pack_id") or "")
        filename = str(pack.get("filename") or "")
        if not pack_id or not filename:
            continue
        try:
            text = (REFERENCE_PACK_DIR / filename).read_text()
        except OSError:
            continue
        summary = _extract_prompt_safe_summary(text)
        if not summary:
            continue
        catalog[pack_id] = {
            "pack_id": pack_id,
            "role": pack.get("role"),
            "priority": pack.get("priority", 0),
            "tags": json_safe(pack.get("tags") or []),
            "prompt_safe_summary": summary,
        }
    return catalog


def _selected_reference_pack_ids(changed_fields: list[str]) -> list[str]:
    selected = list(DEFAULT_REFERENCE_PACK_IDS)
    changed = set(changed_fields)
    if changed.intersection(OPERATIONAL_REFERENCE_FIELDS):
        selected.append("ich_e6_r3_gcp_v1")
    if changed.intersection(ENDPOINT_STATISTICAL_REFERENCE_FIELDS):
        selected.extend(["ich_e9_r1_estimands_v1", "ich_e9_statistical_principles_v1"])
    seen: set[str] = set()
    unique = []
    for pack_id in selected:
        if pack_id not in seen:
            unique.append(pack_id)
            seen.add(pack_id)
    return unique[:5]


def _selected_reference_packs(changed_fields: list[str]) -> list[dict[str, Any]]:
    catalog = _reference_pack_catalog()
    return [
        catalog[pack_id]
        for pack_id in _selected_reference_pack_ids(changed_fields)
        if pack_id in catalog
    ]


def _merge_present_dicts(*sources: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in sources:
        for key, value in source.items():
            if _first_present(value) is not None:
                merged[key] = value
    return merged


def _snapshot_values(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    snapshot = snapshot or {}
    return (
        snapshot.get("structured_features")
        or snapshot.get("compare_values")
        or snapshot.get("submitted_values")
        or {}
    )


def _snapshot_display_values(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    snapshot = snapshot or {}
    return snapshot.get("display_values") or {}


def _snapshot_text_context(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    snapshot = snapshot or {}
    return snapshot.get("text_context") or {}


def _snapshot_trial_identity(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    snapshot = snapshot or {}
    identity = snapshot.get("trial_identity") or {}
    return {
        "nct_id": _first_present(identity.get("nct_id"), snapshot.get("nct_id")),
        "trial_label": identity.get("trial_label"),
        "lead_sponsor_canonical": identity.get("lead_sponsor_canonical"),
        "start_year": identity.get("start_year"),
    }


def _completion_score(snapshot: dict[str, Any] | None) -> int | float | None:
    snapshot = snapshot or {}
    score = _first_present(
        snapshot.get("score"),
        snapshot.get("model_interpretation", {}).get("completion_score"),
        snapshot.get("result", {}).get("score"),
    )
    return json_safe(score)


def _score_delta(current_snapshot: dict[str, Any], previous_snapshot: dict[str, Any] | None) -> int | float | None:
    explicit_delta = _first_present(
        current_snapshot.get("score_delta_points"),
        current_snapshot.get("model_interpretation", {}).get("score_delta"),
    )
    if explicit_delta is not None:
        return json_safe(explicit_delta)

    current = _completion_score(current_snapshot)
    previous = _completion_score(previous_snapshot)
    if isinstance(current, (int, float)) and isinstance(previous, (int, float)):
        return round(float(current) - float(previous), 1)
    return None


def _pillar_impacts(snapshot: dict[str, Any] | None) -> Any:
    snapshot = snapshot or {}
    return json_safe(
        _first_present(
            snapshot.get("pillar_impacts"),
            snapshot.get("model_interpretation", {}).get("pillar_impacts"),
            snapshot.get("result", {}).get("pillar_impacts"),
            {},
        )
    )


def _pillar_deltas(current_snapshot: dict[str, Any], previous_snapshot: dict[str, Any] | None) -> Any:
    explicit = current_snapshot.get("model_interpretation", {}).get("pillar_deltas")
    if explicit:
        return json_safe(explicit)

    current = _pillar_impacts(current_snapshot)
    previous = _pillar_impacts(previous_snapshot)
    if not isinstance(current, list) or not isinstance(previous, list):
        return {}

    previous_by_name = {item.get("Pillar"): item.get("Impact") for item in previous if isinstance(item, dict)}
    deltas: dict[str, float] = {}
    for item in current:
        if not isinstance(item, dict):
            continue
        pillar = item.get("Pillar")
        current_impact = item.get("Impact")
        previous_impact = previous_by_name.get(pillar)
        if isinstance(current_impact, (int, float)) and isinstance(previous_impact, (int, float)):
            delta = round(float(current_impact) - float(previous_impact), 1)
            if delta:
                deltas[str(pillar)] = delta
    return deltas


def _snapshot_feature_impacts(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    snapshot = snapshot or {}
    impacts = _first_present(
        snapshot.get("feature_impacts"),
        snapshot.get("subcat_impacts"),
        snapshot.get("model_interpretation", {}).get("feature_impacts"),
        snapshot.get("model_interpretation", {}).get("subcat_impacts"),
        snapshot.get("result", {}).get("feature_impacts"),
        snapshot.get("result", {}).get("subcat_impacts"),
        [],
    )
    return json_safe(impacts) if isinstance(impacts, list) else []


def _snapshot_feature_level_impacts(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    snapshot = snapshot or {}
    impacts = _first_present(
        snapshot.get("feature_level_impacts"),
        snapshot.get("model_interpretation", {}).get("feature_level_impacts"),
        snapshot.get("result", {}).get("feature_level_impacts"),
        [],
    )
    return json_safe(impacts) if isinstance(impacts, list) else []


def _impact_value(item: dict[str, Any]) -> float | None:
    value = _first_present(item.get("Impact"), item.get("impact"), item.get("value"))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _impact_index(
    snapshot: dict[str, Any] | None,
    *,
    level: str,
) -> dict[str, dict[str, Any]]:
    if level == "pillar":
        items = _pillar_impacts(snapshot)
        if isinstance(items, dict):
            return {
                str(name): {"name": str(name), "impact": float(value)}
                for name, value in items.items()
                if isinstance(value, (int, float))
            }
    else:
        items = _snapshot_feature_impacts(snapshot)

    if not isinstance(items, list):
        return {}

    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if level == "pillar":
            name = item.get("Pillar") or item.get("pillar")
            subcategory = None
        else:
            pillar = item.get("Pillar") or item.get("pillar")
            subcategory = item.get("Subcategory") or item.get("subcategory")
            name = f"{pillar}.{subcategory}" if pillar and subcategory else subcategory
        impact = _impact_value(item)
        if name is None or impact is None:
            continue
        indexed[str(name)] = {
            "name": str(name),
            "pillar": item.get("Pillar") or item.get("pillar"),
            "subcategory": subcategory,
            "impact": round(impact, 1),
        }
    return indexed


def _feature_impact_index(snapshot: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in _snapshot_feature_level_impacts(snapshot):
        if not isinstance(item, dict):
            continue
        feature = _first_present(item.get("Feature"), item.get("feature"), item.get("field"), item.get("Field"))
        impact = _impact_value(item)
        if feature is None or impact is None:
            continue
        indexed[str(feature)] = {
            "name": str(feature),
            "feature": str(feature),
            "label": _first_present(item.get("Label"), item.get("label"), item.get("display_label")),
            "value": _first_present(item.get("Value"), item.get("value"), item.get("display_value")),
            "pillar": item.get("Pillar") or item.get("pillar"),
            "subcategory": item.get("Subcategory") or item.get("subcategory") or item.get("subpillar"),
            "impact": round(impact, 1),
        }
    return indexed


def _impact_changes(
    current_snapshot: dict[str, Any],
    previous_snapshot: dict[str, Any] | None,
    baseline_snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for level in ("pillar", "subcategory"):
        current = _impact_index(current_snapshot, level=level)
        previous = _impact_index(previous_snapshot, level=level)
        baseline = _impact_index(baseline_snapshot, level=level)
        names = sorted(set(current) | set(previous) | set(baseline))
        for name in names:
            current_impact = current.get(name, {}).get("impact")
            previous_impact = previous.get(name, {}).get("impact")
            baseline_impact = baseline.get(name, {}).get("impact")
            if current_impact is None:
                continue

            delta_from_previous = None
            if previous_impact is not None:
                delta_from_previous = round(float(current_impact) - float(previous_impact), 1)

            delta_from_baseline = None
            if baseline_impact is not None:
                delta_from_baseline = round(float(current_impact) - float(baseline_impact), 1)

            if not delta_from_previous and not delta_from_baseline:
                continue

            source = current.get(name) or previous.get(name) or baseline.get(name) or {}
            changes.append({
                "impact_level": level,
                "name": name,
                "pillar": source.get("pillar") or (name if level == "pillar" else None),
                "subcategory": source.get("subcategory"),
                "baseline_impact": baseline_impact,
                "previous_impact": previous_impact,
                "current_impact": current_impact,
                "delta_from_previous": delta_from_previous,
                "delta_from_baseline": delta_from_baseline,
                "changed_since_previous": bool(delta_from_previous),
                "changed_from_baseline": bool(delta_from_baseline),
                "direction_from_previous": _impact_direction(delta_from_previous),
                "direction_from_baseline": _impact_direction(delta_from_baseline),
            })

    return sorted(
        changes,
        key=lambda item: max(
            abs(item.get("delta_from_previous") or 0),
            abs(item.get("delta_from_baseline") or 0),
        ),
        reverse=True,
    )


def _impact_direction(delta: float | None) -> str | None:
    if delta is None:
        return None
    if delta > 0:
        return "increased"
    if delta < 0:
        return "decreased"
    return "unchanged"


def _impact_sign(value: Any) -> str | None:
    numeric = _number(value)
    if numeric is None:
        return None
    if numeric > 0:
        return "positive"
    if numeric < 0:
        return "negative"
    return "neutral"


def _impact_state_label(value: Any) -> str | None:
    sign = _impact_sign(value)
    if sign == "positive":
        return "positive_state"
    if sign == "negative":
        return "negative_state"
    if sign == "neutral":
        return "neutral_state"
    return None


def _crossed_zero(start: Any, end: Any) -> bool | None:
    start_number = _number(start)
    end_number = _number(end)
    if start_number is None or end_number is None:
        return None
    return (start_number < 0 < end_number) or (start_number > 0 > end_number)


def _movement_label(start: Any, end: Any) -> str | None:
    start_number = _number(start)
    end_number = _number(end)
    if start_number is None or end_number is None:
        return None
    if abs(end_number - start_number) <= 1e-9:
        return "unchanged"
    if start_number < 0 < end_number:
        return "negative_to_positive"
    if start_number > 0 > end_number:
        return "positive_to_negative"
    if end_number > start_number:
        if end_number < 0:
            return "still_negative_but_improved"
        if start_number > 0:
            return "still_positive_and_improved"
        return "improved"
    if end_number < start_number:
        if end_number > 0:
            return "still_positive_but_weakened"
        if start_number < 0:
            return "still_negative_and_worsened"
        return "worsened"
    return "unchanged"


def _impact_state_items(snapshot: dict[str, Any] | None, *, level: str) -> list[dict[str, Any]]:
    indexed = _impact_index(snapshot, level=level)
    items: list[dict[str, Any]] = []
    for item in indexed.values():
        impact = _round_number(item.get("impact"), 1)
        if impact is None:
            continue
        items.append({
            "impact_level": "subpillar" if level == "subcategory" else level,
            "name": item.get("name"),
            "pillar": item.get("pillar") or (item.get("name") if level == "pillar" else None),
            "subpillar": item.get("subcategory"),
            "impact": impact,
            "impact_sign": _impact_sign(impact),
            "state": _impact_state_label(impact),
        })
    return items


def _top_signed_impacts(items: list[dict[str, Any]], *, sign: str, limit: int = 5) -> list[dict[str, Any]]:
    signed = [item for item in items if item.get("impact_sign") == sign]
    return sorted(signed, key=lambda item: abs(float(item.get("impact") or 0)), reverse=True)[:limit]


def _feature_state_items(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in _feature_impact_index(snapshot).values():
        impact = _round_number(item.get("impact"), 1)
        if impact is None:
            continue
        items.append({
            "impact_level": "feature",
            "feature": item.get("feature"),
            "label": item.get("label"),
            "value": item.get("value"),
            "pillar": item.get("pillar"),
            "subpillar": item.get("subcategory"),
            "impact": impact,
            "impact_sign": _impact_sign(impact),
            "state": _impact_state_label(impact),
        })
    return items


def _feature_movement_items(
    current_snapshot: dict[str, Any],
    previous_snapshot: dict[str, Any] | None,
    baseline_snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    current = _feature_impact_index(current_snapshot)
    previous = _feature_impact_index(previous_snapshot)
    baseline = _feature_impact_index(baseline_snapshot)
    items: list[dict[str, Any]] = []
    for name in sorted(set(current) | set(previous) | set(baseline)):
        current_impact = current.get(name, {}).get("impact", 0.0)
        previous_impact = previous.get(name, {}).get("impact")
        baseline_impact = baseline.get(name, {}).get("impact")
        delta_from_previous = (
            _round_number(float(current_impact) - float(previous_impact), 1)
            if previous_impact is not None
            else None
        )
        delta_from_baseline = (
            _round_number(float(current_impact) - float(baseline_impact), 1)
            if baseline_impact is not None
            else None
        )
        if not delta_from_previous and not delta_from_baseline:
            continue
        source = current.get(name) or previous.get(name) or baseline.get(name) or {}
        items.append({
            "impact_level": "feature",
            "name": name,
            "feature": source.get("feature") or name,
            "label": source.get("label"),
            "value": source.get("value"),
            "pillar": source.get("pillar"),
            "subpillar": source.get("subcategory"),
            "baseline_impact": _round_number(baseline_impact, 1),
            "previous_impact": _round_number(previous_impact, 1),
            "current_impact": _round_number(current_impact, 1),
            "current_state": _impact_state_label(current_impact),
            "delta_from_baseline": delta_from_baseline,
            "delta_from_previous": delta_from_previous,
            "movement_from_baseline": _movement_label(baseline_impact, current_impact),
            "movement_from_previous": _movement_label(previous_impact, current_impact),
            "crossed_zero_from_baseline": _crossed_zero(baseline_impact, current_impact),
            "crossed_zero_from_previous": _crossed_zero(previous_impact, current_impact),
        })
    return sorted(
        items,
        key=_movement_magnitude_sort_key,
        reverse=True,
    )


def _movement_magnitude_sort_key(item: dict[str, Any]) -> tuple[int, float]:
    previous_delta = item.get("delta_from_previous")
    if previous_delta is not None:
        return (1, abs(float(previous_delta or 0)))
    return (0, abs(float(item.get("delta_from_baseline") or 0)))


def _movement_direction_value(item: dict[str, Any]) -> float:
    previous_delta = item.get("delta_from_previous")
    if previous_delta is not None:
        return float(previous_delta or 0)
    return float(item.get("delta_from_baseline") or 0)


def _impact_movement_items(
    current_snapshot: dict[str, Any],
    previous_snapshot: dict[str, Any] | None,
    baseline_snapshot: dict[str, Any] | None,
    *,
    level: str,
) -> list[dict[str, Any]]:
    current = _impact_index(current_snapshot, level=level)
    previous = _impact_index(previous_snapshot, level=level)
    baseline = _impact_index(baseline_snapshot, level=level)
    items: list[dict[str, Any]] = []
    for name in sorted(set(current) | set(previous) | set(baseline)):
        current_impact = current.get(name, {}).get("impact")
        if current_impact is None:
            continue
        previous_impact = previous.get(name, {}).get("impact")
        baseline_impact = baseline.get(name, {}).get("impact")
        delta_from_previous = (
            _round_number(float(current_impact) - float(previous_impact), 1)
            if previous_impact is not None
            else None
        )
        delta_from_baseline = (
            _round_number(float(current_impact) - float(baseline_impact), 1)
            if baseline_impact is not None
            else None
        )
        if not delta_from_previous and not delta_from_baseline:
            continue
        source = current.get(name) or previous.get(name) or baseline.get(name) or {}
        items.append({
            "impact_level": "subpillar" if level == "subcategory" else level,
            "name": name,
            "pillar": source.get("pillar") or (name if level == "pillar" else None),
            "subpillar": source.get("subcategory"),
            "baseline_impact": _round_number(baseline_impact, 1),
            "previous_impact": _round_number(previous_impact, 1),
            "current_impact": _round_number(current_impact, 1),
            "current_state": _impact_state_label(current_impact),
            "delta_from_baseline": delta_from_baseline,
            "delta_from_previous": delta_from_previous,
            "movement_from_baseline": _movement_label(baseline_impact, current_impact),
            "movement_from_previous": _movement_label(previous_impact, current_impact),
            "crossed_zero_from_baseline": _crossed_zero(baseline_impact, current_impact),
            "crossed_zero_from_previous": _crossed_zero(previous_impact, current_impact),
        })
    return sorted(
        items,
        key=_movement_magnitude_sort_key,
        reverse=True,
    )


def _top_movements(items: list[dict[str, Any]], *, direction: str, limit: int = 5) -> list[dict[str, Any]]:
    if direction == "positive":
        filtered = [item for item in items if _movement_direction_value(item) > 0]
        return sorted(
            filtered,
            key=_movement_magnitude_sort_key,
            reverse=True,
        )[:limit]
    filtered = [item for item in items if _movement_direction_value(item) < 0]
    return sorted(
        filtered,
        key=_movement_magnitude_sort_key,
        reverse=True,
    )[:limit]


def _model_state_evidence(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    pillar_items = _impact_state_items(snapshot, level="pillar")
    subpillar_items = _impact_state_items(snapshot, level="subcategory")
    feature_items = _feature_state_items(snapshot)
    return json_safe({
        "completion_score": _completion_score(snapshot),
        "state_rule": (
            "State is the fixed snapshot of signed model forces. Positive impacts are favorable by definition; "
            "negative impacts are unfavorable by definition."
        ),
        "top_positive_pillar_impacts": _top_signed_impacts(pillar_items, sign="positive"),
        "top_negative_pillar_impacts": _top_signed_impacts(pillar_items, sign="negative"),
        "top_positive_subpillar_impacts": _top_signed_impacts(subpillar_items, sign="positive"),
        "top_negative_subpillar_impacts": _top_signed_impacts(subpillar_items, sign="negative"),
        "top_positive_feature_impacts": _top_signed_impacts(
            feature_items,
            sign="positive",
            limit=MODEL_FEATURE_EVIDENCE_LIMIT,
        ),
        "top_negative_feature_impacts": _top_signed_impacts(
            feature_items,
            sign="negative",
            limit=MODEL_FEATURE_EVIDENCE_LIMIT,
        ),
        "feature_impact_availability": (
            "direct_xgboost_feature_impacts_available"
            if feature_items
            else "not_available_no_direct_xgboost_feature_impacts"
        ),
        "feature_driver_names": {
            "top_positive_feature_drivers": _feature_driver_values(snapshot or {}, "top_positive_feature_drivers"),
            "top_negative_feature_drivers": _feature_driver_values(snapshot or {}, "top_negative_feature_drivers"),
        },
    })


def _model_movement_evidence(
    current_snapshot: dict[str, Any],
    previous_snapshot: dict[str, Any] | None,
    baseline_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    pillar_items = _impact_movement_items(current_snapshot, previous_snapshot, baseline_snapshot, level="pillar")
    subpillar_items = _impact_movement_items(current_snapshot, previous_snapshot, baseline_snapshot, level="subcategory")
    feature_items = _feature_movement_items(current_snapshot, previous_snapshot, baseline_snapshot)
    movement_available = bool(pillar_items or subpillar_items or feature_items)
    return json_safe({
        "available": movement_available,
        "movement_rule": (
            "Movement is the change in signed model forces. Positive deltas are more favorable; negative deltas are "
            "less favorable. Interpret movement together with current_state."
        ),
        "top_positive_pillar_movements": _top_movements(pillar_items, direction="positive"),
        "top_negative_pillar_movements": _top_movements(pillar_items, direction="negative"),
        "top_positive_subpillar_movements": _top_movements(subpillar_items, direction="positive"),
        "top_negative_subpillar_movements": _top_movements(subpillar_items, direction="negative"),
        "top_positive_feature_movements": _top_movements(
            feature_items,
            direction="positive",
            limit=MODEL_FEATURE_EVIDENCE_LIMIT,
        ),
        "top_negative_feature_movements": _top_movements(
            feature_items,
            direction="negative",
            limit=MODEL_FEATURE_EVIDENCE_LIMIT,
        ),
        "feature_movement_availability": (
            "direct_xgboost_feature_movements_available"
            if feature_items
            else "not_available_no_direct_xgboost_feature_movement"
        ),
    })


def _model_signal_guidance() -> dict[str, Any]:
    return {
        "baseline_rule": (
            "For hidden baseline, derive main_model_signals from current_model_state_evidence only; movement "
            "evidence may be empty."
        ),
        "visible_iteration_rule": (
            "For visible iterations, prioritize model_movement_evidence from the previous iteration, then use "
            "current_model_state_evidence as the current-state anchor."
        ),
        "first_visible_iteration_rule": (
            "For first visible iteration, movement from baseline is the relevant movement context because no prior "
            "visible iteration exists."
        ),
        "later_visible_iteration_rule": (
            "For later visible iterations, previous-iteration movement is the primary ranking signal; baseline "
            "movement is context for accumulated drift."
        ),
        "granularity_rule": (
            "Prefer feature-level evidence when available and include parent subpillar and pillar. Fall back to "
            "subpillar, then pillar."
        ),
        "main_model_signals_rule": (
            "Populate main_model_signals with concrete packet-backed signals, not generic pillar slogans. "
            "Movement explains what changed; state explains what still matters."
        ),
        "interpretation_rule": (
            "Positive impacts and deltas are favorable by definition; negative impacts and deltas are unfavorable "
            "by definition."
        ),
        "preferred_signal_format": (
            "Feature Label: Value under Pillar / Subpillar with signed current impact or signed delta, for example "
            "'DMC Involvement Status: Yes under Execution Framework / Methodological Setup (-5.7)' or "
            "'Maximum Primary Endpoint Duration: 38.0 months under Execution Framework / Trial Complexity Footprint (-5.4)'. "
            "Do not emit bare feature values without labels."
        ),
        "avoid": [
            "Scientific Challenge alignment",
            "Patient Profile fit",
            "Execution Framework constraints",
            "pillar-only phrases when feature or subpillar evidence is available",
        ],
    }


def _feature_driver_values(snapshot: dict[str, Any], key: str) -> list[Any]:
    interpretation = snapshot.get("model_interpretation", {})
    value = interpretation.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return json_safe(value)
    return [json_safe(value)]


def _changed_fields(current_snapshot: dict[str, Any]) -> list[str]:
    changed = list(current_snapshot.get("changed_fields") or [])
    changed.extend(
        f"operational_assumptions.{key}"
        for key in current_snapshot.get("changed_operational_assumptions") or []
    )
    changed.extend(
        f"text_context.{key}"
        for key in current_snapshot.get("changed_text_context_fields") or []
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for field in changed:
        field = str(field)
        if field in seen:
            continue
        seen.add(field)
        ordered.append(field)
    return ordered


def _changed_field_entry(
    field_id: str,
    *,
    change_type: str,
    current_value: Any,
    previous_value: Any,
    baseline_value: Any,
    current_label: Any = None,
    previous_label: Any = None,
    baseline_label: Any = None,
    display_label: str | None = None,
) -> dict[str, Any]:
    entry = {
        "field": field_id,
        "change_type": change_type,
        "baseline_value": json_safe(baseline_value),
        "baseline_label": json_safe(_first_present(baseline_label, baseline_value)),
        "previous_value": json_safe(previous_value),
        "previous_label": json_safe(_first_present(previous_label, previous_value)),
        "current_value": json_safe(current_value),
        "current_label": json_safe(_first_present(current_label, current_value)),
        "changed_by_user": True,
    }
    if display_label:
        entry["display_label"] = display_label
    return entry


def _changed_terms(previous_value: Any, current_value: Any) -> tuple[list[str], list[str]]:
    previous_terms = {
        term.strip(".,;:()[]{}").lower()
        for term in str(previous_value or "").split()
        if len(term.strip(".,;:()[]{}")) >= 4
    }
    current_terms = {
        term.strip(".,;:()[]{}").lower()
        for term in str(current_value or "").split()
        if len(term.strip(".,;:()[]{}")) >= 4
    }
    added = sorted(current_terms - previous_terms)[:20]
    removed = sorted(previous_terms - current_terms)[:20]
    return added, removed


def _text_change_evidence(
    current_snapshot: dict[str, Any],
    previous_snapshot: dict[str, Any] | None,
    baseline_snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    current_text = _snapshot_text_context(current_snapshot)
    previous_text = _snapshot_text_context(previous_snapshot)
    baseline_text = _snapshot_text_context(baseline_snapshot)
    for field in _changed_fields(current_snapshot):
        if not field.startswith("text_context."):
            continue
        text_key = field.split(".", 1)[1]
        previous_value = previous_text.get(text_key)
        current_value = current_text.get(text_key)
        added, removed = _changed_terms(previous_value, current_value)
        changed = str(previous_value or "") != str(current_value or "")
        change_type = "minor_cleanup"
        if changed and (added or removed):
            change_type = "new_information"
        evidence.append({
            "field": field,
            "changed": changed,
            "baseline_excerpt": str(baseline_text.get(text_key) or "")[:500],
            "previous_excerpt": str(previous_value or "")[:500],
            "current_excerpt": str(current_value or "")[:500],
            "changed_terms_added": added,
            "changed_terms_removed": removed,
            "change_type": change_type,
        })
    return evidence


def _operational_baseline_confidence(source: Any) -> str:
    source_text = str(source or "").strip()
    if source_text in {
        "completed_registry_facility_count",
        "final_observed_value",
        "completed_actual_primary_completion",
        "final_observed_total_duration",
    }:
        return "high"
    if source_text in {
        "planned_value",
        "estimated_planned_total_duration",
        "current_registry_facility_count_proxy",
        "observed_lower_bound",
        "observed_to_date_lower_bound",
        "actual_primary_completion",
        "actual_total_completion_lower_bound",
        "model_default",
        "benchmark_default",
        "benchmark_default_with_floors",
        "benchmark_imputed_default",
        "benchmark_imputed_default_with_observed_lower_bound",
        "enrollment_coherent_benchmark_default",
        "same_cohort_benchmark",
    }:
        return "medium"
    return "low" if source_text else "unknown"


def _movement_magnitude(baseline_value: float | None, current_value: float | None) -> str:
    if baseline_value is None or current_value is None:
        return "not_comparable"
    change = abs(current_value - baseline_value)
    relative = change / max(abs(baseline_value), 1.0)
    if change <= 1e-9:
        return "none"
    if relative < 0.1:
        return "minor"
    if relative < 0.35:
        return "moderate"
    if relative < 0.75:
        return "major"
    return "extreme"


def _movement_direction(baseline_value: float | None, current_value: float | None) -> str:
    if baseline_value is None or current_value is None:
        return "not_comparable"
    delta = current_value - baseline_value
    if abs(delta) <= 1e-9:
        return "no_change"
    return "increased" if delta > 0 else "decreased"


def _movement_relative_to_p50(
    baseline_value: float | None,
    current_value: float | None,
    p50: float | None,
) -> str:
    if baseline_value is None or current_value is None or p50 is None:
        return "not_available"
    baseline_distance = abs(baseline_value - p50)
    current_distance = abs(current_value - p50)
    if abs(current_distance - baseline_distance) <= 1e-9:
        return "unchanged_distance_to_p50"
    return "toward_p50" if current_distance < baseline_distance else "away_from_p50"


def _benchmark_context_id(assumption: dict[str, Any]) -> Any:
    return _first_present(
        assumption.get("benchmark_snapshot_id"),
        assumption.get("patients_per_site_benchmark_snapshot_id"),
        assumption.get("operational_benchmark_snapshot_id"),
        assumption.get("benchmark_level_used"),
        assumption.get("patients_per_site_benchmark_level_used"),
    )


def _benchmark_context_changed(
    baseline_assumption: dict[str, Any],
    current_assumption: dict[str, Any],
) -> bool | None:
    baseline_id = _benchmark_context_id(baseline_assumption)
    current_id = _benchmark_context_id(current_assumption)
    if baseline_id is None or current_id is None:
        return None
    return str(baseline_id) != str(current_id)


def _operational_status_key(status_kind: str) -> str:
    return {
        "enrollment": "enrollment_status",
        "site_count": "site_count_status",
        "duration": "duration_status",
        "patients_per_site": "patients_per_site_status",
    }.get(status_kind, "benchmark_status")


def _operational_percentiles(
    assumption: dict[str, Any],
    *,
    percentile_prefix: str,
    status_kind: str,
) -> dict[str, Any]:
    return {
        "p25": _round_number(assumption.get(f"{percentile_prefix}_p25")),
        "p50": _round_number(assumption.get(f"{percentile_prefix}_p50")),
        "p75": _round_number(assumption.get(f"{percentile_prefix}_p75")),
        "p90": _round_number(assumption.get(f"{percentile_prefix}_p90")),
        "status": assumption.get(_operational_status_key(status_kind)),
    }


def _operational_value_context(
    field_key: str,
    *,
    baseline_assumption: dict[str, Any],
    current_assumption: dict[str, Any],
    value_key: str = "value",
    percentile_prefix: str = "benchmark",
    status_kind: str = "benchmark",
    value_origin: str = "direct_operational_assumption",
) -> dict[str, Any]:
    baseline_value = _number(baseline_assumption.get(value_key))
    current_value = _number(current_assumption.get(value_key))
    current_p50 = _number(current_assumption.get(f"{percentile_prefix}_p50"))
    baseline_source = _first_present(
        baseline_assumption.get("source"),
        baseline_assumption.get("site_default_basis"),
        "not_available",
    )
    return {
        "field": field_key,
        "value_origin": value_origin,
        "baseline": {
            "value": _round_number(baseline_value),
            "source": baseline_source,
            "confidence": _operational_baseline_confidence(baseline_source),
            "is_neutral_reference": True,
            "benchmark_position": _operational_percentiles(
                baseline_assumption,
                percentile_prefix=percentile_prefix,
                status_kind=status_kind,
            ),
        },
        "current": {
            "value": _round_number(current_value),
            "source": current_assumption.get("source"),
            "benchmark_position": _operational_percentiles(
                current_assumption,
                percentile_prefix=percentile_prefix,
                status_kind=status_kind,
            ),
        },
        "movement_from_baseline": {
            "direction": _movement_direction(baseline_value, current_value),
            "absolute_change": _round_number(
                None if baseline_value is None or current_value is None else current_value - baseline_value
            ),
            "relative_change": _round_number(
                None
                if baseline_value is None or current_value is None
                else (current_value - baseline_value) / max(abs(baseline_value), 1.0)
            ),
            "magnitude": _movement_magnitude(baseline_value, current_value),
            "relative_to_p50": _movement_relative_to_p50(baseline_value, current_value, current_p50),
        },
        "benchmark_context": {
            "baseline_context_id": _benchmark_context_id(baseline_assumption),
            "current_context_id": _benchmark_context_id(current_assumption),
            "changed_from_baseline": _benchmark_context_changed(baseline_assumption, current_assumption),
            "interpretation": (
                "If benchmark context changed, separate scenario-value movement from benchmark-cohort movement before "
                "rating Operational Fit."
            ),
        },
        "interpretation_rule": (
            "Treat the baseline value as the neutral starting assumption. Use benchmark percentiles as residual "
            "context that can counterbalance movement size; do not penalize or credit absolute distance from P50 alone."
        ),
    }


def _operational_movement_context(
    current_snapshot: dict[str, Any],
    baseline_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    current_operational = current_snapshot.get("operational_assumptions") or {}
    baseline_operational = (baseline_snapshot or current_snapshot).get("operational_assumptions") or {}
    current_sites = current_operational.get("planned_sites") or {}
    baseline_sites = baseline_operational.get("planned_sites") or {}
    return json_safe({
        "baseline_is_neutral_reference": True,
        "scoring_rule": (
            "Operational Fit scores movement and coherence versus the neutral baseline first. Residual benchmark "
            "position and percentile distance provide context and may counterbalance a large move from baseline."
        ),
        "fields": {
            "planned_enrollment": _operational_value_context(
                "planned_enrollment",
                baseline_assumption=baseline_operational.get("planned_enrollment") or {},
                current_assumption=current_operational.get("planned_enrollment") or {},
                percentile_prefix="benchmark",
                status_kind="enrollment",
            ),
            "planned_sites": _operational_value_context(
                "planned_sites",
                baseline_assumption=baseline_sites,
                current_assumption=current_sites,
                percentile_prefix="benchmark",
                status_kind="site_count",
            ),
            "patients_per_site": _operational_value_context(
                "patients_per_site",
                baseline_assumption=baseline_sites,
                current_assumption=current_sites,
                value_key="patients_per_site_value",
                percentile_prefix="patients_per_site",
                status_kind="patients_per_site",
                value_origin="calculated_from_enrollment_and_sites",
            ),
            "planned_duration_months": _operational_value_context(
                "planned_duration_months",
                baseline_assumption=baseline_operational.get("planned_duration_months") or {},
                current_assumption=current_operational.get("planned_duration_months") or {},
                percentile_prefix="benchmark",
                status_kind="duration",
            ),
        },
    })


def _field_changes(
    current_snapshot: dict[str, Any],
    previous_snapshot: dict[str, Any] | None,
    baseline_snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    changed_fields = _changed_fields(current_snapshot)

    current_values = _snapshot_values(current_snapshot)
    previous_values = _snapshot_values(previous_snapshot)
    baseline_values = _snapshot_values(baseline_snapshot)
    current_display = _snapshot_display_values(current_snapshot)
    previous_display = _snapshot_display_values(previous_snapshot)
    baseline_display = _snapshot_display_values(baseline_snapshot)

    current_text = _snapshot_text_context(current_snapshot)
    previous_text = _snapshot_text_context(previous_snapshot)
    baseline_text = _snapshot_text_context(baseline_snapshot)

    current_operational = current_snapshot.get("operational_assumptions") or {}
    previous_operational = (previous_snapshot or {}).get("operational_assumptions") or {}
    baseline_operational = (baseline_snapshot or {}).get("operational_assumptions") or {}

    for field in changed_fields:
        if field.startswith("text_context."):
            text_key = field.split(".", 1)[1]
            changes.append(_changed_field_entry(
                field,
                change_type="text_context",
                current_value=current_text.get(text_key),
                previous_value=previous_text.get(text_key),
                baseline_value=baseline_text.get(text_key),
            ))
            continue

        if field.startswith("operational_assumptions."):
            assumption_key = field.split(".", 1)[1]
            changes.append(_changed_field_entry(
                field,
                change_type="operational_assumption",
                current_value=current_operational.get(assumption_key),
                previous_value=previous_operational.get(assumption_key),
                baseline_value=baseline_operational.get(assumption_key),
                display_label=OPERATIONAL_ASSUMPTION_DISPLAY_LABELS.get(assumption_key),
                previous_label=((previous_snapshot or {}).get("previous_operational_display_values") or {}).get(
                    assumption_key
                ),
            ))
            continue

        changes.append(_changed_field_entry(
            field,
            change_type="structured_feature",
            current_value=current_values.get(field),
            previous_value=previous_values.get(field),
            baseline_value=baseline_values.get(field),
            current_label=current_display.get(field),
            previous_label=previous_display.get(field),
            baseline_label=baseline_display.get(field),
        ))

    return changes


def _snapshot_id(snapshot: dict[str, Any] | None, fallback: str | None = None) -> str | None:
    snapshot = snapshot or {}
    return _first_present(snapshot.get("snapshot_id"), snapshot.get("current_snapshot_id"), snapshot.get("timestamp"), fallback)


def _iteration_number(current_snapshot: dict[str, Any], previous_snapshot: dict[str, Any] | None) -> int:
    explicit = current_snapshot.get("iteration_context", {}).get("iteration_number")
    if isinstance(explicit, int):
        return explicit
    if previous_snapshot:
        previous_iteration = previous_snapshot.get("iteration_context", {}).get("iteration_number")
        if isinstance(previous_iteration, int):
            return previous_iteration + 1
    source = str(current_snapshot.get("source") or "")
    return 0 if source == "prerecorded_baseline" else 1


def _compact_review_context(
    trace: dict[str, Any] | None,
    *,
    include_quality_scores: bool = True,
) -> dict[str, Any] | None:
    if not trace:
        return None
    if trace.get("status") not in {"reviewed", "reused_previous_review"}:
        return None

    validated = trace.get("validated_review") or {}
    continuity = validated.get("continuity") or {}
    participant = validated.get("key_questions") or validated.get("participant_review") or {}
    completion_outlook = validated.get("completion_outlook_analysis") or validated.get("completion_outlook_review") or {}
    operational_fit = validated.get("operational_fit") or trace.get("operational_fit") or {}
    draft = validated.get("analytical_narrative_draft") or {}
    development_landscape = str(draft.get("development_landscape_read") or "").strip()
    development_discussion_options = (
        validated.get("development_discussion_options") or trace.get("development_discussion_options") or []
    )
    participant_central_tension = trace.get("participant_central_tension") or {}
    participant_broader_question = trace.get("participant_broader_strategic_question") or {}
    trial_score = trace.get("trial_score")
    storyline_state = merge_storyline_state(trace)
    completion_summary = (
        completion_outlook.get("risk_pattern_summary")
        or completion_outlook.get("score_delta_summary")
        or completion_outlook.get("summary")
    )
    central_tension_summary = (
        participant_central_tension.get("summary")
        or trace.get("central_tension")
    )
    compact_storyline_memory = trace.get("compact_storyline_memory") or ""
    if not include_quality_scores:
        central_tension_summary = ""
        development_discussion_options = []
        participant_central_tension = {}
        participant_broader_question = {}
        storyline_state = deepcopy(storyline_state)
        storyline_state["active_tension"] = ""
        storyline_state["active_tension_status"] = "not_applicable"
        next_watch = (
            storyline_state.get("next_consideration")
            or (validated.get("continuity_update") or {}).get("watch_next")
            or ""
        )
        if next_watch:
            compact_storyline_memory = f"Baseline watch: {next_watch}"
        elif development_landscape:
            compact_storyline_memory = f"Baseline orientation: {development_landscape[:220]}"
    participant_visible_question_history = trace.get("recent_participant_visible_questions")
    if not include_quality_scores:
        participant_visible_question_history = []
    elif not isinstance(participant_visible_question_history, list):
        participant_visible_question_history = merge_participant_visible_question_history(
            [],
            participant_visible_question_entry(
                central_tension=participant_central_tension,
                broader_strategic_question=participant_broader_question,
                fallback_mapped_tension=str(central_tension_summary or ""),
            ),
        )
    compact = {
        "input_hash": trace.get("input_hash"),
        "iteration_id": trace.get("iteration_id"),
        "status": trace.get("status"),
        "validation_status": trace.get("validation_status"),
        "trial_score": trial_score if include_quality_scores else None,
        "operational_fit_points": trace.get("operational_fit_points") if include_quality_scores else None,
        "pre_reality_score": trace.get("pre_reality_score") if include_quality_scores else None,
        "reality_check_points": trace.get("reality_check_points") if include_quality_scores else None,
        "reality_check_assessment": deepcopy(trace.get("reality_check_assessment") or {}),
        "operational_fit": deepcopy(operational_fit),
        "storyline_state": deepcopy(storyline_state),
        "design_numeric_context": "visible_review" if include_quality_scores else "hidden_baseline_qualitative_only",
        "changed_fields": trace.get("changed_fields") or [],
        "score_delta": trace.get("score_delta", trace.get("score_movement")),
        "completion_outlook_summary": completion_summary,
        "central_tension": central_tension_summary,
        "development_discussion_options": deepcopy(development_discussion_options),
        "participant_central_tension": deepcopy(participant_central_tension),
        "participant_broader_strategic_question": deepcopy(participant_broader_question),
        "recent_participant_visible_questions": deepcopy(participant_visible_question_history),
        "key_questions": {
            "completion_outlook_summary": completion_outlook.get("risk_pattern_summary"),
            "medical_clinical_development_question": (
                participant.get("medical_clinical_development_question")
                or participant.get("medical_development_question")
            ),
            "strategic_development_question": (
                participant.get("strategic_development_question")
                or participant.get("strategic_field_question")
                or participant.get("clinical_operations_question")
                or participant.get("clinops_execution_question")
            ),
            "medical_development_question": (
                participant.get("medical_development_question")
                or participant.get("medical_clinical_development_question")
            ),
            "clinical_operations_question": participant.get("clinical_operations_question")
            or participant.get("clinops_execution_question"),
            "strategic_field_question": (
                participant.get("strategic_field_question")
                or participant.get("strategic_development_question")
            ),
        },
        "continuity": {
            "prior_concerns_resolved": continuity.get("prior_concerns_resolved") or [],
            "prior_concerns_worsened": continuity.get("prior_concerns_worsened") or [],
            "prior_concerns_unchanged": continuity.get("prior_concerns_unchanged") or [],
            "new_concerns": continuity.get("new_concerns") or [],
            "storyline_update": continuity.get("storyline_update"),
        },
        "compact_storyline_memory": compact_storyline_memory,
    }

    if not include_quality_scores:
        compact["baseline_completion_outlook_summary"] = completion_summary
        compact["baseline_development_landscape"] = development_landscape
        compact["baseline_consistency_flags"] = {}

    return json_safe(compact)


def _trial_score_continuity(previous_review_trace: dict[str, Any] | None) -> dict[str, Any]:
    previous = _compact_review_context(previous_review_trace)
    if not previous:
        return {
            "available": False,
            "reason": "first_visible_iteration_or_no_prior_visible_review",
            "active_tension": None,
            "previous_trial_score": None,
        }

    storyline_state = previous.get("storyline_state") or {}
    reality_check = previous.get("reality_check_assessment") or {}
    return json_safe({
        "available": True,
        "source_iteration_id": previous.get("iteration_id"),
        "source_input_hash": previous.get("input_hash"),
        "previous_trial_score": previous.get("trial_score"),
        "previous_pre_reality_score": previous.get("pre_reality_score"),
        "previous_operational_fit_points": previous.get("operational_fit_points"),
        "previous_reality_check_points": previous.get("reality_check_points"),
        "active_tension": storyline_state.get("active_tension") or previous.get("central_tension"),
        "last_reality_check_effect": reality_check.get("effect"),
        "protected_gains": storyline_state.get("protected_gains") or [],
        "regression_watch": storyline_state.get("regression_watch") or [],
        "next_consideration": storyline_state.get("next_consideration"),
        "storyline_update": storyline_state.get("storyline_update"),
        "instruction": (
            "Use this compact state to decide whether the latest move resolves, preserves, "
            "reopens, supersedes, or leaves active prior Trial Score discussion topics."
        ),
    })


def _norm_for_state_compare(value: Any) -> str:
    return str(value if value is not None else "").strip().lower()


def _carryover_state_precheck(
    previous_assessment: dict[str, Any],
    field_changes: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(previous_assessment, dict) or not field_changes:
        return {"status": "not_evaluable", "reason": "No field-level carryover precheck was available."}

    evidence_fields = set()
    for field in previous_assessment.get("supported_evidence_fields") or previous_assessment.get("evidence_fields") or []:
        if isinstance(field, str) and field.strip():
            evidence_fields.add(field.strip())
    for allocation in previous_assessment.get("allocation_points") or []:
        if not isinstance(allocation, dict):
            continue
        evidence_fields.update(
            field
            for field in (allocation.get("evidence_fields") or [])
            if isinstance(field, str) and field.strip()
        )
    if not evidence_fields:
        return {"status": "not_evaluable", "reason": "Previous carryover issue has no field-level evidence reference."}

    touched_evidence_changes = [
        change
        for change in field_changes
        if isinstance(change, dict) and str(change.get("field") or "") in evidence_fields
    ]
    if not touched_evidence_changes:
        return {
            "status": "not_touched",
            "evidence_fields": sorted(evidence_fields),
            "reason": "No previous carryover evidence field changed in the latest iteration.",
        }

    restored_changes = []
    unresolved_changes = []
    for change in touched_evidence_changes:
        current = _norm_for_state_compare(change.get("current_value"))
        baseline = _norm_for_state_compare(change.get("baseline_value"))
        previous = _norm_for_state_compare(change.get("previous_value"))
        if current and baseline and current == baseline and current != previous:
            restored_changes.append(change)
        else:
            unresolved_changes.append(change)

    if restored_changes and not unresolved_changes:
        return json_safe({
            "status": "resolved_by_field_return",
            "evidence_fields": sorted(evidence_fields),
            "resolved_fields": [
                {
                    "field": change.get("field"),
                    "previous_value": change.get("previous_value"),
                    "previous_label": change.get("previous_label"),
                    "current_value": change.get("current_value"),
                    "current_label": change.get("current_label"),
                    "baseline_value": change.get("baseline_value"),
                    "baseline_label": change.get("baseline_label"),
                }
                for change in restored_changes
            ],
            "reason": "All touched carryover evidence fields returned to their baseline values.",
        })

    return json_safe({
        "status": "touched_but_not_resolved",
        "evidence_fields": sorted(evidence_fields),
        "touched_fields": [
            {
                "field": change.get("field"),
                "previous_value": change.get("previous_value"),
                "previous_label": change.get("previous_label"),
                "current_value": change.get("current_value"),
                "current_label": change.get("current_label"),
                "baseline_value": change.get("baseline_value"),
                "baseline_label": change.get("baseline_label"),
            }
            for change in touched_evidence_changes
        ],
        "reason": "At least one touched carryover evidence field did not return to its baseline value.",
    })


def _reality_check_carryover_candidate(
    previous_review_trace: dict[str, Any] | None,
    field_changes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    previous = _compact_review_context(previous_review_trace)
    if not previous:
        return {"active": False}
    try:
        previous_points = float(previous.get("reality_check_points"))
    except (TypeError, ValueError):
        return {"active": False}
    if previous_points > REALITY_CHECK_CARRYOVER_MATERIALITY_THRESHOLD:
        return {"active": False}

    assessment = previous.get("reality_check_assessment") or {}
    precheck = _carryover_state_precheck(assessment, field_changes or [])
    return json_safe({
        "active": True,
        "source_iteration_id": previous.get("iteration_id"),
        "source_input_hash": previous.get("input_hash"),
        "previous_reality_check_points": previous_points,
        "previous_reality_check_assessment": deepcopy(assessment),
        "app_state_precheck": precheck,
        "previous_reality_check_allocation_points": deepcopy(
            assessment.get("allocation_points")
            or (previous_review_trace or {}).get("reality_check_allocation_points")
            or []
        ),
        "instruction": (
            "Assess whether this previous negative Reality Check concern is still relevant after the latest scenario "
            "change, partly mitigated, or resolved/superseded. If app_state_precheck.status is "
            "resolved_by_field_return, treat the previous carryover as resolved unless there is a distinct new "
            "independent issue."
        ),
    })


def build_review_packet(
    *,
    current_snapshot: dict[str, Any],
    previous_snapshot: dict[str, Any] | None = None,
    baseline_snapshot: dict[str, Any] | None = None,
    baseline_review_trace: dict[str, Any] | None = None,
    previous_review_trace: dict[str, Any] | None = None,
    trial_identity: dict[str, Any] | None = None,
    text_context: dict[str, Any] | None = None,
    user_clarifications: list[dict[str, Any]] | None = None,
    compact_storyline_memory: str = "",
    mode: str = MODE_EXISTING_STUDY,
) -> dict[str, Any]:
    """Assemble the narrative-review input packet for one prediction snapshot."""
    current_values = _snapshot_values(current_snapshot)
    current_text = {
        **_snapshot_text_context(baseline_snapshot),
        **_snapshot_text_context(previous_snapshot),
        **_snapshot_text_context(current_snapshot),
        **(text_context or {}),
    }
    current_identity = _merge_present_dicts(
        _snapshot_trial_identity(baseline_snapshot),
        _snapshot_trial_identity(previous_snapshot),
        _snapshot_trial_identity(current_snapshot),
        trial_identity or {},
    )
    changed_fields = _changed_fields(current_snapshot)

    field_changes = _field_changes(current_snapshot, previous_snapshot, baseline_snapshot)

    packet = {
        "prompt_version": PROMPT_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "field_dictionary_version": FIELD_DICTIONARY_VERSION,
        "mode": mode,
        "trial_identity": _select_keys(current_identity, TRIAL_IDENTITY_KEYS),
        "text_context": _select_keys(current_text, TEXT_CONTEXT_KEYS),
        "structured_features": _select_keys(current_values, STRUCTURED_FEATURE_KEYS),
        "structured_feature_display_values": _select_keys(
            _snapshot_display_values(current_snapshot),
            STRUCTURED_FEATURE_KEYS,
        ),
        "structured_feature_meanings": _select_field_meanings(STRUCTURED_FEATURE_KEYS),
        "text_context_field_meanings": _select_field_meanings(TEXT_CONTEXT_KEYS),
        "reference_packs": _selected_reference_packs(changed_fields),
        "therapeutic_area_context": _therapeutic_area_context(_select_keys(current_values, STRUCTURED_FEATURE_KEYS)),
        "operational_assumptions": _select_keys(
            current_snapshot.get("operational_assumptions") or {},
            ACTIVE_OPERATIONAL_ASSUMPTION_KEYS,
        ),
        "operational_movement_context": _operational_movement_context(current_snapshot, baseline_snapshot),
        "model_interpretation": {
            "completion_score": _completion_score(current_snapshot),
            "previous_completion_score": _completion_score(previous_snapshot),
            "score_delta": _score_delta(current_snapshot, previous_snapshot),
            "direct_xgboost_shap_fields": list(DIRECT_XGBOOST_SHAP_FIELDS),
            "pillar_impacts": _pillar_impacts(current_snapshot),
            "pillar_deltas": _pillar_deltas(current_snapshot, previous_snapshot),
            "xgboost_impact_changes": _impact_changes(current_snapshot, previous_snapshot, baseline_snapshot),
            "current_model_state_evidence": _model_state_evidence(current_snapshot),
            "model_movement_evidence": _model_movement_evidence(
                current_snapshot,
                previous_snapshot,
                baseline_snapshot,
            ),
            "model_signal_guidance": _model_signal_guidance(),
            "top_positive_feature_drivers": _feature_driver_values(current_snapshot, "top_positive_feature_drivers"),
            "top_negative_feature_drivers": _feature_driver_values(current_snapshot, "top_negative_feature_drivers"),
            "top_feature_impact_changes": _feature_driver_values(current_snapshot, "top_feature_impact_changes"),
        },
        "review_context": {
            "baseline_review": _compact_review_context(
                baseline_review_trace,
                include_quality_scores=False,
            ),
            "previous_review": _compact_review_context(previous_review_trace),
        },
        "clarification_context": {
            "user_clarifications": json_safe(user_clarifications or []),
        },
        "iteration_context": {
            "baseline_snapshot_id": _snapshot_id(baseline_snapshot, "baseline"),
            "previous_snapshot_id": _snapshot_id(previous_snapshot),
            "current_snapshot_id": _snapshot_id(current_snapshot),
            "iteration_number": _iteration_number(current_snapshot, previous_snapshot),
            "changed_fields": changed_fields,
            "field_changes": field_changes,
            "text_change_evidence": _text_change_evidence(current_snapshot, previous_snapshot, baseline_snapshot),
            "trial_score_continuity": _trial_score_continuity(previous_review_trace),
            "reality_check_carryover_candidate": _reality_check_carryover_candidate(
                previous_review_trace,
                field_changes,
            ),
            "compact_storyline_memory": compact_storyline_memory,
        },
    }

    packet["scenario_state_hash"] = scenario_state_hash_from_packet(packet)
    baseline_text = {
        **_snapshot_text_context(baseline_snapshot),
        **(text_context or {}),
    }
    baseline_identity = _merge_present_dicts(
        _snapshot_trial_identity(baseline_snapshot),
        trial_identity or {},
    )
    baseline_state_hash = _baseline_scenario_state_hash(
        packet,
        baseline_snapshot,
        baseline_text_context=baseline_text,
        baseline_trial_identity=baseline_identity,
    )
    if baseline_state_hash:
        packet["iteration_context"]["baseline_scenario_state_hash"] = baseline_state_hash
        packet["iteration_context"]["returned_to_hidden_baseline_state"] = (
            packet["scenario_state_hash"] == baseline_state_hash
            and _iteration_number(current_snapshot, previous_snapshot) > 0
        )
        if packet["iteration_context"]["returned_to_hidden_baseline_state"]:
            packet["iteration_context"]["reality_check_carryover_candidate"] = {"active": False}
    packet["input_hash"] = stable_packet_hash(packet)
    return json_safe(packet)


def build_review_packet_from_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    """Build a packet from a contract fixture for checker and mock-reviewer work."""
    packet = deepcopy(fixture["input_packet"])
    return build_review_packet(
        current_snapshot={
            "snapshot_id": packet["iteration_context"].get("current_snapshot_id"),
            "trial_identity": packet.get("trial_identity", {}),
            "text_context": packet.get("text_context", {}),
            "structured_features": packet.get("structured_features", {}),
            "display_values": packet.get("structured_feature_display_values", {}),
            "operational_assumptions": packet.get("operational_assumptions", {}),
            "model_interpretation": packet.get("model_interpretation", {}),
            "changed_fields": packet["iteration_context"].get("changed_fields", []),
            "source": "prerecorded_baseline" if fixture.get("scenario_type") == "baseline" else "fixture",
        },
        previous_snapshot={
            "snapshot_id": packet["iteration_context"].get("previous_snapshot_id"),
            "score": packet.get("model_interpretation", {}).get("previous_completion_score"),
        }
        if packet["iteration_context"].get("previous_snapshot_id")
        else None,
        baseline_snapshot={
            "snapshot_id": packet["iteration_context"].get("baseline_snapshot_id"),
            "trial_identity": packet.get("trial_identity", {}),
            "text_context": packet.get("text_context", {}),
        },
        user_clarifications=(packet.get("clarification_context") or {}).get("user_clarifications") or [],
        compact_storyline_memory=packet["iteration_context"].get("compact_storyline_memory", ""),
    )
