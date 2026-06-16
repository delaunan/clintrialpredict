"""Deterministic input-packet builder for narrative review.

The builder owns data assembly only. It does not call an LLM, validate LLM
output, calculate Quality Adjustment, or mutate Streamlit session state.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.narratives.contract_fixtures import PROMPT_VERSION, RUBRIC_VERSION

MODE_EXISTING_STUDY = "existing_study"
FIELD_DICTIONARY_VERSION = "taxonomy_01_narrative_v1"

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
DESIGN_CONFIDENCE_SUBCATEGORY_LABELS = {
    "phase_intent_alignment": "Phase & Intent Alignment",
    "endpoint_evidence_strength": "Endpoint & Evidence Strength",
    "target_population_alignment": "Target Population Alignment",
    "operational_burden_balance": "Operational Burden Balance",
}
DESIGN_CONFIDENCE_RELEVANT_FIELDS = {
    "phase_intent_alignment": {
        "biomarker_stratification_ml",
        "phase_ml",
        "strategic_ambition_ml",
        "is_rare_disease_ml",
        "line_of_therapy_ml",
        "patient_severity_ml",
        "target_precedent_ml",
        "target_pathway_class_ml",
        "therapeutic_modality_ml",
        "administration_complexity_ml",
        "innovation_tier_ml",
        "primary_purpose_ml",
        "has_dmc_ml",
        "comparator_benchmark_ml",
        "endpoint_rigor_ml",
        "text_context.title",
        "text_context.summary_ui",
    },
    "endpoint_evidence_strength": {
        "adaptive_design_ml",
        "allocation_ml",
        "biomarker_stratification_ml",
        "comparator_benchmark_ml",
        "endpoint_rigor_ml",
        "endpoint_structure_ml",
        "has_placebo_ml",
        "has_dmc_ml",
        "masking_ml",
        "number_of_arms_ml",
        "primary_duration_months_ml",
        "therapeutic_modality_ml",
        "administration_complexity_ml",
        "text_context.primary_outcomes_ui",
        "text_context.summary_ui",
    },
    "target_population_alignment": {
        "adult_ml",
        "child_ml",
        "gender_ml",
        "gbd_cause_id_3_ml",
        "healthy_volunteers_ml",
        "is_rare_disease_ml",
        "line_of_therapy_ml",
        "older_adult_ml",
        "patient_severity_ml",
        "biomarker_stratification_ml",
        "text_context.conditions_ui",
        "text_context.summary_ui",
    },
    "operational_burden_balance": {
        "administration_complexity_ml",
        "adaptive_design_ml",
        "allocation_ml",
        "biomarker_stratification_ml",
        "comparator_benchmark_ml",
        "endpoint_rigor_ml",
        "endpoint_structure_ml",
        "has_dmc_ml",
        "is_rare_disease_ml",
        "intervention_model_ml",
        "line_of_therapy_ml",
        "masking_ml",
        "number_of_arms_ml",
        "patient_severity_ml",
        "primary_duration_months_ml",
        "sponsor_tier_ml",
        "therapeutic_modality_ml",
        "operational_assumptions.planned_enrollment",
        "operational_assumptions.planned_sites",
        "operational_assumptions.planned_duration_months",
        "text_context.interventions_ui",
        "text_context.primary_outcomes_ui",
        "text_context.summary_ui",
    },
}


def design_confidence_relevant_changed_fields(
    subcategory_name: str,
    changed_fields: list[str],
) -> list[str]:
    """Return changed packet fields that are directly relevant to one Design Confidence subcategory."""
    relevant = DESIGN_CONFIDENCE_RELEVANT_FIELDS.get(subcategory_name, set())
    matched: list[str] = []
    for field in changed_fields:
        field = str(field)
        if field in relevant or any(field.startswith(f"{prefix}.") for prefix in relevant):
            matched.append(field)
    return matched


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


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


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
    design_subcategories = validated.get("design_confidence_subcategories") or {}
    design_confidence_analysis = validated.get("design_confidence_analysis") or {}
    tradeoff_review = validated.get("tradeoff_review") or {}
    design_confidence = trace.get("design_confidence", trace.get("quality_adjustment"))
    total_scenario_score = trace.get("total_scenario_score", trace.get("final_candidate_score"))
    design_assessment = trace.get("design_confidence_assessment") or trace.get("quality_assessment") or {}
    compact = {
        "input_hash": trace.get("input_hash"),
        "iteration_id": trace.get("iteration_id"),
        "status": trace.get("status"),
        "validation_status": trace.get("validation_status"),
        "design_confidence": design_confidence if include_quality_scores else None,
        "total_scenario_score": total_scenario_score if include_quality_scores else None,
        "design_numeric_context": "visible_review" if include_quality_scores else "hidden_baseline_qualitative_only",
        "changed_fields": trace.get("changed_fields") or [],
        "score_delta": trace.get("score_delta", trace.get("score_movement")),
        "completion_outlook_summary": (
            completion_outlook.get("risk_pattern_summary")
            or completion_outlook.get("score_delta_summary")
        ),
        "central_tension": (
            trace.get("central_tension")
            or validated.get("main_tension")
            or design_confidence_analysis.get("confidence_rationale")
            or tradeoff_review.get("central_tension")
        ),
        "design_confidence_subcategory_ratings": {
            subcategory_name: {
                "current_state": subcategory.get("current_state"),
                "movement_direction": subcategory.get("movement_direction"),
                "movement_materiality": subcategory.get("movement_materiality"),
                "effect_role": subcategory.get("effect_role"),
                "rating": subcategory.get("rating"),
                "score_materiality": subcategory.get("score_materiality"),
                "rationale": subcategory.get("rationale"),
                "evidence_fields": subcategory.get("evidence_fields") or [],
            }
            for subcategory_name, subcategory in sorted(design_subcategories.items())
            if isinstance(subcategory, dict)
        },
        "design_confidence_contributions": (
            deepcopy(design_assessment.get("subcategories") or {})
            if include_quality_scores
            else {}
        ),
        "key_questions": {
            "completion_outlook_summary": completion_outlook.get("risk_pattern_summary"),
            "design_confidence_summary": design_confidence_analysis.get("summary"),
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
        "compact_storyline_memory": trace.get("compact_storyline_memory") or "",
    }

    if not include_quality_scores:
        compact["baseline_completion_outlook_summary"] = (
            completion_outlook.get("risk_pattern_summary")
            or completion_outlook.get("score_delta_summary")
        )
        compact["baseline_design_subcategory_ratings"] = compact["design_confidence_subcategory_ratings"]
        compact["baseline_strengths"] = [
            subcategory.get("rationale")
            for subcategory in design_subcategories.values()
            if isinstance(subcategory, dict)
            and subcategory.get("rating")
            in {
                "strong",
                "supportive",
            }
        ]
        compact["baseline_concerns"] = [
            subcategory.get("rationale")
            for subcategory in design_subcategories.values()
            if isinstance(subcategory, dict)
            and subcategory.get("rating") in {
                "weak",
                "conflicting",
            }
        ]
        compact["baseline_consistency_flags"] = {}

    return json_safe(compact)


def _design_confidence_continuity(
    previous_review_trace: dict[str, Any] | None,
    changed_fields: list[str],
) -> dict[str, Any]:
    previous = _compact_review_context(previous_review_trace)
    if not previous:
        return {
            "available": False,
            "reason": "first_visible_iteration_or_no_prior_visible_review",
            "subcategories": {},
        }

    previous_ratings = previous.get("design_confidence_subcategory_ratings") or {}
    previous_contributions = previous.get("design_confidence_contributions") or {}
    subcategories: dict[str, Any] = {}
    for subcategory_name, label in DESIGN_CONFIDENCE_SUBCATEGORY_LABELS.items():
        rating = previous_ratings.get(subcategory_name) or {}
        contribution = previous_contributions.get(subcategory_name) or {}
        relevant_changes = design_confidence_relevant_changed_fields(subcategory_name, changed_fields)
        subcategories[subcategory_name] = {
            "label": label,
            "previous_current_state": rating.get("current_state"),
            "previous_movement_direction": rating.get("movement_direction"),
            "previous_movement_materiality": rating.get("movement_materiality"),
            "previous_effect_role": rating.get("effect_role"),
            "previous_rating": rating.get("rating"),
            "previous_score_materiality": rating.get("score_materiality"),
            "previous_points": contribution.get("points"),
            "previous_raw_points": contribution.get("raw_points"),
            "previous_rationale": rating.get("rationale"),
            "previous_evidence_fields": rating.get("evidence_fields") or [],
            "current_relevant_changed_fields": relevant_changes,
        }
    return json_safe({
        "available": True,
        "source_iteration_id": previous.get("iteration_id"),
        "source_input_hash": previous.get("input_hash"),
        "changed_fields": changed_fields,
        "instruction": (
            "Use this object as deterministic continuity context for Design Confidence subcategories. "
            "The current scenario is still scored fresh, but large subcategory shifts need current relevant evidence."
        ),
        "subcategories": subcategories,
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
        "model_interpretation": {
            "completion_score": _completion_score(current_snapshot),
            "previous_completion_score": _completion_score(previous_snapshot),
            "score_delta": _score_delta(current_snapshot, previous_snapshot),
            "direct_xgboost_shap_fields": list(DIRECT_XGBOOST_SHAP_FIELDS),
            "pillar_impacts": _pillar_impacts(current_snapshot),
            "pillar_deltas": _pillar_deltas(current_snapshot, previous_snapshot),
            "xgboost_impact_changes": _impact_changes(current_snapshot, previous_snapshot, baseline_snapshot),
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
            "field_changes": _field_changes(current_snapshot, previous_snapshot, baseline_snapshot),
            "text_change_evidence": _text_change_evidence(current_snapshot, previous_snapshot, baseline_snapshot),
            "design_confidence_continuity": _design_confidence_continuity(previous_review_trace, changed_fields),
            "compact_storyline_memory": compact_storyline_memory,
        },
    }

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
