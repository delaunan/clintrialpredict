"""Deterministic input-packet builder for narrative review.

The builder owns data assembly only. It does not call an LLM, validate LLM
output, calculate Quality Adjustment, or mutate Streamlit session state.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any

from src.narratives.contract_fixtures import PROMPT_VERSION, RUBRIC_VERSION

MODE_EXISTING_STUDY = "existing_study"

TRIAL_IDENTITY_KEYS = (
    "nct_id",
    "trial_label",
    "lead_sponsor_canonical",
    "start_year",
)

TEXT_CONTEXT_KEYS = (
    "title",
    "summary_ui",
    "primary_outcomes_ui",
    "primary_endpoint_description",
    "interventions_ui",
    "criteria_ui",
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
        or snapshot.get("submitted_values")
        or snapshot.get("compare_values")
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


def _compact_review_context(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    if not trace:
        return None
    if trace.get("status") not in {"reviewed", "reused_previous_review"}:
        return None

    validated = trace.get("validated_review") or {}
    continuity = validated.get("continuity") or {}
    participant = validated.get("participant_review") or {}
    return json_safe({
        "input_hash": trace.get("input_hash"),
        "iteration_id": trace.get("iteration_id"),
        "status": trace.get("status"),
        "validation_status": trace.get("validation_status"),
        "quality_adjustment": trace.get("quality_adjustment"),
        "final_candidate_score": trace.get("final_candidate_score"),
        "changed_fields": trace.get("changed_fields") or [],
        "score_movement": trace.get("score_movement"),
        "participant_review": {
            "what_changed": participant.get("what_changed"),
            "what_the_design_gained": participant.get("what_the_design_gained"),
            "what_the_design_may_have_sacrificed": participant.get("what_the_design_may_have_sacrificed"),
            "challenge_question": participant.get("challenge_question"),
        },
        "continuity": {
            "prior_concerns_resolved": continuity.get("prior_concerns_resolved") or [],
            "prior_concerns_worsened": continuity.get("prior_concerns_worsened") or [],
            "prior_concerns_unchanged": continuity.get("prior_concerns_unchanged") or [],
            "new_concerns": continuity.get("new_concerns") or [],
            "storyline_update": continuity.get("storyline_update"),
        },
        "compact_storyline_memory": trace.get("compact_storyline_memory") or "",
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

    packet = {
        "prompt_version": PROMPT_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "mode": mode,
        "trial_identity": _select_keys(current_identity, TRIAL_IDENTITY_KEYS),
        "text_context": _select_keys(current_text, TEXT_CONTEXT_KEYS),
        "structured_features": _select_keys(current_values, STRUCTURED_FEATURE_KEYS),
        "structured_feature_display_values": _select_keys(
            _snapshot_display_values(current_snapshot),
            STRUCTURED_FEATURE_KEYS,
        ),
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
            "top_positive_feature_drivers": _feature_driver_values(current_snapshot, "top_positive_feature_drivers"),
            "top_negative_feature_drivers": _feature_driver_values(current_snapshot, "top_negative_feature_drivers"),
            "top_feature_impact_changes": _feature_driver_values(current_snapshot, "top_feature_impact_changes"),
        },
        "review_context": {
            "baseline_review": _compact_review_context(baseline_review_trace),
            "previous_review": _compact_review_context(previous_review_trace),
        },
        "iteration_context": {
            "baseline_snapshot_id": _snapshot_id(baseline_snapshot, "baseline"),
            "previous_snapshot_id": _snapshot_id(previous_snapshot),
            "current_snapshot_id": _snapshot_id(current_snapshot),
            "iteration_number": _iteration_number(current_snapshot, previous_snapshot),
            "changed_fields": _changed_fields(current_snapshot),
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
        compact_storyline_memory=packet["iteration_context"].get("compact_storyline_memory", ""),
    )
