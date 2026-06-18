"""Product-level controls for Trial Score narrative boundaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.narratives.packet_builder import stable_packet_hash
from src.narratives.scoring import validate_and_score_review

OPERATIONAL_ONLY_COMPLETION_OUTLOOK_BOUNDARY = (
    "The Completion Outlook remains unchanged because planning assumptions such as enrollment, site count, and "
    "Planned Total Timeline do not directly feed the score. They still matter for whether the scenario feels operationally "
    "proportionate and executable. Therefore, the impact of changes in these variables is reflected in Operational "
    "Fit and, when needed, Reality Check instead."
)

STABLE_NON_SCORE_INPUT_COMPLETION_OUTLOOK = (
    "The Completion Outlook score remains stable because the latest changes are not directly used to calculate the "
    "Completion Outlook score. Nevertheless, the updated scenario details are considered in Reality Check."
)

OPERATIONAL_ASSUMPTION_FIELDS = {
    "operational_assumptions.planned_enrollment",
    "operational_assumptions.planned_sites",
    "operational_assumptions.planned_duration_months",
}

CORE_EVIDENCE_CONTROL_FIELDS = {
    "allocation_ml",
    "masking_ml",
    "comparator_benchmark_ml",
    "intervention_model_ml",
    "number_of_arms_ml",
    "endpoint_rigor_ml",
    "endpoint_structure_ml",
}

CORE_EVIDENCE_WEAKER_VALUES = {
    "allocation_ml": {"NON_RANDOMIZED", "NON-RANDOMIZED", "SINGLE_ARM", "SINGLE-GROUP", "SINGLE_GROUP"},
    "masking_ml": {
        "NONE",
        "OPEN",
        "OPEN_LABEL",
        "OPEN-LABEL",
        "NOT_SPECIFIED",
        "OPEN_LABEL_OR_NOT_SPECIFIED",
        "UNKNOWN",
    },
    "comparator_benchmark_ml": {"NONE", "NO_CONTROL", "SINGLE_ARM", "HISTORICAL", "HISTORICAL_CONTROL"},
    "intervention_model_ml": {"SINGLE_GROUP", "SINGLE-GROUP", "SINGLE_ARM", "SINGLE-ARM"},
    "endpoint_rigor_ml": {"SURROGATE_BIOMARKER", "SURROGATE", "BIOMARKER", "SUBJECTIVE_PRO"},
    "endpoint_structure_ml": {"SINGLE", "SINGLE_GOAL", "SINGLE PRIMARY", "SINGLE_PRIMARY"},
}

SHORTCUT_STRATEGIC_REVIEW_RULE = (
    "When multiple core evidence-quality controls are removed together, such as randomization, masking, "
    "comparator structure, arms, and endpoint rigor, unchanged target-population relevance should not by itself "
    "make Reality Check positive. Reality Check should usually offset the Completion Outlook gain unless "
    "the packet provides a clear safety-extension, exploratory-signal, access, or proportionality rationale for "
    "lower evidence ambition."
)


def _changed_fields(packet: dict[str, Any]) -> set[str]:
    return {str(field) for field in (packet.get("iteration_context") or {}).get("changed_fields") or []}


def _only_field_types(changed_fields: set[str], *, text: bool = False, operational: bool = False) -> bool:
    if not changed_fields:
        return False
    allowed: set[str] = set()
    if text:
        allowed.update(field for field in changed_fields if field.startswith("text_context."))
    if operational:
        allowed.update(field for field in changed_fields if field in OPERATIONAL_ASSUMPTION_FIELDS)
    return changed_fields == allowed


def _has_structured_score_input_change(changed_fields: set[str]) -> bool:
    """Return whether the latest change includes a structured score-input candidate."""
    return any(
        not field.startswith("text_context.") and field not in OPERATIONAL_ASSUMPTION_FIELDS
        for field in changed_fields
    )


def _normalized_value(value: Any) -> str:
    return str(value or "").strip().upper().replace(" / ", "_").replace(" ", "_")


def _is_weakened_core_evidence_change(change: dict[str, Any]) -> bool:
    field = str(change.get("field") or "")
    if field == "number_of_arms_ml":
        try:
            previous = float(change.get("previous_value"))
            current = float(change.get("current_value"))
        except (TypeError, ValueError):
            return False
        return current < previous

    weaker_values = CORE_EVIDENCE_WEAKER_VALUES.get(field)
    if not weaker_values:
        return False
    previous = _normalized_value(change.get("previous_value"))
    current = _normalized_value(change.get("current_value"))
    return current in weaker_values and previous != current


def _weakened_core_evidence_fields(packet: dict[str, Any]) -> set[str]:
    weakened: set[str] = set()
    for change in (packet.get("iteration_context") or {}).get("field_changes") or []:
        if isinstance(change, dict) and _is_weakened_core_evidence_change(change):
            weakened.add(str(change.get("field") or ""))
    return weakened


def review_controls_for_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Return app-owned review controls derived from latest changed fields."""
    changed_fields = _changed_fields(packet)
    changed_operational_fields = changed_fields & OPERATIONAL_ASSUMPTION_FIELDS
    weakened_core_evidence_fields = _weakened_core_evidence_fields(packet)
    if _only_field_types(changed_fields, operational=True):
        return {
            "completion_outlook_mode": "fixed_planning_assumption_boundary",
            "required_completion_outlook_sentence": OPERATIONAL_ONLY_COMPLETION_OUTLOOK_BOUNDARY,
            "completion_outlook_forbidden_latest_fields": sorted(OPERATIONAL_ASSUMPTION_FIELDS),
            "latest_change_focus": "planning_assumptions_only",
        }
    if _only_field_types(changed_fields, text=True) or _only_field_types(
        changed_fields,
        text=True,
        operational=True,
    ):
        return {
            "completion_outlook_mode": "stable_non_score_input_context",
            "required_completion_outlook_sentence": STABLE_NON_SCORE_INPUT_COMPLETION_OUTLOOK,
            "latest_change_focus": "trial_description_without_score_input_change"
            if all(field.startswith("text_context.") for field in changed_fields)
            else "trial_description_and_planning_assumptions_without_score_input_change",
            "completion_outlook_forbidden_latest_fields": sorted(
                field for field in changed_fields if field in OPERATIONAL_ASSUMPTION_FIELDS
            ),
        }
    if changed_operational_fields and _has_structured_score_input_change(changed_fields):
        controls = {
            "completion_outlook_mode": "structured_score_inputs_only",
            "latest_change_focus": "structured_score_input_change_with_planning_assumptions",
            "completion_outlook_forbidden_latest_fields": sorted(changed_operational_fields),
            "completion_outlook_boundary_instruction": (
                "Write the Completion Outlook narrative from changed structured Completion Outlook score inputs and "
                "aligned Trial description field context only. Do not name or use the listed planning assumptions as "
                "Completion Outlook evidence, including proxy phrases such as operational footprint, operational scale, "
                "site expansion, larger enrollment, scaled execution, or site performance; they remain Strategic "
                "Review context."
            ),
        }
        if len(weakened_core_evidence_fields) >= 3:
            controls["latest_change_focus"] = "evidence_shortcut_with_planning_assumptions"
            controls["shortcut_strategic_review_rule"] = SHORTCUT_STRATEGIC_REVIEW_RULE
        return controls
    if len(weakened_core_evidence_fields) >= 3:
        return {
            "latest_change_focus": "evidence_shortcut_and_bias_control",
            "shortcut_strategic_review_rule": SHORTCUT_STRATEGIC_REVIEW_RULE,
        }
    return {}


def attach_review_controls(packet: dict[str, Any], controls: dict[str, Any] | None = None) -> dict[str, Any]:
    """Attach review controls and update the packet hash when controls are present."""
    selected_controls = deepcopy(controls) if controls is not None else review_controls_for_packet(packet)
    if not selected_controls:
        return packet
    controlled = deepcopy(packet)
    controlled["review_controls"] = selected_controls
    controlled["input_hash"] = stable_packet_hash({key: value for key, value in controlled.items() if key != "input_hash"})
    return controlled


def apply_review_control_overrides(packet: dict[str, Any], review_result: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic Completion Outlook wording for hard product-boundary modes."""
    controls = packet.get("review_controls") or {}
    required_sentence = controls.get("required_completion_outlook_sentence")
    completion_outlook_mode = controls.get("completion_outlook_mode")
    if (
        review_result.get("status") != "reviewed"
        or completion_outlook_mode
        not in {"fixed_planning_assumption_boundary", "stable_non_score_input_context"}
        or not isinstance(required_sentence, str)
        or not required_sentence.strip()
    ):
        return review_result

    pre_control_review = deepcopy(review_result.get("review") or {})
    pre_control_validated_review = deepcopy(review_result.get("validated_review") or {})
    pre_control_scoring = deepcopy(review_result.get("scoring") or {})
    review = deepcopy(pre_control_review)
    completion = review.setdefault("completion_outlook_analysis", {})
    completion["risk_pattern_summary"] = required_sentence
    if completion_outlook_mode == "fixed_planning_assumption_boundary":
        driver_summary = "No Completion Outlook score input changed in this planning-assumption-only iteration."
    else:
        driver_summary = "No structured Completion Outlook score input changed in this latest iteration."
    completion["driver_summary"] = driver_summary
    completion["movement_explanation"] = required_sentence
    review.setdefault("trace", {})["completion_outlook_control_applied"] = completion_outlook_mode

    rescored = validate_and_score_review(packet, review)
    return {
        **deepcopy(review_result),
        "review": review,
        "validated_review": rescored["validated_review"],
        "scoring": rescored["scoring"],
        "provider_metadata": {
            **deepcopy(review_result.get("provider_metadata") or {}),
            "review_control_override": completion_outlook_mode,
            "pre_control_review": pre_control_review,
            "pre_control_validated_review": pre_control_validated_review,
            "pre_control_scoring": pre_control_scoring,
        },
    }
