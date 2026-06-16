"""Validation and deterministic scoring for narrative Scenario Reviews."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.narratives.contract_fixtures import REQUIRED_DESIGN_SUBCATEGORIES
from src.narratives.packet_builder import stable_packet_hash

DESIGN_SUBCATEGORY_LABELS = {
    "phase_intent_alignment": "Phase & Intent Alignment",
    "endpoint_evidence_strength": "Endpoint & Evidence Strength",
    "target_population_alignment": "Target Population Alignment",
    "operational_burden_balance": "Operational Burden Balance",
}

DESIGN_SUBCATEGORY_PILLARS = {
    "phase_intent_alignment": "therapeutic_context",
    "endpoint_evidence_strength": "scientific_challenge",
    "target_population_alignment": "patient_profile",
    "operational_burden_balance": "execution_framework",
}

DESIGN_PILLAR_LABELS = {
    "therapeutic_context": "Therapeutic Context",
    "scientific_challenge": "Scientific Challenge",
    "patient_profile": "Patient Profile",
    "execution_framework": "Execution Framework",
}

DESIGN_RATINGS = {
    "strong",
    "supportive",
    "balanced",
    "weak",
    "conflicting",
}

CURRENT_STATE_LEVELS = DESIGN_RATINGS

MOVEMENT_DIRECTIONS = {
    "resolved",
    "improved",
    "partially_resolved",
    "unchanged",
    "offset",
    "weakened",
    "worsened",
    "newly_introduced",
}

MOVEMENT_MATERIALITY_LEVELS = {
    "none",
    "minor",
    "moderate",
    "major",
}

EFFECT_ROLES = {
    "counterweight",
    "confirming",
    "independent",
    "unchanged",
}

# Compatibility alias for older prompt/checker code during migration.
SCORE_MATERIALITY_LEVELS = {
    "minimal",
    "low",
    "moderate",
    "high",
    "very_high",
}

MOVEMENT_MATERIALITY_POINTS = {
    "none": 0.0,
    "minor": 0.5,
    "moderate": 1.0,
    "major": 2.0,
}

POSITIVE_MOVEMENTS = {
    "resolved",
    "improved",
    "partially_resolved",
    "offset",
}

NEGATIVE_MOVEMENTS = {
    "weakened",
    "worsened",
    "newly_introduced",
}

EFFECT_ROLE_MULTIPLIERS = {
    "counterweight": 1.0,
    "independent": 1.0,
    "confirming": 0.5,
    "unchanged": 0.0,
}

# Temporary compatibility alias for prompt/schema code that migrates in Phase 4.
DOMAIN_RATING_POINTS = {
    subcategory_name: {
        rating: 0.0
        for rating in DESIGN_RATINGS
    }
    for subcategory_name in sorted(REQUIRED_DESIGN_SUBCATEGORIES)
}

DESIGN_SUBCATEGORY_MIN = -2.0
DESIGN_SUBCATEGORY_MAX = 2.0
DESIGN_CONFIDENCE_NORMAL_CAP = 4.0
DESIGN_CONFIDENCE_EXCEPTIONAL_CAP = 6.0
TOTAL_SCORE_MIN = 0
TOTAL_SCORE_MAX = 100

STRATEGIC_REVIEW_EFFECT_LABELS = {
    "supports_score_gain",
    "lightly_supports_score_gain",
    "neutral",
    "partly_offsets_score_gain",
    "strongly_offsets_score_gain",
    "critical_reversal",
    "softens_score_decline",
    "lightly_softens_decline",
    "reinforces_score_decline",
    "critical_negative_review",
    "supports_tradeoff_balance",
    "lightly_supports_tradeoff_balance",
    "worsens_active_tension",
    "strongly_worsens_active_tension",
    "reopens_protected_tension",
}

POSITIVE_COMPLETION_REVIEW_FACTORS = {
    "supports_score_gain": 0.25,
    "lightly_supports_score_gain": 0.10,
    "neutral": 0.0,
    "partly_offsets_score_gain": -0.50,
    "strongly_offsets_score_gain": -1.00,
    "critical_reversal": -1.50,
}

NEGATIVE_COMPLETION_REVIEW_FACTORS = {
    "softens_score_decline": 0.25,
    "lightly_softens_decline": 0.10,
    "neutral": 0.0,
    "reinforces_score_decline": -0.75,
    "critical_negative_review": -1.50,
}

FLAT_COMPLETION_REVIEW_FACTORS = {
    "supports_tradeoff_balance": 0.25,
    "lightly_supports_tradeoff_balance": 0.10,
    "neutral": 0.0,
    "worsens_active_tension": -0.50,
    "strongly_worsens_active_tension": -1.00,
    "reopens_protected_tension": -1.50,
}

TENSION_STATUS_FACTORS = {
    "resolved": 0.0,
    "obsolete": 0.0,
    "superseded": -0.10,
    "partially_active": -0.15,
    "still_active_secondary": -0.25,
    "still_active_primary": -0.50,
    "regressed": -1.00,
    "newly_resolved": 0.0,
    "protected_gain_preserved": 0.10,
    "further_improved": 0.20,
    "stable_background_strength": 0.0,
    "not_applicable": 0.0,
}

OPERATIONAL_MATERIALITY_BUDGETS = {
    "minor": 2.0,
    "moderate": 3.0,
    "major": 4.0,
    "extreme": 5.0,
}

STRATEGIC_REVIEW_SUBLEVELS = {
    "current_tension": "Current Tension",
    "carryover_check": "Carryover Check",
    "tradeoff_resolution": "Tradeoff Resolution",
}

APP_OWNED_SCORE_FIELDS = {
    "strategic_review_points",
    "trial_score",
    "strategic_review_assessment",
    "design_confidence",
    "total_scenario_score",
    "design_confidence_assessment",
    "design_confidence_contributions",
    # Legacy names stay app-owned during the migration.
    "quality_adjustment",
    "final_candidate_score",
    "quality_assessment",
}

KEY_QUESTION_FIELDS = {
    "medical_clinical_development_question",
    "strategic_development_question",
}

# Temporary compatibility alias for prompt/schema code during the two-question migration.
PARTICIPANT_REVIEW_KEYS = KEY_QUESTION_FIELDS

SUPPORTED_REVIEW_MODES = {
    "hidden_baseline",
    "first_visible_iteration",
    "later_visible_iteration",
}

COMPLETION_OUTLOOK_ANALYSIS_KEYS = {
    "risk_pattern_summary",
    "driver_summary",
    "main_model_signals",
    "interpretive_hypotheses",
    "movement_explanation",
    "model_boundary_note",
}

DESIGN_CONFIDENCE_ANALYSIS_KEYS = {
    "summary",
    "confidence_rationale",
    "supporting_evidence",
    "limiting_evidence",
}

STRATEGIC_REVIEW_ANALYSIS_KEYS = {
    "summary",
    "overall_score_explanation",
    "pillar_readout",
    "strategic_review_bullet",
    "tension_question",
    "broader_strategic_question",
    "review_rationale",
    "supporting_evidence",
    "limiting_evidence",
}

SCENARIO_CONSISTENCY_NOTE_KEYS = {
    "has_clear_mismatch",
    "message",
    "fields_in_tension",
}


def _clean_points(value: int | float) -> int | float:
    numeric = round(float(value), 1)
    return int(numeric) if numeric.is_integer() else numeric


def clamp(value: int | float, minimum: int | float, maximum: int | float) -> int | float:
    return _clean_points(max(minimum, min(maximum, float(value))))


def _add_nested_evidence_refs(refs: set[str], prefix: str, value: Any) -> None:
    refs.add(prefix)
    if isinstance(value, dict):
        for key, child in value.items():
            _add_nested_evidence_refs(refs, f"{prefix}.{key}", child)


def _evidence_reference_set(packet: dict[str, Any]) -> set[str]:
    refs = {
        "completion_score",
        "score_delta",
        "pillar_impacts",
        "pillar_deltas",
        "field_changes",
        "xgboost_impact_changes",
        "clarification_context.user_clarifications",
    }

    for section_name in ("trial_identity", "text_context", "structured_features", "structured_feature_display_values"):
        section = packet.get(section_name) or {}
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            refs.add(str(key))
            refs.add(f"{section_name}.{key}")
            if isinstance(value, dict):
                _add_nested_evidence_refs(refs, f"{section_name}.{key}", value)

    operational = packet.get("operational_assumptions") or {}
    if isinstance(operational, dict):
        for key, value in operational.items():
            refs.add(str(key))
            _add_nested_evidence_refs(refs, f"operational_assumptions.{key}", value)

    model = packet.get("model_interpretation") or {}
    if isinstance(model, dict):
        for key, value in model.items():
            refs.add(str(key))
            refs.add(f"model_interpretation.{key}")
            if isinstance(value, dict):
                _add_nested_evidence_refs(refs, f"model_interpretation.{key}", value)
        for impact in model.get("xgboost_impact_changes") or []:
            if not isinstance(impact, dict):
                continue
            for key in ("name", "pillar", "subcategory"):
                value = impact.get(key)
                if value:
                    refs.add(str(value))
                    refs.add(f"xgboost_impact_changes.{value}")

    iteration = packet.get("iteration_context") or {}
    if isinstance(iteration, dict):
        for field in iteration.get("changed_fields") or []:
            refs.add(str(field))
        for change in iteration.get("field_changes") or []:
            if not isinstance(change, dict):
                continue
            field = str(change.get("field") or "")
            if field:
                refs.add(field)
                refs.add(f"field_changes.{field}")

    return refs


def _supported_evidence(evidence_fields: list[str], packet: dict[str, Any]) -> tuple[list[str], list[str]]:
    supported_refs = _evidence_reference_set(packet)
    supported = [field for field in evidence_fields if field in supported_refs]
    unsupported = [field for field in evidence_fields if field not in supported_refs]
    return supported, unsupported


def _base_design_points(
    movement_direction: str,
    movement_materiality: str,
    effect_role: str,
) -> float:
    """Map movement-based qualitative review fields to deterministic raw Design Confidence points."""
    if movement_direction in POSITIVE_MOVEMENTS:
        sign = 1.0
    elif movement_direction in NEGATIVE_MOVEMENTS:
        sign = -1.0
    else:
        sign = 0.0
    magnitude = MOVEMENT_MATERIALITY_POINTS.get(movement_materiality, 0.0)
    multiplier = EFFECT_ROLE_MULTIPLIERS.get(effect_role, 1.0)
    return sign * magnitude * multiplier


def _legacy_movement_direction(rating: str | None) -> str:
    return {
        "strong": "improved",
        "supportive": "improved",
        "balanced": "unchanged",
        "weak": "weakened",
        "conflicting": "worsened",
    }.get(str(rating or ""), "unchanged")


def _legacy_movement_materiality(score_materiality: str | None, movement_direction: str) -> str:
    if movement_direction == "unchanged":
        return "none"
    return {
        "minimal": "minor",
        "low": "minor",
        "moderate": "moderate",
        "high": "major",
        "very_high": "major",
    }.get(str(score_materiality or ""), "minor")


def _validated_subcategory(subcategory_name: str, subcategory: Any) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(subcategory, dict):
        return {
            "rating": None,
            "rationale": "",
            "evidence_fields": [],
            "valid": False,
            "validation_notes": ["design subcategory is not an object"],
        }, [f"{subcategory_name}: design subcategory is not an object"]

    current_state = subcategory.get("current_state", subcategory.get("rating"))
    movement_direction = subcategory.get("movement_direction")
    movement_materiality = subcategory.get("movement_materiality")
    effect_role = subcategory.get("effect_role")
    rating = subcategory.get("rating", current_state)
    score_materiality = subcategory.get("score_materiality")
    if movement_direction is None:
        movement_direction = _legacy_movement_direction(rating)
    if movement_materiality is None:
        movement_materiality = _legacy_movement_materiality(score_materiality, str(movement_direction))
    if effect_role is None:
        effect_role = "unchanged" if movement_direction == "unchanged" else "independent"
    rationale = subcategory.get("rationale")
    evidence_fields = subcategory.get("evidence_fields")
    short_rationale = subcategory.get("short_rationale")
    optional_lenses_used = subcategory.get("optional_lenses_used")
    regulatory_or_finance_note = subcategory.get("regulatory_or_finance_note")
    valid = True

    if current_state not in CURRENT_STATE_LEVELS:
        errors.append(f"{subcategory_name}: invalid current_state {current_state!r}")
        valid = False
    if movement_direction not in MOVEMENT_DIRECTIONS:
        errors.append(f"{subcategory_name}: invalid movement_direction {movement_direction!r}")
        valid = False
    if movement_materiality not in MOVEMENT_MATERIALITY_LEVELS:
        errors.append(f"{subcategory_name}: invalid movement_materiality {movement_materiality!r}")
        valid = False
    if effect_role not in EFFECT_ROLES:
        errors.append(f"{subcategory_name}: invalid effect_role {effect_role!r}")
        valid = False
    if not isinstance(rationale, str):
        errors.append(f"{subcategory_name}: rationale must be a string")
        rationale = ""
        valid = False
    if not isinstance(evidence_fields, list):
        errors.append(f"{subcategory_name}: evidence_fields must be a list")
        evidence_fields = []
        valid = False
    if not isinstance(short_rationale, str):
        errors.append(f"{subcategory_name}: short_rationale must be a string")
        short_rationale = ""
        valid = False
    if not isinstance(optional_lenses_used, list):
        errors.append(f"{subcategory_name}: optional_lenses_used must be a list")
        optional_lenses_used = []
        valid = False
    if not isinstance(regulatory_or_finance_note, str):
        errors.append(f"{subcategory_name}: regulatory_or_finance_note must be a string")
        regulatory_or_finance_note = ""
        valid = False

    return {
        "current_state": current_state,
        "movement_direction": movement_direction,
        "movement_materiality": movement_materiality,
        "effect_role": effect_role,
        "rating": rating,
        "score_materiality": score_materiality,
        "rationale": rationale,
        "evidence_fields": [str(field) for field in evidence_fields],
        "short_rationale": short_rationale,
        "optional_lenses_used": [
            str(lens)
            for lens in optional_lenses_used
            if str(lens).strip()
        ],
        "regulatory_or_finance_note": regulatory_or_finance_note,
        "valid": valid,
        "validation_notes": [],
    }, errors


def _validate_key_questions(review: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    questions = review.get("key_questions")
    if not isinstance(questions, dict):
        # Backward compatibility for cached V2 reviews.
        old_participant = review.get("participant_review")
        if isinstance(old_participant, dict):
            questions = {
                "medical_clinical_development_question": (
                    old_participant.get("medical_clinical_development_question")
                    or old_participant.get("medical_development_question")
                ),
                "clinical_operations_question": (
                    old_participant.get("clinical_operations_question")
                    or old_participant.get("clinops_execution_question")
                ),
                "strategic_development_question": (
                    old_participant.get("strategic_development_question")
                    or old_participant.get("strategic_field_question")
                    or old_participant.get("clinical_operations_question")
                    or old_participant.get("clinops_execution_question")
                ),
            }
        else:
            return {}, ["key_questions must be an object"]

    normalized = {
        "medical_clinical_development_question": (
            questions.get("medical_clinical_development_question")
            or questions.get("medical_development_question")
        ),
        "strategic_development_question": (
            questions.get("strategic_development_question")
            or questions.get("strategic_field_question")
            or questions.get("clinical_operations_question")
        ),
    }
    errors = [
        f"key_questions.{key} must be a string"
        for key in sorted(KEY_QUESTION_FIELDS)
        if not isinstance(normalized.get(key), str)
    ]
    if errors:
        return {
            key: value
            for key, value in normalized.items()
            if isinstance(value, str)
        }, errors

    # Keep legacy aliases during migration so existing UI, reports, and cached traces degrade gracefully.
    return {
        "medical_clinical_development_question": normalized["medical_clinical_development_question"],
        "strategic_development_question": normalized["strategic_development_question"],
        "medical_development_question": normalized["medical_clinical_development_question"],
        "clinical_operations_question": (
            questions.get("clinical_operations_question")
            or questions.get("clinops_execution_question")
            or ""
        ),
        "strategic_field_question": normalized["strategic_development_question"],
    }, []


def _validate_object(value: Any, field_name: str, required: bool = True) -> tuple[dict[str, Any], list[str]]:
    if value is None and not required:
        return {}, []
    if not isinstance(value, dict):
        return {}, [f"{field_name} must be an object"]
    return deepcopy(value), []


def _validate_review_metadata(value: Any) -> tuple[dict[str, Any], list[str]]:
    metadata, errors = _validate_object(value, "review_metadata")
    if errors:
        return {}, errors
    review_mode = metadata.get("review_mode")
    visible = metadata.get("visible")
    if visible is None:
        visible = metadata.get("participant_visible")
    if review_mode not in SUPPORTED_REVIEW_MODES:
        errors.append(f"review_metadata.review_mode must be one of {sorted(SUPPORTED_REVIEW_MODES)}")
    if not isinstance(visible, bool):
        errors.append("review_metadata.visible must be a boolean")
    normalized = deepcopy(metadata)
    if isinstance(visible, bool):
        normalized["visible"] = visible
        normalized.setdefault("participant_visible", visible)
    return normalized, errors


def _validate_string_array(value: Any, field_name: str) -> list[str]:
    return [
        f"{field_name} must be an array of strings"
    ] if not isinstance(value, list) or any(not isinstance(item, str) for item in value) else []


def _validate_analysis_object(
    value: Any,
    field_name: str,
    required_fields: set[str],
    array_fields: set[str],
) -> tuple[dict[str, Any], list[str]]:
    obj, errors = _validate_object(value, field_name)
    if errors:
        return {}, errors
    for key in sorted(required_fields):
        if key not in obj:
            errors.append(f"{field_name}.{key} is required")
            continue
        if key in array_fields:
            errors.extend(_validate_string_array(obj.get(key), f"{field_name}.{key}"))
        elif not isinstance(obj.get(key), str):
            errors.append(f"{field_name}.{key} must be a string")
    return obj, errors


def _validate_strategic_review_analysis(value: Any) -> tuple[dict[str, Any], list[str]]:
    obj, errors = _validate_object(value, "strategic_review_analysis")
    if errors:
        return {}, errors

    # Legacy cached/provider traces had only the four fallback fields. Keep them
    # displayable, but require the structured fields for new provider responses.
    legacy_only = not any(key in obj for key in STRATEGIC_REVIEW_ANALYSIS_KEYS - {
        "summary",
        "review_rationale",
        "supporting_evidence",
        "limiting_evidence",
    })
    required_fields = (
        {"summary", "review_rationale", "supporting_evidence", "limiting_evidence"}
        if legacy_only
        else STRATEGIC_REVIEW_ANALYSIS_KEYS
    )
    for key in sorted(required_fields):
        if key not in obj:
            errors.append(f"strategic_review_analysis.{key} is required")
            continue
        if key in {"supporting_evidence", "limiting_evidence"}:
            errors.extend(_validate_string_array(obj.get(key), f"strategic_review_analysis.{key}"))
        elif key == "pillar_readout":
            items = obj.get(key)
            if not isinstance(items, list):
                errors.append("strategic_review_analysis.pillar_readout must be an array")
                continue
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    errors.append(f"strategic_review_analysis.pillar_readout[{index}] must be an object")
                    continue
                if not isinstance(item.get("label"), str):
                    errors.append(f"strategic_review_analysis.pillar_readout[{index}].label must be a string")
                if not isinstance(item.get("interpretation"), str):
                    errors.append(f"strategic_review_analysis.pillar_readout[{index}].interpretation must be a string")
        elif not isinstance(obj.get(key), str):
            errors.append(f"strategic_review_analysis.{key} must be a string")
    return obj, errors


def _validate_completion_outlook_analysis(value: Any, field_name: str) -> tuple[dict[str, Any], list[str]]:
    obj, errors = _validate_object(value, field_name)
    if errors:
        return {}, errors
    for key in sorted(COMPLETION_OUTLOOK_ANALYSIS_KEYS):
        if key not in obj:
            errors.append(f"{field_name}.{key} is required")
            continue
        if key == "main_model_signals":
            errors.extend(_validate_string_array(obj.get(key), f"{field_name}.{key}"))
        elif key == "interpretive_hypotheses":
            hypotheses = obj.get(key)
            if not isinstance(hypotheses, list):
                errors.append(f"{field_name}.{key} must be an array")
            else:
                for index, hypothesis in enumerate(hypotheses):
                    if not isinstance(hypothesis, dict):
                        errors.append(f"{field_name}.{key}[{index}] must be an object")
                        continue
                    for child_key in ("signal", "possible_pattern", "boundary"):
                        if not isinstance(hypothesis.get(child_key), str):
                            errors.append(f"{field_name}.{key}[{index}].{child_key} must be a string")
                    errors.extend(
                        _validate_string_array(
                            hypothesis.get("context_modifiers"),
                            f"{field_name}.{key}[{index}].context_modifiers",
                        )
                    )
        elif not isinstance(obj.get(key), str):
            errors.append(f"{field_name}.{key} must be a string")
    return obj, errors


def _validate_scenario_consistency_note(value: Any) -> tuple[dict[str, Any], list[str]]:
    obj, errors = _validate_object(value, "scenario_consistency_note", required=False)
    if not obj and not errors:
        return {
            "has_clear_mismatch": False,
            "message": "",
            "fields_in_tension": [],
        }, []
    if errors:
        return {}, errors
    for key in sorted(SCENARIO_CONSISTENCY_NOTE_KEYS):
        if key not in obj:
            errors.append(f"scenario_consistency_note.{key} is required")
    if "has_clear_mismatch" in obj and not isinstance(obj.get("has_clear_mismatch"), bool):
        errors.append("scenario_consistency_note.has_clear_mismatch must be a boolean")
    if "message" in obj and not isinstance(obj.get("message"), str):
        errors.append("scenario_consistency_note.message must be a string")
    if "fields_in_tension" in obj:
        errors.extend(_validate_string_array(obj.get("fields_in_tension"), "scenario_consistency_note.fields_in_tension"))
    return obj, errors


def _validate_continuity(value: Any) -> tuple[dict[str, Any], list[str]]:
    obj, errors = _validate_object(value, "continuity")
    if errors:
        return {}, errors
    normalized = deepcopy(obj)
    for key in (
        "prior_concerns_resolved",
        "prior_concerns_worsened",
        "prior_concerns_unchanged",
        "new_concerns",
    ):
        if key not in normalized:
            normalized[key] = []
        errors.extend(_validate_string_array(normalized.get(key), f"continuity.{key}"))
        if not isinstance(normalized.get(key), list):
            normalized[key] = []
    if "storyline_update" not in normalized:
        normalized["storyline_update"] = ""
    if not isinstance(normalized.get("storyline_update"), str):
        errors.append("continuity.storyline_update must be a string")
        normalized["storyline_update"] = ""
    return normalized, errors


def _validate_main_tension(review: dict[str, Any], design_confidence_analysis: dict[str, Any]) -> tuple[str, list[str]]:
    value = review.get("main_tension")
    if value is None:
        value = (review.get("strategic_review") or {}).get("current_tension")
    if value is None:
        value = (review.get("tradeoff_review") or {}).get("central_tension")
    if value is None:
        value = design_confidence_analysis.get("confidence_rationale", "")
    if not isinstance(value, str):
        return "", ["main_tension must be a string"]
    return value, []


def _validate_strategic_review(value: Any) -> tuple[dict[str, Any], list[str]]:
    obj, errors = _validate_object(value, "strategic_review")
    if errors:
        return {}, errors

    required_string_fields = {
        "effect_label",
        "tension_status",
        "current_tension",
        "tradeoff_resolution",
        "rationale",
        "next_consideration",
    }
    for field_name in sorted(required_string_fields):
        if not isinstance(obj.get(field_name), str):
            errors.append(f"strategic_review.{field_name} must be a string")

    carryover = obj.get("carryover_check", "")
    if carryover is not None and not isinstance(carryover, str):
        errors.append("strategic_review.carryover_check must be a string")

    effect_label = obj.get("effect_label")
    if isinstance(effect_label, str) and effect_label not in STRATEGIC_REVIEW_EFFECT_LABELS:
        errors.append(f"strategic_review.effect_label must be one of {sorted(STRATEGIC_REVIEW_EFFECT_LABELS)}")

    tension_status = obj.get("tension_status")
    if isinstance(tension_status, str) and tension_status not in TENSION_STATUS_FACTORS:
        errors.append(f"strategic_review.tension_status must be one of {sorted(TENSION_STATUS_FACTORS)}")

    operational_materiality = obj.get("operational_materiality", "minor")
    if operational_materiality is None:
        operational_materiality = "minor"
    if not isinstance(operational_materiality, str):
        errors.append("strategic_review.operational_materiality must be a string")
        operational_materiality = "minor"
    elif operational_materiality not in OPERATIONAL_MATERIALITY_BUDGETS:
        errors.append(
            "strategic_review.operational_materiality must be one of "
            f"{sorted(OPERATIONAL_MATERIALITY_BUDGETS)}"
        )

    evidence_fields = obj.get("evidence_fields", [])
    if not isinstance(evidence_fields, list) or any(not isinstance(item, str) for item in evidence_fields):
        errors.append("strategic_review.evidence_fields must be an array of strings")
        evidence_fields = []

    move_classification = obj.get("move_classification", [])
    if not isinstance(move_classification, list) or any(not isinstance(item, str) for item in move_classification):
        errors.append("strategic_review.move_classification must be an array of strings")
        move_classification = []

    return {
        **deepcopy(obj),
        "effect_label": effect_label if isinstance(effect_label, str) else "",
        "tension_status": tension_status if isinstance(tension_status, str) else "",
        "operational_materiality": operational_materiality,
        "evidence_fields": [str(field) for field in evidence_fields],
        "move_classification": [str(item) for item in move_classification],
        "carryover_check": carryover if isinstance(carryover, str) else "",
    }, errors


def _is_operational_only_packet(packet: dict[str, Any]) -> bool:
    changed_fields = [
        str(field)
        for field in ((packet.get("iteration_context") or {}).get("changed_fields") or [])
    ]
    if not changed_fields:
        return False
    return all(field.startswith("operational_assumptions.") for field in changed_fields)


def _completion_delta(packet: dict[str, Any]) -> float:
    model = packet.get("model_interpretation") or {}
    score_delta = model.get("score_delta")
    if isinstance(score_delta, (int, float)):
        return float(score_delta)
    current = model.get("completion_score")
    previous = model.get("previous_completion_score")
    if isinstance(current, (int, float)) and isinstance(previous, (int, float)):
        return float(current) - float(previous)
    return 0.0


def _strategic_review_budget(packet: dict[str, Any], strategic_review: dict[str, Any]) -> tuple[float, str]:
    if _is_operational_only_packet(packet):
        materiality = str(strategic_review.get("operational_materiality") or "minor")
        return OPERATIONAL_MATERIALITY_BUDGETS.get(materiality, 2.0), f"operational_{materiality}"
    movement_size = abs(_completion_delta(packet))
    return max(2.0, 0.40 * movement_size), "completion_outlook_delta"


def _latest_move_factor(packet: dict[str, Any], strategic_review: dict[str, Any]) -> float:
    effect_label = str(strategic_review.get("effect_label") or "neutral")
    if _is_operational_only_packet(packet):
        return FLAT_COMPLETION_REVIEW_FACTORS.get(effect_label, 0.0)
    delta = _completion_delta(packet)
    if delta > 0:
        return POSITIVE_COMPLETION_REVIEW_FACTORS.get(effect_label, 0.0)
    if delta < 0:
        return NEGATIVE_COMPLETION_REVIEW_FACTORS.get(effect_label, 0.0)
    return FLAT_COMPLETION_REVIEW_FACTORS.get(effect_label, 0.0)


def _effect_label_movement_error(packet: dict[str, Any], strategic_review: dict[str, Any]) -> str | None:
    effect_label = str(strategic_review.get("effect_label") or "neutral")
    if _is_operational_only_packet(packet):
        allowed = set(FLAT_COMPLETION_REVIEW_FACTORS)
        movement_name = "operational-only or flat Completion Outlook movement"
    else:
        delta = _completion_delta(packet)
        if delta > 0:
            allowed = set(POSITIVE_COMPLETION_REVIEW_FACTORS)
            movement_name = "positive Completion Outlook movement"
        elif delta < 0:
            allowed = set(NEGATIVE_COMPLETION_REVIEW_FACTORS)
            movement_name = "negative Completion Outlook movement"
        else:
            allowed = set(FLAT_COMPLETION_REVIEW_FACTORS)
            movement_name = "flat Completion Outlook movement"
    if effect_label not in allowed:
        return (
            f"strategic_review.effect_label {effect_label!r} is incompatible with "
            f"{movement_name}; expected one of {sorted(allowed)}"
        )
    return None


def _is_hidden_baseline_review(packet: dict[str, Any], validated_review: dict[str, Any]) -> bool:
    metadata = validated_review.get("review_metadata") or {}
    if metadata.get("review_mode") == "hidden_baseline":
        return True
    iteration = packet.get("iteration_context") or {}
    changed_fields = iteration.get("changed_fields") or []
    return (
        not changed_fields
        and iteration.get("previous_snapshot_id") is None
        and iteration.get("current_snapshot_id") == iteration.get("baseline_snapshot_id")
    )


def _strategic_review_contributions(
    packet: dict[str, Any],
    strategic_review: dict[str, Any],
) -> dict[str, Any]:
    evidence_fields = list(strategic_review.get("evidence_fields") or [])
    supported, unsupported = _supported_evidence(evidence_fields, packet)
    budget, budget_source = _strategic_review_budget(packet, strategic_review)
    latest_factor = _latest_move_factor(packet, strategic_review)
    tension_status = str(strategic_review.get("tension_status") or "not_applicable")
    tension_factor = TENSION_STATUS_FACTORS.get(tension_status, 0.0)
    combined_factor = latest_factor + tension_factor
    raw_points = budget * combined_factor
    if raw_points and not supported:
        points = 0.0
        validation_notes = ["strategic review has no point effect because evidence_fields do not reference packet evidence"]
    else:
        points = raw_points
        validation_notes = []

    sublevels = {
        "current_tension": {
            "label": STRATEGIC_REVIEW_SUBLEVELS["current_tension"],
            "text": strategic_review.get("current_tension", ""),
            "factor": _clean_points(latest_factor),
        },
        "tradeoff_resolution": {
            "label": STRATEGIC_REVIEW_SUBLEVELS["tradeoff_resolution"],
            "text": strategic_review.get("tradeoff_resolution", ""),
            "factor": _clean_points(latest_factor),
        },
    }
    carryover_text = str(strategic_review.get("carryover_check") or "").strip()
    if carryover_text or tension_factor:
        sublevels["carryover_check"] = {
            "label": STRATEGIC_REVIEW_SUBLEVELS["carryover_check"],
            "text": carryover_text,
            "factor": _clean_points(tension_factor),
        }

    return {
        "effect_label": strategic_review.get("effect_label"),
        "tension_status": tension_status,
        "operational_materiality": strategic_review.get("operational_materiality"),
        "move_classification": deepcopy(strategic_review.get("move_classification") or []),
        "budget": _clean_points(budget),
        "budget_source": budget_source,
        "latest_move_factor": _clean_points(latest_factor),
        "tension_status_factor": _clean_points(tension_factor),
        "combined_review_factor": _clean_points(combined_factor),
        "raw_points": _clean_points(raw_points),
        "points": _clean_points(points),
        "supported_evidence_fields": supported,
        "unsupported_evidence_fields": unsupported,
        "validation_notes": validation_notes,
        "sublevels": sublevels,
        "rationale": strategic_review.get("rationale", ""),
        "next_consideration": strategic_review.get("next_consideration", ""),
    }


def _has_complete_design_subcategories(validated_review: dict[str, Any]) -> bool:
    subcategories = validated_review.get("design_confidence_subcategories") or {}
    if set(subcategories) != REQUIRED_DESIGN_SUBCATEGORIES:
        return False
    return all(subcategory.get("valid") is True for subcategory in subcategories.values())


def _blocking_validation_errors(validated_review: dict[str, Any]) -> list[str]:
    errors = list(validated_review.get("validation_errors") or [])
    return [
        error
        for error in errors
        if "is application-owned and ignored if returned by provider" not in str(error)
    ]


def _score_subcategory(packet: dict[str, Any], subcategory_name: str, subcategory: dict[str, Any]) -> dict[str, Any]:
    evidence_fields = list(subcategory.get("evidence_fields") or [])
    supported, unsupported = _supported_evidence(evidence_fields, packet)
    raw_points = _base_design_points(
        str(subcategory.get("movement_direction")),
        str(subcategory.get("movement_materiality")),
        str(subcategory.get("effect_role")),
    )
    notes = list(subcategory.get("validation_notes") or [])
    if raw_points and not supported:
        points = 0
        notes.append("rating has no point effect because evidence_fields do not reference packet evidence")
    else:
        points = raw_points
    points = clamp(points, DESIGN_SUBCATEGORY_MIN, DESIGN_SUBCATEGORY_MAX)
    return {
        **deepcopy(subcategory),
        "supported_evidence_fields": supported,
        "unsupported_evidence_fields": unsupported,
        "raw_points": _clean_points(raw_points),
        "points": points,
        "calibration_notes": [],
        "validation_notes": notes,
    }


def _scenario_materiality_cap(packet: dict[str, Any]) -> float:
    model = packet.get("model_interpretation") or {}
    score_delta = model.get("score_delta")
    abs_delta = abs(float(score_delta)) if isinstance(score_delta, (int, float)) else 0.0
    if abs_delta < 1:
        delta_cap = 2.0
    elif abs_delta < 3:
        delta_cap = 3.0
    elif abs_delta < 6:
        delta_cap = 4.0
    else:
        delta_cap = 5.0

    changed_fields = [
        str(field)
        for field in ((packet.get("iteration_context") or {}).get("changed_fields") or [])
    ]
    structured_changes = [
        field for field in changed_fields
        if not field.startswith("text_context.") and not field.startswith("operational_assumptions.")
    ]
    operational_changes = [field for field in changed_fields if field.startswith("operational_assumptions.")]
    text_changes = [field for field in changed_fields if field.startswith("text_context.")]

    if len(structured_changes) >= 4:
        change_cap = 5.0
    elif len(structured_changes) >= 2:
        change_cap = 4.0
    elif len(structured_changes) == 1:
        change_cap = 3.0
    elif operational_changes:
        change_cap = 2.0
    elif text_changes:
        change_cap = 1.5
    else:
        change_cap = 1.0

    # Major coherent redesigns can exceed a flat-score cap, but still stay inside a small adjustment range.
    return min(DESIGN_CONFIDENCE_EXCEPTIONAL_CAP, max(delta_cap, change_cap))


def _apply_net_cap(subcategory_results: dict[str, dict[str, Any]], cap: float) -> None:
    net = sum(float(item.get("points") or 0) for item in subcategory_results.values())
    if abs(net) <= cap or not net:
        return

    if net > cap:
        side_total = sum(float(item.get("points") or 0) for item in subcategory_results.values() if float(item.get("points") or 0) > 0)
        allowed_side_total = side_total - (net - cap)
        scale = max(0.0, allowed_side_total / side_total) if side_total else 0.0
        side = "positive"
    else:
        side_total = sum(abs(float(item.get("points") or 0)) for item in subcategory_results.values() if float(item.get("points") or 0) < 0)
        allowed_side_total = side_total - (abs(net) - cap)
        scale = max(0.0, allowed_side_total / side_total) if side_total else 0.0
        side = "negative"

    for item in subcategory_results.values():
        points = float(item.get("points") or 0)
        if (side == "positive" and points > 0) or (side == "negative" and points < 0):
            item["points"] = _clean_points(points * scale)
            item.setdefault("calibration_notes", []).append(
                f"{side} Design Confidence movement scaled to keep net adjustment within +/-{_clean_points(cap)}"
            )


def _design_contributions(packet: dict[str, Any], validated_subcategories: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pillars = {
        key: {
            "label": label,
            "completion_outlook_component": None,
            "design_subcategories": {},
            "raw_design_points": 0,
            "design_points": 0,
        }
        for key, label in DESIGN_PILLAR_LABELS.items()
    }

    subcategory_results: dict[str, dict[str, Any]] = {}
    for subcategory_name, subcategory in validated_subcategories.items():
        scored = _score_subcategory(packet, subcategory_name, subcategory)
        subcategory_results[subcategory_name] = scored

    design_cap = _scenario_materiality_cap(packet)
    _apply_net_cap(subcategory_results, design_cap)

    for subcategory_name, scored in subcategory_results.items():
        pillar_key = DESIGN_SUBCATEGORY_PILLARS[subcategory_name]
        pillars[pillar_key]["design_subcategories"][subcategory_name] = {
            "label": DESIGN_SUBCATEGORY_LABELS[subcategory_name],
            "current_state": scored.get("current_state"),
            "movement_direction": scored.get("movement_direction"),
            "movement_materiality": scored.get("movement_materiality"),
            "effect_role": scored.get("effect_role"),
            "rating": scored.get("rating"),
            "score_materiality": scored.get("score_materiality"),
            "raw_points": scored.get("raw_points"),
            "points": scored.get("points"),
            "evidence_fields": deepcopy(scored.get("evidence_fields") or []),
            "rationale": scored.get("rationale"),
            "short_rationale": scored.get("short_rationale"),
            "supported_evidence_fields": deepcopy(scored.get("supported_evidence_fields") or []),
            "unsupported_evidence_fields": deepcopy(scored.get("unsupported_evidence_fields") or []),
            "calibration_notes": deepcopy(scored.get("calibration_notes") or []),
            "validation_notes": deepcopy(scored.get("validation_notes") or []),
        }
        pillars[pillar_key]["raw_design_points"] += float(scored.get("raw_points") or 0)
        pillars[pillar_key]["design_points"] += float(scored.get("points") or 0)

    for pillar in pillars.values():
        pillar["raw_design_points"] = _clean_points(pillar["raw_design_points"])
        pillar["design_points"] = _clean_points(pillar["design_points"])

    design_confidence = _clean_points(sum(float(item.get("points") or 0) for item in subcategory_results.values()))
    return {
        "subcategories": subcategory_results,
        "pillars": pillars,
        "design_confidence": design_confidence,
        "design_confidence_cap": _clean_points(design_cap),
    }


def validate_review_json(review: dict[str, Any]) -> dict[str, Any]:
    """Validate provider/mock Scenario Review JSON and return normalized fields."""
    errors: list[str] = []
    if not isinstance(review, dict):
        return {
            "validation_status": "invalid",
            "validation_errors": ["review must be an object"],
            "review_metadata": {},
            "completion_outlook_analysis": {},
            "strategic_review": {},
            "strategic_review_analysis": {},
            "design_confidence_subcategories": {},
            "design_confidence_analysis": {},
            "main_tension": "",
            "key_questions": {},
            "scenario_consistency_note": {},
            "continuity": {},
            "trace": {},
        }

    for field_name in sorted(APP_OWNED_SCORE_FIELDS.intersection(review)):
        errors.append(f"{field_name} is application-owned and ignored if returned by provider")

    strategic_review, strategic_review_errors = _validate_strategic_review(review.get("strategic_review"))

    subcategories = review.get("design_confidence_subcategories")
    validated_subcategories: dict[str, dict[str, Any]] = {}
    if strategic_review and subcategories is None:
        pass
    elif not isinstance(subcategories, dict):
        # Legacy subcategories are tolerated only as ignored context when absent from the
        # new Strategic Review contract. They no longer unlock a score.
        pass
    else:
        missing = REQUIRED_DESIGN_SUBCATEGORIES.difference(subcategories)
        extra = set(subcategories).difference(REQUIRED_DESIGN_SUBCATEGORIES)
        for subcategory_name in sorted(missing):
            errors.append(f"{subcategory_name}: missing required design subcategory")
        for subcategory_name in sorted(extra):
            errors.append(f"{subcategory_name}: unexpected design subcategory")
        for subcategory_name in sorted(REQUIRED_DESIGN_SUBCATEGORIES):
            if subcategory_name not in subcategories:
                continue
            validated, subcategory_errors = _validated_subcategory(
                subcategory_name,
                subcategories[subcategory_name],
            )
            validated_subcategories[subcategory_name] = validated
            errors.extend(subcategory_errors)

    review_metadata, metadata_errors = _validate_review_metadata(review.get("review_metadata"))
    completion_source = review.get("completion_outlook_analysis")
    completion_field_name = "completion_outlook_analysis"
    if completion_source is None and isinstance(review.get("completion_outlook_review"), dict):
        old_completion = review.get("completion_outlook_review") or {}
        completion_source = {
            "risk_pattern_summary": old_completion.get("score_delta_summary", ""),
            "driver_summary": old_completion.get("score_delta_summary", ""),
            "main_model_signals": old_completion.get("model_supported_drivers") or [],
            "interpretive_hypotheses": [],
            "movement_explanation": old_completion.get("score_delta_summary", ""),
            "model_boundary_note": "Legacy completion_outlook_review normalized to completion_outlook_analysis.",
        }
        completion_field_name = "completion_outlook_review"
    completion_outlook_analysis, completion_errors = _validate_completion_outlook_analysis(
        completion_source,
        completion_field_name,
    )
    design_confidence_source = review.get("design_confidence_analysis")
    strategic_review_analysis_source = review.get("strategic_review_analysis")
    if strategic_review_analysis_source is None and strategic_review:
        strategic_review_analysis_source = {
            "summary": strategic_review.get("rationale", ""),
            "review_rationale": strategic_review.get("rationale", ""),
            "supporting_evidence": strategic_review.get("evidence_fields", []),
            "limiting_evidence": [],
        }
    if design_confidence_source is None and isinstance(review.get("tradeoff_review"), dict):
        tradeoff = review.get("tradeoff_review") or {}
        old_participant = review.get("participant_review") or {}
        design_confidence_source = {
            "summary": old_participant.get("overall_design_comment", ""),
            "confidence_rationale": tradeoff.get("central_tension", ""),
            "supporting_evidence": [],
            "limiting_evidence": [],
        }
    if design_confidence_source is None and strategic_review_analysis_source is not None:
        strategic_review_analysis, strategic_analysis_errors = _validate_strategic_review_analysis(
            strategic_review_analysis_source
        )
        design_confidence_analysis = {
            "summary": strategic_review_analysis.get("summary", ""),
            "confidence_rationale": strategic_review_analysis.get("review_rationale", ""),
            "supporting_evidence": strategic_review_analysis.get("supporting_evidence", []),
            "limiting_evidence": strategic_review_analysis.get("limiting_evidence", []),
        }
        design_analysis_errors = []
    else:
        strategic_review_analysis = {}
        strategic_analysis_errors = []
        design_confidence_analysis, design_analysis_errors = _validate_analysis_object(
            design_confidence_source,
            "design_confidence_analysis",
            DESIGN_CONFIDENCE_ANALYSIS_KEYS,
            {"supporting_evidence", "limiting_evidence"},
        )
    main_tension, main_tension_errors = _validate_main_tension(review, design_confidence_analysis)
    if not main_tension and strategic_review:
        main_tension = str(strategic_review.get("current_tension") or "")
    key_questions, question_errors = _validate_key_questions(review)
    consistency_note, consistency_errors = _validate_scenario_consistency_note(
        review.get("scenario_consistency_note")
    )
    continuity, continuity_errors = _validate_continuity(review.get("continuity"))
    trace, trace_errors = _validate_object(review.get("trace"), "trace")

    errors.extend(metadata_errors)
    errors.extend(strategic_review_errors)
    errors.extend(completion_errors)
    errors.extend(strategic_analysis_errors)
    errors.extend(design_analysis_errors)
    errors.extend(main_tension_errors)
    errors.extend(question_errors)
    errors.extend(consistency_errors)
    errors.extend(continuity_errors)
    errors.extend(trace_errors)

    return {
        "validation_status": "valid" if not errors else "partial",
        "validation_errors": errors,
        "review_metadata": review_metadata,
        "completion_outlook_analysis": completion_outlook_analysis,
        "strategic_review": strategic_review,
        "strategic_review_analysis": strategic_review_analysis,
        "design_confidence_subcategories": validated_subcategories,
        "design_confidence_analysis": design_confidence_analysis,
        "main_tension": main_tension,
        "key_questions": key_questions,
        "scenario_consistency_note": consistency_note,
        "continuity": continuity,
        "trace": trace,
    }


def score_validated_review(packet: dict[str, Any], validated_review: dict[str, Any]) -> dict[str, Any]:
    """Calculate app-owned Strategic Review and Trial Score."""
    completion_score = (packet.get("model_interpretation") or {}).get("completion_score")
    input_hash = packet.get("input_hash") or stable_packet_hash(packet)
    if not isinstance(completion_score, (int, float)):
        return {
            "validation_status": "invalid",
            "validation_errors": ["model_interpretation.completion_score must be numeric"],
            "strategic_review": None,
            "trial_score": None,
            "strategic_review_assessment": {},
            "design_confidence": None,
            "total_scenario_score": None,
            "design_confidence_assessment": {},
            "input_hash": input_hash,
        }

    blocking_errors = _blocking_validation_errors(validated_review)
    if _is_hidden_baseline_review(packet, validated_review):
        return {
            "validation_status": validated_review.get("validation_status", "partial"),
            "validation_errors": list(validated_review.get("validation_errors") or []),
            "strategic_review": None,
            "trial_score": None,
            "strategic_review_assessment": {},
            "design_confidence": None,
            "total_scenario_score": None,
            "design_confidence_assessment": {},
            "input_hash": input_hash,
        }
    if validated_review.get("strategic_review"):
        effect_label_error = _effect_label_movement_error(
            packet,
            validated_review.get("strategic_review") or {},
        )
        if effect_label_error:
            blocking_errors = [*blocking_errors, effect_label_error]
        if blocking_errors:
            validation_status = (
                "partial"
                if effect_label_error and validated_review.get("validation_status") == "valid"
                else validated_review.get("validation_status", "partial")
            )
            return {
                "validation_status": validation_status,
                "validation_errors": [
                    *list(validated_review.get("validation_errors") or []),
                    *[
                        error
                        for error in blocking_errors
                        if error not in list(validated_review.get("validation_errors") or [])
                    ],
                ],
                "strategic_review": None,
                "trial_score": None,
                "strategic_review_assessment": {},
                "design_confidence": None,
                "total_scenario_score": None,
                "design_confidence_assessment": {},
                "input_hash": input_hash,
            }
        assessment = _strategic_review_contributions(packet, validated_review.get("strategic_review") or {})
        strategic_review = assessment["points"]
        trial_score = clamp(
            float(completion_score) + float(strategic_review),
            TOTAL_SCORE_MIN,
            TOTAL_SCORE_MAX,
        )
        return {
            "validation_status": validated_review.get("validation_status", "partial"),
            "validation_errors": list(validated_review.get("validation_errors") or []),
            "strategic_review": strategic_review,
            "trial_score": trial_score,
            "strategic_review_assessment": assessment,
            # Compatibility aliases for modules that are migrating from the old labels.
            "design_confidence": strategic_review,
            "total_scenario_score": trial_score,
            "design_confidence_assessment": {
                "strategic_review": assessment,
                "sublevels": assessment.get("sublevels") or {},
                "design_confidence": strategic_review,
            },
            "input_hash": input_hash,
        }

    return {
        "validation_status": validated_review.get("validation_status", "partial"),
        "validation_errors": [
            *list(validated_review.get("validation_errors") or []),
            *([] if validated_review.get("strategic_review") else ["strategic_review is required for scoring"]),
        ],
        "strategic_review": None,
        "trial_score": None,
        "strategic_review_assessment": {},
        "design_confidence": None,
        "total_scenario_score": None,
        "design_confidence_assessment": {},
        "input_hash": input_hash,
    }


def validate_and_score_review(packet: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    """Validate Scenario Review JSON and return deterministic score fields."""
    validated_review = validate_review_json(review)
    scoring = score_validated_review(packet, validated_review)
    return {
        "validated_review": validated_review,
        "scoring": scoring,
    }
