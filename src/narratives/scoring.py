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

SCORE_MATERIALITY_LEVELS = {
    "minimal",
    "low",
    "moderate",
    "high",
    "very_high",
}

DESIGN_RATING_MATERIALITY_POINTS = {
    "strong": {
        "minimal": 3.0,
        "low": 3.5,
        "moderate": 4.0,
        "high": 4.5,
        "very_high": 5.0,
    },
    "supportive": {
        "minimal": 0.5,
        "low": 1.0,
        "moderate": 1.5,
        "high": 2.0,
        "very_high": 2.5,
    },
    "balanced": {
        "minimal": 0.0,
        "low": 0.0,
        "moderate": 0.0,
        "high": 0.0,
        "very_high": 0.0,
    },
    "weak": {
        "minimal": -0.5,
        "low": -1.0,
        "moderate": -1.5,
        "high": -2.0,
        "very_high": -2.5,
    },
    "conflicting": {
        "minimal": -3.0,
        "low": -3.5,
        "moderate": -4.0,
        "high": -4.5,
        "very_high": -5.0,
    },
}

# Temporary compatibility alias for prompt/schema code that migrates in Phase 4.
DOMAIN_RATING_POINTS = {
    subcategory_name: {
        rating: values["minimal"]
        for rating, values in DESIGN_RATING_MATERIALITY_POINTS.items()
    }
    for subcategory_name in sorted(REQUIRED_DESIGN_SUBCATEGORIES)
}

DESIGN_SUBCATEGORY_MIN = -5.0
DESIGN_SUBCATEGORY_MAX = 5.0
TOTAL_SCORE_MIN = 0
TOTAL_SCORE_MAX = 100

APP_OWNED_SCORE_FIELDS = {
    "design_confidence",
    "total_scenario_score",
    "design_confidence_assessment",
    "design_confidence_contributions",
    # Legacy names stay app-owned during the migration.
    "quality_adjustment",
    "final_candidate_score",
    "quality_assessment",
}

PARTICIPANT_REVIEW_KEYS = {
    "medical_development_question",
    "clinical_operations_question",
    "strategic_field_question",
}

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


def _numeric_score_delta(packet: dict[str, Any]) -> float:
    score_delta = (packet.get("model_interpretation") or {}).get("score_delta")
    return float(score_delta) if isinstance(score_delta, (int, float)) else 0.0


def _contains_any(evidence_fields: list[str], tokens: tuple[str, ...]) -> bool:
    joined = " ".join(str(field).lower() for field in evidence_fields)
    return any(token in joined for token in tokens)


def _changed_fields(packet: dict[str, Any]) -> set[str]:
    iteration = packet.get("iteration_context") or {}
    return {str(field) for field in iteration.get("changed_fields") or []}


def _changed_field_supports_evidence(packet: dict[str, Any], evidence_fields: list[str]) -> bool:
    changed = _changed_fields(packet)
    if not changed:
        return False
    for evidence in evidence_fields:
        evidence_text = str(evidence)
        if evidence_text in changed:
            return True
        if any(evidence_text.startswith(f"{field}.") or field.startswith(f"{evidence_text}.") for field in changed):
            return True
    return False


def _is_operational_only_change(packet: dict[str, Any]) -> bool:
    changed = _changed_fields(packet)
    return bool(changed) and all(field.startswith("operational_assumptions.") for field in changed)


def _completion_pillar_value(packet: dict[str, Any], subcategory_name: str) -> float | None:
    pillar_key = DESIGN_SUBCATEGORY_PILLARS.get(subcategory_name)
    pillar_label = DESIGN_PILLAR_LABELS.get(pillar_key or "")
    impacts = (packet.get("model_interpretation") or {}).get("pillar_impacts") or {}
    value = None
    if isinstance(impacts, dict):
        value = impacts.get(pillar_label)
    elif isinstance(impacts, list):
        for impact in impacts:
            if not isinstance(impact, dict):
                continue
            if impact.get("Pillar") == pillar_label:
                value = impact.get("Impact")
                break
    return float(value) if isinstance(value, (int, float)) else None


def _raw_design_points(
    subcategory_name: str,
    rating: str,
    score_materiality: str,
    evidence_fields: list[str],
    packet: dict[str, Any],
) -> float:
    """Map validated qualitative review fields to deterministic Design Confidence points."""
    raw_points = DESIGN_RATING_MATERIALITY_POINTS.get(rating, {}).get(score_materiality, 0.0)

    if (
        subcategory_name == "operational_burden_balance"
        and raw_points > 0
        and _is_operational_only_change(packet)
    ):
        return 0.0

    pillar_value = _completion_pillar_value(packet, subcategory_name)
    if (
        raw_points > 1.0
        and pillar_value is not None
        and pillar_value >= 4.0
        and _numeric_score_delta(packet) >= 0
        and not _changed_field_supports_evidence(packet, evidence_fields)
    ):
        return 1.0

    return raw_points


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

    rating = subcategory.get("rating")
    score_materiality = subcategory.get("score_materiality")
    rationale = subcategory.get("rationale")
    evidence_fields = subcategory.get("evidence_fields")
    short_rationale = subcategory.get("short_rationale")
    optional_lenses_used = subcategory.get("optional_lenses_used")
    regulatory_or_finance_note = subcategory.get("regulatory_or_finance_note")
    valid = True

    if rating not in DESIGN_RATINGS:
        errors.append(f"{subcategory_name}: invalid rating {rating!r}")
        valid = False
    if score_materiality not in SCORE_MATERIALITY_LEVELS:
        errors.append(f"{subcategory_name}: invalid score_materiality {score_materiality!r}")
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
                "medical_development_question": old_participant.get("medical_development_question"),
                "clinical_operations_question": old_participant.get("clinops_execution_question"),
                "strategic_field_question": old_participant.get("strategic_field_question", ""),
            }
        else:
            return {}, ["key_questions must be an object"]

    errors = [
        f"key_questions.{key} must be a string"
        for key in sorted(PARTICIPANT_REVIEW_KEYS)
        if not isinstance(questions.get(key), str)
    ]
    return {
        key: questions.get(key, "")
        for key in sorted(PARTICIPANT_REVIEW_KEYS)
        if isinstance(questions.get(key), str)
    }, errors


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
    participant_visible = metadata.get("participant_visible")
    if review_mode not in SUPPORTED_REVIEW_MODES:
        errors.append(f"review_metadata.review_mode must be one of {sorted(SUPPORTED_REVIEW_MODES)}")
    if not isinstance(participant_visible, bool):
        errors.append("review_metadata.participant_visible must be a boolean")
    return metadata, errors


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
    raw_points = _raw_design_points(
        subcategory_name,
        str(subcategory.get("rating")),
        str(subcategory.get("score_materiality")),
        evidence_fields,
        packet,
    )
    points = raw_points if supported or raw_points == 0 else 0
    notes = list(subcategory.get("validation_notes") or [])
    if raw_points and not supported:
        notes.append("rating has no point effect because evidence_fields do not reference packet evidence")
    points = clamp(points, DESIGN_SUBCATEGORY_MIN, DESIGN_SUBCATEGORY_MAX)
    return {
        **deepcopy(subcategory),
        "supported_evidence_fields": supported,
        "unsupported_evidence_fields": unsupported,
        "raw_points": _clean_points(raw_points),
        "points": points,
        "validation_notes": notes,
    }


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
        pillar_key = DESIGN_SUBCATEGORY_PILLARS[subcategory_name]
        pillars[pillar_key]["design_subcategories"][subcategory_name] = {
            "label": DESIGN_SUBCATEGORY_LABELS[subcategory_name],
            "rating": scored.get("rating"),
            "score_materiality": scored.get("score_materiality"),
            "raw_points": scored.get("raw_points"),
            "points": scored.get("points"),
            "evidence_fields": deepcopy(scored.get("evidence_fields") or []),
            "rationale": scored.get("rationale"),
            "short_rationale": scored.get("short_rationale"),
            "supported_evidence_fields": deepcopy(scored.get("supported_evidence_fields") or []),
            "unsupported_evidence_fields": deepcopy(scored.get("unsupported_evidence_fields") or []),
            "validation_notes": deepcopy(scored.get("validation_notes") or []),
        }
        pillars[pillar_key]["raw_design_points"] += float(scored.get("points") or 0)

    for pillar in pillars.values():
        pillar["design_points"] = _clean_points(pillar["raw_design_points"])

    design_confidence = _clean_points(sum(float(item.get("points") or 0) for item in subcategory_results.values()))
    return {
        "subcategories": subcategory_results,
        "pillars": pillars,
        "design_confidence": design_confidence,
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
            "design_confidence_subcategories": {},
            "design_confidence_analysis": {},
            "key_questions": {},
            "scenario_consistency_note": {},
            "continuity": {},
            "trace": {},
        }

    for field_name in sorted(APP_OWNED_SCORE_FIELDS.intersection(review)):
        errors.append(f"{field_name} is application-owned and ignored if returned by provider")

    subcategories = review.get("design_confidence_subcategories")
    validated_subcategories: dict[str, dict[str, Any]] = {}
    if not isinstance(subcategories, dict):
        errors.append("design_confidence_subcategories must be an object")
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
    if design_confidence_source is None and isinstance(review.get("tradeoff_review"), dict):
        tradeoff = review.get("tradeoff_review") or {}
        old_participant = review.get("participant_review") or {}
        design_confidence_source = {
            "summary": old_participant.get("overall_design_comment", ""),
            "confidence_rationale": tradeoff.get("central_tension", ""),
            "supporting_evidence": [],
            "limiting_evidence": [],
        }
    design_confidence_analysis, design_analysis_errors = _validate_analysis_object(
        design_confidence_source,
        "design_confidence_analysis",
        DESIGN_CONFIDENCE_ANALYSIS_KEYS,
        {"supporting_evidence", "limiting_evidence"},
    )
    key_questions, question_errors = _validate_key_questions(review)
    consistency_note, consistency_errors = _validate_scenario_consistency_note(
        review.get("scenario_consistency_note")
    )
    continuity, continuity_errors = _validate_object(review.get("continuity"), "continuity")
    trace, trace_errors = _validate_object(review.get("trace"), "trace")

    errors.extend(metadata_errors)
    errors.extend(completion_errors)
    errors.extend(design_analysis_errors)
    errors.extend(question_errors)
    errors.extend(consistency_errors)
    errors.extend(continuity_errors)
    errors.extend(trace_errors)

    return {
        "validation_status": "valid" if not errors else "partial",
        "validation_errors": errors,
        "review_metadata": review_metadata,
        "completion_outlook_analysis": completion_outlook_analysis,
        "design_confidence_subcategories": validated_subcategories,
        "design_confidence_analysis": design_confidence_analysis,
        "key_questions": key_questions,
        "scenario_consistency_note": consistency_note,
        "continuity": continuity,
        "trace": trace,
    }


def score_validated_review(packet: dict[str, Any], validated_review: dict[str, Any]) -> dict[str, Any]:
    """Calculate app-owned Design Confidence and Total Scenario Score."""
    completion_score = (packet.get("model_interpretation") or {}).get("completion_score")
    input_hash = packet.get("input_hash") or stable_packet_hash(packet)
    if not isinstance(completion_score, (int, float)):
        return {
            "validation_status": "invalid",
            "validation_errors": ["model_interpretation.completion_score must be numeric"],
            "design_confidence": None,
            "total_scenario_score": None,
            "design_confidence_assessment": {},
            "input_hash": input_hash,
        }

    blocking_errors = _blocking_validation_errors(validated_review)
    if blocking_errors or not _has_complete_design_subcategories(validated_review):
        return {
            "validation_status": validated_review.get("validation_status", "partial"),
            "validation_errors": list(validated_review.get("validation_errors") or []),
            "design_confidence": None,
            "total_scenario_score": None,
            "design_confidence_assessment": {},
            "input_hash": input_hash,
        }

    contributions = _design_contributions(packet, validated_review.get("design_confidence_subcategories") or {})
    total_scenario_score = clamp(
        float(completion_score) + float(contributions["design_confidence"]),
        TOTAL_SCORE_MIN,
        TOTAL_SCORE_MAX,
    )

    return {
        "validation_status": validated_review.get("validation_status", "partial"),
        "validation_errors": list(validated_review.get("validation_errors") or []),
        "design_confidence": contributions["design_confidence"],
        "total_scenario_score": total_scenario_score,
        "design_confidence_assessment": contributions,
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
