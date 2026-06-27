"""V1 Trial Score contract and app validation rails.

This module owns the current accepted score stack documented in
``docs/trial_score_narrative_direction.md`` and
``docs/operational_fit_scoring.md``. It is intentionally provider-agnostic.
The active workflow separates evidence generation, LLM-owned score
adjudication, and participant narrative shaping. The app validates hard rails
and deterministic arithmetic, but it no longer converts symbolic Operational
Fit / Reality Check labels into points.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any

TRIAL_SCORE_CONTRACT_VERSION = "trial_score_v1"
PASS1_SCHEMA_VERSION = "trial_score_evidence_pass_schema_v4"
PASS2_SCHEMA_VERSION = "trial_score_scoring_pass_schema_v1"
PASS3_SCHEMA_VERSION = "trial_score_narrative_pass_schema_v1"
PROMPT_TEMPLATE_VERSION = "trial_score_three_pass_prompt_v2_2"
MIN_PASS2_PILLAR_READINGS = 2
MAX_PASS2_PILLAR_READINGS = 4
SCORE_TRACE_HISTORY_LIMIT = 5
MAX_POSITIVE_REALITY_CHECK_WHEN_PRE_REALITY_ALREADY_IMPROVED = 0.0

XGBOOST_COMPLETION_OUTLOOK_LABEL = "XGBoost Completion Outlook"
COMPLETION_OUTLOOK_LABEL = "Completion Outlook"
OPERATIONAL_FIT_LABEL = "Operational Fit"
REALITY_CHECK_LABEL = "Reality Check"
TRIAL_SCORE_LABEL = "Trial Score"

VISIBLE_SCORE_VIEWS = (
    COMPLETION_OUTLOOK_LABEL,
    REALITY_CHECK_LABEL,
    TRIAL_SCORE_LABEL,
)

OBSOLETE_VISIBLE_TERMS = (
    "Quality Review",
    "Quality Adjustment",
    "Final Candidate Score",
    "Design Confidence",
    "Total Scenario Score",
    "Strategic Review",
)

LEGACY_CACHE_ALIAS_FIELDS = {
    "quality_adjustment",
    "final_candidate_score",
    "quality_assessment",
    "design_confidence",
    "total_scenario_score",
    "design_confidence_assessment",
    "strategic_review",
    "strategic_review_assessment",
}

APP_OWNED_TRIAL_SCORE_FIELDS = {
    "operational_fit_points",
    "pre_reality_score",
    "pre_reality_delta",
    "reality_check_points",
    "reality_check_allocation_points",
    "trial_score",
    "delta_vs_previous_trial_score",
    "delta_vs_previous_pre_reality_score",
    "delta_vs_baseline_xgboost",
}

STRATEGY_SHIFT_STATUSES = {
    "supported",
    "partly_supported",
    "unsupported_or_incoherent",
    "not_applicable",
}

GATED_PREMISE_SENSITIVE_FIELDS = {
    "phase_ml",
    "strategic_ambition_ml",
    "therapeutic_modality_ml",
    "target_pathway_class_ml",
    "primary_purpose_ml",
}

REALITY_CHECK_CARRYOVER_MATERIALITY_THRESHOLD = -1.0

REALITY_CHECK_ALLOCATION_TARGETS = {
    "therapeutic_context.therapeutic_area_profile": {
        "pillar": "Therapeutic Context",
        "subpillar": "Therapeutic Area Profile",
        "description": "Disease context, patient relevance, benchmark context, and calibration limits.",
    },
    "therapeutic_context.development_phase_and_goal": {
        "pillar": "Therapeutic Context",
        "subpillar": "Development Phase and Goal",
        "description": "Whether the evidence standard, endpoint maturity, population scope, and operational scale fit the development decision.",
    },
    "scientific_challenge.biological_profile": {
        "pillar": "Scientific Challenge",
        "subpillar": "Biological Profile",
        "description": "Biological plausibility, novelty, modality risk, and the evidence burden the mechanism creates.",
    },
    "scientific_challenge.protocol_architecture": {
        "pillar": "Scientific Challenge",
        "subpillar": "Protocol Architecture",
        "description": "Whether the trial design architecture can credibly answer the clinical-development question.",
    },
    "patient_profile.clinical_severity": {
        "pillar": "Patient Profile",
        "subpillar": "Clinical Severity",
        "description": "Whether patient burden, acceptable risk, endpoint relevance, and unmet need fit the scenario.",
    },
    "patient_profile.population_scope": {
        "pillar": "Patient Profile",
        "subpillar": "Population Scope",
        "description": "Whether the population definition is credible, generalizable, ethically coherent, and recruitable.",
    },
    "execution_framework.trial_complexity_footprint": {
        "pillar": "Execution Framework",
        "subpillar": "Trial Complexity Footprint",
        "description": "SHAP-derived trial-footprint complexity, follow-up burden, site capability needs, and operational load.",
    },
    "execution_framework.methodological_setup": {
        "pillar": "Execution Framework",
        "subpillar": "Methodological Setup",
        "description": "Bias control, causal interpretability, governance, comparator credibility, and ethical or methodological setup.",
    },
    "execution_framework.operational_fit": {
        "pillar": "Execution Framework",
        "subpillar": "Operational Fit",
        "description": "Operational proportionality of planned enrollment, planned sites, planned duration, patients per site, and benchmarks.",
    },
}

ANALYTICAL_NARRATIVE_DRAFT_FIELDS = (
    "current_state_read",
    "movement_read",
    "operational_fit_read",
    "reality_check_read",
    "development_landscape_read",
)
MIN_ANALYTICAL_DRAFT_SUBSTANTIVE_FIELDS = 4
MIN_VISIBLE_DEVELOPMENT_DISCUSSION_OPTIONS = 1
MAX_VISIBLE_DEVELOPMENT_DISCUSSION_OPTIONS = 1
MIN_STRATEGIC_QUESTION_CANDIDATES = MAX_VISIBLE_DEVELOPMENT_DISCUSSION_OPTIONS

def _clean_points(value: int | float) -> int | float:
    numeric = round(float(value), 1)
    return int(numeric) if numeric.is_integer() else numeric


def clamp(value: int | float, minimum: int | float, maximum: int | float) -> int | float:
    return _clean_points(max(float(minimum), min(float(maximum), float(value))))


def _stable_hash(payload: Any) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def operational_fit_state_payload(packet: dict[str, Any]) -> dict[str, Any]:
    """Return current-state inputs that should preserve Operational Fit equivalence."""
    return {
        "operational_assumptions": deepcopy(packet.get("operational_assumptions") or {}),
        "operational_movement_context": deepcopy(packet.get("operational_movement_context") or {}),
        "structured_features": deepcopy(packet.get("structured_features") or {}),
    }


def operational_fit_state_hash(packet: dict[str, Any]) -> str:
    """Hash operational estimates plus benchmark/context fields relevant to Operational Fit."""
    return _stable_hash(operational_fit_state_payload(packet))


def xgboost_structured_state_payload(packet: dict[str, Any]) -> dict[str, Any]:
    """Return structured model features used to preserve interpretation continuity."""
    return {
        "structured_features": deepcopy(packet.get("structured_features") or {}),
    }


def xgboost_structured_state_hash(packet: dict[str, Any]) -> str:
    """Hash the current XGBoost/scenario structured feature state."""
    return _stable_hash(xgboost_structured_state_payload(packet))


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _normalized_status(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _validate_required_string_fields(
    errors: list[str],
    payload: dict[str, Any],
    *,
    prefix: str,
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{prefix}.{field} is required")


def _validate_required_array_fields(
    errors: list[str],
    payload: dict[str, Any],
    *,
    prefix: str,
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        if not isinstance(payload.get(field), list):
            errors.append(f"{prefix}.{field} must be an array")


def _word_count(value: Any) -> int:
    if isinstance(value, dict):
        text = " ".join(str(item) for item in value.values())
    else:
        text = str(value or "")
    return len([part for part in text.replace("/", " ").split() if part.strip()])


def _normalize_development_discussion_options(raw_options: Any) -> list[dict[str, Any]]:
    """Return canonical visible discussion options only."""
    options: list[dict[str, Any]] = []
    if isinstance(raw_options, list):
        for item in raw_options:
            if not isinstance(item, dict):
                options.append(item)
                continue
            question = deepcopy(item.get("participant_wider_question") or {})
            if isinstance(question, str):
                question = {"question": question}
            options.append({
                "topic": str(item.get("topic") or "").strip(),
                "why_it_matters": str(item.get("why_it_matters") or "").strip(),
                "supporting_evidence": deepcopy(item.get("supporting_evidence") or []),
                "relationship_to_previous_scenario": str(item.get("relationship_to_previous_scenario") or "").strip(),
                "relationship_to_original_baseline": str(item.get("relationship_to_original_baseline") or "").strip(),
                "participant_wider_question": question if isinstance(question, dict) else {},
            })
    return options


def _nested_evidence_refs(value: Any, *, prefix: str) -> set[str]:
    refs: set[str] = set()
    if not isinstance(value, dict):
        return refs
    for key, child_value in value.items():
        key_text = str(key)
        path = f"{prefix}.{key_text}"
        refs.add(path)
        refs.update(_nested_evidence_refs(child_value, prefix=path))
    return refs


def packet_evidence_refs(packet: dict[str, Any]) -> set[str]:
    refs = {
        "completion_score",
        "model_interpretation.completion_score",
        "model_interpretation.score_delta",
        "operational_assumptions",
        "operational_movement_context",
        "field_changes",
        "text_context",
    }
    for field in ((packet.get("iteration_context") or {}).get("changed_fields") or []):
        refs.add(str(field))
    for section_name in (
        "structured_features",
        "text_context",
        "operational_assumptions",
        "operational_movement_context",
        "model_interpretation",
    ):
        section = packet.get(section_name) or {}
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            refs.add(str(key))
            refs.add(f"{section_name}.{key}")
            refs.update(_nested_evidence_refs(value, prefix=f"{section_name}.{key}"))
    return refs


def _packet_evidence_refs(packet: dict[str, Any]) -> set[str]:
    return packet_evidence_refs(packet)


def _canonical_reality_check_target(
    allocation_target_id: Any,
) -> tuple[str, str, str | None, str | None]:
    target_id = str(allocation_target_id or "").strip()
    if target_id:
        target = REALITY_CHECK_ALLOCATION_TARGETS.get(target_id)
        if target:
            return str(target["pillar"]), str(target["subpillar"]), target_id, None
        return "", "", None, f"reality_check allocation_target_id is not allowed: {target_id}"

    return "", "", None, "reality_check.allocations[].allocation_target_id is required"


def _reality_check_display_explanation(target_id: str | None, points: float) -> str:
    positive_labels = {
        "therapeutic_context.therapeutic_area_profile": "Disease fit improved",
        "therapeutic_context.development_phase_and_goal": "Phase fit improved",
        "scientific_challenge.biological_profile": "Biology fit improved",
        "scientific_challenge.protocol_architecture": "Evidence design strengthened",
        "patient_profile.clinical_severity": "Patient fit improved",
        "patient_profile.population_scope": "Population fit improved",
        "execution_framework.trial_complexity_footprint": "Execution realism improved",
        "execution_framework.methodological_setup": "Methodology strengthened",
        "execution_framework.operational_fit": "Operational support improved",
    }
    negative_labels = {
        "therapeutic_context.therapeutic_area_profile": "Disease-context concern",
        "therapeutic_context.development_phase_and_goal": "Phase fit concern",
        "scientific_challenge.biological_profile": "Biology concern",
        "scientific_challenge.protocol_architecture": "Evidence design concern",
        "patient_profile.clinical_severity": "Patient burden",
        "patient_profile.population_scope": "Population concern",
        "execution_framework.trial_complexity_footprint": "Execution burden",
        "execution_framework.methodological_setup": "Methodology concern",
        "execution_framework.operational_fit": "Operational support gap",
    }
    if points > 0:
        return positive_labels.get(str(target_id or ""), "Realism support")
    if points < 0:
        return negative_labels.get(str(target_id or ""), "Realism concern")
    return REALITY_CHECK_LABEL


def _reference_trial_score(packet: dict[str, Any]) -> float | None:
    continuity = (packet.get("iteration_context") or {}).get("trial_score_continuity") or {}
    previous_continuity_trial_score = _number(continuity.get("previous_trial_score"))
    if previous_continuity_trial_score is not None:
        return previous_continuity_trial_score
    model = packet.get("model_interpretation") or {}
    previous_trial_score = _number(model.get("previous_trial_score"))
    if previous_trial_score is not None:
        return previous_trial_score
    previous_completion = _number(model.get("previous_completion_score"))
    if previous_completion is not None:
        return previous_completion
    baseline = _number(model.get("baseline_completion_score"))
    if baseline is not None:
        return baseline
    return _number(model.get("completion_score"))


def _reference_pre_reality_score(packet: dict[str, Any]) -> float | None:
    continuity = (packet.get("iteration_context") or {}).get("trial_score_continuity") or {}
    previous_pre_reality_score = _number(continuity.get("previous_pre_reality_score"))
    if previous_pre_reality_score is not None:
        return previous_pre_reality_score
    model = packet.get("model_interpretation") or {}
    previous_completion = _number(model.get("previous_completion_score"))
    if previous_completion is not None:
        return previous_completion
    baseline = _number(model.get("baseline_completion_score"))
    if baseline is not None:
        return baseline
    return _number(model.get("completion_score"))


def _neutral_operational_fit_assessment(reason: str) -> dict[str, Any]:
    return {
        "points": 0.0,
        "rating": "neutral_or_unclear",
        "materiality": "minor",
        "interaction_with_completion_outlook": "aligned",
        "central_reason": reason,
        "evidence_fields": [],
        "supported_evidence_fields": [],
        "validation_errors": [],
        "validation_notes": [reason],
    }


def _neutral_reality_check_assessment(reason: str) -> dict[str, Any]:
    return {
        "points": 0.0,
        "effect": "neutral",
        "strength": "none",
        "fraction": 0.0,
        "central_reason": reason,
        "allocation_points": [],
        "supported_evidence_fields": [],
        "validation_errors": [],
        "validation_notes": [reason],
    }


def _matching_operational_fit_trace(packet: dict[str, Any]) -> dict[str, Any] | None:
    current_hash = operational_fit_state_hash(packet)
    continuity = (packet.get("iteration_context") or {}).get("trial_score_continuity") or {}
    recent_traces = continuity.get("recent_score_traces") or []
    if not isinstance(recent_traces, list):
        return None
    for trace in reversed(recent_traces):
        if not isinstance(trace, dict):
            continue
        if trace.get("operational_fit_state_hash") != current_hash:
            continue
        if _number(trace.get("operational_fit_points")) is None:
            continue
        return trace
    return None


def _same_state_reuse(packet_or_pass2_input: dict[str, Any]) -> bool:
    if not isinstance(packet_or_pass2_input, dict):
        return False
    history = packet_or_pass2_input.get("participant_visible_history")
    if isinstance(history, dict) and bool(history.get("same_state_reuse")):
        return True
    trajectory = packet_or_pass2_input.get("trajectory_context")
    if isinstance(trajectory, dict) and bool(trajectory.get("same_state_reuse")):
        return True
    state_equivalence = (packet_or_pass2_input.get("iteration_context") or {}).get("state_equivalence_review")
    return isinstance(state_equivalence, dict) and bool(state_equivalence.get("available"))


def validate_pass1_review(packet: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(review, dict):
        return {"validation_status": "invalid", "validation_errors": ["Pass 1 review must be an object"]}

    for field in sorted(APP_OWNED_TRIAL_SCORE_FIELDS.intersection(review)):
        errors.append(f"{field} is application-owned and must not be returned by Pass 1")
    metadata = review.get("review_metadata") or {}
    if not isinstance(metadata, dict):
        errors.append("review_metadata must be an object")
    review_mode = str((metadata or {}).get("review_mode") or "")
    is_hidden_baseline = review_mode == "hidden_baseline"

    strategy_shift = review.get("strategy_shift_check") or {"status": "not_applicable"}
    if not isinstance(strategy_shift, dict):
        errors.append("strategy_shift_check must be an object")
        strategy_shift = {}
    status = strategy_shift.get("status", "not_applicable")
    if status not in STRATEGY_SHIFT_STATUSES:
        errors.append(f"strategy_shift_check.status must be one of {sorted(STRATEGY_SHIFT_STATUSES)}")
    changed_fields = {
        str(field)
        for field in ((packet.get("iteration_context") or {}).get("changed_fields") or [])
    }
    changed_gated_fields = sorted(changed_fields.intersection(GATED_PREMISE_SENSITIVE_FIELDS))
    if changed_gated_fields and status == "not_applicable":
        errors.append(
            "strategy_shift_check.status must not be not_applicable when gated premise-sensitive fields changed: "
            f"{changed_gated_fields}"
        )

    required_objects = (
        "completion_outlook_analysis",
        "evolution_evidence",
        "continuity_update",
        "analytical_narrative_draft",
    )
    for field in required_objects:
        if not isinstance(review.get(field), dict):
            errors.append(f"{field} must be an object")
    evolution_evidence = review.get("evolution_evidence") or {}
    if not isinstance(evolution_evidence, dict):
        evolution_evidence = {}
    else:
        _validate_required_array_fields(
            errors,
            evolution_evidence,
            prefix="evolution_evidence",
            fields=(
                "latest_meaningful_changes",
                "model_movement_evidence",
                "operational_movement_evidence",
                "new_issues",
                "persistent_issues",
                "resolved_or_mitigated_issues",
            ),
        )
        strongest_tension = evolution_evidence.get("strongest_current_development_tension")
        if not isinstance(strongest_tension, dict):
            errors.append("evolution_evidence.strongest_current_development_tension must be an object")
        else:
            _validate_required_string_fields(
                errors,
                strongest_tension,
                prefix="evolution_evidence.strongest_current_development_tension",
                fields=(
                    "topic",
                    "why_this_is_strongest_now",
                    "relationship_to_previous_scenario",
                    "relationship_to_original_baseline",
                ),
            )
            if not isinstance(strongest_tension.get("evidence_fields"), list):
                errors.append("evolution_evidence.strongest_current_development_tension.evidence_fields must be an array")
    discussion_options_provided = "development_discussion_options" in review
    if discussion_options_provided and not isinstance(review.get("development_discussion_options"), list):
        errors.append("development_discussion_options must be an array")
        development_discussion_options = []
    elif not discussion_options_provided:
        if is_hidden_baseline:
            development_discussion_options = []
        else:
            errors.append("development_discussion_options must be an array")
            development_discussion_options = []
    else:
        development_discussion_options = _normalize_development_discussion_options(
            review.get("development_discussion_options")
        )
    if is_hidden_baseline and development_discussion_options:
        errors.append("hidden baseline must not return development_discussion_options")
    draft = review.get("analytical_narrative_draft") or {}
    if isinstance(draft, dict):
        substantive_fields = 0
        for field in ANALYTICAL_NARRATIVE_DRAFT_FIELDS:
            value = draft.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"analytical_narrative_draft.{field} must be a non-empty string")
            elif len(value.split()) >= 8:
                substantive_fields += 1
        if not is_hidden_baseline and substantive_fields < MIN_ANALYTICAL_DRAFT_SUBSTANTIVE_FIELDS:
            errors.append(
                "analytical_narrative_draft must provide substantive interpretation in most required fields"
            )

    if not is_hidden_baseline and isinstance(development_discussion_options, list) and not (
        MIN_VISIBLE_DEVELOPMENT_DISCUSSION_OPTIONS
        <= len(development_discussion_options)
        <= MAX_VISIBLE_DEVELOPMENT_DISCUSSION_OPTIONS
    ):
        errors.append(
            "development_discussion_options must include "
            f"{MIN_VISIBLE_DEVELOPMENT_DISCUSSION_OPTIONS}-{MAX_VISIBLE_DEVELOPMENT_DISCUSSION_OPTIONS} options"
        )
    discussion_topics: list[str] = []
    for index, item in enumerate(
        development_discussion_options if isinstance(development_discussion_options, list) else []
    ):
        if not isinstance(item, dict):
            errors.append(f"development_discussion_options[{index}] must be an object")
            continue
        topic = item.get("topic")
        why_it_matters = item.get("why_it_matters")
        supporting_evidence = item.get("supporting_evidence")
        question = item.get("participant_wider_question")
        if not isinstance(topic, str) or not topic.strip():
            errors.append(f"development_discussion_options[{index}].topic is required")
        if not isinstance(why_it_matters, str) or not why_it_matters.strip():
            errors.append(f"development_discussion_options[{index}].why_it_matters is required")
        if not isinstance(supporting_evidence, list):
            errors.append(f"development_discussion_options[{index}].supporting_evidence must be an array")
        for relationship_field in ("relationship_to_previous_scenario", "relationship_to_original_baseline"):
            if not isinstance(item.get(relationship_field), str) or not item.get(relationship_field, "").strip():
                errors.append(f"development_discussion_options[{index}].{relationship_field} is required")
        topic_text = str(topic or "").strip()
        if topic_text:
            discussion_topics.append(topic_text)
        if not isinstance(question, dict):
            errors.append(f"development_discussion_options[{index}].participant_wider_question must be an object")
            continue
        if not isinstance(question.get("question"), str) or not question.get("question", "").strip():
            errors.append(f"development_discussion_options[{index}].participant_wider_question.question is required")
        if not isinstance(question.get("supporting_evidence"), list):
            errors.append(
                f"development_discussion_options[{index}].participant_wider_question.supporting_evidence must be an array"
            )
    if (
        len(discussion_topics) >= MIN_VISIBLE_DEVELOPMENT_DISCUSSION_OPTIONS
        and len(set(discussion_topics)) != len(discussion_topics)
    ):
        errors.append("development_discussion_options topics must be distinct")
    continuity_update = deepcopy(review.get("continuity_update") or {})
    if is_hidden_baseline:
        continuity_update["active_tension"] = ""

    validated = {
        "validation_status": "valid" if not errors else "invalid",
        "validation_errors": errors,
        "validation_notes": [],
        "review_metadata": deepcopy(metadata),
        "completion_outlook_analysis": deepcopy(review.get("completion_outlook_analysis") or {}),
        "strategy_shift_check": deepcopy(strategy_shift),
        "evolution_evidence": deepcopy(evolution_evidence),
        "continuity_update": continuity_update,
        "analytical_narrative_draft": deepcopy(draft),
    }
    if not is_hidden_baseline:
        validated["development_discussion_options"] = deepcopy(development_discussion_options)
    return validated


def score_pass1_review(packet: dict[str, Any], pass1_review: dict[str, Any]) -> dict[str, Any]:
    """Validate the evidence pass without adjudicating Trial Score points."""
    validated = validate_pass1_review(packet, pass1_review)
    input_hash = packet.get("input_hash")
    model = packet.get("model_interpretation") or {}
    completion_score = _number(model.get("completion_score"))
    metadata = validated.get("review_metadata") or {}
    iteration = packet.get("iteration_context") or {}
    hidden_baseline = (
        metadata.get("review_mode") == "hidden_baseline"
        or (
            not (iteration.get("changed_fields") or [])
            and iteration.get("previous_snapshot_id") is None
            and iteration.get("current_snapshot_id") == iteration.get("baseline_snapshot_id")
        )
    )
    if completion_score is None:
        return {
            "validation_status": "invalid",
            "validation_errors": [*validated.get("validation_errors", []), "model_interpretation.completion_score must be numeric"],
            "input_hash": input_hash,
        }
    return {
        "validation_status": validated.get("validation_status", "invalid"),
        "validation_errors": validated.get("validation_errors", []),
        "validation_notes": validated.get("validation_notes", []),
        "xgboost_completion_outlook": _clean_points(completion_score),
        "operational_fit_points": None,
        "pre_reality_score": None,
        "pre_reality_delta": None,
        "reality_check_points": None,
        "reality_check_allocation_points": [],
        "trial_score": None,
        "input_hash": input_hash,
        "scoring_stage": "not_run_for_hidden_baseline" if hidden_baseline else "awaiting_llm_scoring_pass",
    }


def validate_scoring_adjudication(
    packet: dict[str, Any],
    pass1_review: dict[str, Any],
    scoring_review: dict[str, Any],
) -> dict[str, Any]:
    """Validate the LLM-owned score adjudication pass and app arithmetic rails."""
    errors: list[str] = []
    notes: list[str] = []
    if not isinstance(scoring_review, dict):
        return {"validation_status": "invalid", "validation_errors": ["scoring review must be an object"]}

    input_hash = packet.get("input_hash")
    model = packet.get("model_interpretation") or {}
    completion_score = _number(model.get("completion_score"))
    if completion_score is None:
        return {
            "validation_status": "invalid",
            "validation_errors": ["model_interpretation.completion_score must be numeric"],
            "input_hash": input_hash,
        }

    iteration = packet.get("iteration_context") or {}
    reference_trial_score = _reference_trial_score(packet)
    reference_pre_reality_score = _reference_pre_reality_score(packet)
    hidden_baseline = str(((pass1_review.get("review_metadata") or {}).get("review_mode")) or "") == "hidden_baseline"
    returned_to_hidden_baseline = bool(iteration.get("returned_to_hidden_baseline_state"))

    if hidden_baseline:
        return {
            "validation_status": "valid",
            "validation_errors": [],
            "validation_notes": ["Hidden baseline produces evidence context only; scoring pass is not visible."],
            "xgboost_completion_outlook": _clean_points(completion_score),
            "operational_fit_points": None,
            "pre_reality_score": None,
            "pre_reality_delta": None,
            "reality_check_points": None,
            "reality_check_allocation_points": [],
            "trial_score": None,
            "input_hash": input_hash,
        }

    if returned_to_hidden_baseline:
        pre_reality_delta = (
            float(completion_score) - float(reference_pre_reality_score)
            if reference_pre_reality_score is not None
            else 0.0
        )
        reason = (
            "Current scenario state matches the hidden baseline state; Operational Fit and Reality Check are neutralized "
            "to preserve deterministic same-state behavior."
        )
        return {
            "validation_status": "valid",
            "validation_errors": [],
            "validation_notes": [reason],
            "xgboost_completion_outlook": _clean_points(completion_score),
            "operational_fit_points": 0.0,
            "operational_fit_state_hash": operational_fit_state_hash(packet),
            "operational_fit_state_payload": operational_fit_state_payload(packet),
            "xgboost_structured_state_hash": xgboost_structured_state_hash(packet),
            "xgboost_structured_state_payload": xgboost_structured_state_payload(packet),
            "pre_reality_score": _clean_points(completion_score),
            "pre_reality_delta": _clean_points(pre_reality_delta),
            "reality_check_points": 0.0,
            "reality_check_allocation_points": [],
            "trial_score": _clean_points(completion_score),
            "delta_vs_previous_trial_score": (
                _clean_points(float(completion_score) - float(reference_trial_score))
                if reference_trial_score is not None
                else None
            ),
            "delta_vs_previous_pre_reality_score": _clean_points(pre_reality_delta),
            "delta_vs_baseline_xgboost": 0.0,
            "operational_fit_assessment": _neutral_operational_fit_assessment(reason),
            "reality_check_assessment": _neutral_reality_check_assessment(reason),
            "input_hash": input_hash,
        }

    metadata = scoring_review.get("review_metadata")
    if not isinstance(metadata, dict):
        errors.append("review_metadata must be an object")
        metadata = {}
    else:
        if not isinstance(metadata.get("review_mode"), str) or not metadata.get("review_mode", "").strip():
            errors.append("review_metadata.review_mode is required")
        if not isinstance(metadata.get("visible"), bool):
            errors.append("review_metadata.visible must be a boolean")
    operational_fit = scoring_review.get("operational_fit")
    if not isinstance(operational_fit, dict):
        errors.append("operational_fit must be an object")
        operational_fit = {}
    reality_check = scoring_review.get("reality_check")
    if not isinstance(reality_check, dict):
        errors.append("reality_check must be an object")
        reality_check = {}
    score_read = scoring_review.get("score_evolution_read")
    if not isinstance(score_read, dict):
        errors.append("score_evolution_read must be an object")
        score_read = {}
    else:
        _validate_required_string_fields(
            errors,
            score_read,
            prefix="score_evolution_read",
            fields=("direction", "main_reason", "active_issue_to_carry_forward"),
        )
    _validate_required_string_fields(
        errors,
        operational_fit,
        prefix="operational_fit",
        fields=("relationship_to_previous", "reason", "boundary_check"),
    )
    _validate_required_string_fields(
        errors,
        reality_check,
        prefix="reality_check",
        fields=(
            "relationship_to_previous",
            "carryover_status",
            "new_issue_status",
            "reason",
            "incremental_check",
        ),
    )
    if not isinstance(operational_fit.get("evidence_fields"), list):
        errors.append("operational_fit.evidence_fields must be an array")
    if not isinstance(reality_check.get("evidence_fields"), list):
        errors.append("reality_check.evidence_fields must be an array")
    if not isinstance(reality_check.get("allocations"), list):
        errors.append("reality_check.allocations must be an array")

    operational_points = _number(operational_fit.get("points"))
    if operational_points is None:
        errors.append("operational_fit.points must be numeric")
        operational_points = 0.0
    if operational_points is not None and not -5.0 <= float(operational_points) <= 5.0:
        errors.append("operational_fit.points must be between -5 and 5")
    operational_evidence = _string_list(operational_fit.get("evidence_fields"))
    supported_operational_evidence = [field for field in operational_evidence if field in _packet_evidence_refs(packet)]
    if operational_evidence and not supported_operational_evidence:
        errors.append("operational_fit.evidence_fields do not reference packet evidence")
    matching_operational_trace = _matching_operational_fit_trace(packet)
    matching_operational_points = (
        _number(matching_operational_trace.get("operational_fit_points"))
        if matching_operational_trace
        else None
    )
    if matching_operational_points is not None and abs(
        float(operational_points or 0.0) - float(matching_operational_points)
    ) > 1e-9:
        errors.append(
            "operational_fit.points must match the previous accepted Operational Fit points when "
            "operational estimates and operational benchmark/context are equivalent"
        )
    if abs(float(operational_points or 0.0)) > 1e-9 and not supported_operational_evidence:
        errors.append("operational_fit.evidence_fields must reference packet evidence for non-zero points")

    pre_reality_score = float(completion_score) + float(operational_points or 0.0)
    pre_reality_delta = pre_reality_score - float(
        reference_pre_reality_score if reference_pre_reality_score is not None else completion_score
    )

    reality_points = _number(reality_check.get("points"))
    if reality_points is None:
        errors.append("reality_check.points must be numeric")
        reality_points = 0.0
    if reality_points is not None and not -15.0 <= float(reality_points) <= 15.0:
        errors.append("reality_check.points must be between -15 and 15")
    if (
        pre_reality_delta > 0
        and float(reality_points or 0.0) > MAX_POSITIVE_REALITY_CHECK_WHEN_PRE_REALITY_ALREADY_IMPROVED
    ):
        errors.append(
            "reality_check.points must be <= 0 when the pre-reality check score already improved; accept the gain "
            "with 0 or challenge it with a negative Reality Check"
        )
    reality_evidence = _string_list(reality_check.get("evidence_fields"))
    supported_reality_evidence = [field for field in reality_evidence if field in _packet_evidence_refs(packet)]
    if abs(float(reality_points or 0.0)) > 1e-9 and not supported_reality_evidence:
        errors.append("reality_check.evidence_fields must reference packet evidence for non-zero points")
    incremental_check = str(reality_check.get("incremental_check") or "").strip()
    if abs(float(reality_points or 0.0)) > 1e-9 and len(incremental_check) < 12:
        errors.append("reality_check.incremental_check is required for non-zero points")
    carryover = iteration.get("reality_check_carryover_candidate") or {}
    carryover_precheck = carryover.get("app_state_precheck") if isinstance(carryover, dict) else {}
    carryover_status = _normalized_status(reality_check.get("carryover_status"))
    try:
        previous_reality_points = float(carryover.get("previous_reality_check_points"))
    except (TypeError, ValueError, AttributeError):
        previous_reality_points = None
    if (
        isinstance(carryover, dict)
        and carryover.get("active") is True
        and isinstance(carryover_precheck, dict)
        and carryover_precheck.get("status") == "not_touched"
        and previous_reality_points is not None
        and previous_reality_points <= REALITY_CHECK_CARRYOVER_MATERIALITY_THRESHOLD
        and float(reality_points or 0.0) >= 0.0
        and carryover_status not in {"resolved", "superseded", "no_longer_material"}
    ):
        errors.append(
            "reality_check.points cannot become neutral or positive while a material prior negative carryover "
            "issue is not touched unless carryover_status explicitly says resolved, superseded, or no_longer_material"
        )

    allocation_points: list[dict[str, Any]] = []
    allocations = reality_check.get("allocations") if isinstance(reality_check.get("allocations"), list) else []
    if float(reality_points or 0.0):
        if not isinstance(allocations, list) or not 1 <= len(allocations) <= 4:
            errors.append("reality_check.allocations must include 1-4 allocations for non-zero points")
            allocations = []
        share_total = 0.0
        for index, allocation in enumerate(allocations):
            if not isinstance(allocation, dict):
                errors.append(f"reality_check.allocations[{index}] must be an object")
                continue
            pillar, subpillar, target_id, target_note = _canonical_reality_check_target(
                allocation.get("allocation_target_id")
            )
            if target_note:
                errors.append(target_note)
            share = _number(allocation.get("share"))
            if share is None or share <= 0:
                errors.append(f"reality_check.allocations[{index}].share must be positive")
                share = 0.0
            share_total += float(share)
            for field_name in ("movement_label", "rationale", "incremental_check"):
                if not isinstance(allocation.get(field_name), str) or not allocation.get(field_name, "").strip():
                    errors.append(f"reality_check.allocations[{index}].{field_name} is required")
            movement_label = str(allocation.get("movement_label") or "").strip()
            if float(reality_points or 0.0) > 0 and "negative" in movement_label.lower():
                errors.append(
                    f"reality_check.allocations[{index}].movement_label must not be negative when Reality Check points are positive"
                )
            if float(reality_points or 0.0) < 0 and "positive" in movement_label.lower():
                errors.append(
                    f"reality_check.allocations[{index}].movement_label must not be positive when Reality Check points are negative"
                )
            if target_id:
                rationale = str(allocation.get("rationale") or "")
                allocation_points.append({
                    "allocation_target_id": target_id,
                    "pillar": pillar,
                    "subpillar": REALITY_CHECK_LABEL,
                    "source_subpillar": subpillar,
                    "label": str(allocation.get("movement_label") or REALITY_CHECK_LABEL),
                    "short_explanation": _reality_check_display_explanation(
                        target_id,
                        float(reality_points or 0.0) * float(share),
                    ),
                    "share": float(share),
                    "points": _clean_points(float(reality_points or 0.0) * float(share)),
                    "rationale": rationale,
                    "incremental_check": str(allocation.get("incremental_check") or ""),
                })
        if allocation_points and abs(share_total - 1.0) > 0.02:
            equal_share = 1.0 / len(allocation_points)
            normalized = [equal_share for _ in allocation_points]
            normalized[-1] = max(0.0, 1.0 - sum(normalized[:-1]))
            for row, share in zip(allocation_points, normalized):
                row["share"] = share
                row["points"] = _clean_points(float(reality_points or 0.0) * share)
            notes.append("Reality Check allocation shares were assigned equally by the app")

    trial_score = clamp(pre_reality_score + float(reality_points or 0.0), 0, 100)
    if errors:
        return {
            "validation_status": "invalid",
            "validation_errors": errors,
            "validation_notes": notes,
            "xgboost_completion_outlook": _clean_points(completion_score),
            "operational_fit_points": None,
            "pre_reality_score": None,
            "pre_reality_delta": None,
            "reality_check_points": None,
            "trial_score": None,
            "input_hash": input_hash,
        }

    return {
        "validation_status": "valid",
        "validation_errors": [],
        "validation_notes": notes,
        "xgboost_completion_outlook": _clean_points(completion_score),
        "operational_fit_points": _clean_points(float(operational_points or 0.0)),
        "operational_fit_state_hash": operational_fit_state_hash(packet),
        "operational_fit_state_payload": operational_fit_state_payload(packet),
        "xgboost_structured_state_hash": xgboost_structured_state_hash(packet),
        "xgboost_structured_state_payload": xgboost_structured_state_payload(packet),
        "pre_reality_score": _clean_points(pre_reality_score),
        "pre_reality_delta": _clean_points(pre_reality_delta),
        "reality_check_points": _clean_points(float(reality_points or 0.0)),
        "reality_check_allocation_points": allocation_points,
        "trial_score": _clean_points(trial_score),
        "delta_vs_previous_trial_score": _clean_points(
            float(trial_score) - float(reference_trial_score)
            if reference_trial_score is not None
            else float(trial_score) - float(completion_score)
        ),
        "delta_vs_previous_pre_reality_score": _clean_points(pre_reality_delta),
        "delta_vs_baseline_xgboost": _clean_points(float(trial_score) - float(model.get("baseline_completion_score", completion_score))),
        "operational_fit_assessment": {
            "points": _clean_points(float(operational_points or 0.0)),
            "relationship_to_previous": str(operational_fit.get("relationship_to_previous") or ""),
            "central_reason": str(operational_fit.get("reason") or operational_fit.get("central_reason") or ""),
            "evidence_fields": operational_evidence,
            "supported_evidence_fields": supported_operational_evidence,
            "matched_previous_operational_fit_trace": deepcopy(matching_operational_trace or {}),
        },
        "reality_check_assessment": {
            "points": _clean_points(float(reality_points or 0.0)),
            "relationship_to_previous": str(reality_check.get("relationship_to_previous") or ""),
            "carryover_status": str(reality_check.get("carryover_status") or ""),
            "new_issue_status": str(reality_check.get("new_issue_status") or ""),
            "central_reason": str(reality_check.get("reason") or reality_check.get("central_reason") or ""),
            "incremental_check": incremental_check,
            "evidence_fields": reality_evidence,
            "supported_evidence_fields": supported_reality_evidence,
            "allocation_points": allocation_points,
            "score_evolution_read": deepcopy(score_read or {}),
        },
        "scoring_review": deepcopy(scoring_review),
        "input_hash": input_hash,
    }


def validate_pass2_review(review: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(review, dict):
        return {"validation_status": "invalid", "validation_errors": ["Pass 2 review must be an object"]}
    for field in sorted(APP_OWNED_TRIAL_SCORE_FIELDS.intersection(review)):
        errors.append(f"{field} is application-owned and must not be returned by Pass 2")
    metadata = review.get("review_metadata")
    if not isinstance(metadata, dict):
        errors.append("review_metadata must be an object")
    else:
        if not isinstance(metadata.get("review_mode"), str) or not metadata.get("review_mode", "").strip():
            errors.append("review_metadata.review_mode is required")
        if not isinstance(metadata.get("visible"), bool):
            errors.append("review_metadata.visible must be a boolean")

    narrative = review.get("trial_score_narrative")
    if not isinstance(narrative, dict):
        errors.append("trial_score_narrative must be an object")
        narrative = {}
    for field in ("summary", "movement_reading", "score_interpretation"):
        if not isinstance(narrative.get(field), str) or not narrative.get(field, "").strip():
            errors.append(f"trial_score_narrative.{field} is required")

    pillar_reading = review.get("pillar_reading")
    if not isinstance(pillar_reading, list):
        errors.append("pillar_reading must be an array")
        pillar_reading = []
    elif not MIN_PASS2_PILLAR_READINGS <= len(pillar_reading) <= MAX_PASS2_PILLAR_READINGS:
        errors.append(
            f"pillar_reading must include {MIN_PASS2_PILLAR_READINGS}-{MAX_PASS2_PILLAR_READINGS} material bullets"
        )
    for index, item in enumerate(pillar_reading):
        if not isinstance(item, dict):
            errors.append(f"pillar_reading[{index}] must be an object")
            continue
        for field in ("pillar", "reading"):
            if not isinstance(item.get(field), str) or not item.get(field, "").strip():
                errors.append(f"pillar_reading[{index}].{field} is required")

    central_tension = review.get("central_tension")
    if not isinstance(central_tension, dict):
        errors.append("central_tension must be an object")
        central_tension = {}
    for field in ("summary", "why_it_matters"):
        if not isinstance(central_tension.get(field), str) or not central_tension.get(field, "").strip():
            errors.append(f"central_tension.{field} is required")

    broader_question = review.get("broader_strategic_question")
    if not isinstance(broader_question, dict):
        errors.append("broader_strategic_question must be an object")
        broader_question = {}
    if not isinstance(broader_question.get("mapped_tension"), str) or not broader_question.get("mapped_tension", "").strip():
        errors.append("broader_strategic_question.mapped_tension is required")
    if not isinstance(broader_question.get("question"), str) or not broader_question.get("question", "").strip():
        errors.append("broader_strategic_question.question is required")
    if (
        isinstance(central_tension.get("summary"), str)
        and isinstance(broader_question.get("mapped_tension"), str)
        and central_tension.get("summary", "").strip()
        and broader_question.get("mapped_tension", "").strip()
        and central_tension.get("summary", "").strip() != broader_question.get("mapped_tension", "").strip()
    ):
        errors.append("broader_strategic_question.mapped_tension must match central_tension.summary")
    return {
        "validation_status": "valid" if not errors else "invalid",
        "validation_errors": errors,
        "review_metadata": deepcopy(metadata or {}),
        "trial_score_narrative": deepcopy(narrative or {}),
        "pillar_reading": deepcopy(pillar_reading or []),
        "central_tension": deepcopy(central_tension or {}),
        "broader_strategic_question": deepcopy(broader_question or {}),
    }


def validate_pass2_review_with_input(review: dict[str, Any], pass2_input: dict[str, Any]) -> dict[str, Any]:
    """Validate Pass 2 shape plus its selected discussion topic/question against Pass 1 options."""
    validated = validate_pass2_review(review)
    errors = list(validated.get("validation_errors") or [])
    notes = list(validated.get("validation_notes") or [])
    pass1_analysis = pass2_input.get("pass1_analysis") if isinstance(pass2_input, dict) else {}
    options = (pass1_analysis or {}).get("development_discussion_options") or []
    option_questions_by_summary: dict[str, str] = {}
    if isinstance(options, list):
        for item in options:
            if not isinstance(item, dict):
                continue
            question = item.get("participant_wider_question") or {}
            if not isinstance(question, dict):
                continue
            summary = str(item.get("topic") or "").strip()
            question_text = str(question.get("question") or "").strip()
            if summary:
                option_questions_by_summary[summary] = question_text
    selected_summary = str((validated.get("central_tension") or {}).get("summary") or "").strip()
    selected_question = str((validated.get("broader_strategic_question") or {}).get("question") or "").strip()
    if option_questions_by_summary:
        expected_question = option_questions_by_summary.get(selected_summary)
        if selected_summary and expected_question is None:
            errors.append("central_tension.summary must match one supplied development_discussion_options topic")
        elif expected_question is not None and selected_question and selected_question != expected_question:
            errors.append("broader_strategic_question.question must match the question paired with the selected development_discussion_options topic")
    else:
        errors.append("pass1_analysis.development_discussion_options must include at least one selectable option")
    validated["validation_errors"] = errors
    validated["validation_notes"] = notes
    validated["validation_status"] = "valid" if not errors else "invalid"
    return validated
