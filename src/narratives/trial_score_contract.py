"""V1 Trial Score contract and deterministic app scoring.

This module owns the new app-calculated score stack documented in
``docs/trial_score_narrative_direction.md`` and
``docs/operational_fit_scoring.md``. It is intentionally provider-agnostic:
the LLM returns structured judgments, while the app validates and calculates
Operational Fit, Reality Check, and Trial Score.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

TRIAL_SCORE_CONTRACT_VERSION = "trial_score_v1"
PASS1_SCHEMA_VERSION = "trial_score_pass1_schema_v3"
PASS2_SCHEMA_VERSION = "trial_score_pass2_schema_v2"
PROMPT_TEMPLATE_VERSION = "trial_score_two_pass_prompt_v1_8"
MIN_PASS2_PILLAR_READINGS = 2
MAX_PASS2_PILLAR_READINGS = 4

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

OPERATIONAL_FIT_RATINGS = {
    "strongly_improves_fit",
    "moderately_improves_fit",
    "slightly_improves_fit",
    "neutral_or_unclear",
    "slightly_worsens_fit",
    "moderately_worsens_fit",
    "strongly_worsens_fit",
}

OPERATIONAL_FIT_MATERIALITIES = {
    "minor",
    "moderate",
    "major",
    "extreme",
}

OPERATIONAL_FIT_POINT_TABLE = {
    "minor": {
        "slightly_improves_fit": 0.3,
        "moderately_improves_fit": 0.7,
        "strongly_improves_fit": 1.0,
        "neutral_or_unclear": 0.0,
        "slightly_worsens_fit": -0.3,
        "moderately_worsens_fit": -0.7,
        "strongly_worsens_fit": -1.0,
    },
    "moderate": {
        "slightly_improves_fit": 0.7,
        "moderately_improves_fit": 1.4,
        "strongly_improves_fit": 2.0,
        "neutral_or_unclear": 0.0,
        "slightly_worsens_fit": -0.7,
        "moderately_worsens_fit": -1.4,
        "strongly_worsens_fit": -2.0,
    },
    "major": {
        "slightly_improves_fit": 1.2,
        "moderately_improves_fit": 2.4,
        "strongly_improves_fit": 3.5,
        "neutral_or_unclear": 0.0,
        "slightly_worsens_fit": -1.2,
        "moderately_worsens_fit": -2.4,
        "strongly_worsens_fit": -3.5,
    },
    "extreme": {
        "slightly_improves_fit": 1.8,
        "moderately_improves_fit": 3.5,
        "strongly_improves_fit": 5.0,
        "neutral_or_unclear": 0.0,
        "slightly_worsens_fit": -1.8,
        "moderately_worsens_fit": -3.5,
        "strongly_worsens_fit": -5.0,
    },
}

OPERATIONAL_BASELINE_VALUE_SOURCES = {
    "completed_actual",
    "registered_planned",
    "cohort_p50_estimate",
    "observed_floor_over_estimate",
    "terminated_observed_floor",
}

OPERATIONAL_INTERACTION_LABELS = {
    "aligned",
    "under_supported",
    "overbuilt",
    "unmodeled_support",
    "mixed",
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

REALITY_CHECK_EFFECTS = {
    "reinforce_gain",
    "offset_gain",
    "soften_decline",
    "reinforce_decline",
    "reward_coherence",
    "penalize_incoherence",
    "neutral",
    "reversal",
}

REALITY_CHECK_STRENGTHS = {
    "none",
    "slight",
    "moderate",
    "strong",
    "reversal",
}

REALITY_CHECK_FRACTIONS = {
    "none": 0.0,
    "slight": 0.20,
    "moderate": 0.40,
    "strong": 0.70,
    "reversal": 1.25,
}

REALITY_CHECK_CARRYOVER_STATUSES = {
    "still_relevant",
    "partly_mitigated",
    "resolved_or_superseded",
}

REALITY_CHECK_CURRENT_ISSUE_RELATIONS = {
    "same_issue",
    "new_independent_issue",
    "mixed_or_unclear",
}

REALITY_CHECK_CARRYOVER_MATERIALITY_THRESHOLD = -1.0
REALITY_CHECK_CARRYOVER_NEGATIVE_CAP = -15.0

REALITY_CHECK_ALLOWED_SUBPILLARS = {
    "Therapeutic Context": {
        "Therapeutic Area Profile",
        "Development Phase and Goal",
    },
    "Scientific Challenge": {
        "Biological Profile",
        "Protocol Architecture",
    },
    "Patient Profile": {
        "Clinical Severity",
        "Population Scope",
    },
    "Execution Framework": {
        "Trial Complexity Footprint",
        "Methodological Setup",
        "Operational Fit",
    },
}

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
MIN_ANALYTICAL_DRAFT_WORDS = 320
MIN_HIDDEN_BASELINE_ANALYTICAL_DRAFT_WORDS = 450
MIN_VISIBLE_DEVELOPMENT_DISCUSSION_OPTIONS = 2
MAX_VISIBLE_DEVELOPMENT_DISCUSSION_OPTIONS = 3
MIN_STRATEGIC_QUESTION_CANDIDATES = MAX_VISIBLE_DEVELOPMENT_DISCUSSION_OPTIONS

def _clean_points(value: int | float) -> int | float:
    numeric = round(float(value), 1)
    return int(numeric) if numeric.is_integer() else numeric


def clamp(value: int | float, minimum: int | float, maximum: int | float) -> int | float:
    return _clean_points(max(float(minimum), min(float(maximum), float(value))))


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _payload_number(value: Any) -> float | None:
    if isinstance(value, dict):
        return _number(value.get("value"))
    return _number(value)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


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


def _operational_refs(fields: list[str] | set[str]) -> set[str]:
    return {str(field) for field in fields if str(field).startswith("operational_assumptions.")}


def _baseline_reverted_operational_fields(packet: dict[str, Any], changed_operational: list[str]) -> list[str]:
    changed = set(changed_operational)
    if not changed:
        return []

    comparable: set[str] = set()
    reverted: set[str] = set()
    for change in ((packet.get("iteration_context") or {}).get("field_changes") or []):
        if not isinstance(change, dict):
            continue
        field = str(change.get("field") or "")
        if field not in changed:
            continue
        current = _payload_number(change.get("current_value"))
        baseline = _payload_number(change.get("baseline_value"))
        if current is None or baseline is None:
            continue
        comparable.add(field)
        if abs(current - baseline) <= 1e-9:
            reverted.add(field)

    operational_assumptions = packet.get("operational_assumptions") or {}
    if isinstance(operational_assumptions, dict):
        for field in changed.difference(comparable):
            assumption_key = field.removeprefix("operational_assumptions.")
            assumption = operational_assumptions.get(assumption_key)
            if not isinstance(assumption, dict):
                continue
            current = _payload_number(assumption.get("value"))
            baseline = _payload_number(assumption.get("baseline_value"))
            if current is None or baseline is None:
                continue
            comparable.add(field)
            if abs(current - baseline) <= 1e-9:
                reverted.add(field)

    if comparable == changed and reverted == changed:
        return sorted(reverted)
    return []


def _previous_equivalent_operational_fields(packet: dict[str, Any], changed_operational: list[str]) -> list[str]:
    changed = set(changed_operational)
    if not changed:
        return []

    changed_fields = {
        str(field)
        for field in ((packet.get("iteration_context") or {}).get("changed_fields") or [])
        if str(field)
    }
    if changed_fields.difference(changed):
        return []

    model = packet.get("model_interpretation") or {}
    score_delta = _number(model.get("score_delta"))
    if score_delta is not None and abs(score_delta) > 1e-9:
        return []
    if model.get("xgboost_impact_changes"):
        return []
    if (packet.get("iteration_context") or {}).get("text_change_evidence"):
        return []

    comparable: set[str] = set()
    equivalent: set[str] = set()
    for change in ((packet.get("iteration_context") or {}).get("field_changes") or []):
        if not isinstance(change, dict):
            continue
        field = str(change.get("field") or "")
        if field not in changed:
            continue
        current = _payload_number(change.get("current_value"))
        previous = _payload_number(change.get("previous_value"))
        if current is None or previous is None:
            continue
        comparable.add(field)
        if abs(current - previous) <= 1e-9:
            equivalent.add(field)

    if comparable == changed and equivalent == changed:
        return sorted(equivalent)
    return []


def _previous_operational_fit_points(packet: dict[str, Any]) -> float | None:
    continuity = (packet.get("iteration_context") or {}).get("trial_score_continuity") or {}
    if not isinstance(continuity, dict):
        return None
    return _number(continuity.get("previous_operational_fit_points"))


def _rating_matches_points_direction(rating: Any, points: float) -> bool:
    if points > 0:
        return str(rating).endswith("_improves_fit")
    if points < 0:
        return str(rating).endswith("_worsens_fit")
    return rating == "neutral_or_unclear"


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


def score_operational_fit(operational_fit: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    notes: list[str] = []
    combined = operational_fit.get("combined_operational_fit")
    if not isinstance(combined, dict):
        return {
            "points": None,
            "validation_errors": ["operational_fit.combined_operational_fit must be an object"],
            "validation_notes": [],
        }

    rating = combined.get("rating")
    materiality = combined.get("materiality")
    if rating not in OPERATIONAL_FIT_RATINGS:
        errors.append(f"combined_operational_fit.rating must be one of {sorted(OPERATIONAL_FIT_RATINGS)}")
    if materiality not in OPERATIONAL_FIT_MATERIALITIES:
        errors.append(f"combined_operational_fit.materiality must be one of {sorted(OPERATIONAL_FIT_MATERIALITIES)}")

    interaction = combined.get("interaction_with_completion_outlook")
    if interaction is not None and interaction not in OPERATIONAL_INTERACTION_LABELS:
        errors.append(
            "combined_operational_fit.interaction_with_completion_outlook must be one of "
            f"{sorted(OPERATIONAL_INTERACTION_LABELS)}"
        )

    evidence_fields = _string_list(combined.get("evidence_fields"))
    if not evidence_fields:
        errors.append("combined_operational_fit.evidence_fields must include packet evidence")
    supported = [field for field in evidence_fields if field in _packet_evidence_refs(packet)]
    if evidence_fields and not supported:
        errors.append("combined_operational_fit.evidence_fields do not reference packet evidence")

    changed_operational = [
        str(field)
        for field in ((packet.get("iteration_context") or {}).get("changed_fields") or [])
        if str(field).startswith("operational_assumptions.")
    ]
    baseline_reverted_fields = _baseline_reverted_operational_fields(packet, changed_operational)
    if baseline_reverted_fields and rating != "neutral_or_unclear":
        errors.append(
            "combined_operational_fit.rating must be neutral_or_unclear when all changed operational assumptions "
            f"returned to baseline values: {baseline_reverted_fields}"
        )
    elif baseline_reverted_fields:
        notes.append("Operational Fit is neutral because changed operational assumptions returned to baseline values")

    previous_equivalent_fields = _previous_equivalent_operational_fields(packet, changed_operational)
    previous_points = _previous_operational_fit_points(packet)
    if previous_equivalent_fields and previous_points is not None:
        if not _rating_matches_points_direction(rating, previous_points):
            errors.append(
                "combined_operational_fit.rating must preserve the previous Operational Fit direction when all changed "
                f"operational assumptions returned to previous values and no other scenario inputs moved: {previous_equivalent_fields}"
            )
    if errors:
        return {
            "points": None,
            "rating": rating,
            "materiality": materiality,
            "supported_evidence_fields": supported,
            "validation_errors": errors,
            "validation_notes": notes,
        }

    points = OPERATIONAL_FIT_POINT_TABLE[str(materiality)][str(rating)]
    if points and not changed_operational:
        notes.append("Operational Fit has no point effect because no operational assumptions changed")
        points = 0.0
    if previous_equivalent_fields and previous_points is not None:
        notes.append(
            "Operational Fit reuses previous points because the operational state returned to previous values and no other scenario inputs moved"
        )
        points = previous_points
    operational_evidence = _operational_refs(supported)
    if abs(points) == 5.0 and (
        len(set(changed_operational)) < 2 or len(operational_evidence.intersection(changed_operational)) < 2
    ):
        notes.append(
            "Operational Fit +/-5.0 capped to +/-3.5 because V1 requires at least two changed operational fields with operational evidence"
        )
        points = 3.5 if points > 0 else -3.5

    return {
        "points": clamp(points, -5.0, 5.0),
        "rating": rating,
        "materiality": materiality,
        "interaction_with_completion_outlook": interaction,
        "central_reason": str(combined.get("central_reason") or ""),
        "evidence_fields": evidence_fields,
        "supported_evidence_fields": supported,
        "validation_errors": errors,
        "validation_notes": notes,
    }


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


def _carryover_candidate(packet: dict[str, Any]) -> dict[str, Any]:
    candidate = ((packet.get("iteration_context") or {}).get("reality_check_carryover_candidate") or {})
    return candidate if isinstance(candidate, dict) else {}


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


def _validate_reality_check_carryover_assessment(
    packet: dict[str, Any],
    review: dict[str, Any],
    *,
    is_hidden_baseline: bool,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    candidate = _carryover_candidate(packet)
    candidate_active = bool(candidate.get("active"))
    raw = review.get("reality_check_carryover_assessment")
    returned_to_hidden_baseline = bool((packet.get("iteration_context") or {}).get("returned_to_hidden_baseline_state"))
    if is_hidden_baseline or returned_to_hidden_baseline or not candidate_active:
        if isinstance(raw, dict):
            return deepcopy(raw), errors
        return {}, errors
    if not isinstance(raw, dict):
        return {}, ["reality_check_carryover_assessment must be an object when a carryover candidate is active"]

    status = raw.get("status")
    relation = raw.get("current_issue_relation")
    if status not in REALITY_CHECK_CARRYOVER_STATUSES:
        errors.append(
            "reality_check_carryover_assessment.status must be one of "
            f"{sorted(REALITY_CHECK_CARRYOVER_STATUSES)}"
        )
    if relation not in REALITY_CHECK_CURRENT_ISSUE_RELATIONS:
        errors.append(
            "reality_check_carryover_assessment.current_issue_relation must be one of "
            f"{sorted(REALITY_CHECK_CURRENT_ISSUE_RELATIONS)}"
        )
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("reality_check_carryover_assessment.reason is required")
    evidence_fields = _string_list(raw.get("evidence_fields"))
    if not evidence_fields:
        errors.append("reality_check_carryover_assessment.evidence_fields must include packet evidence")
    supported = [field for field in evidence_fields if field in _packet_evidence_refs(packet)]
    if evidence_fields and not supported:
        errors.append("reality_check_carryover_assessment.evidence_fields do not reference packet evidence")

    return {
        "status": status,
        "current_issue_relation": relation,
        "reason": str(reason or ""),
        "evidence_fields": evidence_fields,
        "supported_evidence_fields": supported,
    }, errors


def _effect_sign(effect: str, delta: float) -> int:
    if effect == "neutral":
        return 0
    if delta > 0:
        return 1 if effect == "reinforce_gain" else -1 if effect in {"offset_gain", "reversal"} else 0
    if delta < 0:
        return 1 if effect in {"soften_decline", "reversal"} else -1 if effect == "reinforce_decline" else 0
    if effect == "reward_coherence":
        return 1
    if effect in {"penalize_incoherence", "reversal"}:
        return -1
    return 0


def _effect_compatible(effect: str, delta: float) -> bool:
    if effect == "neutral":
        return True
    if delta > 0:
        return effect in {"reinforce_gain", "offset_gain", "reversal"}
    if delta < 0:
        return effect in {"soften_decline", "reinforce_decline", "reversal"}
    return effect in {"reward_coherence", "penalize_incoherence", "reversal"}


def score_reality_check(
    reality_check: dict[str, Any],
    *,
    pre_reality_delta: float,
    packet: dict[str, Any],
    operational_fit_assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    notes: list[str] = []
    if not isinstance(reality_check, dict):
        return {
            "points": None,
            "allocation_points": [],
            "validation_errors": ["reality_check must be an object"],
            "validation_notes": [],
        }

    effect = reality_check.get("effect")
    strength = reality_check.get("strength")
    if effect not in REALITY_CHECK_EFFECTS:
        errors.append(f"reality_check.effect must be one of {sorted(REALITY_CHECK_EFFECTS)}")
    if strength not in REALITY_CHECK_STRENGTHS:
        errors.append(f"reality_check.strength must be one of {sorted(REALITY_CHECK_STRENGTHS)}")

    if effect in REALITY_CHECK_EFFECTS and not _effect_compatible(str(effect), pre_reality_delta):
        notes.append("Reality Check effect is incompatible with pre-Reality movement and is downgraded to neutral")
        effect = "neutral"
        strength = "none"

    if effect == "reinforce_gain" and strength in {"moderate", "strong", "reversal"}:
        notes.append("Positive reinforcement is capped at slight in V1")
        strength = "slight"
    if effect == "neutral":
        strength = "none"

    evidence_fields = _string_list(reality_check.get("evidence_fields"))
    supported = [field for field in evidence_fields if field in _packet_evidence_refs(packet)]
    if effect != "neutral" and not supported:
        errors.append("reality_check.evidence_fields must reference packet evidence for non-neutral effects")
    operational_evidence = _operational_refs(
        _string_list((operational_fit_assessment or {}).get("supported_evidence_fields"))
        or _string_list((operational_fit_assessment or {}).get("evidence_fields"))
    )

    if errors:
        return {
            "points": None,
            "effect": effect,
            "strength": strength,
            "allocation_points": [],
            "supported_evidence_fields": supported,
            "validation_errors": errors,
            "validation_notes": notes,
        }

    fraction = REALITY_CHECK_FRACTIONS[str(strength)]
    base = abs(pre_reality_delta)
    if base < 1.0 and effect != "neutral":
        base = 1.0
    points = _clean_points(base * fraction * _effect_sign(str(effect), pre_reality_delta))
    if effect == "neutral":
        points = 0.0

    allocations = reality_check.get("allocations") or []
    allocation_errors: list[str] = []
    allocation_points: list[dict[str, Any]] = []
    if points:
        if not isinstance(allocations, list) or not 1 <= len(allocations) <= 4:
            allocation_errors.append("reality_check.allocations must include 1-4 allocations for non-neutral effects")
        else:
            share_total = 0.0
            provider_shares: list[float | None] = []
            for index, allocation in enumerate(allocations):
                if not isinstance(allocation, dict):
                    allocation_errors.append(f"reality_check.allocations[{index}] must be an object")
                    continue
                pillar, subpillar, target_id, target_note = _canonical_reality_check_target(
                    allocation.get("allocation_target_id"),
                )
                if target_note:
                    notes.append(target_note)
                share = _number(allocation.get("share"))
                if target_id is None or subpillar not in REALITY_CHECK_ALLOWED_SUBPILLARS.get(pillar, set()):
                    allocation_errors.append(
                        f"reality_check.allocations[{index}] must target an allowed allocation_target_id"
                    )
                if share is None or share <= 0:
                    provider_shares.append(None)
                    share = 0.0
                else:
                    provider_shares.append(share)
                    share_total += share
                for field_name in ("movement_label", "rationale", "incremental_check"):
                    if not isinstance(allocation.get(field_name), str) or not allocation.get(field_name).strip():
                        allocation_errors.append(f"reality_check.allocations[{index}].{field_name} is required")
                rationale = str(allocation.get("rationale") or "").strip()
                incremental_check = str(allocation.get("incremental_check") or "").strip()
                incremental_lower = incremental_check.lower()
                if incremental_check and (
                    len(incremental_check) < 12 or incremental_lower == rationale.lower()
                ):
                    allocation_errors.append(
                        f"reality_check.allocations[{index}].incremental_check must explain non-duplication"
                    )
                allocation_evidence = _string_list(allocation.get("evidence_fields")) or evidence_fields
                allocation_operational_overlap = bool(_operational_refs(allocation_evidence).intersection(operational_evidence))
                if subpillar == OPERATIONAL_FIT_LABEL and allocation_operational_overlap:
                    incremental_cues = (
                        "after",
                        "beyond",
                        "incremental",
                        "not already",
                        "rather than",
                        "remaining",
                        "residual",
                    )
                    if not any(cue in incremental_lower for cue in incremental_cues):
                        allocation_errors.append(
                            f"reality_check.allocations[{index}].incremental_check must show the Operational Fit allocation is not duplicate credit"
                        )
                allocation_points.append({
                    "allocation_target_id": target_id,
                    "pillar": pillar,
                    "subpillar": subpillar,
                    "label": str(allocation.get("movement_label") or "Reality Check"),
                    "share": share,
                    "points": _clean_points(float(points) * share),
                    "rationale": str(allocation.get("rationale") or ""),
                    "incremental_check": str(allocation.get("incremental_check") or ""),
                })
            if allocation_points and (
                len(provider_shares) != len(allocation_points)
                or any(share is None for share in provider_shares)
                or abs(share_total - 1.0) > 0.02
            ):
                equal_share = 1.0 / len(allocation_points)
                normalized_shares = [equal_share for _ in allocation_points]
                normalized_shares[-1] = max(0.0, 1.0 - sum(normalized_shares[:-1]))
                for item, share in zip(allocation_points, normalized_shares):
                    item["share"] = share
                    item["points"] = _clean_points(float(points) * share)
                notes.append("Reality Check allocation shares were assigned equally by the app")
    elif allocations:
        notes.append("Neutral Reality Check ignores allocation rows")

    if allocation_errors:
        notes.append(
            "Reality Check downgraded to neutral because allocation rows did not pass V1 traceability checks"
        )
        return {
            "points": 0,
            "effect": "neutral",
            "strength": "none",
            "fraction": 0.0,
            "central_reason": str(reality_check.get("central_reason") or ""),
            "allocation_points": [],
            "supported_evidence_fields": supported,
            "validation_errors": [],
            "validation_notes": [*notes, *allocation_errors],
        }

    return {
        "points": _clean_points(points),
        "effect": effect,
        "strength": strength,
        "fraction": fraction,
        "central_reason": str(reality_check.get("central_reason") or ""),
        "allocation_points": allocation_points,
        "supported_evidence_fields": supported,
        "validation_errors": [],
        "validation_notes": notes,
    }


def _scaled_allocation_points(allocation_points: list[dict[str, Any]], factor: float) -> list[dict[str, Any]]:
    scaled: list[dict[str, Any]] = []
    for allocation in allocation_points or []:
        if not isinstance(allocation, dict):
            continue
        row = deepcopy(allocation)
        points = _number(row.get("points"))
        if points is not None:
            row["points"] = _clean_points(points * factor)
        row["carryover"] = True
        scaled.append(row)
    return scaled


def _apply_reality_check_carryover(
    *,
    packet: dict[str, Any],
    current_assessment: dict[str, Any],
    carryover_assessment: dict[str, Any],
) -> dict[str, Any]:
    candidate = _carryover_candidate(packet)
    previous_points = _number(candidate.get("previous_reality_check_points"))
    if not candidate.get("active") or previous_points is None or previous_points >= 0:
        return current_assessment
    if not isinstance(carryover_assessment, dict) or not carryover_assessment:
        return current_assessment

    precheck = candidate.get("app_state_precheck") or {}
    if isinstance(precheck, dict) and precheck.get("status") == "resolved_by_field_return":
        relation = str(carryover_assessment.get("current_issue_relation") or "mixed_or_unclear")
        if relation == "new_independent_issue":
            resolved = deepcopy(current_assessment)
            resolved["carryover_assessment"] = {
                **deepcopy(carryover_assessment),
                "status": "resolved_or_superseded",
                "app_state_precheck": deepcopy(precheck),
            }
            resolved["carryover_candidate"] = deepcopy(candidate)
            resolved.setdefault("validation_notes", []).append(
                "Previous negative Reality Check carryover was resolved by app state precheck; latest Reality Check kept as a new independent issue"
            )
            return resolved
        if relation != "new_independent_issue":
            reason = str(
                precheck.get("reason")
                or "Previous negative Reality Check carryover evidence returned to baseline; no same-issue carryover was applied."
            )
            resolved = _neutral_reality_check_assessment(reason)
            resolved["carryover_assessment"] = {
                **deepcopy(carryover_assessment),
                "status": "resolved_or_superseded",
                "app_state_precheck": deepcopy(precheck),
            }
            resolved["carryover_candidate"] = deepcopy(candidate)
            resolved["validation_notes"] = [
                *list(resolved.get("validation_notes") or []),
                "Previous negative Reality Check carryover was resolved by app state precheck",
            ]
            return resolved

    status = carryover_assessment.get("status")
    if status == "resolved_or_superseded":
        relation = str(carryover_assessment.get("current_issue_relation") or "mixed_or_unclear")
        if relation != "new_independent_issue":
            resolved = _neutral_reality_check_assessment(
                "Previous negative Reality Check carryover was resolved or superseded; no additional same-issue Reality Check adjustment was applied."
            )
            resolved["carryover_assessment"] = deepcopy(carryover_assessment)
            return resolved
        resolved = deepcopy(current_assessment)
        resolved["carryover_assessment"] = deepcopy(carryover_assessment)
        resolved.setdefault("validation_notes", []).append(
            "Previous negative Reality Check carryover was resolved or superseded; latest Reality Check kept because it was classified as a new independent issue"
        )
        return resolved
    if status not in {"still_relevant", "partly_mitigated"}:
        return current_assessment

    factor = 1.0 if status == "still_relevant" else 0.5
    carryover_points = _clean_points(float(previous_points) * factor)
    current_points = float(current_assessment.get("points") or 0.0)
    relation = str(carryover_assessment.get("current_issue_relation") or "mixed_or_unclear")
    combine = relation == "new_independent_issue" and current_points < 0
    if combine:
        final_points = max(REALITY_CHECK_CARRYOVER_NEGATIVE_CAP, carryover_points + current_points)
    else:
        final_points = min(current_points, carryover_points)

    if final_points == current_points:
        merged = deepcopy(current_assessment)
        merged["carryover_assessment"] = deepcopy(carryover_assessment)
        merged.setdefault("validation_notes", []).append(
            "Previous negative Reality Check carryover remained active but did not exceed the current adjustment"
        )
        return merged

    previous_assessment = candidate.get("previous_reality_check_assessment") or {}
    previous_allocations = (
        candidate.get("previous_reality_check_allocation_points")
        or previous_assessment.get("allocation_points")
        or []
    )
    if combine:
        allocation_points = [
            *_scaled_allocation_points(previous_allocations, factor),
            *deepcopy(current_assessment.get("allocation_points") or []),
        ]
        reason = (
            "Previous negative Reality Check carryover remains active and the latest Reality Check identifies a "
            "new independent concern."
        )
    else:
        allocation_points = _scaled_allocation_points(previous_allocations, factor)
        reason = "Previous negative Reality Check carryover remains active."

    notes = [
        *list(current_assessment.get("validation_notes") or []),
        reason,
    ]
    return {
        **deepcopy(current_assessment),
        "points": _clean_points(final_points),
        "effect": previous_assessment.get("effect") or current_assessment.get("effect"),
        "strength": previous_assessment.get("strength") or current_assessment.get("strength"),
        "fraction": current_assessment.get("fraction"),
        "central_reason": str(carryover_assessment.get("reason") or previous_assessment.get("central_reason") or reason),
        "allocation_points": allocation_points,
        "carryover_assessment": deepcopy(carryover_assessment),
        "carryover_candidate": deepcopy(candidate),
        "validation_notes": notes,
    }


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

    operational = review.get("operational_fit")
    if not isinstance(operational, dict):
        errors.append("operational_fit must be an object")
        operational_score = {"points": None, "validation_errors": ["operational_fit must be an object"]}
    else:
        operational_score = score_operational_fit(operational, packet)
        errors.extend(operational_score.get("validation_errors") or [])

    required_objects = (
        "completion_outlook_analysis",
        "reality_check",
        "continuity_update",
        "analytical_narrative_draft",
    )
    for field in required_objects:
        if not isinstance(review.get(field), dict):
            errors.append(f"{field} must be an object")
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
        for field in ANALYTICAL_NARRATIVE_DRAFT_FIELDS:
            value = draft.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"analytical_narrative_draft.{field} must be a non-empty string")
        minimum_words = (
            MIN_HIDDEN_BASELINE_ANALYTICAL_DRAFT_WORDS
            if is_hidden_baseline
            else MIN_ANALYTICAL_DRAFT_WORDS
        )
        draft_words = _word_count({field: draft.get(field) for field in ANALYTICAL_NARRATIVE_DRAFT_FIELDS})
        if draft_words < minimum_words:
            errors.append(
                "analytical_narrative_draft must be an extensive interpretation "
                f"with at least {minimum_words} words across required fields"
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
    carryover_assessment, carryover_errors = _validate_reality_check_carryover_assessment(
        packet,
        review,
        is_hidden_baseline=is_hidden_baseline,
    )
    errors.extend(carryover_errors)

    continuity_update = deepcopy(review.get("continuity_update") or {})
    reality_check = deepcopy(review.get("reality_check") or {})
    if is_hidden_baseline:
        continuity_update["active_tension"] = ""
        reality_check = {
            **reality_check,
            "effect": "neutral",
            "strength": "none",
            "allocations": [],
        }

    validated = {
        "validation_status": "valid" if not errors else "invalid",
        "validation_errors": errors,
        "validation_notes": list(operational_score.get("validation_notes") or []),
        "review_metadata": deepcopy(metadata),
        "completion_outlook_analysis": deepcopy(review.get("completion_outlook_analysis") or {}),
        "strategy_shift_check": deepcopy(strategy_shift),
        "operational_fit": deepcopy(operational or {}),
        "operational_fit_assessment": operational_score,
        "reality_check": reality_check,
        "reality_check_carryover_assessment": deepcopy(carryover_assessment),
        "continuity_update": continuity_update,
        "analytical_narrative_draft": deepcopy(draft),
    }
    if not is_hidden_baseline:
        validated["development_discussion_options"] = deepcopy(development_discussion_options)
    return validated


def score_pass1_review(packet: dict[str, Any], pass1_review: dict[str, Any]) -> dict[str, Any]:
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
    if hidden_baseline:
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
        }
    if validated["validation_status"] != "valid":
        return {
            "validation_status": "invalid",
            "validation_errors": validated["validation_errors"],
            "validation_notes": validated.get("validation_notes", []),
            "xgboost_completion_outlook": _clean_points(completion_score),
            "operational_fit_points": None,
            "pre_reality_score": None,
            "pre_reality_delta": None,
            "reality_check_points": None,
            "trial_score": None,
            "input_hash": input_hash,
        }

    returned_to_hidden_baseline = bool(iteration.get("returned_to_hidden_baseline_state"))
    if returned_to_hidden_baseline:
        reference_trial_score = _reference_trial_score(packet)
        reference_pre_reality_score = _reference_pre_reality_score(packet)
        pre_reality_delta = (
            float(completion_score) - float(reference_pre_reality_score)
            if reference_pre_reality_score is not None
            else 0.0
        )
        reason = (
            "Current scenario state matches the hidden baseline state; Operational Fit and Reality Check are neutralized "
            "to prevent path-dependent score drift on a full baseline return."
        )
        return {
            "validation_status": "valid",
            "validation_errors": [],
            "validation_notes": [*list(validated.get("validation_notes") or []), reason],
            "xgboost_completion_outlook": _clean_points(completion_score),
            "operational_fit_points": 0.0,
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

    operational_points = float(validated["operational_fit_assessment"]["points"])
    pre_reality_score = float(completion_score) + operational_points
    reference_trial_score = _reference_trial_score(packet)
    reference_pre_reality_score = _reference_pre_reality_score(packet)
    pre_reality_delta = pre_reality_score - float(
        reference_pre_reality_score if reference_pre_reality_score is not None else completion_score
    )
    reality_assessment = score_reality_check(
        validated["reality_check"],
        pre_reality_delta=pre_reality_delta,
        packet=packet,
        operational_fit_assessment=validated["operational_fit_assessment"],
    )
    validation_errors = list(reality_assessment.get("validation_errors") or [])
    validation_notes = [
        *list(validated.get("validation_notes") or []),
        *list(reality_assessment.get("validation_notes") or []),
    ]
    if validation_errors:
        return {
            "validation_status": "invalid",
            "validation_errors": validation_errors,
            "validation_notes": validation_notes,
            "xgboost_completion_outlook": _clean_points(completion_score),
            "operational_fit_points": _clean_points(operational_points),
            "pre_reality_score": _clean_points(pre_reality_score),
            "pre_reality_delta": _clean_points(pre_reality_delta),
            "reality_check_points": None,
            "trial_score": None,
            "input_hash": input_hash,
        }

    reality_assessment = _apply_reality_check_carryover(
        packet=packet,
        current_assessment=reality_assessment,
        carryover_assessment=validated.get("reality_check_carryover_assessment") or {},
    )
    validation_notes = [
        *list(validation_notes),
        *[
            note
            for note in list(reality_assessment.get("validation_notes") or [])
            if note not in validation_notes
        ],
    ]
    reality_points = float(reality_assessment["points"])
    trial_score = clamp(pre_reality_score + reality_points, 0, 100)
    return {
        "validation_status": "valid",
        "validation_errors": [],
        "validation_notes": validation_notes,
        "xgboost_completion_outlook": _clean_points(completion_score),
        "operational_fit_points": _clean_points(operational_points),
        "pre_reality_score": _clean_points(pre_reality_score),
        "pre_reality_delta": _clean_points(pre_reality_delta),
        "reality_check_points": _clean_points(reality_points),
        "reality_check_allocation_points": deepcopy(reality_assessment.get("allocation_points") or []),
        "trial_score": trial_score,
        "delta_vs_previous_trial_score": _clean_points(
            float(trial_score) - float(reference_trial_score)
            if reference_trial_score is not None
            else float(trial_score) - float(completion_score)
        ),
        "delta_vs_previous_pre_reality_score": _clean_points(pre_reality_delta),
        "delta_vs_baseline_xgboost": _clean_points(float(trial_score) - float(model.get("baseline_completion_score", completion_score))),
        "operational_fit_assessment": deepcopy(validated["operational_fit_assessment"]),
        "reality_check_assessment": deepcopy(reality_assessment),
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
