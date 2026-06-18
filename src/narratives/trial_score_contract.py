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
PASS1_SCHEMA_VERSION = "trial_score_pass1_schema_v1"
PASS2_SCHEMA_VERSION = "trial_score_pass2_schema_v1"
PROMPT_TEMPLATE_VERSION = "trial_score_two_pass_prompt_v1_2"

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


def _packet_evidence_refs(packet: dict[str, Any]) -> set[str]:
    refs = {
        "completion_score",
        "model_interpretation.completion_score",
        "model_interpretation.score_delta",
        "operational_assumptions",
        "field_changes",
        "text_context",
    }
    for field in ((packet.get("iteration_context") or {}).get("changed_fields") or []):
        refs.add(str(field))
    for section_name in ("structured_features", "text_context", "operational_assumptions", "model_interpretation"):
        section = packet.get(section_name) or {}
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            refs.add(str(key))
            refs.add(f"{section_name}.{key}")
            if isinstance(value, dict):
                for child_key in value:
                    refs.add(f"{section_name}.{key}.{child_key}")
    return refs


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


def _reference_score(packet: dict[str, Any]) -> float | None:
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
        if not isinstance(allocations, list) or not 1 <= len(allocations) <= 3:
            allocation_errors.append("reality_check.allocations must include 1-3 allocations for non-neutral effects")
        else:
            share_total = 0.0
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
                    allocation_errors.append(f"reality_check.allocations[{index}].share must be positive")
                    share = 0.0
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
            if allocation_points and abs(share_total - 1.0) > 0.02:
                allocation_errors.append("reality_check.allocations shares must sum to 1.0")
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
            "central_tension_candidate": deepcopy(reality_check.get("central_tension_candidate") or {}),
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
        "central_tension_candidate": deepcopy(reality_check.get("central_tension_candidate") or {}),
        "allocation_points": allocation_points,
        "supported_evidence_fields": supported,
        "validation_errors": [],
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
        "central_tension_candidate",
        "broader_strategic_question_candidate",
        "continuity_update",
    )
    for field in required_objects:
        if not isinstance(review.get(field), dict):
            errors.append(f"{field} must be an object")

    return {
        "validation_status": "valid" if not errors else "invalid",
        "validation_errors": errors,
        "validation_notes": list(operational_score.get("validation_notes") or []),
        "review_metadata": deepcopy(metadata),
        "completion_outlook_analysis": deepcopy(review.get("completion_outlook_analysis") or {}),
        "strategy_shift_check": deepcopy(strategy_shift),
        "operational_fit": deepcopy(operational or {}),
        "operational_fit_assessment": operational_score,
        "reality_check": deepcopy(review.get("reality_check") or {}),
        "central_tension_candidate": deepcopy(review.get("central_tension_candidate") or {}),
        "broader_strategic_question_candidate": deepcopy(review.get("broader_strategic_question_candidate") or {}),
        "continuity_update": deepcopy(review.get("continuity_update") or {}),
    }


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

    operational_points = float(validated["operational_fit_assessment"]["points"])
    pre_reality_score = float(completion_score) + operational_points
    reference_score = _reference_score(packet)
    pre_reality_delta = pre_reality_score - float(reference_score if reference_score is not None else completion_score)
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
        "delta_vs_previous_trial_score": _clean_points(float(trial_score) - float(reference_score)),
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
    required = (
        "review_metadata",
        "trial_score_narrative",
        "pillar_reading",
        "central_tension",
        "broader_strategic_question",
    )
    for field in required:
        if field == "pillar_reading":
            if not isinstance(review.get(field), list):
                errors.append("pillar_reading must be an array")
        elif not isinstance(review.get(field), (dict, str)):
            errors.append(f"{field} must be present")
    facilitator_questions = review.get("facilitator_questions") or []
    if not isinstance(facilitator_questions, list):
        errors.append("facilitator_questions must be an array when present")
        facilitator_questions = []
    elif len(facilitator_questions) > 3:
        errors.append("facilitator_questions must include at most 3 questions")
    for index, item in enumerate(facilitator_questions):
        if not isinstance(item, dict):
            errors.append(f"facilitator_questions[{index}] must be an object")
            continue
        if not isinstance(item.get("question"), str) or not item.get("question", "").strip():
            errors.append(f"facilitator_questions[{index}].question is required")
        why_it_matters = item.get("why_it_matters")
        if why_it_matters is not None and (
            not isinstance(why_it_matters, str) or not why_it_matters.strip()
        ):
            errors.append(f"facilitator_questions[{index}].why_it_matters must be non-empty when present")
        related = item.get("related_feature_families", [])
        if related is None:
            related = []
        if not isinstance(related, list) or not all(isinstance(value, str) for value in related):
            errors.append(f"facilitator_questions[{index}].related_feature_families must be an array of strings")
    return {
        "validation_status": "valid" if not errors else "invalid",
        "validation_errors": errors,
        "review_metadata": deepcopy(review.get("review_metadata") or {}),
        "trial_score_narrative": deepcopy(review.get("trial_score_narrative") or {}),
        "pillar_reading": deepcopy(review.get("pillar_reading") or []),
        "central_tension": deepcopy(review.get("central_tension") or {}),
        "broader_strategic_question": deepcopy(review.get("broader_strategic_question") or {}),
        "facilitator_questions": deepcopy(facilitator_questions),
    }
