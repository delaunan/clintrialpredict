"""Validation and deterministic scoring for narrative Quality Reviews."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.narratives.contract_fixtures import REQUIRED_REVIEW_DOMAINS
from src.narratives.packet_builder import stable_packet_hash

STANDARD_RATINGS = {
    "strong": 1,
    "acceptable": 0,
    "weak": -2,
    "conflicting": -4,
}

CHANGE_INTEGRITY_RATINGS = {
    "improved": 1,
    "neutral": 0,
    "simplified": -2,
    "potential_shortcut": -4,
}

TEXT_CONSISTENCY_RATINGS = {
    "consistent": 0,
    "minor_tension": -1,
    "material_tension": -2,
    "contradiction": -4,
}

DOMAIN_RATING_POINTS = {
    "development_question_fit": STANDARD_RATINGS,
    "scientific_rigor": STANDARD_RATINGS,
    "population_relevance": STANDARD_RATINGS,
    "endpoint_and_comparator_logic": STANDARD_RATINGS,
    "operational_scale_fit": STANDARD_RATINGS,
    "change_integrity": CHANGE_INTEGRITY_RATINGS,
    "text_consistency": TEXT_CONSISTENCY_RATINGS,
}

DOMAIN_DEFAULT_PILLARS = {
    "scientific_rigor": "evidence_coherence",
    "endpoint_and_comparator_logic": "evidence_coherence",
    "development_question_fit": "population_strategy_fit",
    "population_relevance": "population_strategy_fit",
    "operational_scale_fit": "execution_plausibility",
    "change_integrity": "execution_plausibility",
}

PILLAR_LABELS = {
    "evidence_coherence": "Evidence Coherence",
    "population_strategy_fit": "Population & Strategy Fit",
    "execution_plausibility": "Execution Plausibility",
}

PILLAR_CAP_MIN = -4
PILLAR_CAP_MAX = 3
SUBCATEGORY_CAP_MIN = -3
SUBCATEGORY_CAP_MAX = 2
QUALITY_ADJUSTMENT_MIN = -10
QUALITY_ADJUSTMENT_MAX = 10
FINAL_SCORE_MIN = 0
FINAL_SCORE_MAX = 100

APP_OWNED_SCORE_FIELDS = {
    "quality_adjustment",
    "final_candidate_score",
    "quality_assessment",
}

PARTICIPANT_REVIEW_KEYS = {
    "what_changed",
    "why_completion_score_may_have_moved",
    "what_the_design_gained",
    "what_the_design_may_have_sacrificed",
    "operational_feasibility_note",
    "text_consistency_note",
    "challenge_question",
}

EVIDENCE_FIELD_TOKENS = (
    "endpoint",
    "primary_outcomes",
    "comparator",
    "placebo",
    "masking",
    "allocation",
    "biomarker",
    "scientific",
    "primary_duration",
)

POPULATION_FIELD_TOKENS = (
    "population",
    "adult",
    "child",
    "older_adult",
    "gender",
    "healthy_volunteers",
    "line_of_therapy",
    "patient_severity",
    "criteria",
    "summary",
    "strategic_ambition",
    "phase",
    "primary_purpose",
)

EXECUTION_FIELD_TOKENS = (
    "operational",
    "planned_enrollment",
    "planned_sites",
    "planned_duration",
    "administration",
    "intervention",
    "number_of_arms",
    "sponsor",
    "has_dmc",
)


def clamp(value: int | float, minimum: int, maximum: int) -> int:
    return int(max(minimum, min(maximum, round(float(value)))))


def _field_matches(evidence_fields: list[str], tokens: tuple[str, ...]) -> bool:
    joined = " ".join(str(field).lower() for field in evidence_fields)
    return any(token in joined for token in tokens)


def route_text_consistency_pillar(evidence_fields: list[str]) -> str:
    """Route text consistency to the most affected Quality Assessment pillar."""
    if _field_matches(evidence_fields, EVIDENCE_FIELD_TOKENS):
        return "evidence_coherence"
    if _field_matches(evidence_fields, POPULATION_FIELD_TOKENS):
        return "population_strategy_fit"
    if _field_matches(evidence_fields, EXECUTION_FIELD_TOKENS):
        return "execution_plausibility"
    return "evidence_coherence"


def _domain_pillar(domain_name: str, evidence_fields: list[str]) -> str:
    if domain_name == "text_consistency":
        return route_text_consistency_pillar(evidence_fields)
    return DOMAIN_DEFAULT_PILLARS[domain_name]


def _is_positive_rating(domain_name: str, rating: str) -> bool:
    return DOMAIN_RATING_POINTS[domain_name].get(rating, 0) > 0


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

    impact_changes = model.get("xgboost_impact_changes") if isinstance(model, dict) else []
    for impact in impact_changes or []:
        if not isinstance(impact, dict):
            continue
        for key in ("name", "pillar", "subcategory"):
            value = impact.get(key)
            if value:
                refs.add(str(value))
                refs.add(f"xgboost_impact_changes.{value}")

    return refs


def _review_with_supported_evidence(packet: dict[str, Any], validated_review: dict[str, Any]) -> dict[str, Any]:
    supported_refs = _evidence_reference_set(packet)
    review = deepcopy(validated_review)
    for domain in (review.get("quality_review_domains") or {}).values():
        evidence_fields = [str(field) for field in domain.get("evidence_fields") or []]
        supported = [field for field in evidence_fields if field in supported_refs]
        unsupported = [field for field in evidence_fields if field not in supported_refs]
        domain["supported_evidence_fields"] = supported
        domain["unsupported_evidence_fields"] = unsupported
        if domain.get("point_effect") and not supported:
            domain["point_effect"] = 0
            notes = list(domain.get("validation_notes") or [])
            notes.append("rating has no point effect because evidence_fields do not reference packet evidence")
            domain["validation_notes"] = notes
    return review


def _domain_supported_evidence_fields(domain: dict[str, Any]) -> list[str]:
    if "supported_evidence_fields" in domain:
        return deepcopy(domain.get("supported_evidence_fields") or [])
    return deepcopy(domain.get("evidence_fields") or [])


def _validated_domain(domain_name: str, domain: Any) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(domain, dict):
        return {
            "rating": None,
            "rationale": "",
            "evidence_fields": [],
            "valid": False,
            "point_effect": 0,
            "validation_notes": ["domain is not an object"],
        }, [f"{domain_name}: domain is not an object"]

    rating = domain.get("rating")
    rationale = domain.get("rationale")
    evidence_fields = domain.get("evidence_fields")

    allowed = DOMAIN_RATING_POINTS[domain_name]
    valid = True
    if rating not in allowed:
        errors.append(f"{domain_name}: invalid rating {rating!r}")
        valid = False
    if not isinstance(rationale, str):
        errors.append(f"{domain_name}: rationale must be a string")
        rationale = ""
        valid = False
    if not isinstance(evidence_fields, list):
        errors.append(f"{domain_name}: evidence_fields must be a list")
        evidence_fields = []
        valid = False

    raw_points = allowed.get(str(rating), 0)
    evidence_required = raw_points != 0
    has_evidence = bool(evidence_fields)
    point_effect = raw_points if valid and (has_evidence or not evidence_required) else 0
    notes = []
    if valid and evidence_required and not has_evidence:
        notes.append("rating has no point effect because evidence_fields is empty")

    return {
        "rating": rating,
        "rationale": rationale,
        "evidence_fields": deepcopy(evidence_fields),
        "valid": valid,
        "point_effect": point_effect,
        "validation_notes": notes,
    }, errors


def _validate_participant_review(review: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    participant = review.get("participant_review")
    if not isinstance(participant, dict):
        return {}, ["participant_review must be an object"]

    errors = [
        f"participant_review.{key} must be a string"
        for key in sorted(PARTICIPANT_REVIEW_KEYS)
        if not isinstance(participant.get(key), str)
    ]
    return {
        key: participant.get(key, "")
        for key in sorted(PARTICIPANT_REVIEW_KEYS)
        if isinstance(participant.get(key), str)
    }, errors


def _validate_object(value: Any, field_name: str, required: bool = True) -> tuple[dict[str, Any], list[str]]:
    if value is None and not required:
        return {}, []
    if not isinstance(value, dict):
        return {}, [f"{field_name} must be an object"]
    return deepcopy(value), []


def _quality_contributions(validated_domains: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pillars = {
        key: {
            "label": label,
            "raw_points": 0,
            "points": 0,
            "domains": {},
        }
        for key, label in PILLAR_LABELS.items()
    }

    positive_domains_with_evidence = 0
    for domain_name, domain in validated_domains.items():
        evidence_fields = domain.get("evidence_fields") or []
        pillar_key = _domain_pillar(domain_name, evidence_fields)
        raw_points = int(domain.get("point_effect", 0))
        subcategory_points = clamp(raw_points, SUBCATEGORY_CAP_MIN, SUBCATEGORY_CAP_MAX)
        pillars[pillar_key]["domains"][domain_name] = {
            "rating": domain.get("rating"),
            "raw_points": raw_points,
            "points": subcategory_points,
            "evidence_fields": deepcopy(evidence_fields),
            "supported_evidence_fields": _domain_supported_evidence_fields(domain),
            "unsupported_evidence_fields": deepcopy(domain.get("unsupported_evidence_fields") or []),
        }
        pillars[pillar_key]["raw_points"] += subcategory_points
        if _is_positive_rating(domain_name, str(domain.get("rating"))) and _domain_supported_evidence_fields(domain):
            positive_domains_with_evidence += 1

    for pillar in pillars.values():
        pillar["points"] = clamp(pillar["raw_points"], PILLAR_CAP_MIN, PILLAR_CAP_MAX)

    rating_points = sum(int(pillar["points"]) for pillar in pillars.values())
    if rating_points > 0 and positive_domains_with_evidence == 0:
        rating_points = 0
    quality_adjustment = clamp(rating_points, QUALITY_ADJUSTMENT_MIN, QUALITY_ADJUSTMENT_MAX)

    return {
        "pillars": pillars,
        "rating_points": rating_points,
        "quality_adjustment": quality_adjustment,
    }


def _has_complete_scoring_domains(validated_review: dict[str, Any]) -> bool:
    domains = validated_review.get("quality_review_domains") or {}
    if set(domains) != REQUIRED_REVIEW_DOMAINS:
        return False
    return all(domain.get("valid") is True for domain in domains.values())


def validate_review_json(review: dict[str, Any]) -> dict[str, Any]:
    """Validate provider/mock review JSON and return normalized review fields."""
    errors: list[str] = []
    if not isinstance(review, dict):
        return {
            "validation_status": "invalid",
            "validation_errors": ["review must be an object"],
            "quality_review_domains": {},
            "participant_review": {},
            "continuity": {},
            "trace": {},
        }

    for field_name in sorted(APP_OWNED_SCORE_FIELDS.intersection(review)):
        errors.append(f"{field_name} is application-owned and ignored if returned by provider")

    domains = review.get("quality_review_domains")
    validated_domains: dict[str, dict[str, Any]] = {}
    if not isinstance(domains, dict):
        errors.append("quality_review_domains must be an object")
    else:
        missing = REQUIRED_REVIEW_DOMAINS.difference(domains)
        for domain_name in sorted(missing):
            errors.append(f"{domain_name}: missing required domain")
        for domain_name in sorted(REQUIRED_REVIEW_DOMAINS):
            if domain_name not in domains:
                continue
            validated, domain_errors = _validated_domain(domain_name, domains[domain_name])
            validated_domains[domain_name] = validated
            errors.extend(domain_errors)

    participant_review, participant_errors = _validate_participant_review(review)
    errors.extend(participant_errors)
    score_movement_review, score_errors = _validate_object(review.get("score_movement_review"), "score_movement_review")
    continuity, continuity_errors = _validate_object(review.get("continuity"), "continuity")
    trace, trace_errors = _validate_object(review.get("trace"), "trace")
    errors.extend(score_errors)
    errors.extend(continuity_errors)
    errors.extend(trace_errors)

    return {
        "validation_status": "valid" if not errors else "partial",
        "validation_errors": errors,
        "score_movement_review": score_movement_review,
        "quality_review_domains": validated_domains,
        "participant_review": participant_review,
        "continuity": continuity,
        "trace": trace,
    }


def score_validated_review(packet: dict[str, Any], validated_review: dict[str, Any]) -> dict[str, Any]:
    """Calculate app-owned Quality Adjustment and Final Candidate Score."""
    completion_score = (packet.get("model_interpretation") or {}).get("completion_score")
    if not isinstance(completion_score, (int, float)):
        return {
            "validation_status": "invalid",
            "validation_errors": ["model_interpretation.completion_score must be numeric"],
            "quality_adjustment": None,
            "final_candidate_score": None,
            "quality_assessment": {},
        }

    if not _has_complete_scoring_domains(validated_review):
        return {
            "validation_status": validated_review.get("validation_status", "partial"),
            "validation_errors": list(validated_review.get("validation_errors") or []),
            "quality_adjustment": None,
            "final_candidate_score": None,
            "quality_assessment": {},
            "input_hash": packet.get("input_hash") or stable_packet_hash(packet),
        }

    scoring_review = _review_with_supported_evidence(packet, validated_review)
    contributions = _quality_contributions(scoring_review.get("quality_review_domains") or {})
    final_candidate_score = clamp(
        float(completion_score) + contributions["quality_adjustment"],
        FINAL_SCORE_MIN,
        FINAL_SCORE_MAX,
    )

    return {
        "validation_status": validated_review.get("validation_status", "partial"),
        "validation_errors": list(validated_review.get("validation_errors") or []),
        "quality_adjustment": contributions["quality_adjustment"],
        "final_candidate_score": final_candidate_score,
        "quality_assessment": contributions,
        "input_hash": packet.get("input_hash") or stable_packet_hash(packet),
    }


def validate_and_score_review(packet: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    """Validate review JSON and return validated review plus deterministic score fields."""
    validated_review = validate_review_json(review)
    scoring = score_validated_review(packet, validated_review)
    return {
        "validated_review": validated_review,
        "scoring": scoring,
    }
