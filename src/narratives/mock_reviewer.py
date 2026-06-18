"""Deterministic mock reviewer for narrative prompt/schema development.

This module is a stand-in for a future LLM provider. It returns fixture-backed
JSON so packet building, validation, scoring, no-op behavior, and failure
handling can be exercised before real provider integration.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.narratives.contract_fixtures import get_contract_fixtures
from src.narratives.packet_builder import build_review_packet_from_fixture, stable_packet_hash
from src.narratives.prompt_builder import build_pass2_input
from src.narratives.scoring import validate_and_score_review
from src.narratives.trial_score_contract import GATED_PREMISE_SENSITIVE_FIELDS, validate_pass2_review

FAILURE_PROVIDER_ERROR = "provider_error"
FAILURE_MALFORMED_JSON = "malformed_json"


def _fixture_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for fixture in get_contract_fixtures():
        packet = build_review_packet_from_fixture(fixture)
        registry[packet["input_hash"]] = fixture
    return registry


def find_fixture_for_packet(packet: dict[str, Any]) -> dict[str, Any] | None:
    """Return the matching contract fixture for a deterministic packet hash."""
    input_hash = packet.get("input_hash")
    registry = _fixture_registry()
    if input_hash and str(input_hash) in registry:
        return registry[str(input_hash)]
    if packet.get("review_controls"):
        uncontrolled = {key: value for key, value in packet.items() if key not in {"input_hash", "review_controls"}}
        return registry.get(stable_packet_hash(uncontrolled))
    return None


def _default_evidence_fields(packet: dict[str, Any]) -> list[str]:
    changed_fields = [
        str(field)
        for field in ((packet.get("iteration_context") or {}).get("changed_fields") or [])
        if str(field).strip()
    ]
    if changed_fields:
        return changed_fields[:3]
    structured = packet.get("structured_features") or {}
    return [next(iter(structured), "phase_ml")]


def _operational_only(packet: dict[str, Any]) -> bool:
    fields = [str(field) for field in ((packet.get("iteration_context") or {}).get("changed_fields") or [])]
    return bool(fields) and all(field.startswith("operational_assumptions.") for field in fields)


def _synthesized_trial_score_pass1_review(packet: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal Trial Score Pass 1 schema for fixture-backed mock paths."""
    source = deepcopy(fixture.get("mock_review") or {})
    expected = fixture.get("expected_behavior") or {}
    model = packet.get("model_interpretation") or {}
    delta = model.get("score_delta")
    if not isinstance(delta, (int, float)):
        delta = 0
    expected_reality_check = expected.get("expected_reality_check_points", 0)
    if not isinstance(expected_reality_check, (int, float)):
        expected_reality_check = 0

    if expected_reality_check == 0:
        reality_effect = "neutral"
        reality_strength = "none"
        allocations = []
    elif _operational_only(packet) or float(delta) == 0:
        reality_effect = "reward_coherence" if expected_reality_check > 0 else "penalize_incoherence"
        reality_strength = "slight" if abs(float(expected_reality_check)) <= 1 else "moderate"
        allocations = [
            {
                "allocation_target_id": "execution_framework.operational_fit",
                "share": 1.0,
                "movement_label": "Reality Check: operational coherence",
                "rationale": "The mock review routes operational-only movement through the operational proportionality lens.",
                "incremental_check": "This checks coherence of the operational plan after Operational Fit rather than repeating the raw operational rating.",
            }
        ]
    elif float(delta) > 0:
        reality_effect = "offset_gain" if expected_reality_check < 0 else "reinforce_gain"
        reality_strength = "moderate" if expected_reality_check < 0 else "slight"
        allocations = [
            {
                "allocation_target_id": "scientific_challenge.protocol_architecture",
                "share": 1.0,
                "movement_label": "Reality Check: evidence robustness",
                "rationale": "The mock review checks whether a favorable score movement remains evidence-ready.",
                "incremental_check": "This is not counted by Operational Fit because it concerns evidence interpretability.",
            }
        ]
    else:
        reality_effect = "soften_decline" if expected_reality_check > 0 else "reinforce_decline"
        reality_strength = "moderate" if expected_reality_check > 0 else "slight"
        allocations = [
            {
                "allocation_target_id": "scientific_challenge.protocol_architecture",
                "share": 1.0,
                "movement_label": "Reality Check: rigor trade-off",
                "rationale": "The mock review checks whether a less favorable score movement reflects useful rigor or unresolved complexity.",
                "incremental_check": "This is a post-score interpretation rather than a rewrite of Completion Outlook.",
            }
        ]

    completion = source.get("completion_outlook_analysis") or source.get("completion_outlook_review") or {}
    if "risk_pattern_summary" not in completion:
        summary = completion.get("score_delta_summary") or "Completion Outlook changed according to the score inputs."
        completion = {
            "risk_pattern_summary": summary,
            "driver_summary": summary,
            "main_model_signals": completion.get("model_supported_drivers") or [],
            "interpretive_hypotheses": [
                {
                    "signal": "score-input movement",
                    "possible_pattern": "The edited scenario changed its historical completion-risk resemblance.",
                    "context_modifiers": [],
                    "boundary": "This is a model-pattern interpretation, not a clinical prediction.",
                }
            ],
            "movement_explanation": summary,
            "model_boundary_note": "Completion Outlook remains model-owned.",
        }

    participant = source.get("key_questions") or source.get("participant_review") or {}
    evidence_fields = _default_evidence_fields(packet)
    operational_changed = [field for field in evidence_fields if field.startswith("operational_assumptions.")]
    gated_changed = sorted(set(evidence_fields).intersection(GATED_PREMISE_SENSITIVE_FIELDS))
    operational_fit_rating = "neutral_or_unclear"
    operational_fit_materiality = "minor"
    operational_interaction = "mixed"
    if operational_changed:
        operational_fit_rating = (
            "slightly_improves_fit"
            if expected_reality_check > 0
            else "slightly_worsens_fit"
            if expected_reality_check < 0
            else "neutral_or_unclear"
        )
        operational_fit_materiality = "moderate" if len(operational_changed) >= 2 else "minor"
        operational_interaction = "unmodeled_support" if expected_reality_check > 0 else "under_supported"
    return {
        "review_metadata": {
            "review_mode": (source.get("review_metadata") or {}).get("review_mode", "first_visible_iteration"),
            "visible": bool((source.get("review_metadata") or {}).get("visible", True)),
        },
        "completion_outlook_analysis": completion,
        "strategy_shift_check": {
            "status": "partly_supported" if gated_changed else "not_applicable",
            "rationale": (
                "Mock Strategy Shift Check applied because gated premise-sensitive fields changed."
                if gated_changed
                else "No mock fixture premise shift is evaluated."
            ),
        },
        "operational_fit": {
            "enrollment_fit": {"rating": "neutral_or_unclear", "materiality": "minor", "rationale": "Mock field-level context."},
            "site_footprint_fit": {"rating": "neutral_or_unclear", "materiality": "minor", "rationale": "Mock field-level context."},
            "timeline_fit": {"rating": "neutral_or_unclear", "materiality": "minor", "rationale": "Mock field-level context."},
            "combined_operational_fit": {
                "rating": operational_fit_rating,
                "materiality": operational_fit_materiality,
                "interaction_with_completion_outlook": operational_interaction,
                "central_reason": "Fixture-backed Operational Fit generated for deterministic local checks.",
                "evidence_fields": operational_changed or evidence_fields,
            },
        },
        "reality_check": {
            "effect": reality_effect,
            "strength": reality_strength,
            "central_reason": "Fixture-backed Reality Check generated for deterministic local checks.",
            "evidence_fields": evidence_fields,
            "allocations": allocations,
        },
        "central_tension_candidate": {
            "summary": source.get("main_tension") or "Feasibility vs Evidence Strength.",
            "why_it_matters": "The scenario should be discussed as a total-score trade-off.",
            "supporting_evidence": evidence_fields,
        },
        "broader_strategic_question_candidate": {
            "question": (
                participant.get("strategic_development_question")
                or participant.get("strategic_field_question")
                or participant.get("clinical_operations_question")
                or "What broader development tension does this scenario expose?"
            ),
        },
        "continuity_update": {
            "active_tension": source.get("main_tension") or "Feasibility vs Evidence Strength.",
            "what_changed": "Fixture-backed mock review evaluated the latest scenario change.",
            "watch_next": "Stress-test whether the score movement remains defensible.",
        },
    }


def _synthesized_pass2_narrative(
    packet: dict[str, Any],
    pass1_review: dict[str, Any],
    scoring: dict[str, Any],
) -> dict[str, Any] | None:
    if scoring.get("trial_score") is None:
        return None

    pass2_input = build_pass2_input(packet, pass1_review, scoring)
    app_scores = pass2_input["app_calculated_scores"]
    analysis = pass2_input["pass1_analysis"]
    operational_assessment = analysis.get("operational_fit_assessment") or {}
    reality_assessment = analysis.get("reality_check_assessment") or {}
    tension = analysis.get("central_tension_candidate") or {}
    broader_question = analysis.get("broader_strategic_question_candidate") or {}
    continuity = analysis.get("continuity_update") or {}
    completion = analysis.get("completion_outlook_analysis") or {}

    operational_points = app_scores.get("operational_fit_points")
    reality_points = app_scores.get("reality_check_points")
    trial_score = app_scores.get("trial_score")
    pre_delta = app_scores.get("pre_reality_delta")
    operational_phrase = (
        f"Operational Fit contributes {operational_points:+g} points"
        if isinstance(operational_points, (int, float))
        else "Operational Fit is not scored for this hidden or unavailable review"
    )
    reality_phrase = (
        f"Reality Check contributes {reality_points:+g} points"
        if isinstance(reality_points, (int, float))
        else "Reality Check is not scored for this hidden or unavailable review"
    )
    movement_phrase = (
        f"The pre-Reality movement is {pre_delta:+g} points versus the reference score"
        if isinstance(pre_delta, (int, float))
        else "The reference movement is unavailable"
    )

    review_mode = (pass2_input.get("review_metadata") or {}).get("review_mode") or "first_visible_iteration"
    return {
        "review_metadata": {
            "review_mode": review_mode,
            "visible": True,
        },
        "trial_score_narrative": {
            "summary": (
                f"The Trial Score is {trial_score:g}. The current reading appears mixed because the model-pattern "
                "Completion Outlook, operational proportionality, and after-review realism check need to be read together."
            ),
            "movement_reading": (
                f"{movement_phrase}. {operational_phrase}, while {reality_phrase}; this should be read as an integrated "
                "scenario judgment rather than separate component essays."
            ),
            "score_interpretation": (
                completion.get("summary")
                or completion.get("risk_pattern_summary")
                or "Completion Outlook remains the protected model-pattern anchor, while the app-owned review layers interpret scenario coherence."
            ),
        },
        "pillar_reading": [
            {
                "pillar": "Execution Framework",
                "reading": (
                    operational_assessment.get("central_reason")
                    or "Operational Fit is interpreted as execution proportionality within the total score."
                ),
            },
            {
                "pillar": "Reality Check",
                "reading": (
                    reality_assessment.get("central_reason")
                    or "Reality Check reads whether the scenario movement appears coherent and realistic."
                ),
            },
        ],
        "central_tension": {
            "summary": tension.get("summary") or continuity.get("active_tension") or "Completion favorability versus scenario defensibility.",
            "why_it_matters": (
                tension.get("why_it_matters")
                or "This tension shapes how participants should defend the current Trial Score movement."
            ),
        },
        "broader_strategic_question": {
            "question": broader_question.get("question") or "What broader development trade-off does this scenario expose?",
        },
        "facilitator_questions": [
            {
                "question": "What evidence would make the current Trial Score movement defensible?",
                "why_it_matters": "It tests whether the score movement reflects a coherent development scenario rather than a shortcut.",
                "related_feature_families": ["evidence", "population", "operations"],
            }
        ],
    }


def review_packet_with_mock(
    packet: dict[str, Any],
    *,
    failure_mode: str | None = None,
) -> dict[str, Any]:
    """Return deterministic mock review behavior for a packet.

    `failure_mode` exists only for validation/storage/UI development. It lets
    callers exercise provider failure and malformed response paths without a
    provider dependency.
    """
    if failure_mode == FAILURE_PROVIDER_ERROR:
        return {
            "review_needed": True,
            "reuse_previous_review": False,
            "provider": "mock",
            "status": "provider_error",
            "failure_reason": "Simulated mock provider failure.",
            "review": None,
            "validated_review": None,
            "scoring": {
                "validation_status": "unavailable",
                "validation_errors": ["Simulated mock provider failure."],
                "operational_fit_points": None,
                "pre_reality_score": None,
                "pre_reality_delta": None,
                "reality_check_points": None,
                "reality_check_assessment": {},
                "trial_score": None,
                "input_hash": packet.get("input_hash"),
            },
        }

    fixture = find_fixture_for_packet(packet)
    if fixture is None:
        return {
            "review_needed": True,
            "reuse_previous_review": False,
            "provider": "mock",
            "status": "no_fixture_match",
            "failure_reason": "No mock fixture matched packet input_hash.",
            "review": None,
            "validated_review": None,
            "scoring": {
                "validation_status": "unavailable",
                "validation_errors": ["No mock fixture matched packet input_hash."],
                "operational_fit_points": None,
                "pre_reality_score": None,
                "pre_reality_delta": None,
                "reality_check_points": None,
                "reality_check_assessment": {},
                "trial_score": None,
                "input_hash": packet.get("input_hash"),
            },
        }

    expected = fixture["expected_behavior"]
    if expected.get("review_needed") is False:
        return {
            "review_needed": False,
            "reuse_previous_review": bool(expected.get("reuse_previous_review")),
            "provider": "mock",
            "status": "reused_previous_review",
            "fixture_id": fixture["fixture_id"],
            "failure_reason": None,
            "review": None,
            "validated_review": None,
            "scoring": {
                "validation_status": "reused",
                "validation_errors": [],
                "operational_fit_points": expected.get("expected_operational_fit_points"),
                "pre_reality_score": expected.get("expected_pre_reality_score"),
                "pre_reality_delta": expected.get("expected_pre_reality_delta"),
                "reality_check_points": expected.get("expected_reality_check_points"),
                "reality_check_assessment": {},
                "trial_score": expected.get("expected_trial_score"),
                "input_hash": packet.get("input_hash"),
            },
        }

    review = deepcopy(fixture["mock_review"])
    if isinstance(review, dict) and "reality_check" not in review:
        review = _synthesized_trial_score_pass1_review(packet, fixture)
    if failure_mode == FAILURE_MALFORMED_JSON:
        review = {"reality_check": "malformed"}

    scored = validate_and_score_review(packet, review)
    pass2_review = None
    validated_pass2 = {"validation_status": "valid", "validation_errors": []}
    if failure_mode != FAILURE_MALFORMED_JSON and scored["scoring"].get("validation_status") == "valid":
        pass2_review = _synthesized_pass2_narrative(packet, scored["validated_review"], scored["scoring"])
        if pass2_review is not None:
            validated_pass2 = validate_pass2_review(pass2_review)
    status = "malformed_response" if failure_mode == FAILURE_MALFORMED_JSON else "reviewed"
    if validated_pass2.get("validation_status") != "valid":
        status = "malformed_response"
    return {
        "review_needed": True,
        "reuse_previous_review": False,
        "provider": "mock",
        "status": status,
        "fixture_id": fixture["fixture_id"],
        "failure_reason": None if status == "reviewed" else "Mock Pass 2 narrative did not satisfy the Trial Score contract.",
        "review": review,
        "validated_review": scored["validated_review"],
        "participant_narrative": pass2_review,
        "validated_participant_narrative": validated_pass2,
        "scoring": scored["scoring"],
    }
