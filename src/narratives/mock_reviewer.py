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
from src.narratives.scoring import validate_and_score_review

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


def _synthesized_strategic_review(packet: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal Strategic Review schema for legacy fixture-backed mock paths."""
    source = deepcopy(fixture.get("mock_review") or {})
    expected = fixture.get("expected_behavior") or {}
    model = packet.get("model_interpretation") or {}
    delta = model.get("score_delta")
    if not isinstance(delta, (int, float)):
        delta = 0
    expected_modifier = expected.get("expected_design_confidence", 0)
    if not isinstance(expected_modifier, (int, float)):
        expected_modifier = 0

    if _operational_only(packet) or float(delta) == 0:
        if expected_modifier < 0:
            effect_label = "strongly_worsens_active_tension"
        elif expected_modifier > 0:
            effect_label = "supports_tradeoff_balance"
        else:
            effect_label = "neutral"
    elif float(delta) > 0:
        if expected_modifier < 0:
            effect_label = "partly_offsets_score_gain"
        elif expected_modifier > 0:
            effect_label = "supports_score_gain"
        else:
            effect_label = "neutral"
    else:
        if expected_modifier > 0:
            effect_label = "softens_score_decline"
        elif expected_modifier < 0:
            effect_label = "reinforces_score_decline"
        else:
            effect_label = "neutral"

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
    strategic_review = {
        "effect_label": effect_label,
        "tension_status": "not_applicable",
        "operational_materiality": "minor",
        "evidence_fields": _default_evidence_fields(packet),
        "move_classification": ["balanced_improvement"] if expected_modifier >= 0 else ["strategic_mismatch"],
        "current_tension": source.get("main_tension") or "Feasibility vs Evidence Strength.",
        "carryover_check": "",
        "tradeoff_resolution": "The latest move is interpreted through the current strategic tradeoff.",
        "rationale": "Fixture-backed Strategic Review generated for deterministic local checks.",
        "next_consideration": "Stress-test whether the score movement remains strategically defensible.",
    }
    return {
        "review_metadata": {
            "review_mode": (source.get("review_metadata") or {}).get("review_mode", "first_visible_iteration"),
            "visible": bool((source.get("review_metadata") or {}).get("visible", True)),
        },
        "completion_outlook_analysis": completion,
        "strategic_review": strategic_review,
        "strategic_review_analysis": {
            "summary": "The Trial Score review combines Completion Outlook movement with a Strategic Review interpretation.",
            "overall_score_explanation": "The Completion Outlook movement is interpreted first, then moderated by Strategic Review when the score pattern is strategically uneven.",
            "pillar_readout": [
                {
                    "label": "Score Pattern",
                    "interpretation": "Changed trial attributes and category movement are interpreted together rather than as isolated UI sections.",
                },
                {
                    "label": "Strategic Review",
                    "interpretation": strategic_review["rationale"],
                },
            ],
            "strategic_review_bullet": strategic_review["rationale"],
            "tension_question": strategic_review.get("next_consideration", ""),
            "broader_strategic_question": (
                participant.get("strategic_development_question")
                or participant.get("strategic_field_question")
                or "What broader development tension does this scenario expose?"
            ),
            "review_rationale": strategic_review["rationale"],
            "supporting_evidence": strategic_review["evidence_fields"],
            "limiting_evidence": [],
        },
        "key_questions": {
            "medical_clinical_development_question": (
                participant.get("medical_clinical_development_question")
                or participant.get("medical_development_question")
                or "What evidence standard would make this scenario defensible?"
            ),
            "strategic_development_question": (
                participant.get("strategic_development_question")
                or participant.get("strategic_field_question")
                or participant.get("clinical_operations_question")
                or "How should the development path balance completion resemblance and evidence value?"
            ),
        },
        "scenario_consistency_note": source.get("scenario_consistency_note") or {
            "has_clear_mismatch": False,
            "message": "",
            "fields_in_tension": [],
        },
        "continuity": source.get("continuity") or {
            "prior_concerns_resolved": [],
            "prior_concerns_worsened": [],
            "prior_concerns_unchanged": [],
            "new_concerns": [],
            "storyline_update": "Fixture-backed Strategic Review storyline update.",
        },
        "trace": {
            "main_features_considered": (source.get("trace") or {}).get("main_features_considered") or [],
            "main_completion_drivers_considered": (source.get("trace") or {}).get("main_completion_drivers_considered") or [],
            "main_strategic_review_signals_considered": _default_evidence_fields(packet),
            "operational_statuses_considered": (source.get("trace") or {}).get("operational_statuses_considered") or [],
            "reference_pack_ids_used": (source.get("trace") or {}).get("reference_pack_ids_used") or [],
            "therapeutic_area_pack_used": (source.get("trace") or {}).get("therapeutic_area_pack_used") or "",
            "compared_against": (source.get("trace") or {}).get("compared_against") or "previous_visible_iteration",
            "should_repeat_prior_warning": bool((source.get("trace") or {}).get("should_repeat_prior_warning", False)),
        },
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
                "design_confidence": None,
                "total_scenario_score": None,
                "design_confidence_assessment": {},
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
                "design_confidence": None,
                "total_scenario_score": None,
                "design_confidence_assessment": {},
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
                "design_confidence": expected.get("expected_design_confidence"),
                "total_scenario_score": expected.get("expected_total_scenario_score"),
                "design_confidence_assessment": {},
                "input_hash": packet.get("input_hash"),
            },
        }

    review = deepcopy(fixture["mock_review"])
    if isinstance(review, dict) and "strategic_review" not in review:
        review = _synthesized_strategic_review(packet, fixture)
    if failure_mode == FAILURE_MALFORMED_JSON:
        review = {"design_confidence_subcategories": "malformed"}

    scored = validate_and_score_review(packet, review)
    status = "malformed_response" if failure_mode == FAILURE_MALFORMED_JSON else "reviewed"
    return {
        "review_needed": True,
        "reuse_previous_review": False,
        "provider": "mock",
        "status": status,
        "fixture_id": fixture["fixture_id"],
        "failure_reason": None,
        "review": review,
        "validated_review": scored["validated_review"],
        "scoring": scored["scoring"],
    }
