#!/usr/bin/env python
"""Validate Strategic Review scoring, storyline, and visual data flow."""

from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.utils.plot import plot_treemap  # noqa: E402
from frontend.utils.scenario_review_plot_data import design_subcategory_impacts  # noqa: E402
from src.narratives.packet_builder import build_review_packet  # noqa: E402
from src.narratives.review_store import store_review_trace  # noqa: E402
from src.narratives.scoring import validate_and_score_review  # noqa: E402
from src.narratives.storyline import build_storyline_state  # noqa: E402


def _packet(*, completion: float, previous: float, changed_fields: list[str]) -> dict:
    return {
        "input_hash": f"flow-{completion}-{previous}-{'-'.join(changed_fields)}",
        "structured_features": {
            "phase_ml": "PHASE3",
            "endpoint_rigor_ml": "HARD_CLINICAL",
            "allocation_ml": "RANDOMIZED",
        },
        "operational_assumptions": {
            "planned_enrollment": {"value": 900, "source": "planned_value"},
            "planned_sites": {"value": 90, "source": "planned_value"},
            "planned_duration_months": {"value": 48.0, "source": "planned_value"},
        },
        "model_interpretation": {
            "completion_score": completion,
            "previous_completion_score": previous,
            "score_delta": completion - previous,
            "pillar_impacts": [{"Pillar": "Execution Framework", "Impact": 2.0}],
            "pillar_deltas": {"Execution Framework": completion - previous},
            "xgboost_impact_changes": [],
        },
        "iteration_context": {
            "baseline_snapshot_id": "baseline",
            "previous_snapshot_id": "previous",
            "current_snapshot_id": "current",
            "iteration_number": 2,
            "changed_fields": changed_fields,
            "field_changes": [{"field": field} for field in changed_fields],
        },
    }


def _review(
    *,
    effect_label: str,
    tension_status: str = "not_applicable",
    operational_materiality: str = "minor",
    evidence_fields: list[str] | None = None,
    continuity: dict | None = None,
) -> dict:
    evidence_fields = evidence_fields or ["phase_ml"]
    return {
        "review_metadata": {"review_mode": "later_visible_iteration", "visible": True},
        "completion_outlook_analysis": {
            "risk_pattern_summary": "Completion Outlook movement is explained from score inputs.",
            "driver_summary": "The latest changed field moved the score pattern.",
            "main_model_signals": evidence_fields,
            "interpretive_hypotheses": [
                {
                    "signal": evidence_fields[0],
                    "possible_pattern": "Score-pattern movement.",
                    "context_modifiers": [],
                    "boundary": "Pattern evidence only.",
                }
            ],
            "movement_explanation": "Movement follows the changed score input.",
            "model_boundary_note": "Completion Outlook remains model-owned.",
        },
        "strategic_review": {
            "effect_label": effect_label,
            "tension_status": tension_status,
            "operational_materiality": operational_materiality,
            "evidence_fields": evidence_fields,
            "move_classification": ["oversimplification"],
            "current_tension": "Feasibility vs Evidence Strength",
            "carryover_check": "Prior evidence tension remains relevant." if tension_status != "not_applicable" else "",
            "tradeoff_resolution": "The latest move changes the strategic tradeoff.",
            "rationale": "The latest move changes whether the Completion Outlook movement should be trusted.",
            "next_consideration": "Test whether evidence credibility remains proportionate.",
        },
        "strategic_review_analysis": {
            "summary": "The Trial Score review integrates Completion Outlook movement and Strategic Review interpretation.",
            "review_rationale": "The app should calculate the numeric modifier from categorical labels.",
            "supporting_evidence": evidence_fields,
            "limiting_evidence": [],
        },
        "key_questions": {
            "medical_clinical_development_question": "What evidence standard would make this tradeoff defensible?",
            "strategic_development_question": "Broadly speaking, how should completion favorability and evidence credibility be balanced?",
        },
        "scenario_consistency_note": {
            "has_clear_mismatch": False,
            "message": "",
            "fields_in_tension": [],
        },
        "continuity": continuity
        or {
            "prior_concerns_resolved": [],
            "prior_concerns_worsened": [],
            "prior_concerns_unchanged": [],
            "new_concerns": [],
            "storyline_update": "Strategic storyline updated.",
        },
        "trace": {
            "main_features_considered": evidence_fields,
            "main_completion_drivers_considered": evidence_fields,
            "main_strategic_review_signals_considered": evidence_fields,
            "operational_statuses_considered": [],
            "reference_pack_ids_used": [],
            "therapeutic_area_pack_used": "",
            "compared_against": "previous_visible_iteration",
            "should_repeat_prior_warning": False,
        },
    }


def _assert_equal(errors: list[str], label: str, actual, expected) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def main() -> int:
    errors: list[str] = []

    packet = _packet(completion=70, previous=60, changed_fields=["phase_ml"])
    support_review = _review(effect_label="supports_score_gain")
    support_result = validate_and_score_review(packet, support_review)
    _assert_equal(errors, "positive support Strategic Review", support_result["scoring"].get("strategic_review"), 1)
    _assert_equal(errors, "positive support Trial Score", support_result["scoring"].get("trial_score"), 71)

    continuity = {
        "prior_concerns_resolved": ["reduced operational burden"],
        "prior_concerns_worsened": [],
        "prior_concerns_unchanged": ["endpoint credibility"],
        "new_concerns": ["population focus"],
        "storyline_update": "Endpoint credibility remains a carryover tension.",
    }
    carryover_review = _review(
        effect_label="partly_offsets_score_gain",
        tension_status="partially_active",
        continuity=continuity,
    )
    carryover_result = validate_and_score_review(
        packet,
        carryover_review,
    )
    _assert_equal(errors, "carryover offset Strategic Review", carryover_result["scoring"].get("strategic_review"), -2.6)
    storyline = build_storyline_state(carryover_result["validated_review"])
    _assert_equal(errors, "storyline active tension", storyline.get("active_tension"), "Feasibility vs Evidence Strength")
    _assert_equal(errors, "storyline protected gains", storyline.get("protected_gains"), ["reduced operational burden"])
    _assert_equal(errors, "storyline active carryover", storyline.get("active_carryover"), ["endpoint credibility"])

    negative_packet = _packet(completion=60, previous=68, changed_fields=["endpoint_rigor_ml"])
    result = validate_and_score_review(
        negative_packet,
        _review(effect_label="softens_score_decline", evidence_fields=["endpoint_rigor_ml"]),
    )
    _assert_equal(errors, "negative softening Strategic Review", result["scoring"].get("strategic_review"), 0.8)
    _assert_equal(errors, "negative softening Trial Score", result["scoring"].get("trial_score"), 60.8)

    result = validate_and_score_review(
        negative_packet,
        _review(effect_label="supports_tradeoff_balance", evidence_fields=["endpoint_rigor_ml"]),
    )
    normalized_review = result["validated_review"].get("strategic_review") or {}
    _assert_equal(
        errors,
        "negative flat-support label normalization",
        normalized_review.get("effect_label"),
        "softens_score_decline",
    )
    _assert_equal(errors, "normalized negative flat-support Strategic Review", result["scoring"].get("strategic_review"), 0.8)

    result = validate_and_score_review(
        negative_packet,
        _review(effect_label="worsens_active_tension", evidence_fields=["endpoint_rigor_ml"]),
    )
    normalized_review = result["validated_review"].get("strategic_review") or {}
    _assert_equal(
        errors,
        "negative flat-worsening label normalization",
        normalized_review.get("effect_label"),
        "reinforces_score_decline",
    )
    _assert_equal(errors, "normalized negative flat-worsening Strategic Review", result["scoring"].get("strategic_review"), -2.4)

    result = validate_and_score_review(
        packet,
        _review(effect_label="worsens_active_tension"),
    )
    normalized_review = result["validated_review"].get("strategic_review") or {}
    _assert_equal(
        errors,
        "positive flat-worsening label normalization",
        normalized_review.get("effect_label"),
        "partly_offsets_score_gain",
    )
    _assert_equal(errors, "normalized positive flat-worsening Strategic Review", result["scoring"].get("strategic_review"), -2)

    operational_packet = _packet(
        completion=70,
        previous=70,
        changed_fields=["operational_assumptions.planned_enrollment"],
    )
    result = validate_and_score_review(
        operational_packet,
        _review(
            effect_label="strongly_worsens_active_tension",
            operational_materiality="major",
            evidence_fields=["operational_assumptions.planned_enrollment"],
        ),
    )
    _assert_equal(errors, "operational-only Strategic Review", result["scoring"].get("strategic_review"), -4)
    _assert_equal(errors, "operational-only Trial Score", result["scoring"].get("trial_score"), 66)

    bad_result = validate_and_score_review(
        packet,
        _review(effect_label="softens_score_decline"),
    )
    if bad_result["scoring"].get("strategic_review") is not None:
        errors.append("incompatible effect label should suppress Strategic Review")
    if not any("incompatible with positive Completion Outlook movement" in error for error in bad_result["scoring"].get("validation_errors") or []):
        errors.append("incompatible effect label should report movement-direction validation error")

    store_state = {}
    stored = store_review_trace(
        store_state,
        packet=packet,
        review_result={
            "review_needed": True,
            "reuse_previous_review": False,
            "provider": "mock",
            "model_name": "strategic-flow-test",
            "provider_metadata": {"deterministic": True},
            "status": "reviewed",
            "failure_reason": None,
            "review": carryover_review,
            "validated_review": carryover_result["validated_review"],
            "scoring": carryover_result["scoring"],
        },
        session_id="strategic-flow",
    )
    stored_state = stored.get("storyline_state") or {}
    if stored_state.get("active_tension") != "Feasibility vs Evidence Strength":
        errors.append("stored trace should preserve app-owned Strategic Review storyline state")

    later_packet = build_review_packet(
        current_snapshot={
            "snapshot_id": "current",
            "structured_features": packet["structured_features"],
            "model_interpretation": packet["model_interpretation"],
            "changed_fields": ["endpoint_rigor_ml"],
        },
        previous_snapshot={"snapshot_id": "previous", "score": 70},
        baseline_snapshot={"snapshot_id": "baseline"},
        previous_review_trace=stored,
    )
    continuity_packet = later_packet.get("iteration_context", {}).get("strategic_review_continuity") or {}
    _assert_equal(errors, "packet active tension continuity", continuity_packet.get("active_tension"), "Feasibility vs Evidence Strength")
    _assert_equal(errors, "packet last effect continuity", continuity_packet.get("last_effect_label"), "partly_offsets_score_gain")
    _assert_equal(errors, "packet protected gains continuity", continuity_packet.get("protected_gains"), ["reduced operational burden"])

    plot_rows = design_subcategory_impacts(stored)
    fig = plot_treemap(plot_rows, [{"Pillar": "Strategic Review", "Impact": stored["strategic_review"]}])
    labels = [str(label) for label in fig.data[0].labels]
    if any("pts" in label and "Current Tension" in label for label in labels):
        errors.append("Strategic Review qualitative treemap leaf should not render point values")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated Strategic Review scoring, storyline continuity, unavailable-state, and treemap flow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
