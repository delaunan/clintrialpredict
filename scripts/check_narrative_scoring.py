#!/usr/bin/env python
"""Validate Strategic Review scoring and provider-output guardrails."""

from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.scoring import validate_and_score_review  # noqa: E402


def _packet(*, completion_score: float = 70, previous_score: float = 60, changed_fields: list[str] | None = None) -> dict:
    return {
        "input_hash": "strategic-review-scoring-check",
        "structured_features": {
            "phase_ml": "Phase 3",
            "endpoint_rigor_ml": "Clinical outcome",
        },
        "operational_assumptions": {
            "planned_enrollment": {"current_value": 1200},
            "planned_sites": {"current_value": 90},
            "planned_duration_months": {"current_value": 36},
        },
        "model_interpretation": {
            "completion_score": completion_score,
            "previous_completion_score": previous_score,
            "score_delta": completion_score - previous_score,
        },
        "iteration_context": {
            "changed_fields": changed_fields or ["phase_ml"],
            "field_changes": [],
        },
    }


def _review(effect_label: str = "supports_score_gain", *, tension_status: str = "not_applicable") -> dict:
    return {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "completion_outlook_analysis": {
            "risk_pattern_summary": "Completion Outlook moved with the selected score inputs.",
            "driver_summary": "The edited score inputs changed the historical risk pattern.",
            "main_model_signals": ["phase_ml"],
            "interpretive_hypotheses": [
                {
                    "signal": "phase_ml",
                    "possible_pattern": "The scenario resembles a different completion-risk pattern.",
                    "context_modifiers": [],
                    "boundary": "This is a score-pattern interpretation, not a clinical claim.",
                }
            ],
            "movement_explanation": "The score moved because score-input fields changed.",
            "model_boundary_note": "Completion Outlook remains model-owned.",
        },
        "strategic_review": {
            "effect_label": effect_label,
            "tension_status": tension_status,
            "operational_materiality": "minor",
            "evidence_fields": ["phase_ml"],
            "move_classification": ["balanced_improvement"],
            "current_tension": "Feasibility vs Evidence Strength.",
            "carryover_check": "",
            "tradeoff_resolution": "The latest move is strategically coherent with the score movement.",
            "rationale": "The move has packet-supported strategic rationale.",
            "next_consideration": "Stress-test whether the evidence standard remains defensible.",
        },
        "strategic_review_analysis": {
            "summary": "The Trial Score combines Completion Outlook movement with a Strategic Review interpretation.",
            "review_rationale": "The Strategic Review classification is packet-supported.",
            "supporting_evidence": ["phase_ml"],
            "limiting_evidence": [],
        },
        "key_questions": {
            "medical_clinical_development_question": "What evidence standard would make this scenario defensible?",
            "strategic_development_question": "How should the development path balance completion resemblance and evidence value?",
        },
        "scenario_consistency_note": {
            "has_clear_mismatch": False,
            "message": "",
            "fields_in_tension": [],
        },
        "continuity": {
            "prior_concerns_resolved": [],
            "prior_concerns_worsened": [],
            "prior_concerns_unchanged": [],
            "new_concerns": [],
            "storyline_update": "The latest move established the active strategic tension.",
        },
        "trace": {
            "main_features_considered": ["phase_ml"],
            "main_completion_drivers_considered": ["phase_ml"],
            "main_strategic_review_signals_considered": ["phase_ml"],
            "operational_statuses_considered": [],
            "reference_pack_ids_used": [],
            "therapeutic_area_pack_used": "",
            "compared_against": "previous_visible_iteration",
            "should_repeat_prior_warning": False,
        },
    }


def _check_positive_support(errors: list[str]) -> None:
    result = validate_and_score_review(_packet(), _review("supports_score_gain"))
    scoring = result["scoring"]
    if scoring.get("validation_status") != "valid":
        errors.append(f"positive support should validate, got {scoring.get('validation_status')}")
    if scoring.get("strategic_review") != 1:
        errors.append("positive support should add +25% of a +10 delta budget")
    if scoring.get("trial_score") != 71:
        errors.append("Trial Score should equal Completion Outlook plus Strategic Review")


def _check_positive_offset_and_carryover(errors: list[str]) -> None:
    review = _review("partly_offsets_score_gain", tension_status="partially_active")
    review["strategic_review"]["carryover_check"] = "The prior evidence tension remains partly active."
    result = validate_and_score_review(_packet(), review)
    assessment = result["scoring"].get("strategic_review_assessment") or {}
    if result["scoring"].get("strategic_review") != -2.6:
        errors.append("positive offset plus partially-active carryover should produce -2.6")
    if assessment.get("combined_review_factor") != -0.7:
        errors.append("Strategic Review assessment should expose the combined review factor")


def _check_negative_softening(errors: list[str]) -> None:
    packet = _packet(completion_score=60, previous_score=68)
    result = validate_and_score_review(packet, _review("softens_score_decline"))
    if result["scoring"].get("strategic_review") != 0.8:
        errors.append("negative softening should add +25% of the decline budget")
    if result["scoring"].get("trial_score") != 60.8:
        errors.append("negative softening Trial Score should use the current Completion Outlook")


def _check_flat_labels_normalize_for_score_movement(errors: list[str]) -> None:
    negative_packet = _packet(completion_score=60, previous_score=68)
    result = validate_and_score_review(negative_packet, _review("supports_tradeoff_balance"))
    strategic_review = result["validated_review"].get("strategic_review") or {}
    if strategic_review.get("effect_label") != "softens_score_decline":
        errors.append("negative movement should normalize flat support label to decline-softening label")
    if result["scoring"].get("strategic_review") != 0.8:
        errors.append("normalized negative support label should score as decline softening")

    result = validate_and_score_review(negative_packet, _review("worsens_active_tension"))
    strategic_review = result["validated_review"].get("strategic_review") or {}
    if strategic_review.get("effect_label") != "reinforces_score_decline":
        errors.append("negative movement should normalize flat worsening label to decline-reinforcing label")
    if result["scoring"].get("strategic_review") != -2.4:
        errors.append("normalized negative worsening label should score as decline reinforcement")

    positive_packet = _packet(completion_score=70, previous_score=60)
    result = validate_and_score_review(positive_packet, _review("worsens_active_tension"))
    strategic_review = result["validated_review"].get("strategic_review") or {}
    if strategic_review.get("effect_label") != "partly_offsets_score_gain":
        errors.append("positive movement should normalize flat worsening label to gain-offset label")
    if result["scoring"].get("strategic_review") != -2:
        errors.append("normalized positive worsening label should score as gain offset")


def _check_operational_only(errors: list[str]) -> None:
    packet = _packet(
        completion_score=70,
        previous_score=70,
        changed_fields=["operational_assumptions.planned_enrollment"],
    )
    review = _review("strongly_worsens_active_tension")
    review["strategic_review"]["operational_materiality"] = "major"
    review["strategic_review"]["evidence_fields"] = ["operational_assumptions.planned_enrollment"]
    result = validate_and_score_review(packet, review)
    if result["scoring"].get("strategic_review") != -4:
        errors.append("operational-only major strongly-worsening review should produce -4")
    if result["scoring"].get("trial_score") != 66:
        errors.append("operational-only Trial Score should use the operational materiality budget")


def _check_invalid_or_legacy_outputs(errors: list[str]) -> None:
    packet = _packet()
    legacy_review = _review()
    legacy_review.pop("strategic_review")
    legacy_review.pop("strategic_review_analysis")
    legacy_review["design_confidence_subcategories"] = {}
    result = validate_and_score_review(packet, legacy_review)
    if result["scoring"].get("strategic_review") is not None:
        errors.append("legacy Design Confidence-only review should not produce Strategic Review")
    if not any("strategic_review" in error for error in result["scoring"].get("validation_errors") or []):
        errors.append("legacy Design Confidence-only review should report missing strategic_review")

    app_score_review = _review()
    app_score_review["trial_score"] = 99
    app_score_review["strategic_review_points"] = 99
    result = validate_and_score_review(packet, app_score_review)
    if result["scoring"].get("validation_status") == "valid":
        errors.append("provider-returned app-owned score fields should prevent a valid provider result")
    if result["scoring"].get("strategic_review") is None:
        errors.append("provider-returned app-owned score fields should be ignored without suppressing Strategic Review")
    if result["scoring"].get("trial_score") is None:
        errors.append("provider-returned app-owned score fields should be ignored without suppressing Trial Score")
    if not any("application-owned" in error for error in result["scoring"].get("validation_errors") or []):
        errors.append("app-owned score fields should be reported")

    incompatible = _review("softens_score_decline")
    result = validate_and_score_review(packet, incompatible)
    if result["scoring"].get("strategic_review") is not None:
        errors.append("movement-incompatible effect label should suppress Strategic Review")
    if not any("incompatible with positive Completion Outlook movement" in error for error in result["scoring"].get("validation_errors") or []):
        errors.append("movement-incompatible effect label should report a validation error")

    malformed_continuity = _review()
    malformed_continuity["continuity"]["prior_concerns_resolved"] = "not-an-array"
    malformed_continuity["continuity"]["storyline_update"] = 123
    result = validate_and_score_review(packet, malformed_continuity)
    if result["scoring"].get("strategic_review") is None:
        errors.append("malformed continuity should warn without suppressing Strategic Review")
    if not any("continuity." in error for error in result["scoring"].get("validation_errors") or []):
        errors.append("malformed continuity should report validation warnings")


def _check_hidden_baseline_suppresses_scores(errors: list[str]) -> None:
    packet = _packet(changed_fields=[])
    packet["iteration_context"] = {
        "baseline_snapshot_id": "baseline",
        "previous_snapshot_id": None,
        "current_snapshot_id": "baseline",
        "changed_fields": [],
    }
    review = _review("neutral")
    review["review_metadata"] = {"review_mode": "hidden_baseline", "visible": False}
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    if scoring.get("validation_status") != "valid":
        errors.append("hidden baseline should validate qualitative Strategic Review context")
    if scoring.get("strategic_review") is not None:
        errors.append("hidden baseline should not calculate Strategic Review")
    if scoring.get("trial_score") is not None:
        errors.append("hidden baseline should not calculate Trial Score")


def main() -> int:
    errors: list[str] = []
    _check_positive_support(errors)
    _check_positive_offset_and_carryover(errors)
    _check_negative_softening(errors)
    _check_flat_labels_normalize_for_score_movement(errors)
    _check_operational_only(errors)
    _check_invalid_or_legacy_outputs(errors)
    _check_hidden_baseline_suppresses_scores(errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated Strategic Review scoring and provider-output guardrails.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
