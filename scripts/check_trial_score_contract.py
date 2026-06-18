#!/usr/bin/env python
"""Validate the V1 Trial Score contract and deterministic scoring."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.trial_score_contract import (  # noqa: E402
    score_pass1_review,
    validate_pass2_review,
)
from frontend.utils.scenario_review_plot_data import design_subcategory_impacts  # noqa: E402


def _packet(
    *,
    completion_score: float = 70,
    previous_trial_score: float = 64,
    changed_fields: list[str] | None = None,
    field_changes: list[dict] | None = None,
    previous_operational_fit_points: float | None = None,
) -> dict:
    iteration_context = {
        "changed_fields": changed_fields
        or [
            "operational_assumptions.planned_enrollment",
            "operational_assumptions.planned_sites",
        ],
        "field_changes": field_changes or [],
    }
    if previous_operational_fit_points is not None:
        iteration_context["trial_score_continuity"] = {
            "previous_operational_fit_points": previous_operational_fit_points,
        }
    return {
        "input_hash": "trial-score-contract-check",
        "structured_features": {
            "phase_ml": "Phase 3",
            "endpoint_rigor_ml": "Clinical outcome",
        },
        "text_context": {
            "summary_ui": "Randomized pivotal study in the target population.",
        },
        "operational_assumptions": {
            "planned_enrollment": {
                "value": 700,
                "baseline_value": 480,
                "baseline_value_source": "cohort_p50_estimate",
            },
            "planned_sites": {
                "value": 40,
                "baseline_value": 20,
                "baseline_value_source": "cohort_p50_estimate",
            },
            "planned_duration_months": {
                "value": 38,
                "baseline_value": 36,
                "baseline_value_source": "registered_planned",
            },
        },
        "model_interpretation": {
            "completion_score": completion_score,
            "previous_completion_score": previous_trial_score,
            "previous_trial_score": previous_trial_score,
            "baseline_completion_score": 65,
            "score_delta": completion_score - previous_trial_score,
        },
        "iteration_context": iteration_context,
    }


def _pass1_review(
    *,
    fit_rating: str = "moderately_improves_fit",
    fit_materiality: str = "major",
    reality_effect: str = "offset_gain",
    reality_strength: str = "moderate",
) -> dict:
    return {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "completion_outlook_analysis": {
            "summary": "Completion Outlook improved on model-visible score inputs.",
            "main_model_signals": ["phase_ml"],
            "model_boundary_note": "XGBoost Completion Outlook remains model-owned.",
        },
        "strategy_shift_check": {
            "status": "not_applicable",
            "rationale": "No gated premise-sensitive field changed.",
        },
        "operational_fit": {
            "enrollment_fit": {
                "rating": "slightly_worsens_fit",
                "materiality": "moderate",
                "baseline_value_source": "cohort_p50_estimate",
                "rationale": "Enrollment became more ambitious.",
            },
            "site_footprint_fit": {
                "rating": "moderately_improves_fit",
                "materiality": "major",
                "baseline_value_source": "cohort_p50_estimate",
                "rationale": "Site footprint reduces patient-per-site burden.",
            },
            "timeline_fit": {
                "rating": "neutral_or_unclear",
                "materiality": "minor",
                "baseline_value_source": "registered_planned",
                "rationale": "Duration changed only slightly.",
            },
            "combined_operational_fit": {
                "rating": fit_rating,
                "materiality": fit_materiality,
                "interaction_with_completion_outlook": "unmodeled_support",
                "central_reason": "The site-footprint change improves executability despite higher enrollment.",
                "evidence_fields": [
                    "operational_assumptions.planned_enrollment",
                    "operational_assumptions.planned_sites",
                ],
            },
        },
        "reality_check": {
            "effect": reality_effect,
            "strength": reality_strength,
            "central_reason": "The gain may be partly shortcut-driven because evidence support did not improve.",
            "evidence_fields": ["phase_ml", "operational_assumptions.planned_sites"],
            "allocations": [
                {
                    "allocation_target_id": "scientific_challenge.protocol_architecture",
                    "share": 0.6,
                    "movement_label": "Reality Check: endpoint robustness",
                    "rationale": "The operational gain does not by itself strengthen endpoint interpretability.",
                    "incremental_check": "This is not already counted by Operational Fit because it concerns evidence interpretability.",
                },
                {
                    "allocation_target_id": "execution_framework.operational_fit",
                    "share": 0.4,
                    "movement_label": "Reality Check: operational support",
                    "rationale": "The site expansion helps but leaves enrollment ambition stretched.",
                    "incremental_check": "This checks residual execution realism after Operational Fit, not the raw site-count credit.",
                },
            ],
        },
        "central_tension_candidate": {
            "summary": "Execution support versus evidence interpretability.",
            "why_it_matters": "The scenario may be more executable without being more decision-ready.",
            "supporting_evidence": ["operational_assumptions.planned_sites", "phase_ml"],
        },
        "broader_strategic_question_candidate": {
            "question": "When should operational support compensate for a scenario whose evidence standard has not strengthened?",
        },
        "continuity_update": {
            "active_tension": "Execution support versus evidence interpretability.",
            "what_changed": "Operational assumptions changed.",
            "watch_next": "Whether evidence support catches up with operational scale.",
        },
    }


def _check_operational_fit_mapping(errors: list[str]) -> None:
    result = score_pass1_review(_packet(), _pass1_review())
    if result.get("validation_status") != "valid":
        errors.append(f"expected valid Trial Score scoring, got {result.get('validation_errors')}")
    if result.get("operational_fit_points") != 2.4:
        errors.append("major/moderately_improves_fit should produce +2.4 Operational Fit")
    if result.get("pre_reality_score") != 72.4:
        errors.append("pre-Reality score should equal XGBoost Completion Outlook plus Operational Fit")
    if result.get("reality_check_points") != -3.4:
        errors.append("moderate offset_gain should subtract 40% of the +8.4 pre-Reality delta")
    if result.get("trial_score") != 69:
        errors.append("Trial Score should equal pre-Reality score plus Reality Check")


def _check_positive_reinforcement_cap(errors: list[str]) -> None:
    result = score_pass1_review(
        _packet(),
        _pass1_review(reality_effect="reinforce_gain", reality_strength="strong"),
    )
    if result.get("reality_check_points") != 1.7:
        errors.append("positive reinforce_gain should be capped at slight in V1")
    if not any("capped at slight" in note for note in result.get("validation_notes") or []):
        errors.append("positive reinforcement cap should be reported as a validation note")


def _check_incompatible_effect_downgrade(errors: list[str]) -> None:
    result = score_pass1_review(
        _packet(completion_score=60, previous_trial_score=70),
        _pass1_review(
            fit_rating="neutral_or_unclear",
            fit_materiality="minor",
            reality_effect="offset_gain",
            reality_strength="strong",
        ),
    )
    if result.get("reality_check_points") != 0:
        errors.append("movement-incompatible Reality Check effect should downgrade to neutral")
    if not any("incompatible" in note for note in result.get("validation_notes") or []):
        errors.append("Reality Check incompatibility downgrade should be reported")


def _check_non_operational_fit_no_points(errors: list[str]) -> None:
    review = _pass1_review(
        fit_rating="strongly_improves_fit",
        fit_materiality="extreme",
        reality_effect="neutral",
        reality_strength="none",
    )
    review["strategy_shift_check"] = {
        "status": "supported",
        "rationale": "Phase-sensitive change is coherent for this Operational Fit isolation test.",
    }
    result = score_pass1_review(
        _packet(changed_fields=["phase_ml"]),
        review,
    )
    if result.get("operational_fit_points") != 0:
        errors.append("Operational Fit should not move when no operational assumption changed")


def _check_gated_strategy_shift_required(errors: list[str]) -> None:
    sponsor_type_change = score_pass1_review(
        _packet(changed_fields=["sponsor_tier_ml"]),
        _pass1_review(reality_effect="neutral", reality_strength="none"),
    )
    if sponsor_type_change.get("validation_status") != "valid":
        errors.append("Sponsor Type is locked in the UI and should not be treated as a gated Strategy Shift field")

    missing_shift = score_pass1_review(
        _packet(changed_fields=["phase_ml"]),
        _pass1_review(reality_effect="neutral", reality_strength="none"),
    )
    if missing_shift.get("validation_status") == "valid":
        errors.append("Gated premise-sensitive field changes must require Strategy Shift Check")
    if not any("strategy_shift_check.status must not be not_applicable" in error for error in missing_shift.get("validation_errors") or []):
        errors.append("Missing Strategy Shift Check should produce a targeted validation error")

    supported_review = _pass1_review(reality_effect="neutral", reality_strength="none")
    supported_review["strategy_shift_check"] = {
        "status": "supported",
        "rationale": "Phase change is supported by endpoint, comparator, and operational context.",
    }
    supported_shift = score_pass1_review(
        _packet(changed_fields=["phase_ml"]),
        supported_review,
    )
    if supported_shift.get("validation_status") != "valid":
        errors.append(f"Supported Strategy Shift Check should validate, got {supported_shift.get('validation_errors')}")


def _check_extreme_operational_fit_guardrail(errors: list[str]) -> None:
    one_field_result = score_pass1_review(
        _packet(changed_fields=["operational_assumptions.planned_enrollment"]),
        _pass1_review(
            fit_rating="strongly_improves_fit",
            fit_materiality="extreme",
            reality_effect="neutral",
            reality_strength="none",
        ),
    )
    if one_field_result.get("operational_fit_points") != 3.5:
        errors.append("Operational Fit +5 should cap to +3.5 when only one operational field changed")
    if not any("capped to +/-3.5" in note for note in one_field_result.get("validation_notes") or []):
        errors.append("Operational Fit +5 guardrail cap should be reported")

    two_field_result = score_pass1_review(
        _packet(),
        _pass1_review(
            fit_rating="strongly_improves_fit",
            fit_materiality="extreme",
            reality_effect="neutral",
            reality_strength="none",
        ),
    )
    if two_field_result.get("operational_fit_points") != 5:
        errors.append("Operational Fit +5 should remain available with two changed operational evidence fields")


def _check_operational_fit_baseline_revert_guardrail(errors: list[str]) -> None:
    revert_field_changes = [
        {
            "field": "operational_assumptions.planned_enrollment",
            "previous_value": {"value": 700},
            "current_value": {"value": 480},
            "baseline_value": {"value": 480},
            "change_type": "operational_assumption",
        },
        {
            "field": "operational_assumptions.planned_sites",
            "previous_value": {"value": 40},
            "current_value": {"value": 20},
            "baseline_value": {"value": 20},
            "change_type": "operational_assumption",
        },
    ]
    stale_positive = score_pass1_review(
        _packet(field_changes=revert_field_changes),
        _pass1_review(reality_effect="neutral", reality_strength="none"),
    )
    if stale_positive.get("validation_status") != "invalid":
        errors.append("Operational Fit should reject stale positive ratings when changed fields returned to baseline")
    if stale_positive.get("operational_fit_points") is not None:
        errors.append("Invalid baseline-revert Operational Fit should not produce points")
    if not any("returned to baseline values" in error for error in stale_positive.get("validation_errors") or []):
        errors.append("Baseline-revert Operational Fit rejection should identify returned-to-baseline fields")

    neutral_review = _pass1_review(
        fit_rating="neutral_or_unclear",
        fit_materiality="minor",
        reality_effect="neutral",
        reality_strength="none",
    )
    neutral_revert = score_pass1_review(
        _packet(field_changes=revert_field_changes),
        neutral_review,
    )
    if neutral_revert.get("validation_status") != "valid":
        errors.append(f"Neutral Operational Fit should validate after baseline revert, got {neutral_revert.get('validation_errors')}")
    if neutral_revert.get("operational_fit_points") != 0:
        errors.append("Neutral Operational Fit baseline revert should score 0")
    if not any("returned to baseline values" in note for note in neutral_revert.get("validation_notes") or []):
        errors.append("Neutral baseline-revert Operational Fit should keep an explicit validation note")


def _check_operational_fit_previous_state_reuse(errors: list[str]) -> None:
    previous_equivalent_changes = [
        {
            "field": "operational_assumptions.planned_enrollment",
            "previous_value": {"value": 700},
            "current_value": {"value": 700},
            "baseline_value": {"value": 480},
            "change_type": "operational_assumption",
        },
        {
            "field": "operational_assumptions.planned_sites",
            "previous_value": {"value": 40},
            "current_value": {"value": 40},
            "baseline_value": {"value": 20},
            "change_type": "operational_assumption",
        },
    ]
    equivalent_positive = score_pass1_review(
        _packet(
            previous_trial_score=70,
            field_changes=previous_equivalent_changes,
            previous_operational_fit_points=1.4,
        ),
        _pass1_review(
            fit_rating="strongly_improves_fit",
            fit_materiality="extreme",
            reality_effect="neutral",
            reality_strength="none",
        ),
    )
    if equivalent_positive.get("validation_status") != "valid":
        errors.append(f"Previous-equivalent Operational Fit should validate, got {equivalent_positive.get('validation_errors')}")
    if equivalent_positive.get("operational_fit_points") != 1.4:
        errors.append("Previous-equivalent Operational Fit should reuse previous points instead of rescoring the same state")
    if not any("returned to previous values" in note for note in equivalent_positive.get("validation_notes") or []):
        errors.append("Previous-equivalent Operational Fit reuse should keep an explicit validation note")

    stale_neutral = score_pass1_review(
        _packet(
            previous_trial_score=70,
            field_changes=previous_equivalent_changes,
            previous_operational_fit_points=1.4,
        ),
        _pass1_review(
            fit_rating="neutral_or_unclear",
            fit_materiality="minor",
            reality_effect="neutral",
            reality_strength="none",
        ),
    )
    if stale_neutral.get("validation_status") != "invalid":
        errors.append("Previous-equivalent Operational Fit should reject a direction inconsistent with previous points")
    if not any("preserve the previous Operational Fit direction" in error for error in stale_neutral.get("validation_errors") or []):
        errors.append("Previous-equivalent Operational Fit rejection should identify direction mismatch")


def _check_allocation_validation(errors: list[str]) -> None:
    review = _pass1_review()
    review["reality_check"]["allocations"][0]["allocation_target_id"] = "invented.target"
    result = score_pass1_review(_packet(), review)
    if result.get("validation_status") != "valid":
        errors.append("Reality Check allocation to invented target ID should downgrade without failing Trial Score")
    if result.get("reality_check_points") != 0:
        errors.append("invalid Reality Check allocation should downgrade Reality Check to neutral")
    if not any("allocation_target_id" in note for note in result.get("validation_notes") or []):
        errors.append("invented target ID downgrade should keep an explicit validation note")


def _check_allocation_target_id_contract(errors: list[str]) -> None:
    result = score_pass1_review(_packet(), _pass1_review())
    allocations = result.get("reality_check_allocation_points") or []
    if not allocations:
        errors.append("canonical allocation_target_id should produce allocation points")
        return
    first = allocations[0]
    second = allocations[1] if len(allocations) > 1 else {}
    if first.get("allocation_target_id") != "scientific_challenge.protocol_architecture":
        errors.append("Protocol Architecture allocation should preserve canonical target ID")
    if first.get("pillar") != "Scientific Challenge" or first.get("subpillar") != "Protocol Architecture":
        errors.append("Protocol Architecture target ID should map to exact display labels")
    if second.get("allocation_target_id") != "execution_framework.operational_fit":
        errors.append("Operational Fit allocation should preserve canonical target ID")


def _check_text_only_allocation_target_rejected(errors: list[str]) -> None:
    review = _pass1_review()
    review["reality_check"]["allocations"] = [
        {
            "pillar": "Execution Framework",
            "subpillar": "Operational Fit",
            "share": 1.0,
            "movement_label": "Reality Check: operational realism",
            "rationale": "The site expansion helps, but the enrollment burden still needs a realism check.",
            "incremental_check": "This checks residual execution realism after Operational Fit rather than repeating the site-count credit.",
            "evidence_fields": ["operational_assumptions.planned_sites"],
        }
    ]
    result = score_pass1_review(_packet(), review)
    allocations = result.get("reality_check_allocation_points") or []
    if result.get("validation_status") != "valid":
        errors.append("text-only allocation target should downgrade without failing Trial Score")
    if allocations:
        errors.append("text-only allocation target should not produce allocation points")
    if result.get("reality_check_points") != 0:
        errors.append("text-only allocation target should downgrade Reality Check to neutral")
    if not any("allocation_target_id is required" in note for note in result.get("validation_notes") or []):
        errors.append("text-only allocation target rejection should require allocation_target_id")


def _check_duplicate_reality_check_allocation(errors: list[str]) -> None:
    review = _pass1_review()
    review["reality_check"]["allocations"] = [
        {
            "allocation_target_id": "execution_framework.operational_fit",
            "share": 1.0,
            "movement_label": "Reality Check: duplicated operational support",
            "rationale": "The site footprint changed.",
            "incremental_check": "The site footprint changed.",
            "evidence_fields": ["operational_assumptions.planned_sites"],
        }
    ]
    result = score_pass1_review(_packet(), review)
    if result.get("validation_status") != "valid":
        errors.append("duplicate Operational Fit allocation should downgrade without failing Trial Score")
    if result.get("reality_check_points") != 0:
        errors.append("duplicate Reality Check allocation should downgrade Reality Check to neutral")
    if not any("incremental_check" in note for note in result.get("validation_notes") or []):
        errors.append("duplicate Reality Check allocation downgrade should point to incremental_check")


def _check_app_owned_fields(errors: list[str]) -> None:
    review = _pass1_review()
    review["trial_score"] = 99
    result = score_pass1_review(_packet(), review)
    if result.get("validation_status") != "invalid":
        errors.append("Pass 1 provider-returned trial_score should invalidate the review")
    if not any("application-owned" in error for error in result.get("validation_errors") or []):
        errors.append("app-owned Pass 1 fields should be reported")

    pass2 = {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "trial_score_narrative": {"summary": "The Trial Score improved, but the gain is mixed."},
        "pillar_reading": [{"pillar": "Execution Framework", "reading": "Operational support improved."}],
        "central_tension": {"summary": "Execution support versus evidence interpretability."},
        "broader_strategic_question": {"question": "What trade-off should the team defend?"},
        "trial_score": 99,
    }
    validated = validate_pass2_review(pass2)
    if validated.get("validation_status") != "invalid":
        errors.append("Pass 2 provider-returned trial_score should invalidate the review")

    pass2_without_questions = {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "trial_score_narrative": {"summary": "The Trial Score improved, but the gain is mixed."},
        "pillar_reading": [{"pillar": "Execution Framework", "reading": "Operational support improved."}],
        "central_tension": {"summary": "Execution support versus evidence interpretability."},
        "broader_strategic_question": {"question": "What trade-off should the team defend?"},
    }
    validated_optional = validate_pass2_review(pass2_without_questions)
    if validated_optional.get("validation_status") != "valid":
        errors.append("Pass 2 facilitator_questions should be optional")

    pass2_too_many_questions = {
        **pass2_without_questions,
        "facilitator_questions": [
            {"question": str(index), "why_it_matters": "test", "related_feature_families": []}
            for index in range(4)
        ],
    }
    validated_too_many = validate_pass2_review(pass2_too_many_questions)
    if validated_too_many.get("validation_status") != "invalid":
        errors.append("Pass 2 facilitator_questions should be capped at 3")

    pass2_malformed_question = {
        **pass2_without_questions,
        "facilitator_questions": [
            {"why_it_matters": "Missing question", "related_feature_families": []},
            {"question": "What should be discussed?", "related_feature_families": "operations"},
        ],
    }
    validated_malformed_question = validate_pass2_review(pass2_malformed_question)
    if validated_malformed_question.get("validation_status") != "invalid":
        errors.append("Pass 2 malformed facilitator question objects should invalidate the review")
    malformed_errors = " ".join(validated_malformed_question.get("validation_errors") or [])
    if "question is required" not in malformed_errors or "array of strings" not in malformed_errors:
        errors.append("Pass 2 malformed facilitator question errors should identify the bad fields")


def _check_neutral_reality_check_has_no_top_level_plot_row(errors: list[str]) -> None:
    result = score_pass1_review(
        _packet(),
        _pass1_review(reality_effect="neutral", reality_strength="none"),
    )
    rows = design_subcategory_impacts({
        "hidden_baseline": False,
        "participant_visible": True,
        "strategic_review": result.get("reality_check_points"),
        "trial_score": result.get("trial_score"),
        "strategic_review_assessment": result.get("reality_check_assessment") or {},
        "reality_check_points": result.get("reality_check_points"),
        "operational_fit_points": result.get("operational_fit_points"),
        "reality_check_allocation_points": result.get("reality_check_allocation_points") or [],
    })
    if rows:
        errors.append("neutral V1 Reality Check should not render a top-level Reality Check plot row")


def main() -> int:
    errors: list[str] = []
    _check_operational_fit_mapping(errors)
    _check_positive_reinforcement_cap(errors)
    _check_incompatible_effect_downgrade(errors)
    _check_non_operational_fit_no_points(errors)
    _check_gated_strategy_shift_required(errors)
    _check_extreme_operational_fit_guardrail(errors)
    _check_operational_fit_baseline_revert_guardrail(errors)
    _check_operational_fit_previous_state_reuse(errors)
    _check_allocation_validation(errors)
    _check_allocation_target_id_contract(errors)
    _check_text_only_allocation_target_rejected(errors)
    _check_duplicate_reality_check_allocation(errors)
    _check_app_owned_fields(errors)
    _check_neutral_reality_check_has_no_top_level_plot_row(errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated Trial Score V1 contract and deterministic scoring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
