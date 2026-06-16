#!/usr/bin/env python
"""Validate Scenario Review scoring against contract fixtures."""

from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.contract_fixtures import (  # noqa: E402
    REQUIRED_DESIGN_SUBCATEGORIES,
    get_contract_fixtures,
)
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402
from src.narratives.scoring import validate_and_score_review  # noqa: E402


def _subcategory(
    rating: str,
    rationale: str,
    evidence_fields: list[str],
    score_materiality: str = "minimal",
    movement_direction: str | None = None,
    movement_materiality: str | None = None,
    effect_role: str | None = None,
) -> dict:
    if movement_direction is None:
        movement_direction = {
            "strong": "improved",
            "supportive": "improved",
            "balanced": "unchanged",
            "weak": "weakened",
            "conflicting": "worsened",
        }.get(rating, "unchanged")
    if movement_materiality is None:
        movement_materiality = {
            "minimal": "minor",
            "low": "minor",
            "moderate": "moderate",
            "high": "major",
            "very_high": "major",
        }.get(score_materiality, "none")
        if movement_direction == "unchanged":
            movement_materiality = "none"
    if effect_role is None:
        effect_role = "unchanged" if movement_direction == "unchanged" else "independent"
    return {
        "current_state": rating,
        "movement_direction": movement_direction,
        "movement_materiality": movement_materiality,
        "effect_role": effect_role,
        "rating": rating,
        "score_materiality": score_materiality,
        "rationale": rationale,
        "evidence_fields": evidence_fields,
        "short_rationale": rationale.split(".", 1)[0],
        "optional_lenses_used": [],
        "regulatory_or_finance_note": "",
    }


def _review_template() -> tuple[dict, dict]:
    fixture = next(item for item in get_contract_fixtures() if item.get("mock_review"))
    packet = build_review_packet_from_fixture(fixture)
    review = deepcopy(fixture["mock_review"])
    review["design_confidence_subcategories"] = {
        subcategory_name: _subcategory("balanced", "Neutral test subcategory.", ["phase_ml"])
        for subcategory_name in sorted(REQUIRED_DESIGN_SUBCATEGORIES)
    }
    return packet, review


def _check_fixture(fixture: dict, errors: list[str]) -> None:
    fixture_id = fixture["fixture_id"]
    expected = fixture["expected_behavior"]
    if expected.get("review_needed") is False:
        if fixture.get("mock_review") is not None:
            errors.append(f"{fixture_id}: non-reviewed fixture should not define mock_review")
        return

    packet = build_review_packet_from_fixture(fixture)
    result = validate_and_score_review(packet, fixture["mock_review"])
    scoring = result["scoring"]

    if scoring.get("validation_status") != "valid":
        errors.append(f"{fixture_id}: expected valid review, got {scoring.get('validation_status')}")
    if scoring.get("design_confidence") != expected["expected_design_confidence"]:
        errors.append(
            f"{fixture_id}: expected design_confidence {expected['expected_design_confidence']}, "
            f"got {scoring.get('design_confidence')}"
        )
    if scoring.get("total_scenario_score") != expected["expected_total_scenario_score"]:
        errors.append(
            f"{fixture_id}: expected total_scenario_score {expected['expected_total_scenario_score']}, "
            f"got {scoring.get('total_scenario_score')}"
        )

    assessment = scoring.get("design_confidence_assessment") or {}
    subcategories = assessment.get("subcategories") or {}
    if set(subcategories) != REQUIRED_DESIGN_SUBCATEGORIES:
        errors.append(f"{fixture_id}: missing expected Design Confidence subcategories")
        return

    for subcategory_name, expected_points in expected["expected_design_subcategories"].items():
        actual = subcategories.get(subcategory_name, {}).get("points")
        if actual != expected_points:
            errors.append(f"{fixture_id}: {subcategory_name} expected {expected_points}, got {actual}")

    subcategory_total = sum(float(item.get("points", 0)) for item in subcategories.values())
    if subcategory_total != scoring.get("design_confidence"):
        errors.append(f"{fixture_id}: subcategory points should add up to Design Confidence")

    pillars = assessment.get("pillars") or {}
    if set(pillars) != {"therapeutic_context", "scientific_challenge", "patient_profile", "execution_framework"}:
        errors.append(f"{fixture_id}: missing expected Design Confidence pillars")
    pillar_total = sum(float(pillar.get("design_points", 0)) for pillar in pillars.values())
    if pillar_total != scoring.get("design_confidence"):
        errors.append(f"{fixture_id}: pillar points should add up to Design Confidence")


def _check_evidence_required(errors: list[str]) -> None:
    fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "operational_only_ambitious_enrollment_v2"
    )
    packet = build_review_packet_from_fixture(fixture)
    review = deepcopy(fixture["mock_review"])
    review["design_confidence_subcategories"]["operational_burden_balance"]["evidence_fields"] = []
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    subcategory = (
        scoring.get("design_confidence_assessment", {})
        .get("subcategories", {})
        .get("operational_burden_balance", {})
    )
    if subcategory.get("points") != 0:
        errors.append("evidence-required guardrail failed: empty evidence_fields should zero point effect")


def _check_unsupported_evidence_required(errors: list[str]) -> None:
    packet, review = _review_template()
    review["design_confidence_subcategories"]["endpoint_evidence_strength"] = _subcategory(
        "weak",
        "Unsupported evidence should not move scoring.",
        ["not_a_packet_field"],
    )
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    if scoring.get("design_confidence") != 0:
        errors.append("unsupported evidence guardrail failed: unsupported evidence_fields should zero point effect")
    endpoint = (
        scoring.get("design_confidence_assessment", {})
        .get("subcategories", {})
        .get("endpoint_evidence_strength", {})
    )
    if endpoint.get("unsupported_evidence_fields") != ["not_a_packet_field"]:
        errors.append("unsupported evidence guardrail should preserve unsupported_evidence_fields for auditability")
    if endpoint.get("supported_evidence_fields") != []:
        errors.append("unsupported evidence guardrail should not report unsupported fields as supported")


def _check_score_reconciliation(errors: list[str]) -> None:
    packet, review = _review_template()
    review["design_confidence_subcategories"]["endpoint_evidence_strength"] = _subcategory(
        "conflicting",
        "Single-subcategory mapping test.",
        ["endpoint_rigor_ml", "comparator_benchmark_ml", "primary_duration_months_ml"],
        "moderate",
        movement_direction="worsened",
        movement_materiality="moderate",
        effect_role="counterweight",
    )
    result = validate_and_score_review(packet, review)
    endpoint = (
        result["scoring"].get("design_confidence_assessment", {})
        .get("subcategories", {})
        .get("endpoint_evidence_strength", {})
    )
    if endpoint.get("points") != -1:
        errors.append("movement mapping failed: worsened moderate counterweight should map to -1")

    packet, review = _review_template()
    packet["model_interpretation"]["completion_score"] = 98
    packet["model_interpretation"]["score_delta"] = 0
    review["design_confidence_subcategories"] = {
        "phase_intent_alignment": _subcategory("strong", "Cap test.", ["phase_ml"], "very_high", movement_materiality="major"),
        "endpoint_evidence_strength": _subcategory(
            "strong",
            "Cap test.",
            ["endpoint_rigor_ml", "comparator_benchmark_ml", "primary_duration_months_ml"],
            "very_high",
            movement_materiality="major",
        ),
        "target_population_alignment": _subcategory("strong", "Cap test.", ["adult_ml"], "very_high", movement_materiality="major"),
        "operational_burden_balance": _subcategory("strong", "Cap test.", ["has_dmc_ml"], "very_high", movement_materiality="major"),
    }
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    if scoring.get("design_confidence") != 2:
        errors.append("net cap failed: flat-score four-major positive movement should scale to +2")
    if scoring.get("total_scenario_score") != 100:
        errors.append("total score cap failed: Total Scenario Score should clamp to 100")

    packet, review = _review_template()
    packet["model_interpretation"]["score_delta"] = -2
    review["design_confidence_subcategories"]["target_population_alignment"] = _subcategory(
        "supportive",
        "Half-point-preserving positive mapping test.",
        ["older_adult_ml"],
        "moderate",
        movement_direction="improved",
        movement_materiality="moderate",
        effect_role="independent",
    )
    review["design_confidence_subcategories"]["operational_burden_balance"] = _subcategory(
        "weak",
        "Half-point-preserving negative mapping test.",
        ["has_dmc_ml"],
        "moderate",
        movement_direction="weakened",
        movement_materiality="moderate",
        effect_role="independent",
    )
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    if scoring.get("design_confidence") != 0:
        errors.append("movement reconciliation failed: +1 and -1 should sum to 0")
    subcategories = scoring.get("design_confidence_assessment", {}).get("subcategories", {})
    if subcategories.get("target_population_alignment", {}).get("points") != 1:
        errors.append("improved moderate mapping should contribute +1")
    if subcategories.get("operational_burden_balance", {}).get("points") != -1:
        errors.append("weakened moderate mapping should contribute -1")

    packet, review = _review_template()
    review["design_confidence_subcategories"]["endpoint_evidence_strength"] = _subcategory(
        "supportive",
        "Offset movement should score as partial positive balancing evidence.",
        ["endpoint_rigor_ml"],
        "moderate",
        movement_direction="offset",
        movement_materiality="moderate",
        effect_role="independent",
    )
    result = validate_and_score_review(packet, review)
    endpoint = (
        result["scoring"].get("design_confidence_assessment", {})
        .get("subcategories", {})
        .get("endpoint_evidence_strength", {})
    )
    if endpoint.get("points") != 1:
        errors.append("offset mapping failed: offset moderate independent should contribute +1")


def _check_score_materiality_outer_bounds(errors: list[str]) -> None:
    packet, review = _review_template()
    review["design_confidence_subcategories"]["endpoint_evidence_strength"] = _subcategory(
        "strong",
        "Very high positive materiality should reach the upper subcategory bound.",
        ["endpoint_rigor_ml"],
        "very_high",
        movement_direction="improved",
        movement_materiality="major",
        effect_role="independent",
    )
    review["design_confidence_subcategories"]["target_population_alignment"] = _subcategory(
        "conflicting",
        "Very high negative materiality should reach the lower subcategory bound.",
        ["older_adult_ml"],
        "very_high",
        movement_direction="worsened",
        movement_materiality="major",
        effect_role="independent",
    )
    result = validate_and_score_review(packet, review)
    subcategories = result["scoring"].get("design_confidence_assessment", {}).get("subcategories", {})
    if subcategories.get("endpoint_evidence_strength", {}).get("points") != 2:
        errors.append("movement_materiality upper bound failed: improved major should map to +2")
    if subcategories.get("target_population_alignment", {}).get("points") != -2:
        errors.append("movement_materiality lower bound failed: worsened major should map to -2")
    if result["scoring"].get("design_confidence") != 0:
        errors.append("movement_materiality reconciliation failed: +2 and -2 should sum to 0 with neutral subcategories")


def _check_score_materiality_guardrails(errors: list[str]) -> None:
    fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "operational_only_ambitious_enrollment_v2"
    )
    packet = build_review_packet_from_fixture(fixture)
    review = deepcopy(fixture["mock_review"])
    review["design_confidence_subcategories"]["operational_burden_balance"] = _subcategory(
        "strong",
        "Supported operational evidence can create positive Operational Burden Balance points.",
        ["operational_assumptions.planned_enrollment.enrollment_status"],
        "very_high",
        movement_direction="improved",
        movement_materiality="major",
        effect_role="independent",
    )
    result = validate_and_score_review(packet, review)
    operational = (
        result["scoring"].get("design_confidence_assessment", {})
        .get("subcategories", {})
        .get("operational_burden_balance", {})
    )
    if operational.get("points") != 2:
        errors.append("operational-only guardrail failed: supported operational evidence should be allowed to score normally in Design Confidence")

    packet, review = _review_template()
    review["design_confidence_subcategories"]["phase_intent_alignment"] = _subcategory(
        "strong",
        "Confirming movement should be softened to avoid double counting.",
        ["phase_ml"],
        "very_high",
        movement_direction="improved",
        movement_materiality="major",
        effect_role="confirming",
    )
    result = validate_and_score_review(packet, review)
    phase = (
        result["scoring"].get("design_confidence_assessment", {})
        .get("subcategories", {})
        .get("phase_intent_alignment", {})
    )
    if phase.get("points") != 1:
        errors.append("confirming effect-role failed: major confirming movement should score half credit")


def _check_design_confidence_calibration(errors: list[str]) -> None:
    packet, review = _review_template()
    packet["model_interpretation"]["score_delta"] = 0
    packet["iteration_context"]["changed_fields"] = [
        "allocation_ml",
        "masking_ml",
        "comparator_benchmark_ml",
        "intervention_model_ml",
    ]
    for subcategory_name in REQUIRED_DESIGN_SUBCATEGORIES:
        review["design_confidence_subcategories"][subcategory_name] = _subcategory(
            "strong",
            "Major redesign movement test.",
            ["phase_ml"],
            "very_high",
            movement_direction="improved",
            movement_materiality="major",
            effect_role="independent",
        )
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    if scoring.get("design_confidence") != 4.8:
        errors.append("major-redesign cap failed: flat Completion delta with four structured changes should scale close to +5 net")
    if (scoring.get("design_confidence_assessment") or {}).get("design_confidence_cap") != 5:
        errors.append("major-redesign cap should be reported in design_confidence_assessment")

    packet, review = _review_template()
    packet["model_interpretation"]["score_delta"] = 0
    review["design_confidence_subcategories"]["endpoint_evidence_strength"] = _subcategory(
        "strong",
        "Unchanged strong current state should not add Design Confidence.",
        ["endpoint_rigor_ml"],
        "very_high",
        movement_direction="unchanged",
        movement_materiality="none",
        effect_role="unchanged",
    )
    result = validate_and_score_review(packet, review)
    endpoint = (
        result["scoring"].get("design_confidence_assessment", {})
        .get("subcategories", {})
        .get("endpoint_evidence_strength", {})
    )
    if endpoint.get("points") != 0:
        errors.append("unchanged movement failed: strong current state with unchanged movement should score 0")


def _check_incomplete_review_suppresses_scores(errors: list[str]) -> None:
    packet, review = _review_template()
    del review["design_confidence_subcategories"]["phase_intent_alignment"]
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    if scoring.get("design_confidence") is not None:
        errors.append("incomplete review should not return Design Confidence")
    if scoring.get("total_scenario_score") is not None:
        errors.append("incomplete review should not return Total Scenario Score")

    packet, review = _review_template()
    del review["completion_outlook_analysis"]["risk_pattern_summary"]
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    if scoring.get("design_confidence") is not None:
        errors.append("incomplete completion_outlook_analysis should not return Design Confidence")
    if not any("completion_outlook_analysis.risk_pattern_summary" in error for error in result["validated_review"].get("validation_errors") or []):
        errors.append("missing completion_outlook_analysis field should be reported")

    packet, review = _review_template()
    del review["design_confidence_subcategories"]["phase_intent_alignment"]["short_rationale"]
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    if scoring.get("design_confidence") is not None:
        errors.append("missing short_rationale should suppress Design Confidence")
    if not any("short_rationale" in error for error in result["validated_review"].get("validation_errors") or []):
        errors.append("missing short_rationale should be reported")


def _check_app_owned_score_fields_ignored(errors: list[str]) -> None:
    packet, review = _review_template()
    review["design_confidence"] = 99
    review["total_scenario_score"] = 99
    review["design_confidence_assessment"] = {"provider": "should_not_be_trusted"}
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    validation_errors = result["validated_review"].get("validation_errors") or []
    if scoring.get("design_confidence") != 0:
        errors.append("provider-owned design_confidence should be ignored in favor of app scoring")
    if scoring.get("total_scenario_score") != packet["model_interpretation"]["completion_score"]:
        errors.append("provider-owned total_scenario_score should be ignored in favor of app scoring")
    if not any("application-owned" in error for error in validation_errors):
        errors.append("provider-owned score fields should be reported as validation errors")


def _check_legacy_participant_review_questions(errors: list[str]) -> None:
    packet, review = _review_template()
    legacy_medical = "What evidence standard matters most?"
    legacy_clinops = "What operational burden is proportionate?"
    review.pop("key_questions", None)
    review["participant_review"] = {
        "medical_development_question": legacy_medical,
        "clinops_execution_question": legacy_clinops,
    }
    result = validate_and_score_review(packet, review)
    validated = result["validated_review"]
    questions = validated.get("key_questions") or {}
    if result["scoring"].get("validation_status") != "valid":
        errors.append("legacy participant_review questions should remain valid after strategic question migration")
    if questions.get("medical_clinical_development_question") != legacy_medical:
        errors.append("legacy participant_review medical question should map to the new medical/clinical-development question")
    if questions.get("medical_development_question") != legacy_medical:
        errors.append("legacy participant_review medical alias should be preserved during migration")
    if questions.get("clinical_operations_question") != legacy_clinops:
        errors.append("legacy participant_review clinops question should be preserved")
    if questions.get("strategic_development_question") != legacy_clinops:
        errors.append("legacy participant_review clinops question should map to the new strategic development question when no strategic question exists")
    if questions.get("strategic_field_question") != legacy_clinops:
        errors.append("legacy participant_review strategic alias should mirror the normalized strategic development question")


def main() -> int:
    errors: list[str] = []
    for fixture in get_contract_fixtures():
        _check_fixture(fixture, errors)
    _check_evidence_required(errors)
    _check_unsupported_evidence_required(errors)
    _check_score_reconciliation(errors)
    _check_score_materiality_outer_bounds(errors)
    _check_score_materiality_guardrails(errors)
    _check_design_confidence_calibration(errors)
    _check_incomplete_review_suppresses_scores(errors)
    _check_app_owned_score_fields_ignored(errors)
    _check_legacy_participant_review_questions(errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated Scenario Review scoring against contract fixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
