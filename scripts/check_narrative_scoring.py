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
) -> dict:
    return {
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
    )
    result = validate_and_score_review(packet, review)
    endpoint = (
        result["scoring"].get("design_confidence_assessment", {})
        .get("subcategories", {})
        .get("endpoint_evidence_strength", {})
    )
    if endpoint.get("points") != -4:
        errors.append("subcategory mapping failed: conflicting moderate should map to -4")

    packet, review = _review_template()
    packet["model_interpretation"]["completion_score"] = 3
    review["design_confidence_subcategories"] = {
        "phase_intent_alignment": {
            **_subcategory("conflicting", "Total reconciliation test.", ["phase_ml"]),
        },
        "endpoint_evidence_strength": _subcategory(
            "conflicting",
            "Total reconciliation test.",
            ["endpoint_rigor_ml", "comparator_benchmark_ml", "primary_duration_months_ml"],
            "moderate",
        ),
        "target_population_alignment": _subcategory("conflicting", "Total reconciliation test.", ["adult_ml"]),
        "operational_burden_balance": _subcategory("conflicting", "Total reconciliation test.", ["has_dmc_ml"]),
    }
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    if scoring.get("design_confidence") != -13:
        errors.append("total reconciliation failed: Design Confidence should preserve uncapped subcategory total")
    if scoring.get("total_scenario_score") != 0:
        errors.append("total score cap failed: Total Scenario Score should clamp to 0")

    packet, review = _review_template()
    packet["model_interpretation"]["score_delta"] = -2
    review["design_confidence_subcategories"]["target_population_alignment"] = _subcategory(
        "supportive",
        "Half-point-preserving positive mapping test.",
        ["older_adult_ml"],
        "moderate",
    )
    review["design_confidence_subcategories"]["operational_burden_balance"] = _subcategory(
        "weak",
        "Half-point-preserving negative mapping test.",
        ["has_dmc_ml"],
        "moderate",
    )
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    if scoring.get("design_confidence") != 0:
        errors.append("half-point reconciliation failed: +1.5 and -1.5 should sum to 0")
    subcategories = scoring.get("design_confidence_assessment", {}).get("subcategories", {})
    if subcategories.get("target_population_alignment", {}).get("points") != 1.5:
        errors.append("supportive moderate mapping should contribute +1.5")
    if subcategories.get("operational_burden_balance", {}).get("points") != -1.5:
        errors.append("weak non-operational-evidence mapping should contribute -1.5")

    packet, review = _review_template()
    packet["model_interpretation"]["completion_score"] = 98
    packet["model_interpretation"]["score_delta"] = -1
    for subcategory_name in REQUIRED_DESIGN_SUBCATEGORIES:
        review["design_confidence_subcategories"][subcategory_name] = _subcategory(
            "strong",
            "Positive ceiling test.",
            ["phase_ml"],
            "minimal",
        )
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    if scoring.get("design_confidence") != 12:
        errors.append("positive reconciliation failed: four strong minimal subcategories should add to +12")
    if scoring.get("total_scenario_score") != 100:
        errors.append("positive final score cap failed: Total Scenario Score should clamp to 100")


def _check_score_materiality_outer_bounds(errors: list[str]) -> None:
    packet, review = _review_template()
    review["design_confidence_subcategories"]["endpoint_evidence_strength"] = _subcategory(
        "strong",
        "Very high positive materiality should reach the upper subcategory bound.",
        ["endpoint_rigor_ml"],
        "very_high",
    )
    review["design_confidence_subcategories"]["target_population_alignment"] = _subcategory(
        "conflicting",
        "Very high negative materiality should reach the lower subcategory bound.",
        ["older_adult_ml"],
        "very_high",
    )
    result = validate_and_score_review(packet, review)
    subcategories = result["scoring"].get("design_confidence_assessment", {}).get("subcategories", {})
    if subcategories.get("endpoint_evidence_strength", {}).get("points") != 5:
        errors.append("score_materiality upper bound failed: strong very_high should map to +5")
    if subcategories.get("target_population_alignment", {}).get("points") != -5:
        errors.append("score_materiality lower bound failed: conflicting very_high should map to -5")
    if result["scoring"].get("design_confidence") != 0:
        errors.append("score_materiality reconciliation failed: +5 and -5 should sum to 0 with neutral subcategories")


def _check_score_materiality_guardrails(errors: list[str]) -> None:
    fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "operational_only_ambitious_enrollment_v2"
    )
    packet = build_review_packet_from_fixture(fixture)
    review = deepcopy(fixture["mock_review"])
    review["design_confidence_subcategories"]["operational_burden_balance"] = _subcategory(
        "strong",
        "Operational-only changes should not create positive Operational Burden Balance points.",
        ["operational_assumptions.planned_enrollment.enrollment_status"],
        "very_high",
    )
    result = validate_and_score_review(packet, review)
    operational = (
        result["scoring"].get("design_confidence_assessment", {})
        .get("subcategories", {})
        .get("operational_burden_balance", {})
    )
    if operational.get("points") != 0:
        errors.append("operational-only guardrail failed: positive operational_burden_balance should be capped at 0")

    packet, review = _review_template()
    review["design_confidence_subcategories"]["phase_intent_alignment"] = _subcategory(
        "strong",
        "Already-positive Completion Outlook pillars need changed evidence before large positive design points.",
        ["phase_ml"],
        "very_high",
    )
    result = validate_and_score_review(packet, review)
    phase = (
        result["scoring"].get("design_confidence_assessment", {})
        .get("subcategories", {})
        .get("phase_intent_alignment", {})
    )
    if phase.get("points") != 1:
        errors.append("anti-overreward guardrail failed: already-positive unchanged pillar should cap positive points at +1")

    packet, review = _review_template()
    packet["model_interpretation"]["pillar_impacts"] = [
        {"Pillar": "Therapeutic Context", "Impact": 4.2},
        {"Pillar": "Scientific Challenge", "Impact": -1.6},
        {"Pillar": "Patient Profile", "Impact": 2.4},
        {"Pillar": "Execution Framework", "Impact": 1.0},
    ]
    review["design_confidence_subcategories"]["phase_intent_alignment"] = _subcategory(
        "strong",
        "Live-style list pillar impacts should still trigger the anti-overreward guardrail.",
        ["phase_ml"],
        "very_high",
    )
    result = validate_and_score_review(packet, review)
    phase = (
        result["scoring"].get("design_confidence_assessment", {})
        .get("subcategories", {})
        .get("phase_intent_alignment", {})
    )
    if phase.get("points") != 1:
        errors.append("anti-overreward guardrail failed for live-style list pillar impacts")


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


def main() -> int:
    errors: list[str] = []
    for fixture in get_contract_fixtures():
        _check_fixture(fixture, errors)
    _check_evidence_required(errors)
    _check_unsupported_evidence_required(errors)
    _check_score_reconciliation(errors)
    _check_score_materiality_outer_bounds(errors)
    _check_score_materiality_guardrails(errors)
    _check_incomplete_review_suppresses_scores(errors)
    _check_app_owned_score_fields_ignored(errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated Scenario Review scoring against contract fixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
