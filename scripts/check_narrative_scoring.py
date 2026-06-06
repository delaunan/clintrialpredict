#!/usr/bin/env python
"""Validate narrative review validation and scoring against contract fixtures."""

from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.contract_fixtures import REQUIRED_REVIEW_DOMAINS  # noqa: E402
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402
from src.narratives.scoring import validate_and_score_review  # noqa: E402


def _review_template() -> tuple[dict, dict]:
    fixture = next(item for item in get_contract_fixtures() if item.get("mock_review"))
    packet = build_review_packet_from_fixture(fixture)
    review = deepcopy(fixture["mock_review"])
    review["quality_review_domains"] = {
        domain_name: {
            "rating": "neutral" if domain_name == "change_integrity" else (
                "consistent" if domain_name == "text_consistency" else "acceptable"
            ),
            "rationale": "Neutral test domain.",
            "evidence_fields": ["phase_ml"],
        }
        for domain_name in sorted(REQUIRED_REVIEW_DOMAINS)
    }
    return packet, review


def _check_fixture(fixture: dict, errors: list[str]) -> None:
    fixture_id = fixture["fixture_id"]
    expected = fixture["expected_behavior"]
    if expected.get("review_needed") is False or expected.get("clarification_needed") is True:
        if fixture.get("mock_review") is not None:
            errors.append(f"{fixture_id}: non-reviewed fixture should not define mock_review")
        return

    packet = build_review_packet_from_fixture(fixture)
    result = validate_and_score_review(packet, fixture["mock_review"])
    scoring = result["scoring"]

    if scoring.get("validation_status") != "valid":
        errors.append(f"{fixture_id}: expected valid review, got {scoring.get('validation_status')}")
    if scoring.get("quality_adjustment") != expected["expected_quality_adjustment"]:
        errors.append(
            f"{fixture_id}: expected quality_adjustment {expected['expected_quality_adjustment']}, "
            f"got {scoring.get('quality_adjustment')}"
        )
    if scoring.get("final_candidate_score") != expected["expected_final_candidate_score"]:
        errors.append(
            f"{fixture_id}: expected final_candidate_score {expected['expected_final_candidate_score']}, "
            f"got {scoring.get('final_candidate_score')}"
        )

    quality_assessment = scoring.get("quality_assessment") or {}
    if set((quality_assessment.get("pillars") or {})) != {
        "evidence_coherence",
        "population_strategy_fit",
        "execution_plausibility",
    }:
        errors.append(f"{fixture_id}: missing expected Quality Assessment pillars")


def _check_evidence_required(errors: list[str]) -> None:
    fixtures = get_contract_fixtures()
    fixture = next(item for item in fixtures if item["fixture_id"] == "operational_only_ambitious_enrollment_v1")
    packet = build_review_packet_from_fixture(fixture)
    review = deepcopy(fixture["mock_review"])
    review["quality_review_domains"]["operational_scale_fit"]["evidence_fields"] = []
    result = validate_and_score_review(packet, review)
    if result["scoring"]["quality_adjustment"] != 0:
        errors.append("evidence-required guardrail failed: empty evidence_fields should zero point effect")


def _check_unsupported_evidence_required(errors: list[str]) -> None:
    packet, review = _review_template()
    review["quality_review_domains"]["endpoint_and_comparator_logic"] = {
        "rating": "weak",
        "rationale": "Unsupported evidence should not move scoring.",
        "evidence_fields": ["not_a_packet_field"],
    }
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    if scoring.get("quality_adjustment") != 0:
        errors.append("unsupported evidence guardrail failed: unsupported evidence_fields should zero point effect")
    evidence_domains = (
        scoring.get("quality_assessment", {})
        .get("pillars", {})
        .get("evidence_coherence", {})
        .get("domains", {})
    )
    endpoint_domain = evidence_domains.get("endpoint_and_comparator_logic", {})
    unsupported = endpoint_domain.get("unsupported_evidence_fields")
    if unsupported != ["not_a_packet_field"]:
        errors.append("unsupported evidence guardrail should preserve unsupported_evidence_fields for auditability")
    if endpoint_domain.get("supported_evidence_fields") != []:
        errors.append("unsupported evidence guardrail should not report unsupported fields as supported")


def _check_cap_reconciliation(errors: list[str]) -> None:
    packet, review = _review_template()
    domains = review["quality_review_domains"]
    domains["scientific_rigor"] = {
        "rating": "conflicting",
        "rationale": "Single-domain cap test.",
        "evidence_fields": ["endpoint_rigor_ml"],
    }
    result = validate_and_score_review(packet, review)
    evidence_domains = (
        result["scoring"].get("quality_assessment", {})
        .get("pillars", {})
        .get("evidence_coherence", {})
        .get("domains", {})
    )
    if evidence_domains.get("scientific_rigor", {}).get("points") != -3:
        errors.append("subcategory cap failed: conflicting should be capped from -4 to -3")

    packet, review = _review_template()
    review["quality_review_domains"]["scientific_rigor"] = {
        "rating": "conflicting",
        "rationale": "Pillar cap test.",
        "evidence_fields": ["endpoint_rigor_ml"],
    }
    review["quality_review_domains"]["endpoint_and_comparator_logic"] = {
        "rating": "conflicting",
        "rationale": "Pillar cap test.",
        "evidence_fields": ["comparator_benchmark_ml"],
    }
    result = validate_and_score_review(packet, review)
    evidence_pillar = result["scoring"]["quality_assessment"]["pillars"]["evidence_coherence"]
    if evidence_pillar.get("raw_points") != -6 or evidence_pillar.get("points") != -4:
        errors.append("pillar cap failed: evidence_coherence should cap raw -6 to -4")

    packet, review = _review_template()
    packet["model_interpretation"]["completion_score"] = 3
    for domain_name in REQUIRED_REVIEW_DOMAINS:
        if domain_name == "change_integrity":
            rating = "potential_shortcut"
        elif domain_name == "text_consistency":
            rating = "contradiction"
        else:
            rating = "conflicting"
        review["quality_review_domains"][domain_name] = {
            "rating": rating,
            "rationale": "Total cap test.",
            "evidence_fields": ["phase_ml"],
        }
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    if scoring.get("quality_adjustment") != -10:
        errors.append("total cap failed: Quality Adjustment should clamp to -10")
    if scoring.get("final_candidate_score") != 0:
        errors.append("final score cap failed: Final Candidate Score should clamp to 0")


def _check_app_owned_score_fields_ignored(errors: list[str]) -> None:
    packet, review = _review_template()
    review["quality_adjustment"] = 99
    review["final_candidate_score"] = 99
    review["quality_assessment"] = {"provider": "should_not_be_trusted"}
    result = validate_and_score_review(packet, review)
    scoring = result["scoring"]
    validation_errors = result["validated_review"].get("validation_errors") or []
    if scoring.get("quality_adjustment") != 0:
        errors.append("provider-owned quality_adjustment should be ignored in favor of app scoring")
    if scoring.get("final_candidate_score") != packet["model_interpretation"]["completion_score"]:
        errors.append("provider-owned final_candidate_score should be ignored in favor of app scoring")
    if not any("application-owned" in error for error in validation_errors):
        errors.append("provider-owned score fields should be reported as validation errors")


def main() -> int:
    errors: list[str] = []
    for fixture in get_contract_fixtures():
        _check_fixture(fixture, errors)
    _check_evidence_required(errors)
    _check_unsupported_evidence_required(errors)
    _check_cap_reconciliation(errors)
    _check_app_owned_score_fields_ignored(errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated narrative review validation and scoring against contract fixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
