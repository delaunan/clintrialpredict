#!/usr/bin/env python
"""Validate narrative review validation and scoring against contract fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402
from src.narratives.scoring import validate_and_score_review  # noqa: E402


def _check_fixture(fixture: dict, errors: list[str]) -> None:
    fixture_id = fixture["fixture_id"]
    expected = fixture["expected_behavior"]
    if expected.get("review_needed") is False:
        if fixture.get("mock_review") is not None:
            errors.append(f"{fixture_id}: no-op fixture should not define mock_review")
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
    review = fixture["mock_review"]
    review["quality_review_domains"]["operational_scale_fit"]["evidence_fields"] = []
    result = validate_and_score_review(packet, review)
    if result["scoring"]["quality_adjustment"] != 0:
        errors.append("evidence-required guardrail failed: empty evidence_fields should zero point effect")


def main() -> int:
    errors: list[str] = []
    for fixture in get_contract_fixtures():
        _check_fixture(fixture, errors)
    _check_evidence_required(errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated narrative review validation and scoring against contract fixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
