#!/usr/bin/env python
"""Validate deterministic mock reviewer behavior against contract fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.mock_reviewer import (  # noqa: E402
    FAILURE_MALFORMED_JSON,
    FAILURE_PROVIDER_ERROR,
    review_packet_with_mock,
)
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402


def _check_fixture(fixture: dict, errors: list[str]) -> None:
    packet = build_review_packet_from_fixture(fixture)
    result = review_packet_with_mock(packet)
    fixture_id = fixture["fixture_id"]
    expected = fixture["expected_behavior"]

    if result.get("fixture_id") != fixture_id:
        errors.append(f"{fixture_id}: mock reviewer matched {result.get('fixture_id')}")
    if result.get("review_needed") != expected.get("review_needed"):
        errors.append(f"{fixture_id}: review_needed mismatch")

    if expected.get("review_needed") is False:
        if result.get("status") != "reused_previous_review":
            errors.append(f"{fixture_id}: no-op should reuse previous review")
        if result.get("review") is not None:
            errors.append(f"{fixture_id}: no-op should not return a fresh review")
        return
    scoring = result.get("scoring") or {}
    if result.get("status") != "reviewed":
        errors.append(f"{fixture_id}: expected reviewed status, got {result.get('status')}")
    if scoring.get("validation_status") != "valid":
        errors.append(f"{fixture_id}: expected valid Trial Score scoring")
    review_mode = ((result.get("validated_review") or {}).get("review_metadata") or {}).get("review_mode")
    if review_mode == "hidden_baseline":
        if scoring.get("reality_check_points") is not None or scoring.get("trial_score") is not None:
            errors.append(f"{fixture_id}: hidden baseline should not produce Reality Check or Trial Score")
        return
    if scoring.get("operational_fit_points") is None:
        errors.append(f"{fixture_id}: expected Operational Fit points")
    if scoring.get("reality_check_points") is None:
        errors.append(f"{fixture_id}: expected Reality Check points")
    if scoring.get("trial_score") is None:
        errors.append(f"{fixture_id}: expected Trial Score")
    if not (result.get("validated_review") or {}).get("operational_fit"):
        errors.append(f"{fixture_id}: expected Operational Fit object in validated review")
    if not (result.get("validated_review") or {}).get("reality_check"):
        errors.append(f"{fixture_id}: expected Reality Check object in validated review")
    pass2 = result.get("validated_participant_narrative") or {}
    if pass2.get("validation_status") != "valid":
        errors.append(f"{fixture_id}: expected valid Pass 2 participant narrative")
    if not (pass2.get("trial_score_narrative") or {}).get("summary"):
        errors.append(f"{fixture_id}: expected Pass 2 Trial Score narrative summary")
    if not (pass2.get("central_tension") or {}).get("summary"):
        errors.append(f"{fixture_id}: expected Pass 2 central tension")
    if scoring.get("reality_check_points") == 0:
        pillar_names = {
            str((item or {}).get("pillar") or "").strip()
            for item in (pass2.get("pillar_reading") or [])
            if isinstance(item, dict)
        }
        movement_reading = str((pass2.get("trial_score_narrative") or {}).get("movement_reading") or "")
        if "Reality Check" in pillar_names:
            errors.append(f"{fixture_id}: neutral Reality Check should not render as a Pass 2 pillar")
        if "Reality Check has none qualitative importance" in movement_reading:
            errors.append(f"{fixture_id}: neutral Reality Check should not produce awkward Pass 2 wording")
    if "trial_score" in (result.get("participant_narrative") or {}):
        errors.append(f"{fixture_id}: Pass 2 provider object should not return app-owned trial_score")


def _check_failure_paths(errors: list[str]) -> None:
    fixture = next(item for item in get_contract_fixtures() if item["expected_behavior"].get("review_needed") is True)
    packet = build_review_packet_from_fixture(fixture)

    provider_failure = review_packet_with_mock(packet, failure_mode=FAILURE_PROVIDER_ERROR)
    if provider_failure.get("status") != "provider_error":
        errors.append("provider failure mode did not return provider_error status")
    if provider_failure.get("scoring", {}).get("reality_check_points") is not None:
        errors.append("provider failure should not return Reality Check")

    malformed = review_packet_with_mock(packet, failure_mode=FAILURE_MALFORMED_JSON)
    if malformed.get("status") != "malformed_response":
        errors.append("malformed failure mode did not return malformed_response status")
    if malformed.get("scoring", {}).get("validation_status") == "valid":
        errors.append("malformed failure mode should not validate as valid")
    if malformed.get("scoring", {}).get("reality_check_points") is not None:
        errors.append("malformed failure mode should not return Reality Check")
    if malformed.get("scoring", {}).get("trial_score") is not None:
        errors.append("malformed failure mode should not return Trial Score")

    unmatched_packet = dict(packet)
    unmatched_packet["input_hash"] = "unmatched"
    unmatched = review_packet_with_mock(unmatched_packet)
    if unmatched.get("status") != "no_fixture_match":
        errors.append("unmatched packet should return no_fixture_match status")


def main() -> int:
    errors: list[str] = []
    for fixture in get_contract_fixtures():
        _check_fixture(fixture, errors)
    _check_failure_paths(errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated deterministic mock reviewer behavior against contract fixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
