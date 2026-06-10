"""Deterministic mock reviewer for narrative V2 development.

This module is a stand-in for a future LLM provider. It returns fixture-backed
JSON so packet building, validation, scoring, no-op behavior, and failure
handling can be exercised before real provider integration.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.narratives.contract_fixtures import get_contract_fixtures
from src.narratives.packet_builder import build_review_packet_from_fixture
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
    if not input_hash:
        return None
    return _fixture_registry().get(str(input_hash))


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
