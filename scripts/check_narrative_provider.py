#!/usr/bin/env python
"""Validate the thin narrative provider boundary."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402
from src.narratives.provider import (  # noqa: E402
    FAILURE_UNSUPPORTED_PROVIDER,
    MOCK_MODEL_NAME,
    PROVIDER_MOCK,
    review_packet_with_provider,
)


def main() -> int:
    errors: list[str] = []
    fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "operational_only_ambitious_enrollment_v1"
    )
    packet = build_review_packet_from_fixture(fixture)

    mock_result = review_packet_with_provider(packet, provider=PROVIDER_MOCK)
    if mock_result.get("provider") != PROVIDER_MOCK:
        errors.append("mock provider result did not preserve provider name")
    if mock_result.get("model_name") != MOCK_MODEL_NAME:
        errors.append("mock provider result did not set normalized model_name")
    if mock_result.get("provider_metadata", {}).get("deterministic") is not True:
        errors.append("mock provider result did not expose deterministic metadata")
    if mock_result.get("scoring", {}).get("quality_adjustment") != fixture["expected_behavior"]["expected_quality_adjustment"]:
        errors.append("mock provider did not preserve scoring result")

    unsupported = review_packet_with_provider(packet, provider="not_configured")
    if unsupported.get("status") != FAILURE_UNSUPPORTED_PROVIDER:
        errors.append("unsupported provider should return unsupported_provider status")
    if unsupported.get("scoring", {}).get("quality_adjustment") is not None:
        errors.append("unsupported provider should not return a Quality Adjustment")
    if unsupported.get("review") is not None:
        errors.append("unsupported provider should not return review JSON")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated narrative provider normalization and unsupported-provider failure behavior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
