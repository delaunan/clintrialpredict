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
    FAILURE_MALFORMED_RESPONSE,
    FAILURE_PROVIDER_UNAVAILABLE,
    FAILURE_UNSUPPORTED_PROVIDER,
    MOCK_MODEL_NAME,
    PROVIDER_MOCK,
    STATUS_CLARIFICATION_NEEDED,
    _score_provider_review,
    review_packet_with_provider_chain,
    review_packet_with_provider,
)
from src.narratives.provider_config import load_narrative_provider_config  # noqa: E402


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

    baseline_fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "baseline_hidden_review_v1"
    )
    baseline_packet = build_review_packet_from_fixture(baseline_fixture)
    baseline_result = review_packet_with_provider(baseline_packet, provider=PROVIDER_MOCK)
    if baseline_result.get("status") == STATUS_CLARIFICATION_NEEDED:
        errors.append("provider should not require participant clarification for hidden baseline review")

    clarification_fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "endpoint_structure_text_alignment_requires_clarification_v1"
    )
    clarification_packet = build_review_packet_from_fixture(clarification_fixture)
    clarification_result = review_packet_with_provider(clarification_packet, provider=PROVIDER_MOCK)
    if clarification_result.get("status") != STATUS_CLARIFICATION_NEEDED:
        errors.append("provider should pause unresolved structured/text mismatch before review")
    if clarification_result.get("review") is not None:
        errors.append("clarification-needed provider result should not return review JSON")
    if clarification_result.get("scoring", {}).get("quality_adjustment") is not None:
        errors.append("clarification-needed provider result should not return Quality Adjustment")

    explained_fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "endpoint_structure_text_alignment_explained_v1"
    )
    explained_packet = build_review_packet_from_fixture(explained_fixture)
    explained_result = review_packet_with_provider(explained_packet, provider=PROVIDER_MOCK)
    if explained_result.get("status") != "reviewed":
        errors.append("provider should continue review when alignment issue has user explanation")
    if explained_result.get("scoring", {}).get("quality_adjustment") != explained_fixture["expected_behavior"]["expected_quality_adjustment"]:
        errors.append("explained alignment fixture did not preserve scoring result")

    unsupported = review_packet_with_provider(packet, provider="not_configured")
    if unsupported.get("status") != FAILURE_UNSUPPORTED_PROVIDER:
        errors.append("unsupported provider should return unsupported_provider status")
    if unsupported.get("scoring", {}).get("quality_adjustment") is not None:
        errors.append("unsupported provider should not return a Quality Adjustment")
    if unsupported.get("review") is not None:
        errors.append("unsupported provider should not return review JSON")

    openai_without_config = review_packet_with_provider(packet, provider="openai")
    if openai_without_config.get("status") != FAILURE_PROVIDER_UNAVAILABLE:
        errors.append("openai provider without config should be unavailable")

    missing_key_config = load_narrative_provider_config({
        "NARRATIVE_LLM_PROVIDER": "openai",
        "NARRATIVE_LLM_FALLBACK_PROVIDER": "mock",
    })
    missing_key_result = review_packet_with_provider(packet, provider="openai", config=missing_key_config)
    if missing_key_result.get("status") != FAILURE_PROVIDER_UNAVAILABLE:
        errors.append("openai provider without API key should be unavailable")

    fallback_result = review_packet_with_provider_chain(packet, config=missing_key_config)
    if fallback_result.get("provider") != PROVIDER_MOCK:
        errors.append("provider chain should fall back to mock after unavailable openai")
    if fallback_result.get("provider_metadata", {}).get("fallback_after", {}).get("provider") != "openai":
        errors.append("fallback provider result should trace primary provider failure")

    invalid_real_review = _score_provider_review(
        packet,
        provider="openai",
        model_name="test-model",
        review={"quality_review_domains": {}},
        provider_metadata={},
    )
    if invalid_real_review.get("status") != FAILURE_MALFORMED_RESPONSE:
        errors.append("contract-invalid real provider review should be malformed_response")
    if invalid_real_review.get("scoring", {}).get("quality_adjustment") is not None:
        errors.append("contract-invalid real provider review should not return Quality Adjustment")

    review_with_app_score = {
        **fixture["mock_review"],
        "quality_adjustment": 99,
    }
    app_score_result = _score_provider_review(
        packet,
        provider="openai",
        model_name="test-model",
        review=review_with_app_score,
        provider_metadata={},
    )
    if app_score_result.get("status") != FAILURE_MALFORMED_RESPONSE:
        errors.append("provider-returned app score field should make result malformed_response")
    if app_score_result.get("scoring", {}).get("quality_adjustment") is not None:
        errors.append("provider-returned app score field should suppress Quality Adjustment")
    if app_score_result.get("scoring", {}).get("final_candidate_score") is not None:
        errors.append("provider-returned app score field should suppress Final Candidate Score")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated narrative provider normalization and unsupported-provider failure behavior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
