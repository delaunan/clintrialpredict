#!/usr/bin/env python
"""Validate the thin narrative provider boundary."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.narratives.provider as provider_module  # noqa: E402
from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402
from src.narratives.provider import (  # noqa: E402
    FAILURE_MALFORMED_RESPONSE,
    FAILURE_PROVIDER_UNAVAILABLE,
    FAILURE_UNSUPPORTED_PROVIDER,
    GEMINI_MALFORMED_JSON_RETRY_ATTEMPTS,
    GEMINI_MIN_SCHEMA_OUTPUT_TOKENS,
    GEMINI_PRIMARY_THINKING_LEVEL,
    GEMINI_RETRY_OUTPUT_TOKENS,
    GEMINI_RETRY_THINKING_LEVEL,
    MOCK_MODEL_NAME,
    PROVIDER_MOCK,
    PROVIDER_VALIDATION_RETRY_ATTEMPTS,
    _gemini_http_options,
    _record_gemini_response_metadata,
    _score_provider_review,
    review_packet_with_provider_chain,
    review_packet_with_provider,
)
from src.narratives.provider_config import load_narrative_provider_config  # noqa: E402


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _check_openai_validation_retry(packet: dict, fixture: dict, errors: list[str]) -> None:
    config = load_narrative_provider_config({
        "NARRATIVE_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_NARRATIVE_MODEL": "test-openai-model",
        "NARRATIVE_LLM_MAX_RETRIES": "0",
    })
    original_post = provider_module.requests.post

    def run_case(first_payload: dict, expected_reason_fragment: str) -> dict:
        calls = {"count": 0}

        def fake_post(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return _FakeResponse(first_payload)
            return _FakeResponse({"output_text": provider_module.json.dumps(fixture["mock_review"])})

        provider_module.requests.post = fake_post
        try:
            result = provider_module.review_packet_with_provider(packet, provider="openai", config=config)
        finally:
            provider_module.requests.post = original_post
        if calls["count"] != 2:
            errors.append("OpenAI validation retry should make exactly one retry call")
        metadata = result.get("provider_metadata") or {}
        if metadata.get("validation_retry_attempts") != 1:
            errors.append("OpenAI validation retry should record one validation_retry_attempt")
        if expected_reason_fragment not in str(metadata.get("validation_retry_reason")):
            errors.append("OpenAI validation retry should record the retry reason")
        return result

    non_json_result = run_case({"output_text": "not json"}, "not a JSON object")
    if non_json_result.get("status") != provider_module.STATUS_REVIEWED:
        errors.append("OpenAI non-JSON response should recover when validation retry returns valid review")
    if non_json_result.get("scoring", {}).get("design_confidence") != fixture["expected_behavior"]["expected_design_confidence"]:
        errors.append("OpenAI non-JSON retry should preserve valid retry scoring")

    invalid_json_result = run_case(
        {"output_text": provider_module.json.dumps({"design_confidence_subcategories": {}})},
        "Scenario Review contract",
    )
    if invalid_json_result.get("status") != provider_module.STATUS_REVIEWED:
        errors.append("OpenAI invalid JSON contract response should recover when validation retry returns valid review")
    if invalid_json_result.get("scoring", {}).get("design_confidence") != fixture["expected_behavior"]["expected_design_confidence"]:
        errors.append("OpenAI invalid-contract retry should preserve valid retry scoring")


def main() -> int:
    errors: list[str] = []
    fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "operational_only_ambitious_enrollment_v2"
    )
    packet = build_review_packet_from_fixture(fixture)

    mock_result = review_packet_with_provider(packet, provider=PROVIDER_MOCK)
    if mock_result.get("provider") != PROVIDER_MOCK:
        errors.append("mock provider result did not preserve provider name")
    if mock_result.get("model_name") != MOCK_MODEL_NAME:
        errors.append("mock provider result did not set normalized model_name")
    if mock_result.get("provider_metadata", {}).get("deterministic") is not True:
        errors.append("mock provider result did not expose deterministic metadata")
    if mock_result.get("scoring", {}).get("design_confidence") != fixture["expected_behavior"]["expected_design_confidence"]:
        errors.append("mock provider did not preserve scoring result")

    baseline_fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "baseline_hidden_review_v2"
    )
    baseline_packet = build_review_packet_from_fixture(baseline_fixture)
    baseline_result = review_packet_with_provider(baseline_packet, provider=PROVIDER_MOCK)
    if baseline_result.get("status") != "reviewed":
        errors.append("provider should review hidden baseline packet through the normal Scenario Review path")

    context_fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "endpoint_text_contradiction_v2"
    )
    context_packet = build_review_packet_from_fixture(context_fixture)
    context_result = review_packet_with_provider(context_packet, provider=PROVIDER_MOCK)
    if context_result.get("status") != "reviewed":
        errors.append("provider should review structured/text context fixture without a clarification gate")
    if context_result.get("scoring", {}).get("design_confidence") != context_fixture["expected_behavior"]["expected_design_confidence"]:
        errors.append("structured/text context fixture did not preserve scoring result")

    unsupported = review_packet_with_provider(packet, provider="not_configured")
    if unsupported.get("status") != FAILURE_UNSUPPORTED_PROVIDER:
        errors.append("unsupported provider should return unsupported_provider status")
    if unsupported.get("scoring", {}).get("design_confidence") is not None:
        errors.append("unsupported provider should not return Design Confidence")
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

    gemini_runtime_config = load_narrative_provider_config({
        "NARRATIVE_LLM_PROVIDER": "gemini",
        "NARRATIVE_LLM_FALLBACK_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test-key",
        "NARRATIVE_LLM_MAX_OUTPUT_TOKENS": "2500",
        "NARRATIVE_LLM_TIMEOUT_SECONDS": "45",
        "NARRATIVE_LLM_MAX_RETRIES": "0",
    })
    if GEMINI_MIN_SCHEMA_OUTPUT_TOKENS < 12000:
        errors.append("Gemini schema output budget should leave margin for longer future reviews")
    if GEMINI_PRIMARY_THINKING_LEVEL != "medium":
        errors.append("Gemini primary thinking level should be medium for clinical-trial coherence reviews")
    if GEMINI_RETRY_THINKING_LEVEL != "low":
        errors.append("Gemini malformed/MAX_TOKENS retry should lower thinking level for completion reliability")
    if GEMINI_RETRY_OUTPUT_TOKENS < 16000:
        errors.append("Gemini retry output budget should be at least 16000 tokens")
    if GEMINI_MALFORMED_JSON_RETRY_ATTEMPTS != 1:
        errors.append("Gemini malformed JSON retry should stay bounded to one explicit retry")
    if PROVIDER_VALIDATION_RETRY_ATTEMPTS != 1:
        errors.append("provider validation retry should stay bounded to one explicit retry")
    fake_usage = type("FakeUsage", (), {
        "prompt_token_count": 100,
        "candidates_token_count": 40,
        "thoughts_token_count": 25,
        "cached_content_token_count": None,
        "total_token_count": 165,
    })()
    fake_candidate = type("FakeCandidate", (), {
        "finish_reason": "STOP",
        "safety_ratings": [object()],
    })()
    fake_response = type("FakeResponse", (), {
        "usage_metadata": fake_usage,
        "candidates": [fake_candidate],
    })()
    fake_metadata = {}
    _record_gemini_response_metadata(fake_metadata, fake_response)
    if fake_metadata.get("usage_metadata", {}).get("thoughts_token_count") != 25:
        errors.append("Gemini provider metadata should include thoughts token count when available")
    if fake_metadata.get("finish_metadata", {}).get("finish_reason") != "STOP":
        errors.append("Gemini provider metadata should include finish reason when available")
    try:
        from google.genai import types
        gemini_http_options = _gemini_http_options(gemini_runtime_config, types)
        if gemini_http_options.timeout != 45000:
            errors.append("gemini provider should convert timeout seconds to SDK milliseconds")
        if gemini_http_options.retry_options.attempts != 1:
            errors.append("gemini provider should disable SDK retries when app max_retries is 0")
    except Exception as exc:
        errors.append(f"gemini SDK HTTP option check failed: {exc.__class__.__name__}")

    invalid_real_review = _score_provider_review(
        packet,
        provider="openai",
        model_name="test-model",
        review={"design_confidence_subcategories": {}},
        provider_metadata={},
    )
    if invalid_real_review.get("status") != FAILURE_MALFORMED_RESPONSE:
        errors.append("contract-invalid real provider review should be malformed_response")
    if invalid_real_review.get("scoring", {}).get("design_confidence") is not None:
        errors.append("contract-invalid real provider review should not return Design Confidence")

    review_with_app_score = {
        **fixture["mock_review"],
        "design_confidence": 99,
        "total_scenario_score": 99,
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
    if app_score_result.get("scoring", {}).get("design_confidence") is not None:
        errors.append("provider-returned app score field should suppress Design Confidence")
    if app_score_result.get("scoring", {}).get("total_scenario_score") is not None:
        errors.append("provider-returned app score field should suppress Total Scenario Score")

    _check_openai_validation_retry(packet, fixture, errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated narrative provider normalization and unsupported-provider failure behavior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
