"""Thin provider boundary for narrative Quality Review calls.

The application owns packet construction, validation/scoring, caching, storage,
and UI rendering. This module owns only provider selection and response
normalization. The only implemented provider is the deterministic mock.
"""

from __future__ import annotations

from typing import Any

from src.narratives.alignment import unresolved_clarification_issues
from src.narratives.mock_reviewer import review_packet_with_mock

PROVIDER_MOCK = "mock"
MOCK_MODEL_NAME = "fixture_hash_mock_v1"
FAILURE_UNSUPPORTED_PROVIDER = "unsupported_provider"
STATUS_CLARIFICATION_NEEDED = "clarification_needed"


def _unavailable_scoring(packet: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "validation_status": "unavailable",
        "validation_errors": [message],
        "quality_adjustment": None,
        "final_candidate_score": None,
        "quality_assessment": {},
        "input_hash": packet.get("input_hash"),
    }


def _normalize_provider_result(
    result: dict[str, Any],
    *,
    provider: str,
    model_name: str,
    provider_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = dict(result)
    normalized["provider"] = provider
    normalized["model_name"] = model_name
    normalized["provider_metadata"] = dict(provider_metadata or {})
    return normalized


def review_packet_with_provider(
    packet: dict[str, Any],
    *,
    provider: str = PROVIDER_MOCK,
    model_name: str | None = None,
    failure_mode: str | None = None,
) -> dict[str, Any]:
    """Invoke a narrative provider and return normalized review JSON fields."""
    provider = str(provider or PROVIDER_MOCK).strip().lower()
    issues = unresolved_clarification_issues(packet)
    if issues:
        message = "Quality Review needs user clarification for apparent structured/text mismatch."
        resolved_model_name = model_name or (MOCK_MODEL_NAME if provider == PROVIDER_MOCK else None)
        return {
            "review_needed": True,
            "reuse_previous_review": False,
            "provider": provider,
            "model_name": resolved_model_name,
            "provider_metadata": {},
            "status": STATUS_CLARIFICATION_NEEDED,
            "failure_reason": message,
            "review": None,
            "validated_review": None,
            "clarification_issues": issues,
            "scoring": _unavailable_scoring(packet, message),
        }

    if provider == PROVIDER_MOCK:
        return _normalize_provider_result(
            review_packet_with_mock(packet, failure_mode=failure_mode),
            provider=PROVIDER_MOCK,
            model_name=model_name or MOCK_MODEL_NAME,
            provider_metadata={"deterministic": True},
        )

    message = f"Unsupported narrative provider: {provider}"
    return {
        "review_needed": True,
        "reuse_previous_review": False,
        "provider": provider,
        "model_name": model_name,
        "provider_metadata": {},
        "status": FAILURE_UNSUPPORTED_PROVIDER,
        "failure_reason": message,
        "review": None,
        "validated_review": None,
        "scoring": _unavailable_scoring(packet, message),
    }
