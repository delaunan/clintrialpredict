"""Thin provider boundary for narrative Quality Review calls.

The application owns packet construction, validation/scoring, caching, storage,
and UI rendering. This module owns provider invocation and response
normalization only.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from src.narratives.alignment import unresolved_clarification_issues
from src.narratives.mock_reviewer import review_packet_with_mock
from src.narratives.provider_config import (
    NarrativeProviderConfig,
    PROVIDER_GEMINI,
    PROVIDER_MOCK,
    PROVIDER_OPENAI,
    ProviderSettings,
)
from src.narratives.scoring import DOMAIN_RATING_POINTS, PARTICIPANT_REVIEW_KEYS, validate_and_score_review

MOCK_MODEL_NAME = "fixture_hash_mock_v1"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
FAILURE_UNSUPPORTED_PROVIDER = "unsupported_provider"
FAILURE_PROVIDER_UNAVAILABLE = "provider_unavailable"
FAILURE_PROVIDER_ERROR = "provider_error"
FAILURE_INCOMPLETE_RESPONSE = "incomplete_response"
FAILURE_MALFORMED_RESPONSE = "malformed_response"
STATUS_CLARIFICATION_NEEDED = "clarification_needed"
STATUS_REVIEWED = "reviewed"

REQUIRED_DOMAIN_NAMES = (
    "change_integrity",
    "development_question_fit",
    "endpoint_and_comparator_logic",
    "operational_scale_fit",
    "population_relevance",
    "scientific_rigor",
    "text_consistency",
)


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


def _failure_result(
    packet: dict[str, Any],
    *,
    provider: str,
    model_name: str | None,
    status: str,
    message: str,
    provider_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "review_needed": True,
        "reuse_previous_review": False,
        "provider": provider,
        "model_name": model_name,
        "provider_metadata": dict(provider_metadata or {}),
        "status": status,
        "failure_reason": message,
        "review": None,
        "validated_review": None,
        "scoring": _unavailable_scoring(packet, message),
    }


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _provider_prompt(packet: dict[str, Any]) -> str:
    participant_keys = ", ".join(sorted(PARTICIPANT_REVIEW_KEYS))
    domain_names = ", ".join(REQUIRED_DOMAIN_NAMES)
    rating_contract = {
        domain: sorted(DOMAIN_RATING_POINTS[domain])
        for domain in REQUIRED_DOMAIN_NAMES
    }
    packet_json = json.dumps(packet, sort_keys=True, separators=(",", ":"), default=str)
    return (
        "You are reviewing a clinical trial design simulation packet. "
        "Return only one valid compact JSON object. Do not include markdown or prose outside JSON. "
        "Do not calculate Quality Adjustment or Final Candidate Score; the application calculates those. "
        "Use field_changes to identify participant edits and xgboost_impact_changes to understand model movement, "
        "but do not treat model movement as proof of clinical causality. "
        f"The JSON must include quality_review_domains for exactly these domains: {domain_names}. "
        "Each domain must contain rating, rationale, and evidence_fields. "
        f"Allowed ratings by domain: {json.dumps(rating_contract, sort_keys=True, separators=(',', ':'))}. "
        f"The JSON must include participant_review with these string keys: {participant_keys}. "
        "Also include score_movement_review, continuity, and trace objects. "
        f"Packet JSON: {packet_json}"
    )


def _openai_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text

    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _real_provider_metadata(
    *,
    provider: str,
    config: NarrativeProviderConfig,
    applied_generation_controls: dict[str, Any],
) -> dict[str, Any]:
    return {
        "configured_generation_controls": {
            "temperature": config.temperature,
            "seed": config.seed,
            "openai_reasoning_effort": config.openai_reasoning_effort,
            "max_output_tokens": config.max_output_tokens,
            "timeout_seconds": config.timeout_seconds,
            "max_retries": config.max_retries,
        },
        "applied_generation_controls": applied_generation_controls,
        "real_provider": provider,
    }


def _score_provider_review(
    packet: dict[str, Any],
    *,
    provider: str,
    model_name: str,
    review: dict[str, Any],
    provider_metadata: dict[str, Any],
) -> dict[str, Any]:
    scored = validate_and_score_review(packet, review)
    scoring = scored["scoring"]
    is_valid = scoring.get("validation_status") == "valid" and scoring.get("quality_adjustment") is not None
    return {
        "review_needed": True,
        "reuse_previous_review": False,
        "provider": provider,
        "model_name": model_name,
        "provider_metadata": provider_metadata,
        "status": STATUS_REVIEWED if is_valid else FAILURE_MALFORMED_RESPONSE,
        "failure_reason": None if is_valid else "Provider review JSON did not satisfy the Quality Review contract.",
        "review": review,
        "validated_review": scored["validated_review"],
        "scoring": scoring,
    }


def _max_attempts(config: NarrativeProviderConfig) -> int:
    return max(1, int(config.max_retries) + 1)


def _call_openai_provider(
    packet: dict[str, Any],
    *,
    config: NarrativeProviderConfig,
    settings: ProviderSettings,
) -> dict[str, Any]:
    applied_controls = {
        "max_output_tokens": config.max_output_tokens,
        "reasoning_effort": config.openai_reasoning_effort,
        "temperature": None,
        "seed": None,
    }
    metadata = _real_provider_metadata(
        provider=PROVIDER_OPENAI,
        config=config,
        applied_generation_controls=applied_controls,
    )
    payload = {
        "model": settings.model,
        "input": _provider_prompt(packet),
        "max_output_tokens": config.max_output_tokens,
        "reasoning": {"effort": config.openai_reasoning_effort},
        "text": {"format": {"type": "json_object"}},
    }
    response_payload = None
    last_error = None
    for attempt in range(1, _max_attempts(config) + 1):
        metadata["attempts"] = attempt
        try:
            response = requests.post(
                OPENAI_RESPONSES_URL,
                headers={
                    "Authorization": f"Bearer {settings.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            response_payload = response.json()
            break
        except Exception as exc:
            last_error = exc
    if response_payload is None:
        return _failure_result(
            packet,
            provider=PROVIDER_OPENAI,
            model_name=settings.model,
            status=FAILURE_PROVIDER_ERROR,
            message=f"OpenAI provider call failed: {last_error.__class__.__name__ if last_error else 'unknown'}",
            provider_metadata=metadata,
        )
    if response_payload.get("status") == "incomplete":
        incomplete_details = response_payload.get("incomplete_details")
        reason = None
        if isinstance(incomplete_details, dict):
            reason = incomplete_details.get("reason")
        return _failure_result(
            packet,
            provider=PROVIDER_OPENAI,
            model_name=settings.model,
            status=FAILURE_INCOMPLETE_RESPONSE,
            message=f"OpenAI provider response was incomplete: {reason or 'unknown'}",
            provider_metadata=metadata,
        )

    review = _parse_json_object(_openai_response_text(response_payload))
    if review is None:
        return _failure_result(
            packet,
            provider=PROVIDER_OPENAI,
            model_name=settings.model,
            status=FAILURE_MALFORMED_RESPONSE,
            message="OpenAI provider response was not a JSON object.",
            provider_metadata=metadata,
        )
    return _score_provider_review(
        packet,
        provider=PROVIDER_OPENAI,
        model_name=settings.model,
        review=review,
        provider_metadata=metadata,
    )


def _call_gemini_provider(
    packet: dict[str, Any],
    *,
    config: NarrativeProviderConfig,
    settings: ProviderSettings,
) -> dict[str, Any]:
    applied_controls = {
        "max_output_tokens": config.max_output_tokens,
        "temperature": config.temperature,
        "seed": config.seed,
        "openai_reasoning_effort": None,
    }
    metadata = _real_provider_metadata(
        provider=PROVIDER_GEMINI,
        config=config,
        applied_generation_controls=applied_controls,
    )
    response = None
    last_error = None
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        last_error = exc

    if last_error is None:
        generation_config = types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            seed=config.seed,
            response_mime_type="application/json",
        )
        client = genai.Client(api_key=settings.api_key)
        for attempt in range(1, _max_attempts(config) + 1):
            metadata["attempts"] = attempt
            try:
                response = client.models.generate_content(
                    model=settings.model,
                    contents=_provider_prompt(packet),
                    config=generation_config,
                )
                break
            except Exception as exc:
                last_error = exc

    if response is None:
        return _failure_result(
            packet,
            provider=PROVIDER_GEMINI,
            model_name=settings.model,
            status=FAILURE_PROVIDER_ERROR,
            message=f"Gemini provider call failed: {last_error.__class__.__name__ if last_error else 'unknown'}",
            provider_metadata=metadata,
        )

    parsed_payload = getattr(response, "parsed", None)
    review = parsed_payload if isinstance(parsed_payload, dict) else _parse_json_object(getattr(response, "text", ""))
    if review is None:
        return _failure_result(
            packet,
            provider=PROVIDER_GEMINI,
            model_name=settings.model,
            status=FAILURE_MALFORMED_RESPONSE,
            message="Gemini provider response was not a JSON object.",
            provider_metadata=metadata,
        )
    return _score_provider_review(
        packet,
        provider=PROVIDER_GEMINI,
        model_name=settings.model,
        review=review,
        provider_metadata=metadata,
    )


def review_packet_with_provider(
    packet: dict[str, Any],
    *,
    provider: str = PROVIDER_MOCK,
    model_name: str | None = None,
    failure_mode: str | None = None,
    config: NarrativeProviderConfig | None = None,
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

    if provider in {PROVIDER_OPENAI, PROVIDER_GEMINI}:
        if config is None:
            return _failure_result(
                packet,
                provider=provider,
                model_name=model_name,
                status=FAILURE_PROVIDER_UNAVAILABLE,
                message=f"Narrative provider config is required for {provider}.",
            )
        settings = config.provider_settings(provider)
        if settings is None or not settings.has_api_key:
            return _failure_result(
                packet,
                provider=provider,
                model_name=model_name or (settings.model if settings else None),
                status=FAILURE_PROVIDER_UNAVAILABLE,
                message=f"{provider} provider is missing an API key.",
            )
        if provider == PROVIDER_OPENAI:
            return _call_openai_provider(packet, config=config, settings=settings)
        return _call_gemini_provider(packet, config=config, settings=settings)

    message = f"Unsupported narrative provider: {provider}"
    return _failure_result(
        packet,
        provider=provider,
        model_name=model_name,
        status=FAILURE_UNSUPPORTED_PROVIDER,
        message=message,
    )


def review_packet_with_provider_chain(
    packet: dict[str, Any],
    *,
    config: NarrativeProviderConfig,
) -> dict[str, Any]:
    """Try configured primary provider, then one fallback for provider failures."""
    primary = review_packet_with_provider(packet, provider=config.provider, config=config)
    if primary.get("status") not in {FAILURE_PROVIDER_ERROR, FAILURE_PROVIDER_UNAVAILABLE, FAILURE_INCOMPLETE_RESPONSE}:
        return primary

    fallback = config.fallback_provider
    if not fallback:
        return primary

    fallback_result = review_packet_with_provider(packet, provider=fallback, config=config)
    fallback_metadata = dict(fallback_result.get("provider_metadata") or {})
    fallback_metadata["fallback_after"] = {
        "provider": primary.get("provider"),
        "model_name": primary.get("model_name"),
        "status": primary.get("status"),
        "failure_reason": primary.get("failure_reason"),
    }
    fallback_result["provider_metadata"] = fallback_metadata
    return fallback_result
