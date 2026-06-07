"""Thin provider boundary for narrative Quality Review calls.

The application owns packet construction, validation/scoring, caching, storage,
and UI rendering. This module owns provider invocation and response
normalization only.
"""

from __future__ import annotations

import json
import time
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
from src.narratives.prompt_builder import (
    PROMPT_TEMPLATE_VERSION,
    RESPONSE_SCHEMA_VERSION,
    build_provider_prompt,
    infer_prompt_mode,
)
from src.narratives.scoring import validate_and_score_review

MOCK_MODEL_NAME = "fixture_hash_mock_v1"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
FAILURE_UNSUPPORTED_PROVIDER = "unsupported_provider"
FAILURE_PROVIDER_UNAVAILABLE = "provider_unavailable"
FAILURE_PROVIDER_ERROR = "provider_error"
FAILURE_INCOMPLETE_RESPONSE = "incomplete_response"
FAILURE_MALFORMED_RESPONSE = "malformed_response"
STATUS_CLARIFICATION_NEEDED = "clarification_needed"
STATUS_REVIEWED = "reviewed"

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
    prompt_mode: str,
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
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "prompt_mode": prompt_mode,
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
    if not is_valid:
        scoring = {
            **scoring,
            "quality_adjustment": None,
            "final_candidate_score": None,
            "quality_assessment": {},
        }
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


def _gemini_http_options(config: NarrativeProviderConfig, types_module: Any) -> Any:
    """Build Gemini SDK HTTP controls from the app-owned runtime config."""
    return types_module.HttpOptions(
        timeout=int(config.timeout_seconds) * 1000,
        retry_options=types_module.HttpRetryOptions(attempts=_max_attempts(config)),
    )


def _call_openai_provider(
    packet: dict[str, Any],
    *,
    config: NarrativeProviderConfig,
    settings: ProviderSettings,
) -> dict[str, Any]:
    prompt_mode = infer_prompt_mode(packet)
    prompt = build_provider_prompt(packet, prompt_mode=prompt_mode)
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
        prompt_mode=prompt_mode,
    )
    payload = {
        "model": settings.model,
        "input": prompt,
        "max_output_tokens": config.max_output_tokens,
        "reasoning": {"effort": config.openai_reasoning_effort},
        "text": {"format": {"type": "json_object"}},
    }
    response_payload = None
    last_error = None
    started_at = time.monotonic()
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
            metadata["http_status"] = response.status_code
            break
        except Exception as exc:
            last_error = exc
            metadata["last_error_type"] = exc.__class__.__name__
    metadata["latency_ms"] = int(round((time.monotonic() - started_at) * 1000))
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

    response_text = _openai_response_text(response_payload)
    metadata["response_status"] = response_payload.get("status")
    metadata["response_text_length"] = len(response_text)
    review = _parse_json_object(response_text)
    metadata["parsed_json_object"] = isinstance(review, dict)
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
    prompt_mode = infer_prompt_mode(packet)
    prompt = build_provider_prompt(packet, prompt_mode=prompt_mode)
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
        prompt_mode=prompt_mode,
    )
    response = None
    last_error = None
    started_at = time.monotonic()
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
        http_options = _gemini_http_options(config, types)
        client = genai.Client(api_key=settings.api_key, http_options=http_options)
        for attempt in range(1, _max_attempts(config) + 1):
            metadata["attempts"] = attempt
            try:
                response = client.models.generate_content(
                    model=settings.model,
                    contents=prompt,
                    config=generation_config,
                )
                break
            except Exception as exc:
                last_error = exc
                metadata["last_error_type"] = exc.__class__.__name__
    metadata["latency_ms"] = int(round((time.monotonic() - started_at) * 1000))

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
    response_text = str(getattr(response, "text", "") or "")
    metadata["parsed_payload_type"] = type(parsed_payload).__name__ if parsed_payload is not None else None
    metadata["response_text_length"] = len(response_text)
    review = parsed_payload if isinstance(parsed_payload, dict) else _parse_json_object(response_text)
    metadata["parsed_json_object"] = isinstance(review, dict)
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
