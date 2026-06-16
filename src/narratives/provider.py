"""Thin provider boundary for narrative Scenario Review calls.

The application owns packet construction, validation/scoring, caching, storage,
and UI rendering. This module owns provider invocation and response
normalization only.
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests
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
    gemini_response_schema,
    infer_prompt_mode,
)
from src.narratives.scoring import validate_and_score_review

MOCK_MODEL_NAME = "fixture_hash_mock_v1"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
GEMINI_MIN_SCHEMA_OUTPUT_TOKENS = 12000
GEMINI_PRIMARY_THINKING_LEVEL = "high"
GEMINI_RETRY_THINKING_LEVEL = "low"
GEMINI_RETRY_OUTPUT_TOKENS = 16000
GEMINI_MALFORMED_JSON_RETRY_ATTEMPTS = 1
PROVIDER_VALIDATION_RETRY_ATTEMPTS = 1
FAILURE_UNSUPPORTED_PROVIDER = "unsupported_provider"
FAILURE_PROVIDER_UNAVAILABLE = "provider_unavailable"
FAILURE_PROVIDER_ERROR = "provider_error"
FAILURE_INCOMPLETE_RESPONSE = "incomplete_response"
FAILURE_MALFORMED_RESPONSE = "malformed_response"
STATUS_REVIEWED = "reviewed"

def _unavailable_scoring(packet: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "validation_status": "unavailable",
        "validation_errors": [message],
        "strategic_review": None,
        "trial_score": None,
        "strategic_review_assessment": {},
        "design_confidence": None,
        "total_scenario_score": None,
        "design_confidence_assessment": {},
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


def _metadata_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    enum_name = getattr(value, "name", None)
    if enum_name:
        return str(enum_name)
    return str(value)


def _gemini_usage_metadata(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    fields = (
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
        "total_token_count",
    )
    return {
        field: _metadata_value(getattr(usage, field, None))
        for field in fields
        if getattr(usage, field, None) is not None
    }


def _gemini_finish_metadata(response: Any) -> dict[str, Any]:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return {}
    first = candidates[0]
    metadata = {
        "finish_reason": _metadata_value(getattr(first, "finish_reason", None)),
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
    safety_ratings = getattr(first, "safety_ratings", None)
    if safety_ratings:
        metadata["safety_rating_count"] = len(safety_ratings)
    return metadata


def _record_gemini_response_metadata(metadata: dict[str, Any], response: Any) -> None:
    usage_metadata = _gemini_usage_metadata(response)
    if usage_metadata:
        metadata["usage_metadata"] = usage_metadata
    finish_metadata = _gemini_finish_metadata(response)
    if finish_metadata:
        metadata["finish_metadata"] = finish_metadata


def _gemini_finish_reason(metadata: dict[str, Any]) -> str | None:
    finish_reason = (metadata.get("finish_metadata") or {}).get("finish_reason")
    return str(finish_reason) if finish_reason is not None else None


def _gemini_finished_max_tokens(metadata: dict[str, Any]) -> bool:
    finish_reason = _gemini_finish_reason(metadata)
    return bool(finish_reason and finish_reason.upper().endswith("MAX_TOKENS"))


def _gemini_generation_config_kwargs(
    config: NarrativeProviderConfig,
    *,
    max_output_tokens: int,
) -> dict[str, Any]:
    kwargs = {
        "max_output_tokens": max_output_tokens,
        "seed": config.seed,
        "response_mime_type": "application/json",
        "response_schema": gemini_response_schema(),
    }
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    return kwargs


def _gemini_primary_thinking_level(config: NarrativeProviderConfig) -> str:
    return config.gemini_thinking_level or GEMINI_PRIMARY_THINKING_LEVEL


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
            "gemini_thinking_level": config.gemini_thinking_level,
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
    prompt_mode = str((scored["validated_review"].get("review_metadata") or {}).get("review_mode") or "")
    hidden_baseline = prompt_mode == "hidden_baseline"
    is_valid = scoring.get("validation_status") == "valid" and (
        hidden_baseline or scoring.get("strategic_review") is not None
    )
    if not is_valid:
        scoring = {
            **scoring,
            "strategic_review": None,
            "trial_score": None,
            "strategic_review_assessment": {},
            "design_confidence": None,
            "total_scenario_score": None,
            "design_confidence_assessment": {},
        }
    return {
        "review_needed": True,
        "reuse_previous_review": False,
        "provider": provider,
        "model_name": model_name,
        "provider_metadata": provider_metadata,
        "status": STATUS_REVIEWED if is_valid else FAILURE_MALFORMED_RESPONSE,
        "failure_reason": None if is_valid else "Provider review JSON did not satisfy the Scenario Review contract.",
        "review": review,
        "validated_review": scored["validated_review"],
        "scoring": scoring,
    }


def _score_openai_review_with_validation_retry(
    packet: dict[str, Any],
    *,
    settings: ProviderSettings,
    payload: dict[str, Any],
    config: NarrativeProviderConfig,
    metadata: dict[str, Any],
    initial_review: dict[str, Any] | None,
    retry_reason: str,
) -> dict[str, Any]:
    result = None
    if initial_review is not None:
        result = _score_provider_review(
            packet,
            provider=PROVIDER_OPENAI,
            model_name=settings.model,
            review=initial_review,
            provider_metadata=metadata,
        )
        if result.get("status") != FAILURE_MALFORMED_RESPONSE:
            return result

    metadata["validation_retry_reason"] = result.get("failure_reason") if result else retry_reason
    for retry_attempt in range(1, PROVIDER_VALIDATION_RETRY_ATTEMPTS + 1):
        metadata["validation_retry_attempts"] = retry_attempt
        try:
            retry_response = requests.post(
                OPENAI_RESPONSES_URL,
                headers={
                    "Authorization": f"Bearer {settings.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=config.timeout_seconds,
            )
            retry_response.raise_for_status()
            retry_payload = retry_response.json()
            retry_text = _openai_response_text(retry_payload)
            metadata["validation_retry_http_status"] = retry_response.status_code
            metadata["validation_retry_response_text_length"] = len(retry_text)
            retry_review = _parse_json_object(retry_text)
            metadata["validation_retry_parsed_json_object"] = isinstance(retry_review, dict)
        except Exception as exc:
            metadata["validation_retry_error_type"] = exc.__class__.__name__
            break
        if retry_review is None:
            continue
        result = _score_provider_review(
            packet,
            provider=PROVIDER_OPENAI,
            model_name=settings.model,
            review=retry_review,
            provider_metadata=metadata,
        )
        if result.get("status") != FAILURE_MALFORMED_RESPONSE:
            return result

    if result is not None:
        return result
    return _failure_result(
        packet,
        provider=PROVIDER_OPENAI,
        model_name=settings.model,
        status=FAILURE_MALFORMED_RESPONSE,
        message=retry_reason,
        provider_metadata=metadata,
    )


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
    if isinstance(response_payload.get("usage"), dict):
        metadata["usage_metadata"] = response_payload.get("usage")
    metadata["response_text_length"] = len(response_text)
    review = _parse_json_object(response_text)
    metadata["parsed_json_object"] = isinstance(review, dict)
    return _score_openai_review_with_validation_retry(
        packet,
        settings=settings,
        payload=payload,
        config=config,
        metadata=metadata,
        initial_review=review,
        retry_reason="OpenAI provider response was not a JSON object.",
    )


def _call_gemini_provider(
    packet: dict[str, Any],
    *,
    config: NarrativeProviderConfig,
    settings: ProviderSettings,
) -> dict[str, Any]:
    prompt_mode = infer_prompt_mode(packet)
    prompt = build_provider_prompt(packet, prompt_mode=prompt_mode)
    max_output_tokens = max(int(config.max_output_tokens), GEMINI_MIN_SCHEMA_OUTPUT_TOKENS)
    primary_thinking_level = _gemini_primary_thinking_level(config)
    applied_controls = {
        "max_output_tokens": max_output_tokens,
        "temperature": config.temperature,
        "seed": config.seed,
        "openai_reasoning_effort": None,
        "response_schema": True,
        "thinking_level": primary_thinking_level,
    }
    metadata = _real_provider_metadata(
        provider=PROVIDER_GEMINI,
        config=config,
        applied_generation_controls=applied_controls,
        prompt_mode=prompt_mode,
    )
    response = None
    client = None
    generation_config = None
    last_error = None
    started_at = time.monotonic()
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        last_error = exc

    if last_error is None:
        generation_config = types.GenerateContentConfig(
            **_gemini_generation_config_kwargs(
                config,
                max_output_tokens=max_output_tokens,
            ),
            thinking_config=types.ThinkingConfig(thinking_level=primary_thinking_level),
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
    _record_gemini_response_metadata(metadata, response)
    metadata["parsed_payload_type"] = type(parsed_payload).__name__ if parsed_payload is not None else None
    metadata["response_text_length"] = len(response_text)
    review = parsed_payload if isinstance(parsed_payload, dict) else _parse_json_object(response_text)
    metadata["parsed_json_object"] = isinstance(review, dict)
    should_retry = review is None or _gemini_finished_max_tokens(metadata)
    if should_retry and client is not None and generation_config is not None:
        retry_generation_config = types.GenerateContentConfig(
            **_gemini_generation_config_kwargs(
                config,
                max_output_tokens=max(GEMINI_RETRY_OUTPUT_TOKENS, max_output_tokens),
            ),
            thinking_config=types.ThinkingConfig(thinking_level=GEMINI_RETRY_THINKING_LEVEL),
        )
        metadata["malformed_json_retry_controls"] = {
            "max_output_tokens": max(GEMINI_RETRY_OUTPUT_TOKENS, max_output_tokens),
            "thinking_level": GEMINI_RETRY_THINKING_LEVEL,
        }
        for retry_attempt in range(1, GEMINI_MALFORMED_JSON_RETRY_ATTEMPTS + 1):
            retry_started_at = time.monotonic()
            try:
                response = client.models.generate_content(
                    model=settings.model,
                    contents=prompt,
                    config=retry_generation_config,
                )
            except Exception as exc:
                metadata["malformed_json_retry_error_type"] = exc.__class__.__name__
                break
            metadata["malformed_json_retry_attempts"] = retry_attempt
            metadata["malformed_json_retry_latency_ms"] = int(round((time.monotonic() - retry_started_at) * 1000))
            parsed_payload = getattr(response, "parsed", None)
            response_text = str(getattr(response, "text", "") or "")
            _record_gemini_response_metadata(metadata, response)
            metadata["parsed_payload_type"] = type(parsed_payload).__name__ if parsed_payload is not None else None
            metadata["response_text_length"] = len(response_text)
            review = parsed_payload if isinstance(parsed_payload, dict) else _parse_json_object(response_text)
            metadata["parsed_json_object"] = isinstance(review, dict)
            metadata["latency_ms"] = int(round((time.monotonic() - started_at) * 1000))
            if review is not None:
                break
    if review is None:
        return _failure_result(
            packet,
            provider=PROVIDER_GEMINI,
            model_name=settings.model,
            status=FAILURE_MALFORMED_RESPONSE,
            message="Gemini provider response was not a JSON object.",
            provider_metadata=metadata,
        )
    result = _score_provider_review(
        packet,
        provider=PROVIDER_GEMINI,
        model_name=settings.model,
        review=review,
        provider_metadata=metadata,
    )
    if result.get("status") != FAILURE_MALFORMED_RESPONSE or client is None or generation_config is None:
        return result

    metadata["validation_retry_reason"] = result.get("failure_reason")
    retry_generation_config = types.GenerateContentConfig(
        **_gemini_generation_config_kwargs(
            config,
            max_output_tokens=max(GEMINI_RETRY_OUTPUT_TOKENS, max_output_tokens),
        ),
        thinking_config=types.ThinkingConfig(thinking_level=GEMINI_RETRY_THINKING_LEVEL),
    )
    metadata["validation_retry_controls"] = {
        "max_output_tokens": max(GEMINI_RETRY_OUTPUT_TOKENS, max_output_tokens),
        "thinking_level": GEMINI_RETRY_THINKING_LEVEL,
    }
    for retry_attempt in range(1, PROVIDER_VALIDATION_RETRY_ATTEMPTS + 1):
        retry_started_at = time.monotonic()
        try:
            response = client.models.generate_content(
                model=settings.model,
                contents=prompt,
                config=retry_generation_config,
            )
        except Exception as exc:
            metadata["validation_retry_error_type"] = exc.__class__.__name__
            break
        metadata["validation_retry_attempts"] = retry_attempt
        metadata["validation_retry_latency_ms"] = int(round((time.monotonic() - retry_started_at) * 1000))
        parsed_payload = getattr(response, "parsed", None)
        response_text = str(getattr(response, "text", "") or "")
        _record_gemini_response_metadata(metadata, response)
        metadata["validation_retry_parsed_payload_type"] = type(parsed_payload).__name__ if parsed_payload is not None else None
        metadata["validation_retry_response_text_length"] = len(response_text)
        retry_review = parsed_payload if isinstance(parsed_payload, dict) else _parse_json_object(response_text)
        metadata["validation_retry_parsed_json_object"] = isinstance(retry_review, dict)
        metadata["latency_ms"] = int(round((time.monotonic() - started_at) * 1000))
        if retry_review is None:
            continue
        result = _score_provider_review(
            packet,
            provider=PROVIDER_GEMINI,
            model_name=settings.model,
            review=retry_review,
            provider_metadata=metadata,
        )
        if result.get("status") != FAILURE_MALFORMED_RESPONSE:
            return result
    return result


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
