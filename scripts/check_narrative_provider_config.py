#!/usr/bin/env python
"""Validate narrative provider config parsing without calling LLM providers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.provider_config import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    load_narrative_provider_config,
)


def _check_defaults(errors: list[str]) -> None:
    config = load_narrative_provider_config({})
    if config.provider != PROVIDER_OPENAI:
        errors.append("default narrative provider should be openai")
    if config.fallback_provider != PROVIDER_GEMINI:
        errors.append("default narrative fallback provider should be gemini")
    if config.provider_available():
        errors.append("openai provider should not be available without OPENAI_API_KEY")
    if config.fallback_available():
        errors.append("gemini fallback should not be available without GEMINI_API_KEY or GOOGLE_API_KEY")
    if config.provider_settings(PROVIDER_OPENAI).model != DEFAULT_OPENAI_MODEL:
        errors.append("openai model should use pinned default when env is absent")
    if config.provider_settings(PROVIDER_GEMINI).model != DEFAULT_GEMINI_MODEL:
        errors.append("gemini model should use pinned default when env is absent")
    if config.max_output_tokens != DEFAULT_MAX_OUTPUT_TOKENS:
        errors.append("default max_output_tokens mismatch")
    if config.timeout_seconds != DEFAULT_TIMEOUT_SECONDS:
        errors.append("default timeout_seconds mismatch")
    if config.max_retries != DEFAULT_MAX_RETRIES:
        errors.append("default max_retries mismatch")


def _check_env_values(errors: list[str]) -> None:
    env = {
        "NARRATIVE_LLM_PROVIDER": " OpenAI ",
        "NARRATIVE_LLM_FALLBACK_PROVIDER": "Gemini",
        "OPENAI_API_KEY": "openai-secret",
        "GOOGLE_API_KEY": "google-secret",
        "OPENAI_NARRATIVE_MODEL": "openai-model",
        "GEMINI_NARRATIVE_MODEL": "gemini-model",
        "NARRATIVE_LLM_TEMPERATURE": "0",
        "NARRATIVE_LLM_SEED": "20260607",
        "NARRATIVE_LLM_MAX_OUTPUT_TOKENS": "3000",
        "NARRATIVE_LLM_TIMEOUT_SECONDS": "45",
        "NARRATIVE_LLM_MAX_RETRIES": "1",
    }
    config = load_narrative_provider_config(env)
    if config.validation_errors:
        errors.append(f"valid env should not produce validation errors: {config.validation_errors}")
    if not config.provider_available():
        errors.append("openai provider should be available when OPENAI_API_KEY is present")
    if not config.fallback_available():
        errors.append("gemini fallback should be available when GOOGLE_API_KEY is present")
    if config.provider_settings().model != "openai-model":
        errors.append("openai model should come from OPENAI_NARRATIVE_MODEL")
    if config.fallback_settings().model != "gemini-model":
        errors.append("gemini model should come from GEMINI_NARRATIVE_MODEL")
    if config.temperature != 0.0:
        errors.append("temperature should parse as float")
    if config.seed != 20260607:
        errors.append("seed should parse as int")
    if config.max_output_tokens != 3000:
        errors.append("max output tokens should parse as int")
    if config.timeout_seconds != 45:
        errors.append("timeout seconds should parse as int")
    if config.max_retries != 1:
        errors.append("max retries should parse as int")

    metadata = config.sanitized_trace_metadata()
    if "openai-secret" in str(metadata) or "google-secret" in str(metadata):
        errors.append("sanitized metadata should not expose API key values")


def _check_gemini_key_precedence(errors: list[str]) -> None:
    env = {
        "GEMINI_API_KEY": "gemini-secret",
        "GOOGLE_API_KEY": "google-secret",
    }
    config = load_narrative_provider_config(env)
    settings = config.provider_settings(PROVIDER_GEMINI)
    if settings.api_key != "gemini-secret":
        errors.append("GEMINI_API_KEY should take precedence over GOOGLE_API_KEY")


def _check_invalid_values(errors: list[str]) -> None:
    env = {
        "NARRATIVE_LLM_PROVIDER": "bad-provider",
        "NARRATIVE_LLM_FALLBACK_PROVIDER": "bad-fallback",
        "NARRATIVE_LLM_TEMPERATURE": "not-a-number",
        "NARRATIVE_LLM_SEED": "not-an-int",
        "NARRATIVE_LLM_MAX_OUTPUT_TOKENS": "0",
        "NARRATIVE_LLM_TIMEOUT_SECONDS": "-1",
        "NARRATIVE_LLM_MAX_RETRIES": "9",
    }
    config = load_narrative_provider_config(env)
    if len(config.validation_errors) != 7:
        errors.append(f"invalid env should produce seven validation errors, got {config.validation_errors}")
    if config.provider != PROVIDER_OPENAI:
        errors.append("invalid primary provider should fall back to openai")
    if config.fallback_provider != PROVIDER_GEMINI:
        errors.append("invalid fallback provider should fall back to gemini")


def _check_same_provider_disables_fallback(errors: list[str]) -> None:
    config = load_narrative_provider_config({
        "NARRATIVE_LLM_PROVIDER": "openai",
        "NARRATIVE_LLM_FALLBACK_PROVIDER": "openai",
    })
    if config.fallback_provider is not None:
        errors.append("fallback should be disabled when it matches primary provider")


def _check_local_env_loads_without_printing_secrets(errors: list[str]) -> None:
    load_dotenv()
    config = load_narrative_provider_config(os.environ)
    if config.provider not in {PROVIDER_OPENAI, PROVIDER_GEMINI, "mock"}:
        errors.append("local env provider should normalize to a supported provider")
    # Intentionally do not print or assert real secret values. This only proves
    # local .env can be parsed without requiring the user's account state.
    _ = config.sanitized_trace_metadata()


def main() -> int:
    errors: list[str] = []
    _check_defaults(errors)
    _check_env_values(errors)
    _check_gemini_key_precedence(errors)
    _check_invalid_values(errors)
    _check_same_provider_disables_fallback(errors)
    _check_local_env_loads_without_printing_secrets(errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated narrative provider config parsing without provider API calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
