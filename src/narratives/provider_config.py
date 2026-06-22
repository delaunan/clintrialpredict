"""Environment-backed configuration for narrative LLM providers.

This module reads provider settings only. It does not call OpenAI, Gemini, or
any other LLM provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

PROVIDER_MOCK = "mock"
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
SUPPORTED_PROVIDERS = {PROVIDER_MOCK, PROVIDER_OPENAI, PROVIDER_GEMINI}

DEFAULT_PRIMARY_PROVIDER = PROVIDER_OPENAI
DEFAULT_FALLBACK_PROVIDER = PROVIDER_GEMINI
DEFAULT_OPENAI_MODEL = "gpt-5.5-2026-04-23"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_TEMPERATURE = None
DEFAULT_MAX_OUTPUT_TOKENS = 20000
DEFAULT_TIMEOUT_SECONDS = 90
DEFAULT_MAX_RETRIES = 1
DEFAULT_OPENAI_REASONING_EFFORT = "high"
OPENAI_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
GEMINI_THINKING_LEVELS = {"low", "medium", "high"}


@dataclass(frozen=True)
class ProviderSettings:
    """Provider-specific settings with secret values hidden from repr."""

    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class NarrativeProviderConfig:
    """Sanitized runtime settings for narrative provider calls."""

    provider: str
    fallback_provider: str | None
    providers: dict[str, ProviderSettings]
    temperature: float | None
    seed: int | None
    gemini_thinking_level: str | None
    openai_reasoning_effort: str
    max_output_tokens: int
    timeout_seconds: int
    max_retries: int
    validation_errors: list[str]

    def provider_settings(self, provider: str | None = None) -> ProviderSettings | None:
        provider_key = normalize_provider(provider or self.provider)
        return self.providers.get(provider_key)

    def fallback_settings(self) -> ProviderSettings | None:
        if not self.fallback_provider:
            return None
        return self.providers.get(self.fallback_provider)

    def provider_available(self, provider: str | None = None) -> bool:
        settings = self.provider_settings(provider)
        if not settings:
            return False
        if settings.provider == PROVIDER_MOCK:
            return True
        return settings.has_api_key

    def fallback_available(self) -> bool:
        settings = self.fallback_settings()
        if not settings:
            return False
        if settings.provider == PROVIDER_MOCK:
            return True
        return settings.has_api_key

    def sanitized_trace_metadata(self) -> dict[str, object]:
        """Return non-secret metadata suitable for traces/debug output."""
        return {
            "provider": self.provider,
            "fallback_provider": self.fallback_provider,
            "provider_available": self.provider_available(),
            "fallback_available": self.fallback_available(),
            "models": {
                name: settings.model
                for name, settings in self.providers.items()
            },
            "temperature": self.temperature,
            "seed": self.seed,
            "gemini_thinking_level": self.gemini_thinking_level,
            "openai_reasoning_effort": self.openai_reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "validation_errors": list(self.validation_errors),
        }


def normalize_provider(value: object, *, default: str | None = None) -> str:
    provider = str(value or default or "").strip().lower()
    return provider


def _env_value(env: Mapping[str, str], key: str, default: str | None = None) -> str | None:
    value = env.get(key)
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def _parse_optional_float_or_omit(
    env: Mapping[str, str],
    key: str,
    default: float | None,
    errors: list[str],
) -> float | None:
    raw = _env_value(env, key)
    if raw is None:
        return default
    if str(raw).strip().lower() in {"omit", "default", "none", "unset"}:
        return None
    try:
        return float(raw)
    except ValueError:
        errors.append(f"{key} must be a number or one of omit/default/none/unset")
        return default


def _parse_optional_int(
    env: Mapping[str, str],
    key: str,
    errors: list[str],
) -> int | None:
    raw = _env_value(env, key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        errors.append(f"{key} must be an integer")
        return None


def _parse_int(
    env: Mapping[str, str],
    key: str,
    default: int,
    errors: list[str],
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = _env_value(env, key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        errors.append(f"{key} must be an integer")
        return default

    if minimum is not None and value < minimum:
        errors.append(f"{key} must be >= {minimum}")
        return default
    if maximum is not None and value > maximum:
        errors.append(f"{key} must be <= {maximum}")
        return default
    return value


def load_narrative_provider_config(env: Mapping[str, str]) -> NarrativeProviderConfig:
    """Load narrative provider settings from an environment-like mapping."""
    errors: list[str] = []
    provider = normalize_provider(
        _env_value(env, "NARRATIVE_LLM_PROVIDER", DEFAULT_PRIMARY_PROVIDER),
        default=DEFAULT_PRIMARY_PROVIDER,
    )
    fallback = normalize_provider(
        _env_value(env, "NARRATIVE_LLM_FALLBACK_PROVIDER", DEFAULT_FALLBACK_PROVIDER),
        default=DEFAULT_FALLBACK_PROVIDER,
    )

    if provider not in SUPPORTED_PROVIDERS:
        errors.append(f"NARRATIVE_LLM_PROVIDER must be one of {sorted(SUPPORTED_PROVIDERS)}")
        provider = DEFAULT_PRIMARY_PROVIDER
    if fallback and fallback not in SUPPORTED_PROVIDERS:
        errors.append(f"NARRATIVE_LLM_FALLBACK_PROVIDER must be one of {sorted(SUPPORTED_PROVIDERS)}")
        fallback = DEFAULT_FALLBACK_PROVIDER
    if fallback == provider:
        fallback = None

    providers = {
        PROVIDER_MOCK: ProviderSettings(
            provider=PROVIDER_MOCK,
            model="fixture_hash_mock_v1",
            api_key=None,
        ),
        PROVIDER_OPENAI: ProviderSettings(
            provider=PROVIDER_OPENAI,
            model=_env_value(env, "OPENAI_NARRATIVE_MODEL", DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL,
            api_key=_env_value(env, "OPENAI_API_KEY"),
        ),
        PROVIDER_GEMINI: ProviderSettings(
            provider=PROVIDER_GEMINI,
            model=_env_value(env, "GEMINI_NARRATIVE_MODEL", DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL,
            api_key=_env_value(env, "GEMINI_API_KEY") or _env_value(env, "GOOGLE_API_KEY"),
        ),
    }

    temperature = _parse_optional_float_or_omit(env, "NARRATIVE_LLM_TEMPERATURE", DEFAULT_TEMPERATURE, errors)
    seed = _parse_optional_int(env, "NARRATIVE_LLM_SEED", errors)
    gemini_thinking_level = _env_value(env, "GEMINI_THINKING_LEVEL")
    if gemini_thinking_level is not None:
        gemini_thinking_level = str(gemini_thinking_level).strip().lower()
        if gemini_thinking_level not in GEMINI_THINKING_LEVELS:
            errors.append(f"GEMINI_THINKING_LEVEL must be one of {sorted(GEMINI_THINKING_LEVELS)}")
            gemini_thinking_level = None
    openai_reasoning_effort = str(
        _env_value(env, "OPENAI_REASONING_EFFORT", DEFAULT_OPENAI_REASONING_EFFORT)
        or DEFAULT_OPENAI_REASONING_EFFORT
    ).strip().lower()
    if openai_reasoning_effort not in OPENAI_REASONING_EFFORTS:
        errors.append(f"OPENAI_REASONING_EFFORT must be one of {sorted(OPENAI_REASONING_EFFORTS)}")
        openai_reasoning_effort = DEFAULT_OPENAI_REASONING_EFFORT
    max_output_tokens = _parse_int(
        env,
        "NARRATIVE_LLM_MAX_OUTPUT_TOKENS",
        DEFAULT_MAX_OUTPUT_TOKENS,
        errors,
        minimum=1,
    )
    timeout_seconds = _parse_int(
        env,
        "NARRATIVE_LLM_TIMEOUT_SECONDS",
        DEFAULT_TIMEOUT_SECONDS,
        errors,
        minimum=1,
    )
    max_retries = _parse_int(
        env,
        "NARRATIVE_LLM_MAX_RETRIES",
        DEFAULT_MAX_RETRIES,
        errors,
        minimum=0,
        maximum=3,
    )

    return NarrativeProviderConfig(
        provider=provider,
        fallback_provider=fallback,
        providers=providers,
        temperature=temperature,
        seed=seed,
        gemini_thinking_level=gemini_thinking_level,
        openai_reasoning_effort=openai_reasoning_effort,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        validation_errors=errors,
    )


def provider_config_cache_namespace(config: NarrativeProviderConfig) -> str:
    """Return a non-secret cache namespace for live provider-chain reviews."""
    openai = config.provider_settings(PROVIDER_OPENAI)
    gemini = config.provider_settings(PROVIDER_GEMINI)
    return "|".join([
        f"provider={config.provider}",
        f"fallback={config.fallback_provider or 'none'}",
        f"openai_model={(openai.model if openai else '')}",
        f"gemini_model={(gemini.model if gemini else '')}",
        f"temperature={config.temperature}",
        f"seed={config.seed}",
        f"gemini_thinking_level={config.gemini_thinking_level or 'default'}",
        f"openai_reasoning_effort={config.openai_reasoning_effort}",
        f"max_output_tokens={config.max_output_tokens}",
    ])
