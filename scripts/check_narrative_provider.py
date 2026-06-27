#!/usr/bin/env python
"""Validate simplified narrative provider boundary behavior."""

from __future__ import annotations

from copy import deepcopy
import json
import sys
import types as py_types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402
from src.narratives import provider as provider_module  # noqa: E402
from src.narratives.provider import (  # noqa: E402
    FAILURE_MALFORMED_RESPONSE,
    FAILURE_PROVIDER_UNAVAILABLE,
    PROVIDER_GEMINI,
    PROVIDER_MOCK,
    PROVIDER_OPENAI,
    STATUS_REVIEWED,
    _call_gemini_scoring,
    _call_gemini_pass2,
    _call_openai_scoring,
    pass1_result_needs_repair,
    review_packet_pass2_with_provider,
    review_packet_scoring_with_provider,
    review_packet_with_provider,
)
from src.narratives.provider_config import NarrativeProviderConfig, ProviderSettings  # noqa: E402


def _fixture(fixture_id: str) -> dict:
    return next(item for item in get_contract_fixtures() if item["fixture_id"] == fixture_id)


def _config(provider: str) -> NarrativeProviderConfig:
    return NarrativeProviderConfig(
        provider=provider,
        fallback_provider=None,
        providers={provider: ProviderSettings(provider=provider, model=f"mock-{provider}", api_key="fake-key")},
        temperature=None,
        seed=None,
        gemini_thinking_level=None,
        openai_reasoning_effort="low",
        max_output_tokens=12000,
        timeout_seconds=30,
        max_retries=0,
        validation_errors=[],
    )


def main() -> int:
    errors: list[str] = []
    packet = build_review_packet_from_fixture(_fixture("operational_only_ambitious_enrollment_v2"))
    hidden_packet = build_review_packet_from_fixture(_fixture("baseline_hidden_review_v2"))

    mock_result = review_packet_with_provider(packet, provider=PROVIDER_MOCK)
    if mock_result.get("status") != STATUS_REVIEWED:
        errors.append(f"mock provider should produce reviewed result, got {mock_result.get('status')}")
    if mock_result.get("scoring", {}).get("trial_score") is None:
        errors.append("mock provider should include accepted Trial Score after scoring pass")
    if not mock_result.get("participant_narrative"):
        errors.append("mock provider should include synthesized final narrative")
    if pass1_result_needs_repair(mock_result):
        errors.append("valid mock result should not need Pass 1 repair")
    if not mock_result.get("scoring_review"):
        errors.append("mock result should preserve scoring_review for trace/debug")

    malformed = review_packet_with_provider(packet, provider=PROVIDER_MOCK, failure_mode="malformed_json")
    if malformed.get("status") != FAILURE_MALFORMED_RESPONSE:
        errors.append("mock malformed mode should return malformed_response")
    if malformed.get("scoring", {}).get("trial_score") is not None:
        errors.append("malformed review should not expose Trial Score")

    unavailable = review_packet_with_provider(
        packet,
        provider=PROVIDER_GEMINI,
        config=NarrativeProviderConfig(
            provider=PROVIDER_GEMINI,
            fallback_provider=None,
            providers={PROVIDER_GEMINI: ProviderSettings(provider=PROVIDER_GEMINI, model="mock-gemini", api_key="")},
            temperature=None,
            seed=None,
            gemini_thinking_level=None,
            openai_reasoning_effort="low",
            max_output_tokens=12000,
            timeout_seconds=30,
            max_retries=0,
            validation_errors=[],
        ),
    )
    if unavailable.get("status") != FAILURE_PROVIDER_UNAVAILABLE:
        errors.append("missing live provider key should return provider_unavailable")

    scored_again = review_packet_scoring_with_provider(packet, mock_result, provider=PROVIDER_MOCK)
    if scored_again is not mock_result:
        errors.append("mock scoring wrapper should leave full mock result unchanged")
    narrative_again = review_packet_pass2_with_provider(packet, mock_result, provider=PROVIDER_MOCK)
    if narrative_again is not mock_result:
        errors.append("mock narrative wrapper should leave full mock result unchanged")

    hidden_seed = provider_module._hidden_baseline_fallback_result(
        hidden_packet,
        provider=PROVIDER_GEMINI,
        model_name="mock-gemini",
        provider_metadata={"workflow_stage": "pass1_initial"},
        reason="test hidden-baseline chain guard",
    )
    if review_packet_scoring_with_provider(
        hidden_packet,
        hidden_seed,
        provider=PROVIDER_GEMINI,
        config=_config(PROVIDER_GEMINI),
    ) is not hidden_seed:
        errors.append("hidden-baseline scoring wrapper should return the Pass 1 result unchanged")
    if review_packet_pass2_with_provider(
        hidden_packet,
        hidden_seed,
        provider=PROVIDER_GEMINI,
        config=_config(PROVIDER_GEMINI),
    ) is not hidden_seed:
        errors.append("hidden-baseline narrative wrapper should return the Pass 1 result unchanged")

    chain_calls: list[str] = []
    original_pass1 = provider_module.review_packet_pass1_initial_with_provider
    original_repair = provider_module.review_packet_pass1_repair_with_provider
    original_scoring = provider_module.review_packet_scoring_with_provider
    original_pass2 = provider_module.review_packet_pass2_with_provider

    def _fake_pass1(*args, **kwargs):
        chain_calls.append("pass1")
        return deepcopy(hidden_seed)

    def _fake_repair(*args, **kwargs):
        chain_calls.append("repair")
        return args[1]

    def _fake_scoring(*args, **kwargs):
        chain_calls.append("scoring")
        return args[1]

    def _fake_pass2(*args, **kwargs):
        chain_calls.append("pass2")
        return args[1]

    provider_module.review_packet_pass1_initial_with_provider = _fake_pass1
    provider_module.review_packet_pass1_repair_with_provider = _fake_repair
    provider_module.review_packet_scoring_with_provider = _fake_scoring
    provider_module.review_packet_pass2_with_provider = _fake_pass2
    try:
        hidden_chain_result = provider_module._review_packet_two_pass_with_provider(
            hidden_packet,
            provider=PROVIDER_GEMINI,
            config=_config(PROVIDER_GEMINI),
        )
    finally:
        provider_module.review_packet_pass1_initial_with_provider = original_pass1
        provider_module.review_packet_pass1_repair_with_provider = original_repair
        provider_module.review_packet_scoring_with_provider = original_scoring
        provider_module.review_packet_pass2_with_provider = original_pass2
    if chain_calls != ["pass1"]:
        errors.append(f"hidden-baseline provider chain should stop after Pass 1, got calls {chain_calls}")
    if hidden_chain_result.get("status") != STATUS_REVIEWED:
        errors.append("hidden-baseline provider chain should keep the reviewed Pass 1 result")
    if hidden_chain_result.get("participant_narrative"):
        errors.append("hidden-baseline provider chain should not attach participant narrative")
    chain_metadata = hidden_chain_result.get("provider_metadata") or {}
    if "pass2_scoring_latency_ms" in chain_metadata or "pass2_latency_ms" in chain_metadata:
        errors.append("hidden-baseline provider chain should not include Pass 2 latency metadata")

    hidden_invalid = provider_module._failure_result(
        hidden_packet,
        provider=PROVIDER_GEMINI,
        model_name="mock-gemini",
        status=FAILURE_MALFORMED_RESPONSE,
        message="Hidden baseline initial response was not valid JSON",
        provider_metadata={"workflow_stage": "pass1_initial"},
    )
    chain_calls = []

    def _fake_invalid_pass1(*args, **kwargs):
        chain_calls.append("pass1")
        return deepcopy(hidden_invalid)

    def _fake_successful_repair(*args, **kwargs):
        chain_calls.append("repair")
        return deepcopy(hidden_seed)

    provider_module.review_packet_pass1_initial_with_provider = _fake_invalid_pass1
    provider_module.review_packet_pass1_repair_with_provider = _fake_successful_repair
    provider_module.review_packet_scoring_with_provider = _fake_scoring
    provider_module.review_packet_pass2_with_provider = _fake_pass2
    try:
        repaired_hidden_chain_result = provider_module._review_packet_two_pass_with_provider(
            hidden_packet,
            provider=PROVIDER_GEMINI,
            config=_config(PROVIDER_GEMINI),
        )
    finally:
        provider_module.review_packet_pass1_initial_with_provider = original_pass1
        provider_module.review_packet_pass1_repair_with_provider = original_repair
        provider_module.review_packet_scoring_with_provider = original_scoring
        provider_module.review_packet_pass2_with_provider = original_pass2
    if chain_calls != ["pass1", "repair"]:
        errors.append(f"hidden-baseline provider chain should allow one repair then stop, got calls {chain_calls}")
    if repaired_hidden_chain_result.get("status") != STATUS_REVIEWED:
        errors.append("hidden-baseline provider chain should keep a reviewed result after successful repair")
    if repaired_hidden_chain_result.get("participant_narrative"):
        errors.append("repaired hidden-baseline provider chain should not attach participant narrative")

    valid_scoring_review = deepcopy(mock_result.get("scoring_review") or {})
    invalid_scoring_review = deepcopy(valid_scoring_review)
    invalid_scoring_review.pop("score_evolution_read", None)
    fake_payloads = [invalid_scoring_review, valid_scoring_review]
    post_inputs: list[str] = []

    class _FakeResponse:
        status_code = 200

        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"output_text": json.dumps(self._payload)}

    def _fake_post(*args, **kwargs):
        post_inputs.append(str((kwargs.get("json") or {}).get("input") or ""))
        return _FakeResponse(fake_payloads.pop(0))

    original_post = provider_module.requests.post
    provider_module.requests.post = _fake_post
    try:
        repaired_scoring = _call_openai_scoring(
            packet,
            mock_result,
            settings=ProviderSettings(provider=PROVIDER_OPENAI, model="mock-openai", api_key="fake-key"),
            config=_config(PROVIDER_OPENAI),
        )
    finally:
        provider_module.requests.post = original_post
    repair_metadata = repaired_scoring.get("provider_metadata") or {}
    if repaired_scoring.get("status") != STATUS_REVIEWED:
        errors.append(f"scoring repair should restore reviewed status, got {repaired_scoring.get('status')}")
    if repaired_scoring.get("scoring", {}).get("validation_status") != "valid":
        errors.append("scoring repair should produce valid scoring")
    if repair_metadata.get("pass2_scoring_retry_attempts") != 1:
        errors.append("scoring repair should run one retry")
    if repair_metadata.get("pass2_scoring_retry_validation_status") != "valid":
        errors.append("scoring repair metadata should record valid retry validation")
    if repair_metadata.get("pass2_scoring_retry_validation_errors"):
        errors.append("scoring repair should not retain stale retry validation errors after a valid retry")
    if repair_metadata.get("pass2_scoring_validation_errors"):
        errors.append("scoring repair should clear stale initial validation errors after a valid retry")
    if len(post_inputs) != 2 or "repairing a previous Pass 2 Score Adjudication" not in post_inputs[-1]:
        errors.append("scoring repair should call provider with the targeted repair prompt")

    class _FakeGeminiResponse:
        def __init__(self, parsed: dict):
            self.parsed = parsed
            self.text = json.dumps(parsed)

    class _FakeModels:
        def __init__(self, payloads: list[dict]):
            self.payloads = payloads
            self.prompts: list[str] = []
            self.configs: list[object] = []

        def generate_content(self, **kwargs):
            self.prompts.append(str(kwargs.get("contents") or ""))
            self.configs.append(kwargs.get("config"))
            return _FakeGeminiResponse(self.payloads.pop(0))

    class _FakeClient:
        def __init__(self, payloads: list[dict]):
            self.models = _FakeModels(payloads)

    class _FakeTypes:
        class GenerateContentConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class ThinkingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HttpOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class HttpRetryOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    gemini_invalid = deepcopy(valid_scoring_review)
    gemini_invalid["reality_check"].pop("allocations", None)
    fake_client = _FakeClient([gemini_invalid, deepcopy(valid_scoring_review)])
    gemini_repaired = _call_gemini_scoring(
        packet,
        mock_result,
        client=fake_client,
        types_module=_FakeTypes,
        settings=ProviderSettings(provider=PROVIDER_GEMINI, model="mock-gemini", api_key="fake-key"),
        config=_config(PROVIDER_GEMINI),
        thinking_level="low",
        max_output_tokens=12000,
    )
    gemini_metadata = gemini_repaired.get("provider_metadata") or {}
    if gemini_repaired.get("status") != STATUS_REVIEWED:
        errors.append(f"Gemini scoring repair should restore reviewed status, got {gemini_repaired.get('status')}")
    if gemini_metadata.get("pass2_scoring_retry_validation_status") != "valid":
        errors.append("Gemini scoring repair metadata should record valid retry validation")
    if len(fake_client.models.prompts) != 2 or "repairing a previous Pass 2 Score Adjudication" not in fake_client.models.prompts[-1]:
        errors.append("Gemini scoring repair should call the targeted repair prompt")
    if gemini_metadata.get("pass2_scoring_validation_errors") or gemini_metadata.get("pass2_scoring_retry_validation_errors"):
        errors.append("Gemini scoring repair should not retain stale validation errors after a valid retry")
    if provider_module._gemini_primary_thinking_level(_config(PROVIDER_GEMINI)) != "medium":
        errors.append("Gemini primary thinking helper should default Pass 2/3 to medium")

    gemini_scoring_medium_client = _FakeClient([deepcopy(valid_scoring_review)])
    _call_gemini_scoring(
        packet,
        mock_result,
        client=gemini_scoring_medium_client,
        types_module=_FakeTypes,
        settings=ProviderSettings(provider=PROVIDER_GEMINI, model="mock-gemini", api_key="fake-key"),
        config=_config(PROVIDER_GEMINI),
        thinking_level=provider_module._gemini_primary_thinking_level(_config(PROVIDER_GEMINI)),
        max_output_tokens=12000,
    )
    scoring_config = gemini_scoring_medium_client.models.configs[0]
    scoring_thinking = getattr(scoring_config, "kwargs", {}).get("thinking_config")
    if getattr(scoring_thinking, "kwargs", {}).get("thinking_level") != "medium":
        errors.append("Gemini Pass 2 scoring generation config should use medium thinking")

    gemini_narrative_medium_client = _FakeClient([deepcopy(mock_result.get("participant_narrative") or {})])
    _call_gemini_pass2(
        packet,
        mock_result,
        client=gemini_narrative_medium_client,
        types_module=_FakeTypes,
        settings=ProviderSettings(provider=PROVIDER_GEMINI, model="mock-gemini", api_key="fake-key"),
        config=_config(PROVIDER_GEMINI),
        thinking_level=provider_module._gemini_primary_thinking_level(_config(PROVIDER_GEMINI)),
        max_output_tokens=12000,
    )
    narrative_config = gemini_narrative_medium_client.models.configs[0]
    narrative_thinking = getattr(narrative_config, "kwargs", {}).get("thinking_config")
    if getattr(narrative_thinking, "kwargs", {}).get("thinking_level") != "medium":
        errors.append("Gemini Pass 3 narrative generation config should use medium thinking")

    original_google = sys.modules.get("google")
    original_google_genai = sys.modules.get("google.genai")
    created_clients: list[_FakeClient] = []
    fake_google = py_types.ModuleType("google")
    fake_genai = py_types.ModuleType("google.genai")

    class _FakeGenaiClient(_FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__([deepcopy(mock_result.get("review") or mock_result.get("validated_review") or {})])
            created_clients.append(self)

    fake_genai.Client = _FakeGenaiClient
    fake_genai.types = _FakeTypes
    fake_google.genai = fake_genai
    sys.modules["google"] = fake_google
    sys.modules["google.genai"] = fake_genai
    try:
        pass1_gemini = provider_module._call_gemini_pass1_initial(
            packet,
            config=_config(PROVIDER_GEMINI),
            settings=ProviderSettings(provider=PROVIDER_GEMINI, model="mock-gemini", api_key="fake-key"),
        )
    finally:
        if original_google is None:
            sys.modules.pop("google", None)
        else:
            sys.modules["google"] = original_google
        if original_google_genai is None:
            sys.modules.pop("google.genai", None)
        else:
            sys.modules["google.genai"] = original_google_genai
    pass1_metadata = pass1_gemini.get("provider_metadata") or {}
    if pass1_metadata.get("applied_generation_controls", {}).get("thinking_level") != "medium":
        errors.append("Gemini visible Pass 1 should apply medium thinking level")
    if not created_clients or not created_clients[0].models.configs:
        errors.append("Gemini visible Pass 1 test should capture a generated config")
    else:
        config_obj = created_clients[0].models.configs[0]
        thinking_config = getattr(config_obj, "kwargs", {}).get("thinking_config")
        if getattr(thinking_config, "kwargs", {}).get("thinking_level") != "medium":
            errors.append("Gemini visible Pass 1 generation config should use medium thinking")

    provider_source = (ROOT / "src/narratives/provider.py").read_text()
    for term in (
        'GEMINI_PRIMARY_THINKING_LEVEL = "medium"',
        'GEMINI_PASS1_THINKING_LEVEL = "medium"',
        'GEMINI_HIDDEN_BASELINE_THINKING_LEVEL = "medium"',
        "GEMINI_HIDDEN_BASELINE_OUTPUT_TOKENS = DEFAULT_HIDDEN_BASELINE_MAX_OUTPUT_TOKENS",
        "GEMINI_HIDDEN_BASELINE_TIMEOUT_SECONDS = 100",
        "GEMINI_HIDDEN_BASELINE_REPAIR_ATTEMPTS = 2",
        'metadata["hidden_baseline_repair_profile"] = "bounded_compact_repair"',
        '"hidden_baseline_fast_profile": hidden_baseline',
        '"timeout_seconds": request_timeout_seconds',
        "hidden_baseline_fallback_used",
        "Hidden baseline initial response hit MAX_TOKENS",
        "if _is_hidden_baseline_packet(packet):",
    ):
        if term not in provider_source:
            errors.append(f"Gemini hidden-baseline fast path missing term: {term}")

    simulator_source = (ROOT / "frontend/views/trial_simulator.py").read_text()
    if "SCENARIO_REVIEW_PHASE_SCORING = \"pass2_scoring\"" not in simulator_source:
        errors.append("staged simulator should define a distinct pass2_scoring phase")
    if "SCENARIO_REVIEW_PHASE_NARRATIVE = \"pass3_narrative\"" not in simulator_source:
        errors.append("staged simulator should define a distinct pass3_narrative phase")
    if "workflow[\"phase\"] = SCENARIO_REVIEW_PHASE_SCORING" not in simulator_source:
        errors.append("staged simulator should transition to scoring before narrative")
    if "workflow.get(\"scoring_result\") or workflow.get(\"pass1_result\")" not in simulator_source:
        errors.append("staged simulator narrative phase should consume the scoring result")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Validated simplified narrative provider boundary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
