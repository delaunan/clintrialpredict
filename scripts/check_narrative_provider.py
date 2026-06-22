#!/usr/bin/env python
"""Validate the thin narrative provider boundary."""

from __future__ import annotations

import sys
import types as py_types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.narratives.provider as provider_module  # noqa: E402
from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402
from src.narratives.mock_reviewer import _synthesized_trial_score_pass1_review  # noqa: E402
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
    NARRATIVE_REPAIR_RETRY_ATTEMPTS,
    PROVIDER_MOCK,
    PROVIDER_VALIDATION_RETRY_ATTEMPTS,
    PASS2_VALIDATION_RETRY_ATTEMPTS,
    STATUS_REVIEWED,
    _gemini_http_options,
    _pass1_repair_stage,
    _record_gemini_response_metadata,
    _score_provider_review,
    pass1_result_needs_repair,
    review_packet_pass1_initial_with_provider,
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


class _FakeGeminiResponse:
    def __init__(self, parsed: dict | None = None, text: str = "") -> None:
        self.parsed = parsed
        self.text = text
        self.usage_metadata = None
        self.candidates = []


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
        retry_review = _synthesized_trial_score_pass1_review(packet, fixture)
        pass2_review = {
            "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
            "trial_score_narrative": {
                "summary": "The Trial Score reading is mixed but defensible.",
                "movement_reading": "Operational Fit and Reality Check should be read together.",
                "score_interpretation": "Completion Outlook remains the protected model-pattern anchor.",
            },
            "pillar_reading": [
                {"pillar": "Execution Framework", "reading": "Operational proportionality matters."},
                {"pillar": "Scientific Challenge", "reading": "Evidence interpretability remains relevant to the participant read."},
            ],
            "central_tension": {
                "summary": "The main tension is whether completion favorability and design defensibility move in the same direction.",
                "why_it_matters": "It frames the participant discussion.",
            },
            "broader_strategic_question": {
                "mapped_tension": "The main tension is whether completion favorability and design defensibility move in the same direction.",
                "question": "When should operational feasibility change confidence in the development story, and when does it only make an uncertain evidence package easier to run?",
            },
        }

        def fake_post(*args, **kwargs):
            calls["count"] += 1
            prompt_text = str((kwargs.get("json") or {}).get("input") or "")
            if calls["count"] == 1:
                return _FakeResponse(first_payload)
            if calls["count"] == 2:
                if "repairing a previous Pass 1 Trial Score JSON response" not in prompt_text:
                    errors.append("OpenAI validation retry should use a targeted repair prompt")
                if "Allowed Reality Check allocation_target_id values" not in prompt_text:
                    errors.append("OpenAI validation retry prompt should include canonical allocation target IDs")
                if "Allowed packet evidence references" not in prompt_text:
                    errors.append("OpenAI validation retry prompt should include allowed packet evidence refs")
                if "usable beyond this exact trial" not in prompt_text:
                    errors.append("OpenAI validation retry prompt should include broader-question abstraction guidance")
            if calls["count"] == 3:
                return _FakeResponse({"output_text": provider_module.json.dumps(pass2_review)})
            return _FakeResponse({"output_text": provider_module.json.dumps(retry_review)})

        provider_module.requests.post = fake_post
        try:
            result = provider_module.review_packet_with_provider(packet, provider="openai", config=config)
        finally:
            provider_module.requests.post = original_post
        if calls["count"] != 3:
            errors.append("OpenAI validation retry should make one retry call plus one Pass 2 call")
        metadata = result.get("provider_metadata") or {}
        if metadata.get("validation_retry_attempts") != 1:
            errors.append("OpenAI validation retry should record one validation_retry_attempt")
        if metadata.get("validation_retry_max_attempts") != PROVIDER_VALIDATION_RETRY_ATTEMPTS:
            errors.append("OpenAI validation retry should record configured max attempts")
        if not metadata.get("validation_retry_history"):
            errors.append("OpenAI validation retry should record retry history")
        if metadata.get("pass2_validation_status") != "valid":
            errors.append("OpenAI recovered review should validate Pass 2 participant narrative")
        if not (result.get("validated_participant_narrative") or {}).get("trial_score_narrative"):
            errors.append("OpenAI recovered review should attach validated Pass 2 narrative")
        if expected_reason_fragment not in str(metadata.get("validation_retry_reason")):
            errors.append("OpenAI validation retry should record the retry reason")
        return result

    non_json_result = run_case({"output_text": "not json"}, "not a JSON object")
    if non_json_result.get("status") != provider_module.STATUS_REVIEWED:
        errors.append("OpenAI non-JSON response should recover when validation retry returns valid review")
    if non_json_result.get("scoring", {}).get("trial_score") is None:
        errors.append("OpenAI non-JSON retry should preserve valid Trial Score scoring")

    invalid_json_result = run_case(
        {"output_text": provider_module.json.dumps({"reality_check": "malformed"})},
        "Pass 1 Trial Score JSON shape",
    )
    if invalid_json_result.get("status") != provider_module.STATUS_REVIEWED:
        errors.append("OpenAI invalid JSON contract response should recover when validation retry returns valid review")
    if invalid_json_result.get("scoring", {}).get("trial_score") is None:
        errors.append("OpenAI invalid-contract retry should preserve valid Trial Score scoring")

    invalid_operational_fit_review = _synthesized_trial_score_pass1_review(packet, fixture)
    invalid_operational_fit_review["operational_fit"]["combined_operational_fit"]["rating"] = "invented_rating"
    operational_fit_result = run_case(
        {"output_text": provider_module.json.dumps(invalid_operational_fit_review)},
        "combined_operational_fit.rating",
    )
    metadata = operational_fit_result.get("provider_metadata") or {}
    if metadata.get("validation_retry_stage") != provider_module.PASS1_REPAIR_STAGE_OPERATIONAL_FIT:
        errors.append("OpenAI Operational Fit contract failure should use Operational Fit repair stage")
    if operational_fit_result.get("status") != provider_module.STATUS_REVIEWED:
        errors.append("OpenAI Operational Fit contract response should recover when validation retry returns valid review")
    if operational_fit_result.get("scoring", {}).get("trial_score") is None:
        errors.append("OpenAI Operational Fit contract retry should preserve valid Trial Score scoring")

    calls = {"count": 0}
    invalid_initial = _synthesized_trial_score_pass1_review(packet, fixture)
    invalid_initial["operational_fit"]["combined_operational_fit"]["rating"] = "invented_rating"
    invalid_repair = _synthesized_trial_score_pass1_review(packet, fixture)
    invalid_repair["operational_fit"]["combined_operational_fit"]["rating"] = "still_invented"
    valid_repair = _synthesized_trial_score_pass1_review(packet, fixture)
    pass2_review = {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "trial_score_narrative": {
            "summary": "The Trial Score reading is mixed but defensible.",
            "movement_reading": "Operational Fit and Reality Check should be read together.",
            "score_interpretation": "Completion Outlook remains the protected model-pattern anchor.",
        },
        "pillar_reading": [
                {"pillar": "Execution Framework", "reading": "Operational proportionality matters."},
                {"pillar": "Scientific Challenge", "reading": "Evidence interpretability remains relevant to the participant read."},
            ],
        "central_tension": {
            "summary": "The main tension is whether completion favorability and design defensibility move in the same direction.",
            "why_it_matters": "It frames the participant discussion.",
        },
        "broader_strategic_question": {
            "mapped_tension": "The main tension is whether completion favorability and design defensibility move in the same direction.",
            "question": "When should operational feasibility change confidence in the development story, and when does it only make an uncertain evidence package easier to run?",
        },
    }

    def fake_post_multi_repair(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResponse({"output_text": provider_module.json.dumps(invalid_initial)})
        if calls["count"] == 2:
            return _FakeResponse({"output_text": provider_module.json.dumps(invalid_repair)})
        if calls["count"] == 3:
            return _FakeResponse({"output_text": provider_module.json.dumps(valid_repair)})
        return _FakeResponse({"output_text": provider_module.json.dumps(pass2_review)})

    provider_module.requests.post = fake_post_multi_repair
    try:
        multi_repair_result = provider_module.review_packet_with_provider(packet, provider="openai", config=config)
    finally:
        provider_module.requests.post = original_post
    metadata = multi_repair_result.get("provider_metadata") or {}
    if multi_repair_result.get("status") != provider_module.STATUS_REVIEWED:
        errors.append("OpenAI validation repair should allow a second targeted repair when the first repair remains invalid")
    if metadata.get("validation_retry_attempts") != 2:
        errors.append("OpenAI multi-repair recovery should record two validation_retry_attempts")
    if len(metadata.get("validation_retry_history") or []) != 2:
        errors.append("OpenAI multi-repair recovery should record two retry-history entries")


def _check_openai_pass2_retry(packet: dict, fixture: dict, errors: list[str]) -> None:
    config = load_narrative_provider_config({
        "NARRATIVE_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_NARRATIVE_MODEL": "test-openai-model",
        "NARRATIVE_LLM_MAX_RETRIES": "0",
    })
    original_post = provider_module.requests.post
    calls = {"count": 0}
    pass1_review = _synthesized_trial_score_pass1_review(packet, fixture)
    invalid_pass2 = {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "trial_score_narrative": {"summary": "Missing required narrative fields."},
    }
    repaired_pass2 = {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "trial_score_narrative": {
            "summary": "The Trial Score reading is mixed but defensible.",
            "movement_reading": "Operational Fit and Reality Check should be read together.",
            "score_interpretation": "Completion Outlook remains the protected model-pattern anchor.",
        },
        "pillar_reading": [
                {"pillar": "Execution Framework", "reading": "Operational proportionality matters."},
                {"pillar": "Scientific Challenge", "reading": "Evidence interpretability remains relevant to the participant read."},
            ],
        "central_tension": {
            "summary": "The main tension is whether completion favorability and design defensibility move in the same direction.",
            "why_it_matters": "It frames the participant discussion.",
        },
        "broader_strategic_question": {
            "mapped_tension": "The main tension is whether completion favorability and design defensibility move in the same direction.",
            "question": "When should operational feasibility change confidence in the development story, and when does it only make an uncertain evidence package easier to run?",
        },
    }

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        prompt_text = str((kwargs.get("json") or {}).get("input") or "")
        if calls["count"] == 1:
            return _FakeResponse({"output_text": provider_module.json.dumps(pass1_review)})
        if calls["count"] == 2:
            return _FakeResponse({"output_text": provider_module.json.dumps(invalid_pass2)})
        if "repairing a previous Pass 2 Participant Narrative JSON response" not in prompt_text:
            errors.append("OpenAI Pass 2 retry should use a targeted Pass 2 repair prompt")
        if "Do not rerun Pass 1" not in prompt_text or "Preserve app_calculated_scores exactly" not in prompt_text:
            errors.append("OpenAI Pass 2 retry should preserve Pass 1 scoring basis")
        return _FakeResponse({"output_text": provider_module.json.dumps(repaired_pass2)})

    provider_module.requests.post = fake_post
    try:
        result = provider_module.review_packet_with_provider(packet, provider="openai", config=config)
    finally:
        provider_module.requests.post = original_post
    if calls["count"] != 3:
        errors.append("OpenAI Pass 2 retry should add only one participant-narrative retry call")
    metadata = result.get("provider_metadata") or {}
    if metadata.get("pass2_retry_attempts") != 1:
        errors.append("OpenAI Pass 2 retry should record one pass2_retry_attempt")
    if result.get("participant_narrative_status") != "valid":
        errors.append("OpenAI Pass 2 retry should attach a valid participant narrative")
    if result.get("scoring", {}).get("trial_score") is None:
        errors.append("OpenAI Pass 2 retry should preserve Trial Score scoring")


def _check_openai_pass2_non_json_repair(packet: dict, fixture: dict, errors: list[str]) -> None:
    config = load_narrative_provider_config({
        "NARRATIVE_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_NARRATIVE_MODEL": "test-openai-model",
        "NARRATIVE_LLM_MAX_RETRIES": "0",
    })
    original_post = provider_module.requests.post
    calls = {"count": 0, "raw_text_seen": False}
    pass1_review = _synthesized_trial_score_pass1_review(packet, fixture)
    raw_pass2_text = "This is not JSON but it contains useful Pass 2 narrative text."
    repaired_pass2 = {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "trial_score_narrative": {
            "summary": "The Trial Score reading is mixed but defensible.",
            "movement_reading": "The latest movement should be read through the supplied score alignment notes.",
            "score_interpretation": "The final read follows the app-calibrated direction without exposing points.",
        },
        "pillar_reading": [
            {"pillar": "Execution Framework", "reading": "Execution evidence is material to this participant read."},
            {"pillar": "Scientific Challenge", "reading": "Evidence interpretability remains part of the selected read."},
        ],
        "central_tension": {
            "summary": "The main tension is whether completion favorability and design defensibility move in the same direction.",
            "why_it_matters": "This frames whether the scenario is easier to run or actually more development-informative.",
        },
        "broader_strategic_question": {
            "mapped_tension": "The main tension is whether completion favorability and design defensibility move in the same direction.",
            "question": "When should operational feasibility change confidence in the development story, and when does it only make an uncertain evidence package easier to run?",
        },
    }

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        prompt_text = str((kwargs.get("json") or {}).get("input") or "")
        if calls["count"] == 1:
            return _FakeResponse({"output_text": provider_module.json.dumps(pass1_review)})
        if calls["count"] == 2:
            return _FakeResponse({"output_text": raw_pass2_text})
        if "Previous raw Pass 2 response text" not in prompt_text or raw_pass2_text not in prompt_text:
            errors.append("OpenAI Pass 2 non-JSON repair prompt should include the raw malformed Pass 2 response")
        calls["raw_text_seen"] = True
        return _FakeResponse({"output_text": provider_module.json.dumps(repaired_pass2)})

    provider_module.requests.post = fake_post
    try:
        result = provider_module.review_packet_with_provider(packet, provider="openai", config=config)
    finally:
        provider_module.requests.post = original_post
    metadata = result.get("provider_metadata") or {}
    if calls["count"] != 3:
        errors.append("OpenAI non-JSON Pass 2 should make one repair call after the initial malformed text")
    if not calls["raw_text_seen"]:
        errors.append("OpenAI non-JSON Pass 2 repair path should observe raw text in the repair prompt")
    if metadata.get("pass2_failure_stage") != "initial_validation_failed":
        errors.append("OpenAI non-JSON Pass 2 should classify the initial failure as validation/parsing failure before repair")
    if metadata.get("pass2_retry_attempts") != 1:
        errors.append("OpenAI non-JSON Pass 2 should record one repair attempt")
    if result.get("participant_narrative_status") != "valid":
        errors.append("OpenAI non-JSON Pass 2 should recover when repair returns valid JSON")


def _check_openai_pass2_exception_warning(packet: dict, fixture: dict, errors: list[str]) -> None:
    config = load_narrative_provider_config({
        "NARRATIVE_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_NARRATIVE_MODEL": "test-openai-model",
        "NARRATIVE_LLM_MAX_RETRIES": "0",
    })
    original_post = provider_module.requests.post
    calls = {"count": 0}
    pass1_review = _synthesized_trial_score_pass1_review(packet, fixture)

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResponse({"output_text": provider_module.json.dumps(pass1_review)})
        raise provider_module.requests.exceptions.Timeout("pass2 timeout")

    provider_module.requests.post = fake_post
    try:
        result = provider_module.review_packet_with_provider(packet, provider="openai", config=config)
    finally:
        provider_module.requests.post = original_post
    metadata = result.get("provider_metadata") or {}
    if result.get("status") != provider_module.STATUS_REVIEWED:
        errors.append("OpenAI Pass 2 exception should preserve reviewed Pass 1 status")
    if result.get("scoring", {}).get("trial_score") is None:
        errors.append("OpenAI Pass 2 exception should preserve Trial Score scoring")
    if result.get("participant_narrative_status") != "invalid":
        errors.append("OpenAI Pass 2 exception should mark participant narrative invalid")
    if "Pass 2 generation" not in str(result.get("participant_narrative_warning") or ""):
        errors.append("OpenAI Pass 2 exception should expose a participant narrative warning")
    if metadata.get("pass2_error_type") != "Timeout":
        errors.append("OpenAI Pass 2 exception should record pass2_error_type")


def _check_gemini_validation_retry(packet: dict, fixture: dict, errors: list[str]) -> None:
    config = load_narrative_provider_config({
        "NARRATIVE_LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test-key",
        "GEMINI_NARRATIVE_MODEL": "test-gemini-model",
        "NARRATIVE_LLM_MAX_RETRIES": "0",
    })
    original_google = sys.modules.get("google")
    original_genai = sys.modules.get("google.genai")
    original_types = sys.modules.get("google.genai.types")
    calls = {"count": 0, "repair_prompt_seen": False}
    retry_review = _synthesized_trial_score_pass1_review(packet, fixture)
    invalid_pass2 = {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "trial_score_narrative": {"summary": "Missing required narrative fields."},
    }
    pass2_review = {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "trial_score_narrative": {
            "summary": "The Trial Score reading is mixed but defensible.",
            "movement_reading": "Operational Fit and Reality Check should be read together.",
            "score_interpretation": "Completion Outlook remains the protected model-pattern anchor.",
        },
        "pillar_reading": [
                {"pillar": "Execution Framework", "reading": "Operational proportionality matters."},
                {"pillar": "Scientific Challenge", "reading": "Evidence interpretability remains relevant to the participant read."},
            ],
        "central_tension": {
            "summary": "The main tension is whether completion favorability and design defensibility move in the same direction.",
            "why_it_matters": "It frames the participant discussion.",
        },
        "broader_strategic_question": {
            "mapped_tension": "The main tension is whether completion favorability and design defensibility move in the same direction.",
            "question": "When should operational feasibility change confidence in the development story, and when does it only make an uncertain evidence package easier to run?",
        },
    }

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls["count"] += 1
            prompt_text = str(contents or "")
            if calls["count"] == 1:
                return _FakeGeminiResponse(parsed={"reality_check": "malformed"})
            if calls["count"] == 2:
                if "repairing a previous Pass 1 Trial Score JSON response" not in prompt_text:
                    errors.append("Gemini validation retry should use a targeted repair prompt")
                if "Allowed Reality Check allocation_target_id values" not in prompt_text:
                    errors.append("Gemini validation retry prompt should include canonical allocation target IDs")
                if "usable beyond this exact trial" not in prompt_text:
                    errors.append("Gemini validation retry prompt should include broader-question abstraction guidance")
                calls["repair_prompt_seen"] = True
                return _FakeGeminiResponse(parsed=retry_review)
            if calls["count"] == 3:
                return _FakeGeminiResponse(parsed=invalid_pass2)
            if "repairing a previous Pass 2 Participant Narrative JSON response" not in prompt_text:
                errors.append("Gemini Pass 2 retry should use a targeted Pass 2 repair prompt")
            if "usable beyond this exact trial" not in prompt_text:
                errors.append("Gemini Pass 2 retry prompt should include broader-question abstraction guidance")
            return _FakeGeminiResponse(parsed=pass2_review)

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.models = FakeModels()

    class FakeTypesModule:
        class ThinkingConfig:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        class GenerateContentConfig:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        class HttpRetryOptions:
            def __init__(self, attempts) -> None:
                self.attempts = attempts

        class HttpOptions:
            def __init__(self, timeout, retry_options) -> None:
                self.timeout = timeout
                self.retry_options = retry_options

    fake_genai = py_types.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.types = FakeTypesModule
    fake_google = py_types.ModuleType("google")
    fake_google.genai = fake_genai
    sys.modules["google"] = fake_google
    sys.modules["google.genai"] = fake_genai
    sys.modules["google.genai.types"] = FakeTypesModule
    try:
        result = provider_module.review_packet_with_provider(packet, provider="gemini", config=config)
    finally:
        if original_google is None:
            sys.modules.pop("google", None)
        else:
            sys.modules["google"] = original_google
        if original_genai is None:
            sys.modules.pop("google.genai", None)
        else:
            sys.modules["google.genai"] = original_genai
        if original_types is None:
            sys.modules.pop("google.genai.types", None)
        else:
            sys.modules["google.genai.types"] = original_types

    if calls["count"] != 4:
        errors.append("Gemini validation retry should make one Pass 1 retry, one Pass 2 call, and one Pass 2 retry")
    if not calls["repair_prompt_seen"]:
        errors.append("Gemini validation retry prompt should be observed")
    metadata = result.get("provider_metadata") or {}
    if metadata.get("validation_retry_attempts") != 1:
        errors.append("Gemini validation retry should record one validation_retry_attempt")
    if metadata.get("validation_retry_max_attempts") != PROVIDER_VALIDATION_RETRY_ATTEMPTS:
        errors.append("Gemini validation retry should record configured max attempts")
    if not metadata.get("validation_retry_history"):
        errors.append("Gemini validation retry should record retry history")
    if metadata.get("pass2_retry_attempts") != 1:
        errors.append("Gemini Pass 2 retry should record one pass2_retry_attempt")
    if metadata.get("validation_retry_stage") != provider_module.PASS1_REPAIR_STAGE_JSON_SHAPE:
        errors.append("Gemini validation retry should record the repair stage")
    if result.get("status") != provider_module.STATUS_REVIEWED:
        errors.append("Gemini validation retry should recover to reviewed status")
    if metadata.get("pass2_validation_status") != "valid":
        errors.append("Gemini recovered review should validate Pass 2 participant narrative after retry")


def _check_gemini_multi_validation_retry(packet: dict, fixture: dict, errors: list[str]) -> None:
    config = load_narrative_provider_config({
        "NARRATIVE_LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test-key",
        "GEMINI_NARRATIVE_MODEL": "test-gemini-model",
        "NARRATIVE_LLM_MAX_RETRIES": "0",
    })
    original_google = sys.modules.get("google")
    original_genai = sys.modules.get("google.genai")
    original_types = sys.modules.get("google.genai.types")
    calls = {"count": 0}
    invalid_initial = _synthesized_trial_score_pass1_review(packet, fixture)
    invalid_initial["operational_fit"]["combined_operational_fit"]["rating"] = "invented_rating"
    invalid_repair = _synthesized_trial_score_pass1_review(packet, fixture)
    invalid_repair["operational_fit"]["combined_operational_fit"]["rating"] = "still_invented"
    valid_repair = _synthesized_trial_score_pass1_review(packet, fixture)
    pass2_review = {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "trial_score_narrative": {
            "summary": "The Trial Score reading is mixed but defensible.",
            "movement_reading": "Operational Fit and Reality Check should be read together.",
            "score_interpretation": "Completion Outlook remains the protected model-pattern anchor.",
        },
        "pillar_reading": [
                {"pillar": "Execution Framework", "reading": "Operational proportionality matters."},
                {"pillar": "Scientific Challenge", "reading": "Evidence interpretability remains relevant to the participant read."},
            ],
        "central_tension": {
            "summary": "The main tension is whether completion favorability and design defensibility move in the same direction.",
            "why_it_matters": "It frames the participant discussion.",
        },
        "broader_strategic_question": {
            "mapped_tension": "The main tension is whether completion favorability and design defensibility move in the same direction.",
            "question": "When should operational feasibility change confidence in the development story, and when does it only make an uncertain evidence package easier to run?",
        },
    }

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls["count"] += 1
            if calls["count"] == 1:
                return _FakeGeminiResponse(parsed=invalid_initial, text=provider_module.json.dumps(invalid_initial))
            if calls["count"] == 2:
                return _FakeGeminiResponse(parsed=invalid_repair, text=provider_module.json.dumps(invalid_repair))
            if calls["count"] == 3:
                return _FakeGeminiResponse(parsed=valid_repair, text=provider_module.json.dumps(valid_repair))
            return _FakeGeminiResponse(parsed=pass2_review, text=provider_module.json.dumps(pass2_review))

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.models = FakeModels()

    class FakeTypesModule:
        class ThinkingConfig:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        class GenerateContentConfig:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        class HttpRetryOptions:
            def __init__(self, attempts) -> None:
                self.attempts = attempts

        class HttpOptions:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

    fake_genai = py_types.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.types = FakeTypesModule
    fake_google = py_types.ModuleType("google")
    fake_google.genai = fake_genai
    sys.modules["google"] = fake_google
    sys.modules["google.genai"] = fake_genai
    sys.modules["google.genai.types"] = FakeTypesModule
    try:
        result = provider_module.review_packet_with_provider(packet, provider="gemini", config=config)
    finally:
        if original_google is None:
            sys.modules.pop("google", None)
        else:
            sys.modules["google"] = original_google
        if original_genai is None:
            sys.modules.pop("google.genai", None)
        else:
            sys.modules["google.genai"] = original_genai
        if original_types is None:
            sys.modules.pop("google.genai.types", None)
        else:
            sys.modules["google.genai.types"] = original_types

    metadata = result.get("provider_metadata") or {}
    if result.get("status") != provider_module.STATUS_REVIEWED:
        errors.append("Gemini validation repair should allow a second targeted repair when the first repair remains invalid")
    if metadata.get("validation_retry_attempts") != 2:
        errors.append("Gemini multi-repair recovery should record two validation_retry_attempts")
    history = metadata.get("validation_retry_history") or []
    if len(history) != 2:
        errors.append("Gemini multi-repair recovery should record two retry-history entries")
    if not all(item.get("prompt_text") and item.get("response_text") for item in history):
        errors.append("Gemini retry history should preserve per-attempt prompt and response text")


def _check_gemini_staged_pass1_malformed_retry(packet: dict, fixture: dict, errors: list[str]) -> None:
    config = load_narrative_provider_config({
        "NARRATIVE_LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test-key",
        "GEMINI_NARRATIVE_MODEL": "test-gemini-model",
        "NARRATIVE_LLM_MAX_RETRIES": "0",
    })
    original_google = sys.modules.get("google")
    original_genai = sys.modules.get("google.genai")
    original_types = sys.modules.get("google.genai.types")
    calls = {"count": 0}
    retry_review = _synthesized_trial_score_pass1_review(packet, fixture)

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls["count"] += 1
            if calls["count"] == 1:
                return _FakeGeminiResponse(parsed=None, text="not json")
            return _FakeGeminiResponse(parsed=retry_review)

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.models = FakeModels()

    class FakeTypesModule:
        class ThinkingConfig:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        class GenerateContentConfig:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        class HttpRetryOptions:
            def __init__(self, attempts) -> None:
                self.attempts = attempts

        class HttpOptions:
            def __init__(self, timeout, retry_options) -> None:
                self.timeout = timeout
                self.retry_options = retry_options

    fake_genai = py_types.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.types = FakeTypesModule
    fake_google = py_types.ModuleType("google")
    fake_google.genai = fake_genai
    sys.modules["google"] = fake_google
    sys.modules["google.genai"] = fake_genai
    sys.modules["google.genai.types"] = FakeTypesModule
    try:
        result = review_packet_pass1_initial_with_provider(packet, provider="gemini", config=config)
    finally:
        if original_google is None:
            sys.modules.pop("google", None)
        else:
            sys.modules["google"] = original_google
        if original_genai is None:
            sys.modules.pop("google.genai", None)
        else:
            sys.modules["google.genai"] = original_genai
        if original_types is None:
            sys.modules.pop("google.genai.types", None)
        else:
            sys.modules["google.genai.types"] = original_types

    if calls["count"] != 2:
        errors.append("Staged Gemini Pass 1 should retry once after malformed JSON")
    metadata = result.get("provider_metadata") or {}
    if metadata.get("malformed_json_retry_attempts") != 1:
        errors.append("Staged Gemini malformed retry should record malformed_json_retry_attempts")
    if result.get("status") != STATUS_REVIEWED:
        errors.append("Staged Gemini malformed retry should recover to reviewed status")


def _check_validation_stage_classifier(errors: list[str]) -> None:
    cases = [
        (
            ["operational_fit.combined_operational_fit must be an object"],
            provider_module.PASS1_REPAIR_STAGE_OPERATIONAL_FIT,
        ),
        (
            ["combined_operational_fit.rating must be one of ['neutral_or_unclear']"],
            provider_module.PASS1_REPAIR_STAGE_OPERATIONAL_FIT,
        ),
        (
            ["reality_check.allocations[0] must target an allowed allocation_target_id"],
            provider_module.PASS1_REPAIR_STAGE_REALITY_CHECK,
        ),
        (
            ["reality_check.allocations[].allocation_target_id is required"],
            provider_module.PASS1_REPAIR_STAGE_REALITY_CHECK,
        ),
        (
            ["strategy_shift_check.status must not be not_applicable when gated premise-sensitive fields changed"],
            provider_module.PASS1_REPAIR_STAGE_STRATEGY_SHIFT,
        ),
        (
            ["combined_operational_fit.evidence_fields do not reference packet evidence"],
            provider_module.PASS1_REPAIR_STAGE_OPERATIONAL_FIT,
        ),
        (
            ["Pass 1 review must be an object"],
            provider_module.PASS1_REPAIR_STAGE_JSON_SHAPE,
        ),
    ]
    for messages, expected in cases:
        actual = _pass1_repair_stage(messages)
        if actual != expected:
            errors.append(f"expected validation stage {expected!r}, got {actual!r} for {messages!r}")


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
    if mock_result.get("scoring", {}).get("trial_score") is None:
        errors.append("mock provider did not preserve Trial Score scoring result")
    staged_mock_result = review_packet_pass1_initial_with_provider(packet, provider=PROVIDER_MOCK)
    if staged_mock_result.get("provider_metadata", {}).get("workflow_stage") != provider_module.PASS1_INITIAL_STAGE:
        errors.append("staged mock Pass 1 should expose the pass1_initial workflow stage")
    if pass1_result_needs_repair(staged_mock_result):
        errors.append("staged mock Pass 1 should not require repair for the valid operational fixture")
    if staged_mock_result.get("scoring", {}).get("trial_score") is None:
        errors.append("staged mock Pass 1 should preserve Trial Score scoring result")

    baseline_fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "baseline_hidden_review_v2"
    )
    baseline_packet = build_review_packet_from_fixture(baseline_fixture)
    baseline_result = review_packet_with_provider(baseline_packet, provider=PROVIDER_MOCK)
    if baseline_result.get("status") != "reviewed":
        errors.append("provider should review hidden baseline packet through the normal Trial Score review path")
    if baseline_result.get("scoring", {}).get("reality_check_points") is not None:
        errors.append("hidden baseline provider result should not calculate Reality Check")
    if baseline_result.get("scoring", {}).get("trial_score") is not None:
        errors.append("hidden baseline provider result should not calculate Trial Score")

    context_fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "endpoint_text_contradiction_v2"
    )
    context_packet = build_review_packet_from_fixture(context_fixture)
    context_result = review_packet_with_provider(context_packet, provider=PROVIDER_MOCK)
    if context_result.get("status") != "reviewed":
        errors.append("provider should review structured/text context fixture without a clarification gate")
    if context_result.get("scoring", {}).get("trial_score") is None:
        errors.append("structured/text context fixture did not preserve Trial Score scoring result")

    unsupported = review_packet_with_provider(packet, provider="not_configured")
    if unsupported.get("status") != FAILURE_UNSUPPORTED_PROVIDER:
        errors.append("unsupported provider should return unsupported_provider status")
    if unsupported.get("scoring", {}).get("reality_check_points") is not None:
        errors.append("unsupported provider should not return Reality Check")
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
    if GEMINI_PRIMARY_THINKING_LEVEL != "high":
        errors.append("Gemini primary thinking level should be high for clinical-trial coherence reviews")
    if GEMINI_RETRY_THINKING_LEVEL != "low":
        errors.append("Gemini malformed/MAX_TOKENS retry should lower thinking level for completion reliability")
    if GEMINI_RETRY_OUTPUT_TOKENS < 16000:
        errors.append("Gemini retry output budget should be at least 16000 tokens")
    if NARRATIVE_REPAIR_RETRY_ATTEMPTS != 2:
        errors.append("narrative repair/retry cap should stay bounded to two explicit retries")
    if GEMINI_MALFORMED_JSON_RETRY_ATTEMPTS != NARRATIVE_REPAIR_RETRY_ATTEMPTS:
        errors.append("Gemini malformed JSON retry cap should match narrative repair/retry cap")
    if PROVIDER_VALIDATION_RETRY_ATTEMPTS != 3:
        errors.append("provider validation repair cap should stay bounded to three explicit retries")
    if PASS2_VALIDATION_RETRY_ATTEMPTS != NARRATIVE_REPAIR_RETRY_ATTEMPTS:
        errors.append("Pass 2 validation repair cap should match narrative repair/retry cap")
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
        review={"reality_check": "malformed"},
        provider_metadata={},
    )
    if invalid_real_review.get("status") != FAILURE_MALFORMED_RESPONSE:
        errors.append("contract-invalid real provider review should be malformed_response")
    if invalid_real_review.get("scoring", {}).get("reality_check_points") is not None:
        errors.append("contract-invalid real provider review should not return Reality Check")

    review_with_app_score = {
        **_synthesized_trial_score_pass1_review(packet, fixture),
        "reality_check_points": 99,
        "trial_score": 99,
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
    if app_score_result.get("scoring", {}).get("reality_check_points") is not None:
        errors.append("provider-returned app score field should suppress Reality Check")
    if app_score_result.get("scoring", {}).get("trial_score") is not None:
        errors.append("provider-returned app score field should suppress Trial Score")
    if not any(
        "application-owned" in str(error)
        for error in app_score_result.get("scoring", {}).get("validation_errors") or []
    ):
        errors.append("provider-returned app score field should stay visible as a validation warning")

    _check_validation_stage_classifier(errors)
    _check_openai_validation_retry(packet, fixture, errors)
    _check_openai_pass2_retry(packet, fixture, errors)
    _check_openai_pass2_non_json_repair(packet, fixture, errors)
    _check_openai_pass2_exception_warning(packet, fixture, errors)
    _check_gemini_validation_retry(packet, fixture, errors)
    _check_gemini_multi_validation_retry(packet, fixture, errors)
    _check_gemini_staged_pass1_malformed_retry(packet, fixture, errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated narrative provider normalization and unsupported-provider failure behavior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
