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
    build_pass2_input,
    build_pass2_provider_prompt,
    build_provider_prompt,
    gemini_response_schema,
    infer_prompt_mode,
    pass2_gemini_response_schema,
)
from src.narratives.scoring import validate_and_score_review
from src.narratives.trial_score_contract import (
    GATED_PREMISE_SENSITIVE_FIELDS,
    OPERATIONAL_FIT_MATERIALITIES,
    OPERATIONAL_FIT_RATINGS,
    OPERATIONAL_INTERACTION_LABELS,
    REALITY_CHECK_ALLOCATION_TARGETS,
    REALITY_CHECK_EFFECTS,
    REALITY_CHECK_STRENGTHS,
    packet_evidence_refs,
    validate_pass2_review,
)

MOCK_MODEL_NAME = "fixture_hash_mock_v1"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
GEMINI_MIN_SCHEMA_OUTPUT_TOKENS = 12000
GEMINI_PRIMARY_THINKING_LEVEL = "high"
GEMINI_RETRY_THINKING_LEVEL = "low"
GEMINI_RETRY_OUTPUT_TOKENS = 16000
GEMINI_MALFORMED_JSON_RETRY_ATTEMPTS = 1
PROVIDER_VALIDATION_RETRY_ATTEMPTS = 3
PASS2_VALIDATION_RETRY_ATTEMPTS = 1
FAILURE_UNSUPPORTED_PROVIDER = "unsupported_provider"
FAILURE_PROVIDER_UNAVAILABLE = "provider_unavailable"
FAILURE_PROVIDER_ERROR = "provider_error"
FAILURE_INCOMPLETE_RESPONSE = "incomplete_response"
FAILURE_MALFORMED_RESPONSE = "malformed_response"
STATUS_REVIEWED = "reviewed"
PASS1_INITIAL_STAGE = "pass1_initial"
PASS1_REPAIR_STAGE = "pass1_repair"
PASS2_STAGE = "pass2"

PASS1_REPAIR_STAGE_HARD_BOUNDARY = "hard_boundary"
PASS1_REPAIR_STAGE_JSON_SHAPE = "json_shape"
PASS1_REPAIR_STAGE_OPERATIONAL_FIT = "operational_fit"
PASS1_REPAIR_STAGE_REALITY_CHECK = "reality_check"
PASS1_REPAIR_STAGE_EVIDENCE = "evidence_fields"
PASS1_REPAIR_STAGE_STRATEGY_SHIFT = "strategy_shift"
PASS1_REPAIR_STAGE_NARRATIVE_SCAFFOLD = "narrative_scaffold"
PASS1_REPAIR_STAGE_UNKNOWN = "unknown"

def _unavailable_scoring(packet: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "validation_status": "unavailable",
        "validation_errors": [message],
        "operational_fit_points": None,
        "pre_reality_score": None,
        "pre_reality_delta": None,
        "reality_check_points": None,
        "trial_score": None,
        "reality_check_assessment": {},
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
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = {
        "max_output_tokens": max_output_tokens,
        "seed": config.seed,
        "response_mime_type": "application/json",
        "response_schema": response_schema or gemini_response_schema(),
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


def _packet_evidence_refs(packet: dict[str, Any]) -> list[str]:
    return sorted(packet_evidence_refs(packet))


def _pass1_validation_messages(result: dict[str, Any] | None) -> list[str]:
    if not result:
        return []
    scoring = result.get("scoring") or {}
    messages = [str(item) for item in scoring.get("validation_errors") or [] if str(item).strip()]
    notes = [str(item) for item in scoring.get("validation_notes") or [] if str(item).strip()]
    repair_notes = [
        note for note in notes
        if "allocation_target_id" in note
        or "downgraded to neutral because allocation" in note
        or "incremental_check" in note
    ]
    return [*messages, *repair_notes]


def _pass1_repair_stage(messages: list[str]) -> str | None:
    if not messages:
        return None
    joined = " ".join(messages).lower()
    if "application-owned" in joined:
        return PASS1_REPAIR_STAGE_HARD_BOUNDARY
    top_level_shape_errors = [
        message for message in messages
        if " must be an object" in str(message).lower() or " is required" in str(message).lower()
    ]
    if len(top_level_shape_errors) >= 2:
        return PASS1_REPAIR_STAGE_JSON_SHAPE
    if "operational_fit" in joined or "combined_operational_fit" in joined:
        return PASS1_REPAIR_STAGE_OPERATIONAL_FIT
    if "reality_check" in joined or "allocation_target_id" in joined or "incremental_check" in joined:
        return PASS1_REPAIR_STAGE_REALITY_CHECK
    if "strategy_shift_check" in joined or "gated premise-sensitive" in joined:
        return PASS1_REPAIR_STAGE_STRATEGY_SHIFT
    if (
        "analytical_narrative_draft" in joined
        or "alternative_tension_candidates" in joined
        or "alternative_strategic_question_candidates" in joined
    ):
        return PASS1_REPAIR_STAGE_NARRATIVE_SCAFFOLD
    if "evidence_fields" in joined or "packet evidence" in joined:
        return PASS1_REPAIR_STAGE_EVIDENCE
    if "json object" in joined or "must be an object" in joined or "is required" in joined:
        return PASS1_REPAIR_STAGE_JSON_SHAPE
    return PASS1_REPAIR_STAGE_UNKNOWN


def _pass1_repair_failure_message(stage: str | None, messages: list[str], *, after_retry: bool = False) -> str:
    level = {
        PASS1_REPAIR_STAGE_HARD_BOUNDARY: "hard app-owned score boundary",
        PASS1_REPAIR_STAGE_JSON_SHAPE: "Pass 1 Trial Score JSON shape",
        PASS1_REPAIR_STAGE_OPERATIONAL_FIT: "Operational Fit contract",
        PASS1_REPAIR_STAGE_REALITY_CHECK: "Reality Check contract",
        PASS1_REPAIR_STAGE_STRATEGY_SHIFT: "Strategy Shift Check contract",
        PASS1_REPAIR_STAGE_EVIDENCE: "packet evidence references",
        PASS1_REPAIR_STAGE_NARRATIVE_SCAFFOLD: "narrative scaffold",
        PASS1_REPAIR_STAGE_UNKNOWN: "Pass 1 Trial Score contract",
    }.get(stage or PASS1_REPAIR_STAGE_UNKNOWN, "Pass 1 Trial Score contract")
    detail = "; ".join(messages[:4]) if messages else "unknown validation error"
    if after_retry:
        return f"Provider review failed after the repair retry at {level}: {detail}"
    return f"Provider review failed at {level}: {detail}"


def _pass1_repair_prompt(packet: dict[str, Any], review: dict[str, Any] | None, stage: str | None, messages: list[str]) -> str:
    review_json = json.dumps(review or {}, sort_keys=True, separators=(",", ":"), default=str)
    target_lines = [
        f"- {target_id}: {target.get('pillar')} / {target.get('subpillar')} - {target.get('description')}"
        for target_id, target in REALITY_CHECK_ALLOCATION_TARGETS.items()
    ]
    stage_instruction = {
        PASS1_REPAIR_STAGE_OPERATIONAL_FIT: (
            "Change only invalid operational_fit enum fields or evidence_fields. "
            "Do not change the clinical judgment unless required to use an allowed enum."
        ),
        PASS1_REPAIR_STAGE_REALITY_CHECK: (
            "Change only invalid reality_check fields, especially allocations[].allocation_target_id, "
            "effect, strength, evidence_fields, or incremental_check. Do not add pillar/subpillar fields."
        ),
        PASS1_REPAIR_STAGE_STRATEGY_SHIFT: (
            "Change only strategy_shift_check. Because a gated premise-sensitive field changed, "
            "strategy_shift_check.status must be supported, partly_supported, or unsupported_or_incoherent; "
            "do not use not_applicable."
        ),
        PASS1_REPAIR_STAGE_EVIDENCE: (
            "Change only evidence_fields arrays so every item exactly matches one allowed packet evidence reference."
        ),
        PASS1_REPAIR_STAGE_JSON_SHAPE: (
            "Return the same review as a complete Pass 1 JSON object with the missing required objects restored."
        ),
        PASS1_REPAIR_STAGE_NARRATIVE_SCAFFOLD: (
            "Change only analytical_narrative_draft and alternative_tension_candidates. Add or repair the required "
            "qualitative draft fields, at least two non-prescriptive tension options, and at least three wider strategic "
            "question candidates mapped to the selected and alternative tensions. Do not change valid ratings, evidence "
            "fields, strategy_shift_check, or Reality Check allocations. Do not recommend actions or field changes."
        ),
    }.get(stage or "", "Change only the fields named by the validation errors.")
    return (
        "You are repairing a previous Pass 1 Trial Score JSON response. "
        "Do not restart the review and do not rewrite unrelated rationale.\n"
        f"Validation errors/notes:\n{json.dumps(messages, indent=2, default=str)}\n"
        f"Repair scope: {stage or PASS1_REPAIR_STAGE_UNKNOWN}. {stage_instruction}\n"
        "Hard rules:\n"
        "- Return exactly one JSON object and no markdown.\n"
        "- Do not return app-owned score fields such as operational_fit_points, reality_check_points, or trial_score.\n"
        "- analytical_narrative_draft may be extensive and score-aware because it is not participant-facing; do not add app-owned score fields as structured fields.\n"
        "- Tension candidates must frame trade-offs and questions, not recommendations, next steps, or sponsor instructions.\n"
        "- Strategic question candidates must be wider facilitator debate questions mapped to tensions, not narrow checklist questions.\n"
        "- Use only canonical Reality Check allocation_target_id values; do not use free-text pillar/subpillar targets.\n"
        "Allowed Reality Check allocation_target_id values:\n"
        f"{chr(10).join(target_lines)}\n"
        f"Allowed Operational Fit ratings: {sorted(OPERATIONAL_FIT_RATINGS)}\n"
        f"Allowed Operational Fit materiality: {sorted(OPERATIONAL_FIT_MATERIALITIES)}\n"
        f"Allowed Operational Fit interaction labels: {sorted(OPERATIONAL_INTERACTION_LABELS)}\n"
        f"Allowed Reality Check effects: {sorted(REALITY_CHECK_EFFECTS)}\n"
        f"Allowed Reality Check strengths: {sorted(REALITY_CHECK_STRENGTHS)}\n"
        f"Gated premise-sensitive fields: {sorted(GATED_PREMISE_SENSITIVE_FIELDS)}\n"
        "Allowed packet evidence references:\n"
        f"{json.dumps(_packet_evidence_refs(packet), separators=(',', ':'), default=str)}\n"
        "Previous JSON to repair:\n"
        f"{review_json}"
    )


def _pass1_needs_repair(result: dict[str, Any] | None) -> tuple[bool, str | None, list[str]]:
    messages = _pass1_validation_messages(result)
    stage = _pass1_repair_stage(messages)
    if stage == PASS1_REPAIR_STAGE_HARD_BOUNDARY:
        return False, stage, messages
    if result and result.get("status") == FAILURE_MALFORMED_RESPONSE:
        return True, stage, messages
    if stage in {
        PASS1_REPAIR_STAGE_OPERATIONAL_FIT,
        PASS1_REPAIR_STAGE_REALITY_CHECK,
        PASS1_REPAIR_STAGE_STRATEGY_SHIFT,
        PASS1_REPAIR_STAGE_NARRATIVE_SCAFFOLD,
        PASS1_REPAIR_STAGE_EVIDENCE,
    }:
        return True, stage, messages
    return False, stage, messages


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
    is_valid = hidden_baseline or scoring.get("trial_score") is not None
    validation_messages = [
        *[str(item) for item in scoring.get("validation_errors") or [] if str(item).strip()],
        *[str(item) for item in scoring.get("validation_notes") or [] if str(item).strip()],
    ]
    failure_stage = _pass1_repair_stage(validation_messages)
    provider_metadata = dict(provider_metadata or {})
    if validation_messages:
        provider_metadata["pass1_validation_messages"] = validation_messages
    if failure_stage:
        provider_metadata["pass1_validation_stage"] = failure_stage
    if not is_valid:
        scoring = {
            **scoring,
            "operational_fit_points": None,
            "pre_reality_score": None,
            "pre_reality_delta": None,
            "reality_check_points": None,
            "reality_check_assessment": {},
            "trial_score": None,
        }
        failure_reason = _pass1_repair_failure_message(failure_stage, validation_messages)
    else:
        failure_reason = None
    return {
        "review_needed": True,
        "reuse_previous_review": False,
        "provider": provider,
        "model_name": model_name,
        "provider_metadata": provider_metadata,
        "status": STATUS_REVIEWED if is_valid else FAILURE_MALFORMED_RESPONSE,
        "failure_reason": failure_reason,
        "review": review,
        "validated_review": scored["validated_review"],
        "scoring": scoring,
    }


def _attach_participant_narrative(result: dict[str, Any], narrative: dict[str, Any] | None) -> dict[str, Any]:
    if narrative is None:
        return result
    validated = validate_pass2_review(narrative)
    metadata = dict(result.get("provider_metadata") or {})
    metadata["pass2_validation_status"] = validated.get("validation_status")
    if validated.get("validation_errors"):
        metadata["pass2_validation_errors"] = validated.get("validation_errors")
    updated = dict(result)
    updated["provider_metadata"] = metadata
    if validated.get("validation_status") == "valid":
        updated["participant_narrative"] = narrative
        updated["validated_participant_narrative"] = validated
        updated["participant_narrative_status"] = "valid"
    return updated


def _pass2_validation_messages(narrative: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(narrative, dict):
        return (
            {"validation_status": "invalid", "validation_errors": ["Pass 2 provider response was not a JSON object"]},
            ["Pass 2 provider response was not a JSON object"],
        )
    validated = validate_pass2_review(narrative)
    return validated, [str(item) for item in validated.get("validation_errors") or [] if str(item).strip()]


def _pass2_repair_failure_message(messages: list[str], *, after_retry: bool = False) -> str:
    detail = "; ".join(messages[:4]) if messages else "unknown validation error"
    if after_retry:
        return f"Participant narrative failed after the Pass 2 repair retry: {detail}"
    return f"Participant narrative failed Pass 2 validation: {detail}"


def _pass2_provider_error_message(error_type: str, *, retry: bool = False) -> str:
    stage = "Pass 2 repair retry" if retry else "Pass 2 generation"
    return f"Participant narrative failed during {stage}: {error_type}"


def _pass2_repair_prompt(pass2_input: dict[str, Any], narrative: dict[str, Any] | None, messages: list[str]) -> str:
    input_json = json.dumps(pass2_input, sort_keys=True, separators=(",", ":"), default=str)
    previous_json = json.dumps(narrative or {}, sort_keys=True, separators=(",", ":"), default=str)
    return (
        "You are repairing a previous Pass 2 Participant Narrative JSON response. "
        "Do not rerun Pass 1, do not change scores, and do not change the analytical basis.\n"
        f"Validation errors:\n{json.dumps(messages, indent=2, default=str)}\n"
        "Repair scope: change only invalid or missing participant-narrative fields.\n"
        "Hard rules:\n"
        "- Return exactly one JSON object and no markdown.\n"
        "- Do not return app-owned score fields such as operational_fit_points, reality_check_points, or trial_score.\n"
        "- Preserve app_calculated_scores exactly as supplied in Pass 2 input.\n"
        "- Use score_alignment_notes only to calibrate qualitative direction/materiality.\n"
        "- Remove exact Trial Score values, point values, and numeric contribution language from participant-facing prose.\n"
        "- Do not reanalyze, re-rate Operational Fit, re-decide Reality Check, or introduce new claims beyond Pass 1.\n"
        "- Keep one integrated Trial Score narrative, central tension, and broader strategic question.\n"
        "- facilitator_questions are optional and must include at most three questions.\n"
        "Pass 2 input JSON:\n"
        f"{input_json}\n"
        "Previous Pass 2 JSON to repair:\n"
        f"{previous_json}"
    )


def _pass2_result_with_warning(
    result: dict[str, Any],
    *,
    metadata: dict[str, Any],
    message: str,
    narrative: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(result)
    updated["provider_metadata"] = metadata
    updated["participant_narrative"] = narrative
    updated["participant_narrative_status"] = "invalid"
    updated["participant_narrative_warning"] = message
    return updated


def _pass2_input_for_result(packet: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    scoring = result.get("scoring") or {}
    pass1_review = result.get("validated_review") or {}
    review_mode = str((pass1_review.get("review_metadata") or {}).get("review_mode") or "")
    if review_mode == "hidden_baseline" or scoring.get("trial_score") is None:
        return None
    return build_pass2_input(packet, pass1_review, scoring)


def _call_openai_pass2(
    packet: dict[str, Any],
    result: dict[str, Any],
    *,
    settings: ProviderSettings,
    config: NarrativeProviderConfig,
) -> dict[str, Any]:
    pass2_input = _pass2_input_for_result(packet, result)
    if pass2_input is None:
        return result
    metadata = dict(result.get("provider_metadata") or {})
    prompt = build_pass2_provider_prompt(pass2_input)
    metadata["pass2_input"] = pass2_input
    metadata["pass2_prompt_text"] = prompt
    metadata["pass2_prompt_template_version"] = PROMPT_TEMPLATE_VERSION
    metadata["pass2_response_schema_version"] = pass2_input.get("schema_version")
    started_at = time.monotonic()
    try:
        response = requests.post(
            OPENAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.model,
                "input": prompt,
                "max_output_tokens": config.max_output_tokens,
                "reasoning": {"effort": config.openai_reasoning_effort},
                "text": {"format": {"type": "json_object"}},
            },
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        response_text = _openai_response_text(response.json())
        metadata["pass2_http_status"] = response.status_code
        metadata["pass2_response_text"] = response_text
        metadata["pass2_response_text_length"] = len(response_text)
        narrative = _parse_json_object(response_text)
        metadata["pass2_parsed_json_object"] = isinstance(narrative, dict)
    except Exception as exc:
        metadata["pass2_error_type"] = exc.__class__.__name__
        return _pass2_result_with_warning(
            result,
            metadata=metadata,
            message=_pass2_provider_error_message(exc.__class__.__name__),
            narrative=None,
        )
    finally:
        metadata["pass2_latency_ms"] = int(round((time.monotonic() - started_at) * 1000))
    updated = dict(result)
    updated["provider_metadata"] = metadata
    validated, pass2_errors = _pass2_validation_messages(narrative)
    metadata["pass2_validation_status"] = validated.get("validation_status")
    if pass2_errors:
        metadata["pass2_validation_errors"] = pass2_errors
    if validated.get("validation_status") == "valid":
        return _attach_participant_narrative(updated, narrative)

    repair_prompt = _pass2_repair_prompt(pass2_input, narrative, pass2_errors)
    metadata["pass2_repair_prompt_text"] = repair_prompt
    metadata["pass2_retry_reason"] = _pass2_repair_failure_message(pass2_errors)
    for retry_attempt in range(1, PASS2_VALIDATION_RETRY_ATTEMPTS + 1):
        metadata["pass2_retry_attempts"] = retry_attempt
        retry_started_at = time.monotonic()
        try:
            retry_response = requests.post(
                OPENAI_RESPONSES_URL,
                headers={
                    "Authorization": f"Bearer {settings.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.model,
                    "input": repair_prompt,
                    "max_output_tokens": config.max_output_tokens,
                    "reasoning": {"effort": config.openai_reasoning_effort},
                    "text": {"format": {"type": "json_object"}},
                },
                timeout=config.timeout_seconds,
            )
            retry_response.raise_for_status()
            retry_text = _openai_response_text(retry_response.json())
            metadata["pass2_retry_http_status"] = retry_response.status_code
            metadata["pass2_retry_response_text"] = retry_text
            metadata["pass2_retry_response_text_length"] = len(retry_text)
            retry_narrative = _parse_json_object(retry_text)
            metadata["pass2_retry_parsed_json_object"] = isinstance(retry_narrative, dict)
        except Exception as exc:
            metadata["pass2_retry_error_type"] = exc.__class__.__name__
            pass2_errors = [_pass2_provider_error_message(exc.__class__.__name__, retry=True)]
            break
        finally:
            metadata["pass2_retry_latency_ms"] = int(round((time.monotonic() - retry_started_at) * 1000))
        retry_validated, retry_errors = _pass2_validation_messages(retry_narrative)
        metadata["pass2_retry_validation_status"] = retry_validated.get("validation_status")
        if retry_errors:
            metadata["pass2_retry_validation_errors"] = retry_errors
        if retry_validated.get("validation_status") == "valid":
            updated["provider_metadata"] = metadata
            return _attach_participant_narrative(updated, retry_narrative)
        pass2_errors = retry_errors
        narrative = retry_narrative

    final_error = _pass2_repair_failure_message(pass2_errors, after_retry=True)
    metadata["pass2_retry_final_error"] = final_error
    return _pass2_result_with_warning(updated, metadata=metadata, message=final_error, narrative=narrative)


def _call_gemini_pass2(
    packet: dict[str, Any],
    result: dict[str, Any],
    *,
    client: Any,
    types_module: Any,
    settings: ProviderSettings,
    config: NarrativeProviderConfig,
    thinking_level: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    pass2_input = _pass2_input_for_result(packet, result)
    if pass2_input is None:
        return result
    metadata = dict(result.get("provider_metadata") or {})
    prompt = build_pass2_provider_prompt(pass2_input)
    metadata["pass2_input"] = pass2_input
    metadata["pass2_prompt_text"] = prompt
    generation_config = types_module.GenerateContentConfig(
        **_gemini_generation_config_kwargs(
            config,
            max_output_tokens=max_output_tokens,
            response_schema=pass2_gemini_response_schema(),
        ),
        thinking_config=types_module.ThinkingConfig(thinking_level=thinking_level),
    )
    metadata["pass2_prompt_template_version"] = PROMPT_TEMPLATE_VERSION
    metadata["pass2_response_schema_version"] = pass2_input.get("schema_version")
    started_at = time.monotonic()
    try:
        response = client.models.generate_content(
            model=settings.model,
            contents=prompt,
            config=generation_config,
        )
        parsed_payload = getattr(response, "parsed", None)
        response_text = str(getattr(response, "text", "") or "")
        metadata["pass2_parsed_payload_type"] = type(parsed_payload).__name__ if parsed_payload is not None else None
        metadata["pass2_response_text"] = response_text
        metadata["pass2_response_text_length"] = len(response_text)
        narrative = parsed_payload if isinstance(parsed_payload, dict) else _parse_json_object(response_text)
        metadata["pass2_parsed_json_object"] = isinstance(narrative, dict)
    except Exception as exc:
        metadata["pass2_error_type"] = exc.__class__.__name__
        return _pass2_result_with_warning(
            result,
            metadata=metadata,
            message=_pass2_provider_error_message(exc.__class__.__name__),
            narrative=None,
        )
    finally:
        metadata["pass2_latency_ms"] = int(round((time.monotonic() - started_at) * 1000))
    updated = dict(result)
    updated["provider_metadata"] = metadata
    validated, pass2_errors = _pass2_validation_messages(narrative)
    metadata["pass2_validation_status"] = validated.get("validation_status")
    if pass2_errors:
        metadata["pass2_validation_errors"] = pass2_errors
    if validated.get("validation_status") == "valid":
        return _attach_participant_narrative(updated, narrative)

    repair_prompt = _pass2_repair_prompt(pass2_input, narrative, pass2_errors)
    metadata["pass2_repair_prompt_text"] = repair_prompt
    metadata["pass2_retry_reason"] = _pass2_repair_failure_message(pass2_errors)
    for retry_attempt in range(1, PASS2_VALIDATION_RETRY_ATTEMPTS + 1):
        metadata["pass2_retry_attempts"] = retry_attempt
        retry_started_at = time.monotonic()
        try:
            response = client.models.generate_content(
                model=settings.model,
                contents=repair_prompt,
                config=generation_config,
            )
            parsed_payload = getattr(response, "parsed", None)
            response_text = str(getattr(response, "text", "") or "")
            metadata["pass2_retry_parsed_payload_type"] = type(parsed_payload).__name__ if parsed_payload is not None else None
            metadata["pass2_retry_response_text"] = response_text
            metadata["pass2_retry_response_text_length"] = len(response_text)
            retry_narrative = parsed_payload if isinstance(parsed_payload, dict) else _parse_json_object(response_text)
            metadata["pass2_retry_parsed_json_object"] = isinstance(retry_narrative, dict)
        except Exception as exc:
            metadata["pass2_retry_error_type"] = exc.__class__.__name__
            pass2_errors = [_pass2_provider_error_message(exc.__class__.__name__, retry=True)]
            break
        finally:
            metadata["pass2_retry_latency_ms"] = int(round((time.monotonic() - retry_started_at) * 1000))
        retry_validated, retry_errors = _pass2_validation_messages(retry_narrative)
        metadata["pass2_retry_validation_status"] = retry_validated.get("validation_status")
        if retry_errors:
            metadata["pass2_retry_validation_errors"] = retry_errors
        if retry_validated.get("validation_status") == "valid":
            updated["provider_metadata"] = metadata
            return _attach_participant_narrative(updated, retry_narrative)
        pass2_errors = retry_errors
        narrative = retry_narrative

    final_error = _pass2_repair_failure_message(pass2_errors, after_retry=True)
    metadata["pass2_retry_final_error"] = final_error
    return _pass2_result_with_warning(updated, metadata=metadata, message=final_error, narrative=narrative)


def _max_attempts(config: NarrativeProviderConfig) -> int:
    return max(1, int(config.max_retries) + 1)


def _gemini_http_options(config: NarrativeProviderConfig, types_module: Any) -> Any:
    """Build Gemini SDK HTTP controls from the app-owned runtime config."""
    return types_module.HttpOptions(
        timeout=int(config.timeout_seconds) * 1000,
        retry_options=types_module.HttpRetryOptions(attempts=_max_attempts(config)),
    )


def _review_packet_two_pass_with_provider(
    packet: dict[str, Any],
    *,
    provider: str,
    config: NarrativeProviderConfig,
) -> dict[str, Any]:
    result = review_packet_pass1_initial_with_provider(packet, provider=provider, config=config)
    if pass1_result_needs_repair(result):
        result = review_packet_pass1_repair_with_provider(packet, result, provider=provider, config=config)
    if result.get("status") == STATUS_REVIEWED:
        result = review_packet_pass2_with_provider(packet, result, provider=provider, config=config)
    return result


def _call_openai_pass1_initial(
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
    metadata["workflow_stage"] = PASS1_INITIAL_STAGE
    metadata["pass1_prompt_text"] = prompt
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
        reason = incomplete_details.get("reason") if isinstance(incomplete_details, dict) else None
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
    metadata["pass1_response_text"] = response_text
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


def _call_gemini_pass1_initial(
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
    metadata["workflow_stage"] = PASS1_INITIAL_STAGE
    metadata["pass1_prompt_text"] = prompt
    response = None
    client = None
    generation_config = None
    last_error = None
    started_at = time.monotonic()
    try:
        from google import genai
        from google.genai import types
        generation_config = types.GenerateContentConfig(
            **_gemini_generation_config_kwargs(config, max_output_tokens=max_output_tokens),
            thinking_config=types.ThinkingConfig(thinking_level=primary_thinking_level),
        )
        client = genai.Client(api_key=settings.api_key, http_options=_gemini_http_options(config, types))
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
    except Exception as exc:
        last_error = exc
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
    metadata["pass1_response_text"] = response_text
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
            metadata["malformed_json_retry_response_text"] = response_text
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
    return _score_provider_review(
        packet,
        provider=PROVIDER_GEMINI,
        model_name=settings.model,
        review=review,
        provider_metadata=metadata,
    )


def _call_openai_pass1_repair(
    packet: dict[str, Any],
    result: dict[str, Any],
    *,
    config: NarrativeProviderConfig,
    settings: ProviderSettings,
) -> dict[str, Any]:
    needs_repair, repair_stage, repair_messages = _pass1_needs_repair(result)
    if not needs_repair:
        return result
    metadata = dict(result.get("provider_metadata") or {})
    metadata["workflow_stage"] = PASS1_REPAIR_STAGE
    metadata["validation_retry_reason"] = result.get("failure_reason")
    metadata["validation_retry_stage"] = repair_stage
    metadata["validation_retry_max_attempts"] = PROVIDER_VALIDATION_RETRY_ATTEMPTS
    retry_history: list[dict[str, Any]] = []
    metadata["validation_retry_history"] = retry_history
    current_result = result
    current_stage = repair_stage
    current_messages = repair_messages
    for retry_attempt in range(1, PROVIDER_VALIDATION_RETRY_ATTEMPTS + 1):
        attempt_record = {
            "attempt": retry_attempt,
            "stage": current_stage,
            "messages": list(current_messages or []),
        }
        retry_history.append(attempt_record)
        repair_prompt = _pass1_repair_prompt(packet, current_result.get("review"), current_stage, current_messages)
        attempt_record["prompt_text"] = repair_prompt
        metadata["pass1_repair_prompt_text"] = repair_prompt
        metadata["validation_retry_stage"] = current_stage
        metadata["validation_retry_attempts"] = retry_attempt
        started_at = time.monotonic()
        try:
            response = requests.post(
                OPENAI_RESPONSES_URL,
                headers={
                    "Authorization": f"Bearer {settings.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.model,
                    "input": repair_prompt,
                    "max_output_tokens": config.max_output_tokens,
                    "reasoning": {"effort": config.openai_reasoning_effort},
                    "text": {"format": {"type": "json_object"}},
                },
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            response_text = _openai_response_text(response.json())
            metadata["validation_retry_http_status"] = response.status_code
            metadata["validation_retry_response_text"] = response_text
            metadata["validation_retry_response_text_length"] = len(response_text)
            attempt_record["response_text"] = response_text
            retry_review = _parse_json_object(response_text)
            metadata["validation_retry_parsed_json_object"] = isinstance(retry_review, dict)
            attempt_record["parsed_json_object"] = isinstance(retry_review, dict)
            attempt_record["response_text_length"] = len(response_text)
        except Exception as exc:
            metadata["validation_retry_error_type"] = exc.__class__.__name__
            attempt_record["error_type"] = exc.__class__.__name__
            break
        finally:
            latency_ms = int(round((time.monotonic() - started_at) * 1000))
            metadata["validation_retry_latency_ms"] = latency_ms
            attempt_record["latency_ms"] = latency_ms
        if retry_review is None:
            current_messages = ["Provider repair response was not a JSON object."]
            attempt_record["validation_status"] = "invalid"
            attempt_record["remaining_messages"] = list(current_messages)
            continue
        repaired = _score_provider_review(
            packet,
            provider=PROVIDER_OPENAI,
            model_name=settings.model,
            review=retry_review,
            provider_metadata=metadata,
        )
        retry_needs_repair, current_stage, current_messages = _pass1_needs_repair(repaired)
        attempt_record["validation_status"] = (repaired.get("scoring") or {}).get("validation_status")
        attempt_record["remaining_stage"] = current_stage if retry_needs_repair else None
        attempt_record["remaining_messages"] = list(current_messages or []) if retry_needs_repair else []
        if not retry_needs_repair:
            return repaired
        current_result = repaired
        metadata.update(repaired.get("provider_metadata") or {})

    metadata["validation_retry_final_stage"] = current_stage
    metadata["validation_retry_final_error"] = _pass1_repair_failure_message(
        current_stage,
        current_messages,
        after_retry=True,
    )
    updated = dict(current_result)
    updated["provider_metadata"] = metadata
    updated["failure_reason"] = metadata["validation_retry_final_error"]
    return updated


def _call_gemini_pass1_repair(
    packet: dict[str, Any],
    result: dict[str, Any],
    *,
    config: NarrativeProviderConfig,
    settings: ProviderSettings,
) -> dict[str, Any]:
    needs_repair, repair_stage, repair_messages = _pass1_needs_repair(result)
    if not needs_repair:
        return result
    metadata = dict(result.get("provider_metadata") or {})
    metadata["workflow_stage"] = PASS1_REPAIR_STAGE
    metadata["validation_retry_reason"] = result.get("failure_reason")
    metadata["validation_retry_stage"] = repair_stage
    metadata["validation_retry_max_attempts"] = PROVIDER_VALIDATION_RETRY_ATTEMPTS
    retry_history: list[dict[str, Any]] = []
    metadata["validation_retry_history"] = retry_history
    max_output_tokens = max(GEMINI_RETRY_OUTPUT_TOKENS, max(int(config.max_output_tokens), GEMINI_MIN_SCHEMA_OUTPUT_TOKENS))
    try:
        from google import genai
        from google.genai import types
        generation_config = types.GenerateContentConfig(
            **_gemini_generation_config_kwargs(config, max_output_tokens=max_output_tokens),
            thinking_config=types.ThinkingConfig(thinking_level=GEMINI_RETRY_THINKING_LEVEL),
        )
        client = genai.Client(api_key=settings.api_key, http_options=_gemini_http_options(config, types))
    except Exception as exc:
        metadata["validation_retry_error_type"] = exc.__class__.__name__
        updated = dict(result)
        updated["provider_metadata"] = metadata
        updated["failure_reason"] = _pass1_repair_failure_message(repair_stage, repair_messages, after_retry=True)
        return updated
    current_result = result
    current_stage = repair_stage
    current_messages = repair_messages
    for retry_attempt in range(1, PROVIDER_VALIDATION_RETRY_ATTEMPTS + 1):
        attempt_record = {
            "attempt": retry_attempt,
            "stage": current_stage,
            "messages": list(current_messages or []),
        }
        retry_history.append(attempt_record)
        repair_prompt = _pass1_repair_prompt(packet, current_result.get("review"), current_stage, current_messages)
        attempt_record["prompt_text"] = repair_prompt
        metadata["pass1_repair_prompt_text"] = repair_prompt
        metadata["validation_retry_stage"] = current_stage
        metadata["validation_retry_attempts"] = retry_attempt
        started_at = time.monotonic()
        try:
            response = client.models.generate_content(
                model=settings.model,
                contents=repair_prompt,
                config=generation_config,
            )
            parsed_payload = getattr(response, "parsed", None)
            response_text = str(getattr(response, "text", "") or "")
            _record_gemini_response_metadata(metadata, response)
            metadata["validation_retry_parsed_payload_type"] = type(parsed_payload).__name__ if parsed_payload is not None else None
            metadata["validation_retry_response_text"] = response_text
            metadata["validation_retry_response_text_length"] = len(response_text)
            attempt_record["response_text"] = response_text
            retry_review = parsed_payload if isinstance(parsed_payload, dict) else _parse_json_object(response_text)
            metadata["validation_retry_parsed_json_object"] = isinstance(retry_review, dict)
            attempt_record["parsed_json_object"] = isinstance(retry_review, dict)
            attempt_record["parsed_payload_type"] = type(parsed_payload).__name__ if parsed_payload is not None else None
            attempt_record["response_text_length"] = len(response_text)
        except Exception as exc:
            metadata["validation_retry_error_type"] = exc.__class__.__name__
            attempt_record["error_type"] = exc.__class__.__name__
            break
        finally:
            latency_ms = int(round((time.monotonic() - started_at) * 1000))
            metadata["validation_retry_latency_ms"] = latency_ms
            attempt_record["latency_ms"] = latency_ms
        if retry_review is None:
            current_messages = ["Provider repair response was not a JSON object."]
            attempt_record["validation_status"] = "invalid"
            attempt_record["remaining_messages"] = list(current_messages)
            continue
        repaired = _score_provider_review(
            packet,
            provider=PROVIDER_GEMINI,
            model_name=settings.model,
            review=retry_review,
            provider_metadata=metadata,
        )
        retry_needs_repair, current_stage, current_messages = _pass1_needs_repair(repaired)
        attempt_record["validation_status"] = (repaired.get("scoring") or {}).get("validation_status")
        attempt_record["remaining_stage"] = current_stage if retry_needs_repair else None
        attempt_record["remaining_messages"] = list(current_messages or []) if retry_needs_repair else []
        if not retry_needs_repair:
            return repaired
        current_result = repaired
        metadata.update(repaired.get("provider_metadata") or {})

    metadata["validation_retry_final_stage"] = current_stage
    metadata["validation_retry_final_error"] = _pass1_repair_failure_message(
        current_stage,
        current_messages,
        after_retry=True,
    )
    updated = dict(current_result)
    updated["provider_metadata"] = metadata
    updated["failure_reason"] = metadata["validation_retry_final_error"]
    return updated


def review_packet_pass1_initial_with_provider(
    packet: dict[str, Any],
    *,
    provider: str = PROVIDER_MOCK,
    config: NarrativeProviderConfig | None = None,
) -> dict[str, Any]:
    provider = str(provider or PROVIDER_MOCK).strip().lower()
    if provider == PROVIDER_MOCK:
        return _normalize_provider_result(
            review_packet_with_mock(packet),
            provider=PROVIDER_MOCK,
            model_name=MOCK_MODEL_NAME,
            provider_metadata={"deterministic": True, "workflow_stage": PASS1_INITIAL_STAGE},
        )
    if config is None:
        return _failure_result(
            packet,
            provider=provider,
            model_name=None,
            status=FAILURE_PROVIDER_UNAVAILABLE,
            message=f"Narrative provider config is required for {provider}.",
        )
    settings = config.provider_settings(provider)
    if settings is None or not settings.has_api_key:
        return _failure_result(
            packet,
            provider=provider,
            model_name=settings.model if settings else None,
            status=FAILURE_PROVIDER_UNAVAILABLE,
            message=f"{provider} provider is missing an API key.",
        )
    if provider == PROVIDER_OPENAI:
        return _call_openai_pass1_initial(packet, config=config, settings=settings)
    if provider == PROVIDER_GEMINI:
        return _call_gemini_pass1_initial(packet, config=config, settings=settings)
    return _failure_result(
        packet,
        provider=provider,
        model_name=None,
        status=FAILURE_UNSUPPORTED_PROVIDER,
        message=f"Unsupported narrative provider: {provider}",
    )


def review_packet_pass1_repair_with_provider(
    packet: dict[str, Any],
    result: dict[str, Any],
    *,
    provider: str = PROVIDER_MOCK,
    config: NarrativeProviderConfig | None = None,
) -> dict[str, Any]:
    provider = str(provider or PROVIDER_MOCK).strip().lower()
    if not _pass1_needs_repair(result)[0]:
        return result
    if provider == PROVIDER_MOCK:
        return result
    if config is None:
        return result
    settings = config.provider_settings(provider)
    if settings is None or not settings.has_api_key:
        return result
    if provider == PROVIDER_OPENAI:
        return _call_openai_pass1_repair(packet, result, config=config, settings=settings)
    if provider == PROVIDER_GEMINI:
        return _call_gemini_pass1_repair(packet, result, config=config, settings=settings)
    return result


def pass1_result_needs_repair(result: dict[str, Any] | None) -> bool:
    return _pass1_needs_repair(result)[0]


def review_packet_pass2_with_provider(
    packet: dict[str, Any],
    result: dict[str, Any],
    *,
    provider: str = PROVIDER_MOCK,
    config: NarrativeProviderConfig | None = None,
) -> dict[str, Any]:
    provider = str(provider or PROVIDER_MOCK).strip().lower()
    if provider == PROVIDER_MOCK or result.get("status") != STATUS_REVIEWED:
        return result
    if config is None:
        return result
    settings = config.provider_settings(provider)
    if settings is None or not settings.has_api_key:
        return result
    if provider == PROVIDER_OPENAI:
        return _call_openai_pass2(packet, result, settings=settings, config=config)
    if provider == PROVIDER_GEMINI:
        max_output_tokens = max(int(config.max_output_tokens), GEMINI_MIN_SCHEMA_OUTPUT_TOKENS)
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=settings.api_key, http_options=_gemini_http_options(config, types))
        except Exception as exc:
            metadata = dict(result.get("provider_metadata") or {})
            metadata["pass2_error_type"] = exc.__class__.__name__
            return _pass2_result_with_warning(
                result,
                metadata=metadata,
                message=_pass2_provider_error_message(exc.__class__.__name__),
                narrative=None,
            )
        return _call_gemini_pass2(
            packet,
            result,
            client=client,
            types_module=types,
            settings=settings,
            config=config,
            thinking_level=_gemini_primary_thinking_level(config),
            max_output_tokens=max_output_tokens,
        )
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
        return _review_packet_two_pass_with_provider(packet, provider=provider, config=config)

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
