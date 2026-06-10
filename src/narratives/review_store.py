"""Session-state-compatible storage and replay for narrative reviews."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, MutableMapping

from src.narratives.provider import (
    MOCK_MODEL_NAME,
    PROVIDER_MOCK,
    review_packet_with_provider,
    review_packet_with_provider_chain,
)
from src.narratives.provider_config import NarrativeProviderConfig, provider_config_cache_namespace

NARRATIVE_REVIEW_STATE_KEY = "narrative_review_store_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _initial_store() -> dict[str, Any]:
    return {
        "reviews_by_hash": {},
        "trace_history": [],
        "latest_trace_by_session": {},
    }


def get_review_store(state: MutableMapping[str, Any]) -> dict[str, Any]:
    """Return the narrative review store from a session-state-like mapping."""
    store = state.setdefault(NARRATIVE_REVIEW_STATE_KEY, _initial_store())
    store.setdefault("reviews_by_hash", {})
    store.setdefault("trace_history", [])
    store.setdefault("latest_trace_by_session", {})
    return store


def compact_storyline_from_trace(trace: dict[str, Any] | None) -> str:
    """Extract compact storyline memory from a stored trace."""
    if not trace:
        return ""
    validated = trace.get("validated_review") or {}
    continuity = validated.get("continuity") or {}
    storyline = continuity.get("storyline_update")
    if isinstance(storyline, str) and storyline.strip():
        return storyline.strip()
    return str(trace.get("compact_storyline_memory") or "").strip()


def latest_trace_for_session(state: MutableMapping[str, Any], session_id: str) -> dict[str, Any] | None:
    store = get_review_store(state)
    trace_id = store["latest_trace_by_session"].get(str(session_id))
    if trace_id is None:
        return None
    for trace in reversed(store["trace_history"]):
        if trace.get("trace_id") == trace_id:
            return deepcopy(trace)
    return None


def _trace_id(session_id: str, input_hash: str, iteration_id: Any) -> str:
    return f"{session_id}:{iteration_id}:{input_hash}"


def _cache_key(
    input_hash: str | None,
    provider: str | None,
    model_name: str | None,
    cache_namespace: str | None = None,
) -> str | None:
    if not input_hash:
        return None
    provider_key = str(provider or PROVIDER_MOCK).strip().lower()
    model_key = str(model_name or (MOCK_MODEL_NAME if provider_key == PROVIDER_MOCK else "")).strip()
    namespace_key = str(cache_namespace or "").strip()
    return f"{provider_key}:{model_key}:{namespace_key}:{input_hash}"


def _build_trace(
    *,
    packet: dict[str, Any],
    review_result: dict[str, Any],
    session_id: str,
    baseline_id: str | None = None,
    cached: bool = False,
    cache_namespace: str | None = None,
) -> dict[str, Any]:
    iteration_context = packet.get("iteration_context") or {}
    scoring = review_result.get("scoring") or {}
    validated_review = review_result.get("validated_review")
    design_confidence = scoring.get("design_confidence")
    total_scenario_score = scoring.get("total_scenario_score")
    design_confidence_assessment = deepcopy(scoring.get("design_confidence_assessment") or {})
    provider_trace = (validated_review or {}).get("trace") or {}
    tradeoff_review = (validated_review or {}).get("tradeoff_review") or {}
    reference_pack_ids_available = [
        pack.get("pack_id")
        for pack in packet.get("reference_packs") or []
        if isinstance(pack, dict) and pack.get("pack_id")
    ]
    available_reference_pack_set = set(reference_pack_ids_available)
    raw_reference_pack_ids_used = [
        str(pack_id)
        for pack_id in provider_trace.get("reference_pack_ids_used") or []
        if str(pack_id).strip()
    ]
    supported_reference_pack_ids_used = [
        pack_id
        for pack_id in raw_reference_pack_ids_used
        if pack_id in available_reference_pack_set
    ]
    unsupported_reference_pack_ids_used = [
        pack_id
        for pack_id in raw_reference_pack_ids_used
        if pack_id not in available_reference_pack_set
    ]
    input_hash = str(packet.get("input_hash") or scoring.get("input_hash") or "")
    iteration_id = iteration_context.get("iteration_number")
    trace = {
        "trace_id": _trace_id(session_id, input_hash, iteration_id),
        "session_id": session_id,
        "baseline_id": baseline_id or iteration_context.get("baseline_snapshot_id"),
        "iteration_id": iteration_id,
        "timestamp": _utc_now(),
        "provider": review_result.get("provider"),
        "model_name": review_result.get("model_name"),
        "cache_namespace": cache_namespace,
        "provider_metadata": deepcopy(review_result.get("provider_metadata") or {}),
        "status": review_result.get("status"),
        "cached": cached,
        "review_needed": review_result.get("review_needed"),
        "reuse_previous_review": review_result.get("reuse_previous_review"),
        "input_hash": input_hash,
        "prompt_version": packet.get("prompt_version"),
        "rubric_version": packet.get("rubric_version"),
        "input_packet": deepcopy(packet),
        "output_json": deepcopy(review_result.get("review")),
        "validated_review": deepcopy(validated_review),
        "validation_status": scoring.get("validation_status"),
        "validation_errors": deepcopy(scoring.get("validation_errors") or []),
        "design_confidence": design_confidence,
        "total_scenario_score": total_scenario_score,
        "design_confidence_assessment": design_confidence_assessment,
        "design_confidence_subcategories": deepcopy(
            (validated_review or {}).get("design_confidence_subcategories") or {}
        ),
        "design_confidence_contributions": deepcopy(design_confidence_assessment.get("subcategories") or {}),
        "central_tension": tradeoff_review.get("central_tension"),
        "reference_pack_ids_available": reference_pack_ids_available,
        "reference_pack_ids_used": supported_reference_pack_ids_used,
        "unsupported_reference_pack_ids_used": unsupported_reference_pack_ids_used,
        # Temporary aliases for the old simulator panel until Phase 6 migrates UI labels.
        "quality_adjustment": design_confidence,
        "final_candidate_score": total_scenario_score,
        "quality_assessment": design_confidence_assessment,
        "failure_reason": review_result.get("failure_reason"),
        "clarification_issues": deepcopy(review_result.get("clarification_issues") or []),
        "user_clarifications": deepcopy((packet.get("clarification_context") or {}).get("user_clarifications") or []),
        "changed_fields": deepcopy(iteration_context.get("changed_fields") or []),
        "score_delta": (packet.get("model_interpretation") or {}).get("score_delta"),
        "score_movement": (packet.get("model_interpretation") or {}).get("score_delta"),
        "compact_storyline_memory": compact_storyline_from_trace({"validated_review": validated_review}),
    }
    return trace


def store_review_trace(
    state: MutableMapping[str, Any],
    *,
    packet: dict[str, Any],
    review_result: dict[str, Any],
    session_id: str,
    baseline_id: str | None = None,
    cached: bool = False,
    cache_namespace: str | None = None,
) -> dict[str, Any]:
    """Persist a review trace and cache it by input hash when appropriate."""
    store = get_review_store(state)
    trace = _build_trace(
        packet=packet,
        review_result=review_result,
        session_id=session_id,
        baseline_id=baseline_id,
        cached=cached,
        cache_namespace=cache_namespace,
    )
    input_hash = trace["input_hash"]

    if input_hash and review_result.get("status") in {"reviewed", "reused_previous_review"}:
        cache_key = _cache_key(
            input_hash,
            trace.get("provider"),
            trace.get("model_name"),
            trace.get("cache_namespace"),
        )
        if cache_key:
            store["reviews_by_hash"][cache_key] = deepcopy(trace)

    store["trace_history"].append(deepcopy(trace))
    store["latest_trace_by_session"][str(session_id)] = trace["trace_id"]
    return deepcopy(trace)


def cached_review_trace(
    state: MutableMapping[str, Any],
    input_hash: str | None,
    *,
    provider: str = PROVIDER_MOCK,
    model_name: str | None = None,
    cache_namespace: str | None = None,
) -> dict[str, Any] | None:
    cache_key = _cache_key(input_hash, provider, model_name, cache_namespace)
    if not cache_key:
        return None
    trace = get_review_store(state)["reviews_by_hash"].get(cache_key)
    return deepcopy(trace) if trace else None


def cached_review_trace_for_namespace(
    state: MutableMapping[str, Any],
    input_hash: str | None,
    *,
    cache_namespace: str | None,
) -> dict[str, Any] | None:
    """Return any reusable provider-chain review for the same input/settings."""
    if not input_hash or not cache_namespace:
        return None
    for trace in reversed(get_review_store(state)["trace_history"]):
        if trace.get("input_hash") != input_hash:
            continue
        if trace.get("cache_namespace") != cache_namespace:
            continue
        if trace.get("status") not in {"reviewed", "reused_previous_review"}:
            continue
        return deepcopy(trace)
    return None


def replay_or_review_with_provider(
    state: MutableMapping[str, Any],
    *,
    packet: dict[str, Any],
    session_id: str,
    baseline_id: str | None = None,
    provider: str = PROVIDER_MOCK,
    model_name: str | None = None,
    failure_mode: str | None = None,
    config: NarrativeProviderConfig | None = None,
    use_provider_chain: bool = False,
) -> dict[str, Any]:
    """Reuse cached review traces for identical inputs, otherwise call provider."""
    input_hash = packet.get("input_hash")
    cache_namespace = provider_config_cache_namespace(config) if use_provider_chain and config is not None else None
    if failure_mode is None:
        if use_provider_chain and config is not None:
            cached = cached_review_trace_for_namespace(
                state,
                str(input_hash) if input_hash else None,
                cache_namespace=cache_namespace,
            )
        else:
            cached = cached_review_trace(
                state,
                str(input_hash) if input_hash else None,
                provider=provider,
                model_name=model_name,
                cache_namespace=cache_namespace,
            )
        if cached is not None:
            iteration_id = (packet.get("iteration_context") or {}).get("iteration_number")
            cached["cached"] = True
            cached["timestamp"] = _utc_now()
            cached["session_id"] = session_id
            cached["baseline_id"] = baseline_id or (packet.get("iteration_context") or {}).get("baseline_snapshot_id")
            cached["iteration_id"] = iteration_id
            cached["trace_id"] = _trace_id(str(session_id), str(input_hash), iteration_id)
            store = get_review_store(state)
            store["trace_history"].append(deepcopy(cached))
            store["latest_trace_by_session"][str(session_id)] = cached["trace_id"]
            return cached

    previous_session_trace = latest_trace_for_session(state, session_id)
    if use_provider_chain and config is not None and failure_mode is None:
        review_result = review_packet_with_provider_chain(packet, config=config)
    else:
        review_result = review_packet_with_provider(
            packet,
            provider=provider,
            model_name=model_name,
            failure_mode=failure_mode,
            config=config,
        )
    if (
        review_result.get("status") == "reused_previous_review"
        and previous_session_trace
        and previous_session_trace.get("status") in {"reviewed", "reused_previous_review"}
    ):
        review_result = {
            **review_result,
            "review": deepcopy(previous_session_trace.get("output_json")),
            "validated_review": deepcopy(previous_session_trace.get("validated_review")),
            "scoring": {
                "validation_status": previous_session_trace.get("validation_status"),
                "validation_errors": deepcopy(previous_session_trace.get("validation_errors") or []),
                "design_confidence": previous_session_trace.get("design_confidence"),
                "total_scenario_score": previous_session_trace.get("total_scenario_score"),
                "design_confidence_assessment": deepcopy(
                    previous_session_trace.get("design_confidence_assessment") or {}
                ),
                "input_hash": packet.get("input_hash"),
            },
        }
    return store_review_trace(
        state,
        packet=packet,
        review_result=review_result,
        session_id=session_id,
        baseline_id=baseline_id,
        cached=False,
        cache_namespace=cache_namespace,
    )


def replay_or_review_with_mock(
    state: MutableMapping[str, Any],
    *,
    packet: dict[str, Any],
    session_id: str,
    baseline_id: str | None = None,
    failure_mode: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible mock-provider replay helper."""
    return replay_or_review_with_provider(
        state,
        packet=packet,
        session_id=session_id,
        baseline_id=baseline_id,
        provider=PROVIDER_MOCK,
        failure_mode=failure_mode,
    )
