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
from src.narratives.review_controls import apply_review_control_overrides, attach_review_controls
from src.narratives.storyline import build_storyline_state

NARRATIVE_REVIEW_STATE_KEY = "narrative_review_store_v2"


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
    metadata = validated.get("review_metadata") or {}
    central_tension = validated.get("central_tension_candidate") or {}
    continuity_update = validated.get("continuity_update") or {}
    if str(metadata.get("review_mode") or "") == "hidden_baseline":
        summary = str(central_tension.get("summary") or "").strip()
        if summary:
            watch_next = str(continuity_update.get("watch_next") or "").strip()
            memory = f"Baseline tension: {summary}"
            if watch_next:
                memory = f"{memory} Next watch: {watch_next}"
            return memory
    if isinstance(continuity_update.get("what_changed"), str) and continuity_update.get("what_changed").strip():
        return continuity_update["what_changed"].strip()
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
    participant_narrative = review_result.get("participant_narrative")
    validated_participant_narrative = review_result.get("validated_participant_narrative")
    operational_fit_points = scoring.get("operational_fit_points")
    pre_reality_score = scoring.get("pre_reality_score")
    pre_reality_delta = scoring.get("pre_reality_delta")
    reality_check_points = scoring.get("reality_check_points")
    reality_check_assessment = deepcopy(scoring.get("reality_check_assessment") or {})
    reality_check_allocation_points = deepcopy(scoring.get("reality_check_allocation_points") or [])
    trial_score = scoring.get("trial_score")
    provider_trace = (validated_review or {}).get("trace") or {}
    operational_fit = (validated_review or {}).get("operational_fit") or {}
    central_tension_candidate = (validated_review or {}).get("central_tension_candidate") or {}
    broader_strategic_question_candidate = (validated_review or {}).get("broader_strategic_question_candidate") or {}
    continuity_update = (validated_review or {}).get("continuity_update") or {}
    tradeoff_review = (validated_review or {}).get("tradeoff_review") or {}
    main_tension = (
        central_tension_candidate.get("summary")
        or (validated_review or {}).get("main_tension")
        or tradeoff_review.get("central_tension")
    )
    storyline_state = build_storyline_state(validated_review)
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
        "repair_warning": (review_result.get("provider_metadata") or {}).get("validation_retry_final_error"),
        "cached": cached,
        "review_needed": review_result.get("review_needed"),
        "reuse_previous_review": review_result.get("reuse_previous_review"),
        "input_hash": input_hash,
        "scenario_state_hash": packet.get("scenario_state_hash"),
        "prompt_version": packet.get("prompt_version"),
        "rubric_version": packet.get("rubric_version"),
        "input_packet": deepcopy(packet),
        "output_json": deepcopy(review_result.get("review")),
        "participant_narrative_json": deepcopy(participant_narrative),
        "participant_narrative_status": review_result.get("participant_narrative_status"),
        "participant_narrative_warning": review_result.get("participant_narrative_warning"),
        "validated_review": deepcopy(validated_review),
        "validated_participant_narrative": deepcopy(validated_participant_narrative),
        "validation_status": scoring.get("validation_status"),
        "validation_errors": deepcopy(scoring.get("validation_errors") or []),
        "xgboost_completion_outlook": scoring.get("xgboost_completion_outlook"),
        "operational_fit_points": operational_fit_points,
        "operational_fit_assessment": deepcopy(scoring.get("operational_fit_assessment") or {}),
        "pre_reality_score": pre_reality_score,
        "pre_reality_delta": pre_reality_delta,
        "reality_check_points": reality_check_points,
        "reality_check_assessment": reality_check_assessment,
        "reality_check_allocation_points": reality_check_allocation_points,
        "trial_score_diagnostics": {
            "xgboost_completion_outlook": scoring.get("xgboost_completion_outlook"),
            "operational_fit_points": operational_fit_points,
            "pre_reality_score": pre_reality_score,
            "pre_reality_delta": pre_reality_delta,
            "reality_check_points": reality_check_points,
            "trial_score": trial_score,
            "delta_vs_previous_trial_score": scoring.get("delta_vs_previous_trial_score"),
            "delta_vs_previous_pre_reality_score": scoring.get("delta_vs_previous_pre_reality_score"),
            "delta_vs_baseline_xgboost": scoring.get("delta_vs_baseline_xgboost"),
        },
        "trial_score": trial_score,
        "operational_fit": deepcopy(operational_fit),
        "central_tension_candidate": deepcopy(central_tension_candidate),
        "broader_strategic_question_candidate": deepcopy(broader_strategic_question_candidate),
        "continuity_update": deepcopy(continuity_update),
        "storyline_state": deepcopy(storyline_state),
        "completion_outlook_analysis": deepcopy(
            (validated_review or {}).get("completion_outlook_analysis") or {}
        ),
        "key_questions": deepcopy((validated_review or {}).get("key_questions") or {}),
        "trial_score_narrative": deepcopy((validated_participant_narrative or {}).get("trial_score_narrative") or {}),
        "participant_pillar_reading": deepcopy((validated_participant_narrative or {}).get("pillar_reading") or []),
        "participant_central_tension": deepcopy((validated_participant_narrative or {}).get("central_tension") or {}),
        "participant_broader_strategic_question": deepcopy(
            (validated_participant_narrative or {}).get("broader_strategic_question") or {}
        ),
        "facilitator_questions": deepcopy((validated_participant_narrative or {}).get("facilitator_questions") or []),
        "scenario_consistency_note": deepcopy((validated_review or {}).get("scenario_consistency_note") or {}),
        "central_tension": main_tension,
        "reference_pack_ids_available": reference_pack_ids_available,
        "reference_pack_ids_used": supported_reference_pack_ids_used,
        "unsupported_reference_pack_ids_used": unsupported_reference_pack_ids_used,
        "therapeutic_area_pack_used": provider_trace.get("therapeutic_area_pack_used"),
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
    packet = attach_review_controls(packet)
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
                "xgboost_completion_outlook": previous_session_trace.get("xgboost_completion_outlook"),
                "operational_fit_points": previous_session_trace.get("operational_fit_points"),
                "operational_fit_assessment": deepcopy(
                    previous_session_trace.get("operational_fit_assessment") or {}
                ),
                "pre_reality_score": previous_session_trace.get("pre_reality_score"),
                "pre_reality_delta": previous_session_trace.get("pre_reality_delta"),
                "reality_check_points": previous_session_trace.get("reality_check_points"),
                "reality_check_assessment": deepcopy(
                    previous_session_trace.get("reality_check_assessment") or {}
                ),
                "reality_check_allocation_points": deepcopy(
                    previous_session_trace.get("reality_check_allocation_points") or []
                ),
                "trial_score": previous_session_trace.get("trial_score"),
                "input_hash": packet.get("input_hash"),
            },
        }
    review_result = apply_review_control_overrides(packet, review_result)
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
