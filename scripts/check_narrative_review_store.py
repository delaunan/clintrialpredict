#!/usr/bin/env python
"""Validate session-state-compatible narrative review storage and replay."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.mock_reviewer import FAILURE_PROVIDER_ERROR  # noqa: E402
from src.narratives.packet_builder import build_review_packet, build_review_packet_from_fixture  # noqa: E402
from src.narratives.review_store import (  # noqa: E402
    NARRATIVE_REVIEW_STATE_KEY,
    cached_review_trace,
    compact_storyline_from_trace,
    latest_trace_for_session,
    replay_or_review_with_mock,
    replay_or_review_with_provider,
    store_review_trace,
)
from src.narratives.provider_config import load_narrative_provider_config  # noqa: E402
from src.narratives.trial_score_contract import (  # noqa: E402
    SCORE_TRACE_HISTORY_LIMIT,
    operational_fit_state_hash,
    operational_fit_state_payload,
    xgboost_structured_state_hash,
    xgboost_structured_state_payload,
)


def main() -> int:
    errors: list[str] = []
    state: dict = {}
    session_id = "fixture-session"

    fixtures = get_contract_fixtures()
    review_fixture = next(item for item in fixtures if item["fixture_id"] == "score_improves_evidence_weakens_v2")
    packet = build_review_packet_from_fixture(review_fixture)

    first = replay_or_review_with_mock(state, packet=packet, session_id=session_id)
    if first.get("cached") is not False:
        errors.append("first review should not be cached")
    if first.get("operational_fit_points") is None:
        errors.append("stored trace did not preserve Operational Fit points")
    if not first.get("operational_fit_state_hash"):
        errors.append("stored trace did not preserve Operational Fit state hash")
    if not first.get("operational_fit_state_payload"):
        errors.append("stored trace did not preserve Operational Fit state payload")
    if not first.get("xgboost_structured_state_hash"):
        errors.append("stored trace did not preserve XGBoost structured state hash")
    if not first.get("xgboost_structured_state_payload"):
        errors.append("stored trace did not preserve XGBoost structured state payload")
    if first.get("reality_check_points") is None:
        errors.append("stored trace did not preserve Reality Check points")
    if first.get("trial_score") is None:
        errors.append("stored trace did not preserve Trial Score")
    if not first.get("operational_fit_assessment"):
        errors.append("stored trace should preserve Operational Fit assessment")
    if not first.get("reality_check_assessment"):
        errors.append("stored trace should preserve Reality Check assessment")
    if not first.get("central_tension"):
        errors.append("stored trace should preserve central_tension")
    if not (first.get("trial_score_narrative") or {}).get("summary"):
        errors.append("stored trace should preserve Pass 2 Trial Score narrative")
    if not (first.get("participant_central_tension") or {}).get("summary"):
        errors.append("stored trace should preserve Pass 2 participant central discussion topic")
    if not (first.get("participant_broader_strategic_question") or {}).get("question"):
        errors.append("stored trace should preserve Pass 2 broader strategic question")
    if not (first.get("participant_broader_strategic_question") or {}).get("mapped_tension"):
        errors.append("stored trace should preserve Pass 2 broader strategic question mapped development issue")
    first_visible_history = first.get("recent_participant_visible_questions") or []
    if len(first_visible_history) != 1:
        errors.append("first stored trace should initialize participant-visible question history")
    elif first_visible_history[-1].get("mapped_tension") != (
        first.get("participant_broader_strategic_question") or {}
    ).get("mapped_tension"):
        errors.append("participant-visible question history should preserve selected question mapped development issue")
    storyline_state = first.get("storyline_state") or {}
    if not storyline_state:
        errors.append("stored trace should expose app-owned storyline_state")
    if storyline_state.get("active_tension") != first.get("central_tension"):
        errors.append("storyline_state active_tension should mirror stored central_tension")
    expected_main_tension = (first.get("participant_central_tension") or {}).get("summary")
    if first.get("central_tension") != expected_main_tension:
        errors.append("stored trace central_tension should prefer Pass 2 participant central_tension.summary")
    if "strategic_context_2026_v1" not in set(first.get("reference_pack_ids_available") or []):
        errors.append("stored trace should preserve available reference pack IDs")
    if first.get("unsupported_reference_pack_ids_used"):
        errors.append("fixture-backed review should not report unsupported reference pack IDs")
    if first.get("model_name") != "fixture_hash_mock_v1":
        errors.append("stored trace did not preserve provider model name")
    if first.get("provider_metadata", {}).get("deterministic") is not True:
        errors.append("stored trace did not preserve provider metadata")
    if not compact_storyline_from_trace(first):
        errors.append("stored trace should expose compact storyline memory")
    first_score_history = first.get("recent_score_traces") or []
    if len(first_score_history) != 1:
        errors.append("first stored trace should initialize compact score trace history")
    elif first_score_history[-1].get("trial_score") != first.get("trial_score"):
        errors.append("first compact score trace should describe the stored score")
    elif not first_score_history[-1].get("operational_fit_state_hash"):
        errors.append("first compact score trace should preserve Operational Fit state hash")
    forbidden_score_trace_keys = {
        "input_packet",
        "output_json",
        "participant_narrative_json",
        "validated_review",
        "validated_participant_narrative",
        "trial_score_narrative",
    }
    if any(forbidden_score_trace_keys.intersection(item) for item in first_score_history if isinstance(item, dict)):
        errors.append("compact score trace history should not store bulky narrative payloads")

    cached = cached_review_trace(state, packet["input_hash"])
    if not cached:
        errors.append("review trace was not cached by input hash")

    second = replay_or_review_with_mock(state, packet=packet, session_id=session_id)
    if second.get("cached") is not True:
        errors.append("second identical review should replay from cache")

    cross_session = replay_or_review_with_mock(state, packet=packet, session_id="second-session")
    if cross_session.get("cached") is not True:
        errors.append("cross-session identical review should replay from cache")
    if cross_session.get("session_id") != "second-session":
        errors.append("cached replay should use the current session_id")

    noop_fixture = next(item for item in fixtures if item["expected_behavior"].get("review_needed") is False)
    noop_packet = build_review_packet_from_fixture(noop_fixture)
    noop_trace = replay_or_review_with_mock(state, packet=noop_packet, session_id=session_id)
    if noop_trace.get("status") != "reused_previous_review":
        errors.append("no-op fixture should store reused_previous_review status")
    if noop_trace.get("review_needed") is not False:
        errors.append("no-op fixture should not require a fresh review")
    if not noop_trace.get("validated_review"):
        errors.append("no-op fixture should carry forward the previous validated review")

    failure_trace = replay_or_review_with_mock(
        state,
        packet=packet,
        session_id="failure-session",
        failure_mode=FAILURE_PROVIDER_ERROR,
    )
    if failure_trace.get("reality_check_points") is not None:
        errors.append("provider failure should not store Reality Check")
    if failure_trace.get("failure_reason") is None:
        errors.append("provider failure should store a failure reason")
    cached_failure = cached_review_trace(state, failure_trace.get("input_hash"))
    if cached_failure and cached_failure.get("status") == "provider_error":
        errors.append("provider failure traces should not be cached as reusable reviews")

    unsupported_pack_result = {
        "review_needed": True,
        "reuse_previous_review": False,
        "provider": "mock",
        "model_name": "fixture_hash_mock_v1",
        "provider_metadata": {"deterministic": True},
        "status": "reviewed",
        "failure_reason": None,
        "review": first.get("output_json"),
        "validated_review": {
            **(first.get("validated_review") or {}),
            "trace": {
                **((first.get("validated_review") or {}).get("trace") or {}),
                "reference_pack_ids_used": [
                    "strategic_context_2026_v1",
                    "not_in_packet_v1",
                ],
            },
        },
        "scoring": {
            "validation_status": first.get("validation_status"),
            "validation_errors": first.get("validation_errors") or [],
            "operational_fit_points": first.get("operational_fit_points"),
            "operational_fit_state_hash": first.get("operational_fit_state_hash"),
            "operational_fit_state_payload": first.get("operational_fit_state_payload") or {},
            "reality_check_points": first.get("reality_check_points"),
            "trial_score": first.get("trial_score"),
            "operational_fit_assessment": first.get("operational_fit_assessment") or {},
            "reality_check_assessment": first.get("reality_check_assessment") or {},
            "input_hash": packet.get("input_hash"),
        },
    }
    unsupported_pack_trace = store_review_trace(
        state,
        packet={**packet, "input_hash": f"{packet['input_hash']}-manual-unsupported-pack-check"},
        review_result=unsupported_pack_result,
        session_id="unsupported-pack-session-manual",
    )
    if unsupported_pack_trace.get("reference_pack_ids_used") != ["strategic_context_2026_v1"]:
        errors.append("store should keep only provider-used reference pack IDs available in the packet")
    if unsupported_pack_trace.get("unsupported_reference_pack_ids_used") != ["not_in_packet_v1"]:
        errors.append("store should preserve unsupported provider-returned reference pack IDs for audit")

    history_result = {
        "review_needed": True,
        "reuse_previous_review": False,
        "provider": "mock",
        "model_name": "fixture_hash_mock_v1",
        "provider_metadata": {"deterministic": True},
        "status": "reviewed",
        "failure_reason": None,
        "review": first.get("output_json"),
        "participant_narrative": first.get("participant_narrative_json"),
        "validated_review": first.get("validated_review"),
        "validated_participant_narrative": first.get("validated_participant_narrative"),
        "scoring": {
            "validation_status": first.get("validation_status"),
            "validation_errors": first.get("validation_errors") or [],
            "operational_fit_points": first.get("operational_fit_points"),
            "operational_fit_state_hash": first.get("operational_fit_state_hash"),
            "operational_fit_state_payload": first.get("operational_fit_state_payload") or {},
            "reality_check_points": first.get("reality_check_points"),
            "trial_score": first.get("trial_score"),
            "operational_fit_assessment": first.get("operational_fit_assessment") or {},
            "reality_check_assessment": first.get("reality_check_assessment") or {},
            "input_hash": f"{packet['input_hash']}-manual-history-check",
        },
    }
    prior_score_trace = {
        **(first_score_history[-1] if first_score_history else {}),
        "input_packet": {"should_not": "persist_in_score_history"},
        "output_json": {"should_not": "persist_in_score_history"},
    }
    history_trace = store_review_trace(
        state,
        packet={
            **packet,
            "input_hash": f"{packet['input_hash']}-manual-history-check",
            "iteration_context": {
                **(packet.get("iteration_context") or {}),
                "trial_score_continuity": {
                    "recent_score_traces": [prior_score_trace],
                },
            },
            "review_context": {
                "previous_review": {
                    "recent_participant_visible_questions": [
                        {
                            "question": "How should earlier evidence ambition be debated?",
                            "mapped_tension": "Evidence ambition versus feasibility.",
                        },
                        {
                            "question": "How should earlier governance burden be debated?",
                            "mapped_tension": "Governance burden versus interpretability.",
                        },
                    ],
                },
            },
        },
        review_result=history_result,
        session_id="history-session-manual",
    )
    visible_history = history_trace.get("recent_participant_visible_questions") or []
    current_question = (first.get("participant_broader_strategic_question") or {}).get("question")
    if len(visible_history) != 3:
        errors.append("participant-visible question history should retain previous questions and append current question")
    elif visible_history[-1].get("question") != current_question:
        errors.append("participant-visible question history should put the current question last")
    score_history = history_trace.get("recent_score_traces") or []
    if len(score_history) != 2:
        errors.append("stored trace should carry previous compact score trace history and append current trace")
    elif score_history[-1].get("input_hash") != f"{packet['input_hash']}-manual-history-check":
        errors.append("stored trace should put the current compact score trace last")
    if any(forbidden_score_trace_keys.intersection(item) for item in score_history if isinstance(item, dict)):
        errors.append("stored rolling score trace history should sanitize raw prior trace payloads")
    if score_history and not score_history[0].get("operational_fit_state_hash"):
        errors.append("stored rolling score trace history should preserve prior Operational Fit state hash")

    long_prior_score_traces = [
        {
            "iteration_id": index,
            "input_hash": f"older-score-trace-{index}",
            "trial_score": 50 + index,
            "pre_reality_score": 49 + index,
            "operational_fit_points": 1.0,
            "operational_fit_state_hash": f"operational-state-{index}",
            "operational_fit_state_payload": {"index": index},
            "xgboost_structured_state_hash": f"structured-state-{index}",
            "xgboost_structured_state_payload": {"index": index},
            "reality_check_points": 0.0,
        }
        for index in range(SCORE_TRACE_HISTORY_LIMIT + 5)
    ]
    long_history_packet = {
        **packet,
        "input_hash": f"{packet['input_hash']}-manual-long-history-check",
        "iteration_context": {
            **(packet.get("iteration_context") or {}),
            "trial_score_continuity": {
                "recent_score_traces": long_prior_score_traces,
            },
        },
    }
    long_history_hash = operational_fit_state_hash(long_history_packet)
    long_structured_hash = xgboost_structured_state_hash(long_history_packet)
    long_history_trace = store_review_trace(
        state,
        packet=long_history_packet,
        review_result={
            **history_result,
            "scoring": {
                **history_result["scoring"],
                "input_hash": long_history_packet.get("input_hash"),
                "operational_fit_state_hash": long_history_hash,
                "operational_fit_state_payload": operational_fit_state_payload(long_history_packet),
                "xgboost_structured_state_hash": long_structured_hash,
                "xgboost_structured_state_payload": xgboost_structured_state_payload(long_history_packet),
            },
        },
        session_id="long-score-history-manual",
    )
    long_score_history = long_history_trace.get("recent_score_traces") or []
    if len(long_score_history) != SCORE_TRACE_HISTORY_LIMIT:
        errors.append(f"stored score trace history should retain exactly {SCORE_TRACE_HISTORY_LIMIT} compact traces")
    elif long_score_history[-1].get("operational_fit_state_hash") != long_history_hash:
        errors.append("stored score trace history should retain current Operational Fit state hash after trimming")
    elif long_score_history[-1].get("xgboost_structured_state_hash") != long_structured_hash:
        errors.append("stored score trace history should retain current XGBoost structured state hash after trimming")

    lifecycle_packet = build_review_packet(
        current_snapshot={
            "snapshot_id": "score-history-lifecycle-current",
            "structured_features": packet.get("structured_features", {}),
            "operational_assumptions": packet.get("operational_assumptions", {}),
            "model_interpretation": packet.get("model_interpretation", {}),
            "changed_fields": ["phase_ml"],
            "changed_operational_assumptions": [],
        },
        previous_snapshot={"snapshot_id": "score-history-lifecycle-previous", "score": first.get("trial_score")},
        baseline_snapshot={"snapshot_id": "score-history-lifecycle-baseline"},
        previous_review_trace=first,
        compact_storyline_memory=first.get("compact_storyline_memory"),
    )
    lifecycle_continuity = lifecycle_packet.get("iteration_context", {}).get("trial_score_continuity") or {}
    lifecycle_previous_history = lifecycle_continuity.get("recent_score_traces") or []
    if len(lifecycle_previous_history) != 1:
        errors.append("packet builder should not duplicate previous trace already present in stored score history")
    lifecycle_trace = store_review_trace(
        state,
        packet=lifecycle_packet,
        review_result={
            **history_result,
            "scoring": {
                **history_result["scoring"],
                "input_hash": lifecycle_packet.get("input_hash"),
            },
        },
        session_id="score-history-lifecycle-session",
    )
    lifecycle_score_history = lifecycle_trace.get("recent_score_traces") or []
    lifecycle_identities = [
        (item.get("input_hash"), item.get("iteration_id"))
        for item in lifecycle_score_history
        if isinstance(item, dict)
    ]
    if len(lifecycle_score_history) != 2:
        errors.append("stored lifecycle score history should contain previous trace and current trace once each")
    if len(set(lifecycle_identities)) != len(lifecycle_identities):
        errors.append("stored lifecycle score history should not duplicate trace identities")

    invalid_pass2_result = {
        "review_needed": True,
        "reuse_previous_review": False,
        "provider": "mock",
        "model_name": "fixture_hash_mock_v1",
        "provider_metadata": {"deterministic": True},
        "status": "reviewed",
        "failure_reason": None,
        "review": first.get("output_json"),
        "participant_narrative": {"central_tension": {"summary": "Invalid Pass 2 was not selected."}},
        "participant_narrative_status": "invalid",
        "participant_narrative_warning": "Synthetic invalid Pass 2 check.",
        "validated_review": {
            **(first.get("validated_review") or {}),
            "development_discussion_options": [
                {
                    "topic": "First Pass 1 option should remain only a candidate.",
                    "why_it_matters": "Pass 2 did not validly select it.",
                    "supporting_evidence": ["phase_ml"],
                    "participant_wider_question": {
                        "question": "This candidate question should not become visible history.",
                        "supporting_evidence": ["phase_ml"],
                    },
                },
                {
                    "topic": "Second Pass 1 option should also remain only a candidate.",
                    "why_it_matters": "Pass 2 did not validly select it.",
                    "supporting_evidence": ["endpoint_rigor_ml"],
                    "participant_wider_question": {
                        "question": "This second candidate should not become visible history either.",
                        "supporting_evidence": ["endpoint_rigor_ml"],
                    },
                },
            ],
        },
        "validated_participant_narrative": {},
        "scoring": {
            "validation_status": first.get("validation_status"),
            "validation_errors": first.get("validation_errors") or [],
            "operational_fit_points": first.get("operational_fit_points"),
            "reality_check_points": first.get("reality_check_points"),
            "trial_score": first.get("trial_score"),
            "operational_fit_assessment": first.get("operational_fit_assessment") or {},
            "reality_check_assessment": first.get("reality_check_assessment") or {},
            "input_hash": f"{packet['input_hash']}-manual-invalid-pass2-check",
        },
    }
    invalid_pass2_trace = store_review_trace(
        state,
        packet={
            **packet,
            "input_hash": f"{packet['input_hash']}-manual-invalid-pass2-check",
            "review_context": {
                "previous_review": {
                    "recent_participant_visible_questions": [
                        {
                            "question": "Previously visible question should remain the only history item.",
                            "mapped_tension": "Previously visible development issue.",
                        },
                    ],
                },
            },
        },
        review_result=invalid_pass2_result,
        session_id="invalid-pass2-session-manual",
    )
    if invalid_pass2_trace.get("central_tension"):
        errors.append("invalid Pass 2 should not promote the first Pass 1 option to central_tension")
    invalid_storyline = invalid_pass2_trace.get("storyline_state") or {}
    if invalid_storyline.get("active_tension"):
        errors.append("invalid Pass 2 should not promote the first Pass 1 option to storyline_state.active_tension")
    invalid_history = invalid_pass2_trace.get("recent_participant_visible_questions") or []
    if len(invalid_history) != 1 or invalid_history[-1].get("question") != "Previously visible question should remain the only history item.":
        errors.append("invalid Pass 2 should not append Pass 1 candidate questions to participant-visible history")

    second_option_result = {
        **invalid_pass2_result,
        "participant_narrative": {
            "central_tension": {
                "summary": "Second Pass 1 option should be selectable.",
                "why_it_matters": "Pass 2 selected the second option.",
            },
            "broader_strategic_question": {
                "mapped_tension": "Second Pass 1 option should be selectable.",
                "question": "When should the second option become the visible debate?",
            },
        },
        "participant_narrative_status": "valid",
        "participant_narrative_warning": None,
        "validated_review": {
            **(first.get("validated_review") or {}),
            "development_discussion_options": [
                {
                    "topic": "First Pass 1 option should not be selected.",
                    "why_it_matters": "Pass 2 chose another option.",
                    "supporting_evidence": ["phase_ml"],
                    "participant_wider_question": {
                        "question": "This first candidate should not become visible.",
                        "supporting_evidence": ["phase_ml"],
                    },
                },
                {
                    "topic": "Second Pass 1 option should be selectable.",
                    "why_it_matters": "Pass 2 selected the second option.",
                    "supporting_evidence": ["endpoint_rigor_ml"],
                    "participant_wider_question": {
                        "question": "When should the second option become the visible debate?",
                        "supporting_evidence": ["endpoint_rigor_ml"],
                    },
                },
            ],
        },
        "validated_participant_narrative": {
            "central_tension": {
                "summary": "Second Pass 1 option should be selectable.",
                "why_it_matters": "Pass 2 selected the second option.",
            },
            "broader_strategic_question": {
                "mapped_tension": "Second Pass 1 option should be selectable.",
                "question": "When should the second option become the visible debate?",
            },
        },
        "scoring": {
            **invalid_pass2_result["scoring"],
            "input_hash": f"{packet['input_hash']}-manual-second-option-check",
        },
    }
    second_option_trace = store_review_trace(
        state,
        packet={**packet, "input_hash": f"{packet['input_hash']}-manual-second-option-check"},
        review_result=second_option_result,
        session_id="second-option-session-manual",
    )
    if second_option_trace.get("central_tension") != "Second Pass 1 option should be selectable.":
        errors.append("valid Pass 2 second-option selection should become stored central_tension")
    second_option_history = second_option_trace.get("recent_participant_visible_questions") or []
    if not second_option_history or second_option_history[-1].get("mapped_tension") != "Second Pass 1 option should be selectable.":
        errors.append("valid Pass 2 second-option selection should become participant-visible history")

    app_reality_effect_result = {
        **second_option_result,
        "validated_review": {
            **(second_option_result.get("validated_review") or {}),
            "reality_check": {
                **((second_option_result.get("validated_review") or {}).get("reality_check") or {}),
                "effect": "penalize_incoherence",
                "strength": "moderate",
            },
        },
        "scoring": {
            **second_option_result["scoring"],
            "input_hash": f"{packet['input_hash']}-manual-app-reality-effect-check",
            "reality_check_assessment": {
                **(second_option_result["scoring"].get("reality_check_assessment") or {}),
                "effect": "neutral",
                "strength": "none",
            },
        },
    }
    app_reality_effect_trace = store_review_trace(
        state,
        packet={**packet, "input_hash": f"{packet['input_hash']}-manual-app-reality-effect-check"},
        review_result=app_reality_effect_result,
        session_id="app-reality-effect-session-manual",
    )
    if (app_reality_effect_trace.get("storyline_state") or {}).get("last_effect_label") != "neutral":
        errors.append("storyline_state.last_effect_label should use accepted Reality Check effect")

    missing_key_config = load_narrative_provider_config({
        "NARRATIVE_LLM_PROVIDER": "openai",
        "NARRATIVE_LLM_FALLBACK_PROVIDER": "mock",
    })
    chain_trace = replay_or_review_with_provider(
        state,
        packet=packet,
        session_id="chain-session",
        config=missing_key_config,
        use_provider_chain=True,
    )
    if chain_trace.get("provider") != "mock":
        errors.append("review store provider-chain path should use fallback provider")
    if chain_trace.get("provider_metadata", {}).get("fallback_after", {}).get("provider") != "openai":
        errors.append("review store provider-chain trace should preserve fallback metadata")

    cached_chain_trace = replay_or_review_with_provider(
        state,
        packet=packet,
        session_id="chain-session-repeat",
        config=missing_key_config,
        use_provider_chain=True,
    )
    if cached_chain_trace.get("provider") != "mock" or cached_chain_trace.get("cached") is not True:
        errors.append("provider-chain fallback result should replay from fallback cache when primary is unavailable")

    available_primary_same_settings = load_narrative_provider_config({
        "NARRATIVE_LLM_PROVIDER": "openai",
        "NARRATIVE_LLM_FALLBACK_PROVIDER": "mock",
        "OPENAI_API_KEY": "fake-key-for-cache-check",
    })
    cached_despite_primary_key = replay_or_review_with_provider(
        state,
        packet=packet,
        session_id="chain-session-primary-now-configured",
        config=available_primary_same_settings,
        use_provider_chain=True,
    )
    if cached_despite_primary_key.get("provider") != "mock" or cached_despite_primary_key.get("cached") is not True:
        errors.append("provider-chain cache should reuse same-scenario review before calling newly available primary")

    latest = latest_trace_for_session(state, session_id)
    if not latest:
        errors.append("latest trace lookup failed")

    store = state.get(NARRATIVE_REVIEW_STATE_KEY, {})
    if len(store.get("trace_history", [])) < 5:
        errors.append("trace history should include first, cached replays, no-op, and failure traces")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated narrative review storage, cache replay, no-op storage, and failure traces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
