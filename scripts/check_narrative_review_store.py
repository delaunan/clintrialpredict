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
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402
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
    if first.get("design_confidence") != review_fixture["expected_behavior"]["expected_design_confidence"]:
        errors.append("stored trace did not preserve Design Confidence")
    if first.get("total_scenario_score") != review_fixture["expected_behavior"]["expected_total_scenario_score"]:
        errors.append("stored trace did not preserve Total Scenario Score")
    if first.get("quality_adjustment") != first.get("design_confidence"):
        errors.append("temporary Quality Adjustment alias should mirror Design Confidence until UI migration")
    if first.get("final_candidate_score") != first.get("total_scenario_score"):
        errors.append("temporary Final Candidate Score alias should mirror Total Scenario Score until UI migration")
    if not first.get("design_confidence_assessment", {}).get("subcategories"):
        errors.append("stored trace should preserve Design Confidence assessment")
    if not first.get("central_tension"):
        errors.append("stored trace should preserve central_tension")
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
    if failure_trace.get("design_confidence") is not None:
        errors.append("provider failure should not store Design Confidence")
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
            "design_confidence": first.get("design_confidence"),
            "total_scenario_score": first.get("total_scenario_score"),
            "design_confidence_assessment": first.get("design_confidence_assessment") or {},
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
