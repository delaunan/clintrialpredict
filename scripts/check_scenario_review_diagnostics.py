#!/usr/bin/env python
"""Validate Scenario Review diagnostics expose staged timing clearly."""

from __future__ import annotations

import sys

from frontend.utils.scenario_review_diagnostics import diagnostics_payload


def main() -> int:
    trace = {
        "status": "reviewed",
        "validation_status": "valid",
        "provider": "gemini",
        "model_name": "gemini-3.1-flash-lite",
        "provider_metadata": {
            "prompt_mode": "later_visible_iteration",
            "latency_ms": 14045,
            "validation_retry_latency_ms": 11076,
            "pass2_scoring_latency_ms": 20199,
            "pass2_scoring_retry_latency_ms": 9020,
            "pass2_latency_ms": 13814,
            "response_text_length": 8362,
            "validation_retry_response_text_length": 9953,
            "pass2_scoring_response_text_length": 2451,
            "pass2_scoring_retry_response_text_length": 1877,
            "pass2_response_text_length": 2842,
            "pass2_scoring_usage_metadata": {
                "prompt_token_count": 12000,
                "candidates_token_count": 400,
                "thoughts_token_count": 900,
                "total_token_count": 13300,
            },
            "pass2_usage_metadata": {
                "prompt_token_count": 9000,
                "candidates_token_count": 500,
                "thoughts_token_count": 700,
                "total_token_count": 10200,
            },
            "pass2_scoring_finish_metadata": {"finish_reason": "STOP"},
            "pass2_finish_metadata": {"finish_reason": "STOP"},
            "validation_retry_attempts": 1,
            "validation_retry_stage": "narrative_scaffold",
        },
        "workflow_metadata": {
            "review_phase": "later_visible_iteration",
            "workflow_latency_ms": 59851,
            "visible_provider_or_store_latency_ms": 59851,
            "session_cache_hit": False,
            "review_store_cache_hit": False,
        },
    }
    diagnostics = diagnostics_payload(trace)
    errors: list[str] = []

    expected_stage_latency = {
        "pass1_initial": 14045,
        "pass1_repair": 11076,
        "pass2_scoring": 20199,
        "pass2_scoring_repair": 9020,
        "pass3_narrative": 13814,
    }
    if diagnostics.get("provider_latency_ms") != 14045:
        errors.append("provider_latency_ms should preserve the first provider call for backward compatibility")
    if diagnostics.get("provider_latency_scope") != "pass1_initial_only":
        errors.append("provider_latency_scope should make clear provider_latency_ms is not total workflow latency")
    if diagnostics.get("stage_latency_ms") != expected_stage_latency:
        errors.append("stage_latency_ms should expose pass1, repair, scoring, and narrative timings")
    if diagnostics.get("model_call_latency_ms") != sum(expected_stage_latency.values()):
        errors.append("model_call_latency_ms should sum visible staged model-call timings")
    if diagnostics.get("response_text_length_scope") != "pass1_initial_only":
        errors.append("response_text_length_scope should make clear response_text_length is not all response text")
    expected_response_lengths = {
        "pass1_initial": 8362,
        "pass1_repair": 9953,
        "pass2_scoring": 2451,
        "pass2_scoring_repair": 1877,
        "pass3_narrative": 2842,
    }
    if diagnostics.get("stage_response_text_length") != expected_response_lengths:
        errors.append("stage_response_text_length should expose per-stage response sizes")
    expected_usage = {
        "pass2_scoring": {
            "prompt_token_count": 12000,
            "candidates_token_count": 400,
            "thoughts_token_count": 900,
            "total_token_count": 13300,
        },
        "pass3_narrative": {
            "prompt_token_count": 9000,
            "candidates_token_count": 500,
            "thoughts_token_count": 700,
            "total_token_count": 10200,
        },
    }
    if diagnostics.get("stage_usage_metadata") != expected_usage:
        errors.append("stage_usage_metadata should expose per-stage token usage when provider returns it")
    expected_finish = {
        "pass2_scoring": {"finish_reason": "STOP"},
        "pass3_narrative": {"finish_reason": "STOP"},
    }
    if diagnostics.get("stage_finish_metadata") != expected_finish:
        errors.append("stage_finish_metadata should expose per-stage finish metadata when provider returns it")
    single_call = diagnostics_payload({
        "provider_metadata": {
            "latency_ms": 2500,
            "response_text_length": 900,
        },
    })
    if "provider_latency_scope" in single_call:
        errors.append("single-call diagnostics should not add a provider latency scope warning")
    if single_call.get("model_call_latency_ms") != 2500:
        errors.append("single-call diagnostics should still expose total model-call latency")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Validated Scenario Review staged diagnostics payload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
