#!/usr/bin/env python
"""Benchmark live narrative latency without printing secrets or raw prompts.

This script is opt-in because it can call OpenAI/Gemini and spend API credits.
Use it to compare provider latency, cache replay time, and the current
baseline-plus-first-iteration timing profile.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402
from src.narratives.provider import review_packet_with_provider, review_packet_with_provider_chain  # noqa: E402
from src.narratives.provider_config import PROVIDER_GEMINI, PROVIDER_OPENAI  # noqa: E402
from src.narratives.provider_config import load_narrative_provider_config  # noqa: E402
from src.narratives.review_store import replay_or_review_with_provider  # noqa: E402


BASELINE_FIXTURE_ID = "baseline_hidden_review_v1"
ITERATION_FIXTURE_ID = "model_facing_endpoint_shortcut_v1"


def _elapsed_call(fn) -> tuple[float, Any]:
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def _summary(label: str, elapsed: float, result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("provider_metadata") or {}
    scoring = result.get("scoring") or {}
    return {
        "label": label,
        "elapsed_s": round(elapsed, 2),
        "status": result.get("status"),
        "provider": result.get("provider"),
        "model_name": result.get("model_name"),
        "prompt_mode": metadata.get("prompt_mode"),
        "attempts": metadata.get("attempts"),
        "latency_ms": metadata.get("latency_ms"),
        "response_text_length": metadata.get("response_text_length"),
        "parsed_json_object": metadata.get("parsed_json_object"),
        "failure_reason": result.get("failure_reason"),
        "fallback_after": metadata.get("fallback_after"),
        "quality_adjustment": scoring.get("quality_adjustment"),
        "validation_status": scoring.get("validation_status"),
        "validation_errors": scoring.get("validation_errors") or [],
    }


def _print_json(item: dict[str, Any]) -> None:
    print(json.dumps(item, sort_keys=True), flush=True)


def _packet(fixture_id: str) -> dict[str, Any]:
    fixtures = {fixture["fixture_id"]: fixture for fixture in get_contract_fixtures()}
    return build_review_packet_from_fixture(fixtures[fixture_id])


def _with_overrides(config, args: argparse.Namespace):
    updated = config
    if args.timeout_seconds is not None:
        updated = replace(updated, timeout_seconds=args.timeout_seconds)
    if args.max_retries is not None:
        updated = replace(updated, max_retries=args.max_retries)
    if args.max_output_tokens is not None:
        updated = replace(updated, max_output_tokens=args.max_output_tokens)
    if args.openai_reasoning_effort is not None:
        updated = replace(updated, openai_reasoning_effort=args.openai_reasoning_effort)
    if args.temperature is not None:
        updated = replace(updated, temperature=args.temperature)
    if args.seed is not None:
        updated = replace(updated, seed=args.seed)
    return updated


def _worst_case_note(config) -> dict[str, Any]:
    attempts_per_provider = max(1, int(config.max_retries) + 1)
    provider_count = 1 + (1 if config.fallback_provider else 0)
    return {
        "configured_provider": config.provider,
        "configured_fallback": config.fallback_provider,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "attempts_per_provider": attempts_per_provider,
        "provider_count_if_fallback": provider_count,
        "worst_case_wait_before_success_or_failure_s": config.timeout_seconds * attempts_per_provider * provider_count,
        "note": "Worst case excludes actual model generation time after a provider responds.",
    }


def run(args: argparse.Namespace) -> int:
    load_dotenv(".env")
    config = _with_overrides(load_narrative_provider_config(os.environ), args)
    baseline_packet = _packet(BASELINE_FIXTURE_ID)
    iteration_packet = _packet(ITERATION_FIXTURE_ID)

    _print_json({
        "kind": "config",
        "provider_available": config.provider_available(),
        "fallback_available": config.fallback_available(),
        "models": {key: settings.model for key, settings in config.providers.items()},
        "generation_controls": {
            "temperature": config.temperature,
            "seed": config.seed,
            "openai_reasoning_effort": config.openai_reasoning_effort,
            "max_output_tokens": config.max_output_tokens,
            "timeout_seconds": config.timeout_seconds,
            "max_retries": config.max_retries,
        },
        "validation_errors": config.validation_errors,
    })
    _print_json({"kind": "worst_case_timeout_budget", **_worst_case_note(config)})

    if not args.run_live:
        _print_json({
            "kind": "skipped",
            "reason": "Pass --run-live to make provider API calls.",
        })
        return 0

    packets = {
        "baseline_hidden": baseline_packet,
        "visible_iteration": iteration_packet,
    }

    if args.compare_providers:
        for packet_label, packet in packets.items():
            for provider in (PROVIDER_OPENAI, PROVIDER_GEMINI):
                if not config.provider_available(provider):
                    _print_json({
                        "label": f"{packet_label}:{provider}",
                        "status": "skipped",
                        "reason": f"{provider} is not available",
                    })
                    continue
                elapsed, result = _elapsed_call(
                    lambda packet=packet, provider=provider: review_packet_with_provider(
                        deepcopy(packet),
                        provider=provider,
                        config=config,
                    )
                )
                _print_json(_summary(f"{packet_label}:{provider}", elapsed, result))

    if args.chain:
        for packet_label, packet in packets.items():
            elapsed, result = _elapsed_call(
                lambda packet=packet: review_packet_with_provider_chain(deepcopy(packet), config=config)
            )
            _print_json(_summary(f"{packet_label}:provider_chain", elapsed, result))

    if args.combined:
        state: dict[str, Any] = {}
        baseline_elapsed, baseline_trace = _elapsed_call(
            lambda: replay_or_review_with_provider(
                state,
                packet=deepcopy(baseline_packet),
                session_id="latency-benchmark:hidden_baseline",
                provider=config.provider,
                config=config,
                use_provider_chain=True,
            )
        )
        _print_json(_summary("combined_step_1_baseline", baseline_elapsed, baseline_trace))

        iteration_elapsed, iteration_trace = _elapsed_call(
            lambda: replay_or_review_with_provider(
                state,
                packet=deepcopy(iteration_packet),
                session_id="latency-benchmark",
                provider=config.provider,
                config=config,
                use_provider_chain=True,
            )
        )
        _print_json(_summary("combined_step_2_visible_iteration", iteration_elapsed, iteration_trace))
        _print_json({
            "label": "combined_total_without_prediction_api",
            "elapsed_s": round(baseline_elapsed + iteration_elapsed, 2),
        })

        replay_elapsed, replay_trace = _elapsed_call(
            lambda: replay_or_review_with_provider(
                state,
                packet=deepcopy(iteration_packet),
                session_id="latency-benchmark",
                provider=config.provider,
                config=config,
                use_provider_chain=True,
            )
        )
        _print_json(_summary("cache_replay_visible_iteration", replay_elapsed, replay_trace))

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true", help="Make live provider API calls.")
    parser.add_argument("--compare-providers", action="store_true", help="Call OpenAI and Gemini directly on the same packets.")
    parser.add_argument("--chain", action="store_true", help="Call the configured provider chain on baseline and iteration packets.")
    parser.add_argument("--combined", action="store_true", help="Measure current baseline-then-first-iteration flow plus cache replay.")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="Override NARRATIVE_LLM_TIMEOUT_SECONDS.")
    parser.add_argument("--max-retries", type=int, default=None, help="Override NARRATIVE_LLM_MAX_RETRIES.")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Override NARRATIVE_LLM_MAX_OUTPUT_TOKENS.")
    parser.add_argument("--openai-reasoning-effort", default=None, help="Override OPENAI_REASONING_EFFORT.")
    parser.add_argument("--temperature", type=float, default=None, help="Override NARRATIVE_LLM_TEMPERATURE.")
    parser.add_argument("--seed", type=int, default=None, help="Override NARRATIVE_LLM_SEED.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
