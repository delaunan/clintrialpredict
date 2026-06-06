#!/usr/bin/env python
"""Validate narrative behavior for frontend-like prediction snapshots."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.packet_builder import build_review_packet  # noqa: E402
from src.narratives.review_store import (  # noqa: E402
    cached_review_trace,
    replay_or_review_with_provider,
)


def _result(score: int) -> dict:
    return {
        "score": score,
        "pillar_impacts": [{"Pillar": "Execution Framework", "Impact": 1.0}],
        "subcat_impacts": [],
    }


def main() -> int:
    errors: list[str] = []
    state: dict = {}

    baseline_snapshot = {
        "timestamp": "baseline-ts",
        "nct_id": "NCT-LIVE-FLOW",
        "source": "prerecorded_baseline",
        "submitted_values": {"phase_ml": 4, "endpoint_rigor_ml": 3},
        "compare_values": {"phase_ml": "PHASE3", "endpoint_rigor_ml": "HARD_CLINICAL"},
        "display_values": {"phase_ml": "Phase 3", "endpoint_rigor_ml": "Clinical outcome"},
        "result": _result(68),
        "score": 68,
        "operational_assumptions": {
            "planned_enrollment": {"value": 620},
            "planned_sites": {"value": 75},
            "planned_duration_months": {"value": 42.0},
        },
        "text_context": {"summary_ui": "Original summary."},
        "iteration_context": {"iteration_number": 0},
    }
    current_snapshot = {
        **baseline_snapshot,
        "timestamp": "current-ts",
        "source": "simulation_text_update",
        "score": 68,
        "result": _result(68),
        "changed_text_context_fields": ["summary_ui"],
        "text_context": {"summary_ui": "Materially revised summary."},
        "iteration_context": {"iteration_number": 1},
    }

    baseline_packet = build_review_packet(current_snapshot=baseline_snapshot, baseline_snapshot=baseline_snapshot)
    baseline_trace = {
        "input_hash": baseline_packet["input_hash"],
        "iteration_id": 0,
        "status": "reviewed",
        "validation_status": "valid",
        "quality_adjustment": 0,
        "final_candidate_score": 68,
        "changed_fields": [],
        "score_movement": 0,
        "validated_review": {
            "participant_review": {"what_changed": "Baseline"},
            "continuity": {"storyline_update": "Baseline reviewed."},
        },
        "compact_storyline_memory": "Baseline reviewed.",
    }

    current_packet = build_review_packet(
        current_snapshot=current_snapshot,
        previous_snapshot=baseline_snapshot,
        baseline_snapshot=baseline_snapshot,
        baseline_review_trace=baseline_trace,
        previous_review_trace=baseline_trace,
        compact_storyline_memory=baseline_trace.get("compact_storyline_memory", ""),
    )

    changed_fields = current_packet.get("iteration_context", {}).get("changed_fields") or []
    if "text_context.summary_ui" not in changed_fields:
        errors.append("text-context changes should appear in packet changed_fields")
    if current_packet.get("iteration_context", {}).get("iteration_number") != 1:
        errors.append("current packet should preserve frontend iteration_number")
    if current_packet.get("structured_features", {}).get("phase_ml") != "PHASE3":
        errors.append("packet should prefer taxonomy option-key compare_values over model-facing submitted_values")
    if current_packet.get("structured_feature_display_values", {}).get("phase_ml") != "Phase 3":
        errors.append("packet should expose display values separately")
    baseline_context = (current_packet.get("review_context") or {}).get("baseline_review") or {}
    if not baseline_context.get("input_hash"):
        errors.append("packet should include compact baseline review context by input_hash")
    if "trace_id" in baseline_context:
        errors.append("packet review_context should not include session-specific trace_id")

    trace = replay_or_review_with_provider(
        state,
        packet=current_packet,
        session_id="live-flow",
        provider="mock",
    )
    if trace.get("iteration_id") != 1:
        errors.append("stored trace should preserve frontend iteration_id")
    if cached_review_trace(state, current_packet["input_hash"], provider="not_configured"):
        errors.append("provider cache should be namespaced by provider/model")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated narrative live-style snapshot flow, text deltas, iteration IDs, and cache namespacing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
