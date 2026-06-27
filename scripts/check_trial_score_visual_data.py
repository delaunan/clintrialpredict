#!/usr/bin/env python
"""Validate Trial Score view chart-data composition."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.provider_config import load_narrative_provider_config  # noqa: E402
from src.narratives.review_store import get_review_store  # noqa: E402
from frontend.views import trial_simulator as ts  # noqa: E402


NCT_ID = "NCT-VISUAL-DATA"


def _pillars(scientific=19.8, patient=8.5, therapeutic=-3.2, execution=-10.4):
    return [
        {"Pillar": "Scientific Challenge", "Impact": scientific},
        {"Pillar": "Patient Profile", "Impact": patient},
        {"Pillar": "Therapeutic Context", "Impact": therapeutic},
        {"Pillar": "Execution Framework", "Impact": execution},
    ]


def _trace(iteration, *, xgb_pillars, operational_fit=0.0, reality_check=0.0, allocations=None):
    return {
        "input_hash": f"hash-{iteration}",
        "status": "reviewed",
        "validation_status": "valid",
        "hidden_baseline": False,
        "participant_visible": True,
        "iteration_id": iteration,
        "trial_score": 65.0 + operational_fit + reality_check,
        "pre_reality_score": 65.0 + operational_fit,
        "operational_fit_points": operational_fit,
        "reality_check_points": reality_check,
        "reality_check_allocation_points": allocations or [],
        "reality_check_assessment": {
            "points": reality_check,
            "sublevels": {},
        },
        "input_packet": {
            "trial_identity": {"nct_id": NCT_ID},
            "iteration_context": {
                "iteration_number": iteration,
                "current_snapshot_id": f"snapshot-{iteration}",
            },
            "model_interpretation": {
                "pillar_impacts": xgb_pillars,
            },
            "operational_assumptions": {
                "planned_enrollment": {"value": 140},
                "planned_sites": {"value": 18},
                "planned_duration_months": {"value": 38.0},
            },
        },
    }


def _impact(rows, pillar):
    for row in rows:
        if row.get("Pillar") == pillar:
            return row.get("Impact")
    raise AssertionError(f"missing pillar {pillar!r} in {rows!r}")


def main() -> int:
    errors: list[str] = []

    previous_trace = _trace(
        1,
        xgb_pillars=_pillars(scientific=19.0, execution=-10.0),
        operational_fit=0.7,
    )
    current_trace = _trace(
        2,
        xgb_pillars=_pillars(scientific=19.8, execution=-9.0),
        operational_fit=1.0,
    )

    ts.st.session_state["selected_nct_id"] = NCT_ID
    store = get_review_store(ts.st.session_state)
    store["trace_history"] = [previous_trace]

    completion_rows = ts._completion_outlook_pillar_impacts(
        previous_trace["input_packet"]["model_interpretation"]["pillar_impacts"],
        previous_trace,
    )
    if _impact(completion_rows, "Execution Framework") != -9.3:
        errors.append("Completion Outlook rows did not add Operational Fit to Execution Framework.")

    delta_map = ts._score_view_pillar_delta_map(
        ts.SCORE_VIEW_COMPLETION,
        {"previous_pillar_impacts": previous_trace["input_packet"]["model_interpretation"]["pillar_impacts"]},
        current_trace,
        current_trace["input_packet"]["model_interpretation"]["pillar_impacts"],
        [],
    )
    if delta_map.get("Execution Framework") != 1.3:
        errors.append(f"Completion delta should compare against previous XGBoost + Operational Fit, got {delta_map!r}.")

    previous_completion_score = ts._previous_completion_outlook_score(
        {"previous_score": 65.0},
        current_trace,
    )
    if previous_completion_score != 65.7:
        errors.append(
            "Completion gauge delta should compare against previous visible Completion Outlook "
            f"(65.7), got {previous_completion_score!r}."
        )

    locked_row = pd.Series({
        "nct_id": NCT_ID,
        "therapeutic_area_ui": "Infections",
        "therapeutic_area_ml": "INFECTIOUS_DISEASES",
        "gbd_cause_id_3_ml": 357,
        "sponsor_tier_ml": "Top-Tier Pharma",
    })
    trial_key = locked_row["nct_id"]
    stale_locked_values = {
        "therapeutic_area_ml": "Oncology",
        "gbd_cause_id_3_ml": 999,
        "sponsor_tier_ml": "Small Biotech",
    }
    for field_id, stale_value in stale_locked_values.items():
        ts.st.session_state[ts.get_simulation_feature_state_key(trial_key, field_id)] = stale_value

    if ts._simulation_feature_value_for_current_state(locked_row, "therapeutic_area_ml") != "Infections":
        errors.append("Hard-locked Therapeutic Area should ignore stale scenario state.")
    locked_indication = pd.to_numeric(
        ts._simulation_feature_value_for_current_state(locked_row, "gbd_cause_id_3_ml"),
        errors="coerce",
    )
    if pd.isna(locked_indication) or int(locked_indication) != 357:
        errors.append("Hard-locked Indication should ignore stale scenario state.")
    if ts._simulation_feature_value_for_current_state(locked_row, "sponsor_tier_ml") != "Top-Tier Pharma":
        errors.append("Hard-locked Sponsor Type should ignore stale scenario state.")

    neutral_rows = ts._design_pillar_impacts(current_trace)
    if [row.get("Pillar") for row in neutral_rows] != list(ts.SCORE_DRIVER_PILLARS):
        errors.append(f"Neutral Reality Check did not return the four score-driver pillars: {neutral_rows!r}.")
    if any(row.get("Impact") != 0.0 for row in neutral_rows):
        errors.append(f"Neutral Reality Check rows must all be zero-impact: {neutral_rows!r}.")

    duplicate_pillar_trace = _trace(
        3,
        xgb_pillars=_pillars(),
        reality_check=5.0,
        allocations=[
            {
                "pillar": "Execution Framework",
                "subpillar": "Reality Check",
                "points": 3.0,
                "short_explanation": "Methodology strengthened",
            },
            {
                "pillar": "Execution Framework",
                "subpillar": "Reality Check",
                "points": 2.0,
                "short_explanation": "Execution realism improved",
            },
        ],
    )
    duplicate_pillar_rows = ts._design_pillar_impacts(duplicate_pillar_trace)
    if duplicate_pillar_rows != [{"Pillar": "Execution Framework", "Impact": 5.0}]:
        errors.append(
            "Reality Check barchart should aggregate multiple allocations in the same pillar: "
            f"{duplicate_pillar_rows!r}."
        )
    duplicate_subcat_rows = ts.design_subcategory_impacts(duplicate_pillar_trace)
    if len(duplicate_subcat_rows) != 1 or duplicate_subcat_rows[0].get("Impact") != 5.0:
        errors.append(
            "Reality Check treemap should aggregate duplicate pillar/subcategory allocations: "
            f"{duplicate_subcat_rows!r}."
        )
    duplicate_details = duplicate_subcat_rows[0].get("FeatureDetails") if duplicate_subcat_rows else []
    if duplicate_details != ["Methodology strengthened", "Execution realism improved"]:
        errors.append(f"Reality Check details should keep concise unique phrases: {duplicate_details!r}.")

    completion_subcats = [
        {"Pillar": "Scientific Challenge", "Subcategory": "Biological Profile", "Impact": 1.2}
    ]
    reality_plot = ts._score_view_plot_impacts(
        ts.SCORE_VIEW_DESIGN,
        current_trace,
        current_trace["input_packet"]["model_interpretation"]["pillar_impacts"],
        completion_subcats,
    )
    if _impact(reality_plot["treemap_pillar_impacts"], "Execution Framework") != -8.0:
        errors.append("Reality Check treemap did not use the full Completion Outlook + Reality Check composition.")
    if not any(row.get("Subcategory") == "Operational Fit" for row in reality_plot["treemap_subcat_impacts"]):
        errors.append("Reality Check treemap did not include the Operational Fit subpillar.")

    operational_fit_rows = ts._operational_fit_subcategory_impacts(current_trace)
    details = operational_fit_rows[0].get("FeatureDetails") if operational_fit_rows else []
    expected_details = {
        "Planned enrollment: <b>140</b>",
        "Planned sites: <b>18</b>",
        "Planned duration: <b>38 months</b>",
    }
    if set(details) != expected_details:
        errors.append(f"Operational Fit treemap details should be feature-value lines, got {details!r}.")

    fallback_config = load_narrative_provider_config({
        "NARRATIVE_LLM_PROVIDER": "openai",
        "NARRATIVE_LLM_FALLBACK_PROVIDER": "mock",
    })
    fallback_runtime = {"config": fallback_config}
    primary_failure = {
        "provider": "openai",
        "model_name": "missing-key-model",
        "status": "provider_unavailable",
        "failure_reason": "missing key",
    }
    if ts._staged_provider_with_fallback(primary_failure, fallback_runtime) != "mock":
        errors.append("Staged review should select configured fallback for primary provider unavailability.")
    fallback_result = {
        "provider": "mock",
        "model_name": "fixture_hash_mock_v1",
        "provider_metadata": {"deterministic": True},
    }
    fallback_with_metadata = ts._apply_staged_fallback_metadata(primary_failure, fallback_result)
    if fallback_with_metadata.get("provider_metadata", {}).get("fallback_after", {}).get("provider") != "openai":
        errors.append("Staged fallback metadata should preserve the primary provider failure.")

    active_runtime = ts.narrative_review_runtime()
    replay_packet = {
        "input_hash": "staged-cache-hash",
        "trial_identity": {"nct_id": NCT_ID},
        "iteration_context": {
            "iteration_number": 3,
            "baseline_snapshot_id": "baseline-1",
            "current_snapshot_id": "snapshot-3",
        },
    }
    cached_trace = {
        "trace_id": "old-session:1:staged-cache-hash",
        "session_id": "old-session",
        "input_hash": "staged-cache-hash",
        "cached": False,
        "status": "reviewed",
        "validation_status": "valid",
        "hidden_baseline": False,
        "participant_visible": True,
        "reality_check_points": 0.0,
        "trial_score": 65.0,
        "review_runtime_key": active_runtime["runtime_key"],
        "input_packet": {
            "iteration_context": {"current_snapshot_id": "old-snapshot"},
        },
    }
    replayed = ts._replay_staged_cached_trace(
        cached_trace,
        packet=replay_packet,
        session_id="new-session",
        baseline_id="baseline-1",
    )
    if replayed.get("session_id") != "new-session" or replayed.get("cached") is not True:
        errors.append("Staged cached replay should stamp the current session and cached flag.")
    replay_store = get_review_store(ts.st.session_state)
    if replay_store.get("latest_trace_by_session", {}).get("new-session") != replayed.get("trace_id"):
        errors.append("Staged cached replay should update latest_trace_by_session.")
    replay_snapshot = {
        "nct_id": NCT_ID,
        "source": "simulation_ptc",
        "snapshot_id": "snapshot-3",
    }
    replay_bound_trace = ts.get_cached_quality_review_trace_for_snapshot(replay_snapshot)
    if (replay_bound_trace or {}).get("trace_id") != replayed.get("trace_id"):
        errors.append("Staged cached replay should bind the visible trace to the Trial Score renderer cache.")

    same_state_packet = {
        "input_hash": "same-state-current-hash",
        "scenario_state_hash": "same-state-1",
        "trial_identity": {"nct_id": NCT_ID},
        "iteration_context": {
            "iteration_number": 5,
            "baseline_snapshot_id": "baseline-1",
            "current_snapshot_id": "snapshot-5",
            "trial_score_continuity": {
                "previous_trial_score": 66.1,
                "previous_pre_reality_score": 66.1,
            },
            "state_equivalence_review": {
                "available": True,
                "source_iteration_id": 2,
                "source_scenario_state_hash": "same-state-1",
                "operational_fit_points": 0.0,
                "trial_score": 64.7,
            },
        },
        "model_interpretation": {
            "completion_score": 64.7,
            "previous_completion_score": 66.1,
            "baseline_completion_score": 64.7,
            "pillar_impacts": _pillars(),
        },
    }
    same_state_prior = {
        "trace_id": "old-session:2:same-state-prior-hash",
        "session_id": "old-session",
        "input_hash": "same-state-prior-hash",
        "scenario_state_hash": "same-state-1",
        "cached": False,
        "status": "reviewed",
        "validation_status": "valid",
        "hidden_baseline": False,
        "participant_visible": True,
        "provider": "mock",
        "model_name": "fixture_hash_mock_v1",
        "iteration_id": 2,
        "xgboost_completion_outlook": 64.7,
        "operational_fit_points": 0.0,
        "pre_reality_score": 64.7,
        "reality_check_points": -4.2,
        "trial_score": 60.5,
        "reality_check_assessment": {
            "points": -4.2,
            "central_reason": "Prior same-state Reality Check should be restored with the same visual allocation.",
            "allocation_points": [
                {
                    "allocation_target_id": "execution_framework.methodological_setup",
                    "pillar": "Execution Framework",
                    "subpillar": "Reality Check",
                    "source_subpillar": "Methodological Setup",
                    "label": "Reality Check",
                    "short_explanation": "Prior methodological concern",
                    "share": 1.0,
                    "points": -4.2,
                    "rationale": "Prior methodological concern.",
                    "incremental_check": "Prior concern was incremental to Completion Outlook.",
                }
            ],
        },
        "validated_review": {
            "review_metadata": {"review_mode": "later_visible_iteration", "visible": True},
            "completion_outlook_analysis": {},
            "operational_fit": {},
            "reality_check": {},
            "development_discussion_options": [],
            "continuity_update": {"what_changed": "Stale prior movement"},
        },
        "validated_participant_narrative": {
            "validation_status": "valid",
            "validation_errors": [],
            "review_metadata": {"review_mode": "later_visible_iteration", "visible": True},
            "trial_score_narrative": {
                "summary": "Prior same-state narrative.",
                "movement_reading": "Prior movement.",
                "score_interpretation": "Prior interpretation.",
            },
            "pillar_reading": [],
            "central_tension": {"summary": "Prior discussion topic.", "why_it_matters": "Prior reason."},
            "broader_strategic_question": {
                "mapped_tension": "Prior discussion topic.",
                "question": "Prior question?",
            },
        },
        "output_json": {},
        "input_packet": {
            "scenario_state_hash": "same-state-1",
            "trial_identity": {"nct_id": NCT_ID},
            "iteration_context": {"current_snapshot_id": "snapshot-2"},
        },
    }
    same_state_trace = ts._same_state_replay_trace_for_packet(
        packet=same_state_packet,
        prior_trace=same_state_prior,
        runtime={"provider": "mock", "config": None, "runtime_key": active_runtime["runtime_key"]},
        session_id="same-state-session",
        baseline_trace={},
        baseline_latency_ms=0,
        current_snapshot_id="snapshot-5",
        workflow_started_at=ts.time.monotonic(),
    )
    if same_state_trace.get("provider_metadata", {}).get("pass1_skipped") is not True:
        errors.append("Same-state replay should skip Pass 1 scoring reinterpretation.")
    if same_state_trace.get("trial_score") != 60.5 or same_state_trace.get("operational_fit_points") != 0:
        errors.append("Same-state replay should reuse prior app-owned scores.")
    same_state_delta = (same_state_trace.get("trial_score_diagnostics") or {}).get("delta_vs_previous_trial_score")
    if same_state_delta != -5.6:
        errors.append("Same-state replay should recalculate delta versus the immediate previous iteration.")
    same_state_allocations = same_state_trace.get("reality_check_allocation_points") or []
    if not same_state_allocations or same_state_allocations[0].get("points") != -4.2:
        errors.append("Same-state replay should restore Reality Check allocation impacts from the prior trace.")
    same_state_design_impacts = ts._design_pillar_impacts(same_state_trace)
    if same_state_design_impacts != [{"Pillar": "Execution Framework", "Impact": -4.2}]:
        errors.append("Same-state replay should render the same Reality Check visual impact as the prior state.")
    same_state_design_subcats = ts.design_subcategory_impacts(same_state_trace)
    if not same_state_design_subcats or same_state_design_subcats[0].get("Subcategory") != "Reality Check":
        errors.append("Same-state replay should render Reality Check as the pillar-level Reality Check subgroup.")
    if not ts._trace_is_successful_visible_review(same_state_trace):
        errors.append("Same-state replay should be treated as a successful visible review using current V1 fields.")
    same_state_continuity = (same_state_trace.get("validated_review") or {}).get("continuity_update") or {}
    if "Returned" not in str(same_state_continuity.get("what_changed") or ""):
        errors.append("Same-state replay should store current return-to-prior-state continuity, not stale prior continuity.")
    if same_state_trace.get("participant_narrative_status") != "valid":
        errors.append("Same-state replay should retain the prior participant narrative.")
    same_state_metadata = same_state_trace.get("provider_metadata") or {}
    if same_state_metadata.get("pass2_skipped") is not True:
        errors.append("Same-state replay should skip Pass 2 narrative regeneration.")
    if same_state_metadata.get("same_state_participant_narrative_reused") is not True:
        errors.append("Same-state replay should explicitly mark participant narrative reuse.")
    if same_state_trace.get("participant_narrative_warning"):
        errors.append("Same-state replay should not warn when prior participant narrative was intentionally reused.")
    long_path_store = get_review_store(ts.st.session_state)
    long_path_store["trace_history"] = [
        {
            **same_state_prior,
            "trace_id": "old-session:1:long-path-match",
            "iteration_id": 1,
            "review_runtime_key": active_runtime["runtime_key"],
            "scenario_state_hash": "long-path-state",
            "input_hash": "long-path-match",
            "input_packet": {
                **(same_state_prior.get("input_packet") or {}),
                "scenario_state_hash": "long-path-state",
                "iteration_context": {"current_snapshot_id": "snapshot-long-1"},
            },
        },
        *[
            {
                **same_state_prior,
                "trace_id": f"old-session:{index}:long-path-distractor",
                "iteration_id": index,
                "review_runtime_key": active_runtime["runtime_key"],
                "scenario_state_hash": f"other-state-{index}",
                "input_hash": f"long-path-distractor-{index}",
                "input_packet": {
                    **(same_state_prior.get("input_packet") or {}),
                    "scenario_state_hash": f"other-state-{index}",
                    "iteration_context": {"current_snapshot_id": f"snapshot-long-{index}"},
                },
            }
            for index in range(2, 9)
        ],
    ]
    long_path_packet = {
        **same_state_packet,
        "scenario_state_hash": "long-path-state",
        "iteration_context": {
            **(same_state_packet.get("iteration_context") or {}),
            "current_snapshot_id": "snapshot-long-current",
        },
    }
    long_path_prior = ts._prior_state_equivalent_review_trace(
        long_path_packet,
        runtime={"provider": "mock", "config": None, "runtime_key": active_runtime["runtime_key"]},
        nct_id=NCT_ID,
        current_snapshot_id="snapshot-long-current",
    )
    if (long_path_prior or {}).get("input_hash") != "long-path-match":
        errors.append("Same-state lookup should find an older full-state match after several later iterations")
    long_path_scoring = ts._same_state_reused_scoring(long_path_prior or {}, long_path_packet)
    if (
        long_path_scoring.get("operational_fit_points") != same_state_prior.get("operational_fit_points")
        or long_path_scoring.get("reality_check_points") != same_state_prior.get("reality_check_points")
    ):
        errors.append("Long-path same-state scoring should restore both Operational Fit and Reality Check")
    same_state_snapshot = {
        "nct_id": NCT_ID,
        "source": "simulation_ptc",
        "snapshot_id": "snapshot-5",
    }
    same_state_bound_trace = ts.get_cached_quality_review_trace_for_snapshot(same_state_snapshot)
    if (same_state_bound_trace or {}).get("trace_id") != same_state_trace.get("trace_id"):
        errors.append("Same-state replay should bind the visible trace to the current snapshot cache.")

    ts.st.session_state.pop(ts.get_quality_review_trace_state_key(NCT_ID), None)
    finalized_packet = {
        "input_hash": "fresh-staged-hash",
        "trial_identity": {"nct_id": NCT_ID},
        "iteration_context": {
            "iteration_number": 4,
            "baseline_snapshot_id": "baseline-1",
            "current_snapshot_id": "snapshot-4",
        },
        "model_interpretation": {
            "completion_score": 64.7,
            "pillar_impacts": _pillars(),
        },
    }
    finalized_result = {
        "provider": "mock",
        "model_name": "fixture_hash_mock_v1",
        "status": "reviewed",
        "review_needed": True,
        "reuse_previous_review": False,
        "provider_metadata": {
            "validation_retry_attempts": 2,
            "validation_retry_max_attempts": 3,
            "validation_retry_history": [
                {
                    "attempt": 1,
                    "stage": "operational_fit",
                    "messages": ["first repair target"],
                    "prompt_text": "first repair prompt",
                    "response_text": "{\"status\":\"still invalid\"}",
                    "parsed_json_object": True,
                    "validation_status": "invalid",
                    "remaining_messages": ["still invalid"],
                },
                {
                    "attempt": 2,
                    "stage": "operational_fit",
                    "messages": ["second repair target"],
                    "prompt_text": "second repair prompt",
                    "response_text": "{\"status\":\"valid\"}",
                    "parsed_json_object": True,
                    "validation_status": "valid",
                    "remaining_messages": [],
                },
            ],
        },
        "validated_review": {},
        "scoring": {
            "validation_status": "valid",
            "xgboost_completion_outlook": 64.7,
            "operational_fit_points": 0.3,
            "pre_reality_score": 65.0,
            "reality_check_points": 0.0,
            "trial_score": 65.0,
        },
    }
    finalized = ts._finalize_staged_scenario_review_trace(
        packet=finalized_packet,
        review_result=finalized_result,
        session_id="fresh-session",
        baseline_id="baseline-1",
        runtime={
            "provider": "mock",
            "config": None,
            "runtime_key": active_runtime["runtime_key"],
        },
        workflow_started_at=ts.time.monotonic(),
        visible_review_started_at=ts.time.monotonic(),
        baseline_latency_ms=0,
        baseline_trace={},
        current_snapshot_id="snapshot-4",
    )
    finalized_snapshot = {
        "nct_id": NCT_ID,
        "source": "simulation_ptc",
        "snapshot_id": "snapshot-4",
    }
    finalized_bound_trace = ts.get_cached_quality_review_trace_for_snapshot(finalized_snapshot)
    if (finalized_bound_trace or {}).get("trace_id") != finalized.get("trace_id"):
        errors.append("Fresh staged finalization should bind the visible trace to the Trial Score renderer cache.")
    bundle_dir = ts.persist_scenario_review_audit_bundle(finalized, snapshot=finalized_snapshot)
    expected_bundle_files = {
        "00_manifest.json",
        "01_input_packet.json",
        "02_pass1_prompt.txt",
        "03_pass1_response_contract.json",
        "10_app_scoring.json",
        "11_pass2_input.json",
        "12_pass2_prompt.txt",
        "13_pass2_response_contract.json",
        "17_final_trace.json",
        "18_ui_binding.json",
        "19_decision_rating_narrative_map.json",
        "20_walkthrough.md",
    }
    missing_bundle_files = [
        name
        for name in sorted(expected_bundle_files)
        if not bundle_dir or not (bundle_dir / name).exists()
    ]
    if missing_bundle_files:
        errors.append(f"Scenario Review audit bundle is missing files: {missing_bundle_files!r}.")
    if bundle_dir:
        with (bundle_dir / "00_manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        with (bundle_dir / "10_app_scoring.json").open(encoding="utf-8") as handle:
            audit_scoring = json.load(handle)
        with (bundle_dir / "17_final_trace.json").open(encoding="utf-8") as handle:
            audit_trace = json.load(handle)
        with (bundle_dir / "18_ui_binding.json").open(encoding="utf-8") as handle:
            ui_binding = json.load(handle)
        with (bundle_dir / "19_decision_rating_narrative_map.json").open(encoding="utf-8") as handle:
            decision_map = json.load(handle)
        if manifest.get("bundle_dir") != audit_trace.get("audit_bundle_dir"):
            errors.append("Audit bundle final trace should carry the same audit_bundle_dir as the manifest.")
        if audit_scoring.get("operational_fit_points") != 0.3 or audit_scoring.get("trial_score") != 65.0:
            errors.append(f"Audit bundle app scoring has unexpected values: {audit_scoring!r}.")
        if ui_binding.get("renderer_cache_match") is not True:
            errors.append(f"Audit bundle UI binding should confirm renderer cache match: {ui_binding!r}.")
        if decision_map.get("operational_fit", {}).get("points") != 0.3:
            errors.append(f"Audit bundle decision map should expose Operational Fit points: {decision_map!r}.")
        discussion_map = decision_map.get("development_discussion_and_questions") or {}
        if "pass1_development_discussion_options" not in discussion_map:
            errors.append("Audit bundle decision map should expose current Pass 1 development_discussion_options.")
        for obsolete_key in (
            "pass1_central_tension_candidate",
            "pass1_broader_question_candidate",
            "central_tension_candidate",
            "broader_strategic_question_candidate",
        ):
            if obsolete_key in discussion_map:
                errors.append(f"Audit bundle decision map should not expose obsolete discussion key {obsolete_key!r}.")
        retry_dir = bundle_dir / "pass1_repair_attempts"
        expected_retry_files = {
            "attempt_01_summary.json",
            "attempt_01_prompt.txt",
            "attempt_01_response.txt",
            "attempt_02_summary.json",
            "attempt_02_prompt.txt",
            "attempt_02_response.txt",
        }
        missing_retry_files = [
            name
            for name in sorted(expected_retry_files)
            if not (retry_dir / name).exists()
        ]
        if missing_retry_files:
            errors.append(f"Audit bundle should persist per-attempt repair artifacts: {missing_retry_files!r}.")
        if (retry_dir / "attempt_01_prompt.txt").exists():
            first_prompt = (retry_dir / "attempt_01_prompt.txt").read_text(encoding="utf-8").strip()
            if first_prompt != "first repair prompt":
                errors.append("Audit bundle should preserve the first repair prompt text.")
        repair_history = decision_map.get("pass1_validation", {}).get("repair_history") or []
        if repair_history and any("prompt_text" in item or "response_text" in item for item in repair_history):
            errors.append("Decision map retry history should omit bulky raw prompt/response text.")

    pass1_debug = ts._pass2_pass1_analysis_debug_view({
        "completion_outlook_analysis": {},
        "operational_fit": {},
        "reality_check": {},
        "central_tension_candidate": {"summary": "obsolete"},
        "broader_strategic_question_candidate": {"question": "obsolete?"},
        "development_discussion_options": [
            {
                "topic": "Current option.",
                "why_it_matters": "Current.",
                "supporting_evidence": [],
                "participant_wider_question": {"question": "Current question?", "supporting_evidence": []},
            }
        ],
    })
    if "development_discussion_options" not in pass1_debug:
        errors.append("Pass 2 debug summary should expose current development discussion option structures.")
    for obsolete_key in (
        "central_tension_candidate",
        "alternative_tension_candidates",
        "broader_strategic_question_candidate",
        "alternative_strategic_question_candidates",
    ):
        if obsolete_key in pass1_debug:
            errors.append(f"Pass 2 debug summary should not expose obsolete key {obsolete_key!r}.")

    frontend_source = (ROOT / "frontend/views/trial_simulator.py").read_text(encoding="utf-8")
    for required_token in (
        "_participant_pillar_reading_html",
        "_scenario_review_multi_text_block",
        "_discussion_point_html",
        "SCENARIO_SOURCE_OF_TRUTH_PREFACE",
        "In case of misalignment across Trial description fields and structured fields",
        "_scenario_consistency_html",
        "participant_pillar_reading",
        "Trial Score",
        "Overall Evolution",
        "Completion Outlook",
        "Reality Check",
        "What Is Driving The Score",
        "Discussion Point",
        "Discussion Point:",
    ):
        if required_token not in frontend_source:
            errors.append(f"frontend trial simulator should render Pass 2 participant format token {required_token!r}.")
    renderer_start = frontend_source.find("def render_scenario_review_report")
    renderer_end = frontend_source.find("def get_simulation_pillar_delta_map")
    renderer_source = frontend_source[renderer_start:renderer_end]
    if "_scenario_consistency_html(consistency_text)" not in renderer_source or "_trial_score_narrative_html(trial_score_narrative)" not in renderer_source:
        errors.append("frontend trial simulator should render consistency preface and Trial Score title")
    elif renderer_source.find("_scenario_consistency_html(consistency_text)") > renderer_source.find("_trial_score_narrative_html(trial_score_narrative)"):
        errors.append("frontend trial simulator should render consistency preface before the Trial Score title")
    for obsolete_token in (
        "central_tension_candidate",
        "alternative_tension_candidates",
        "broader_strategic_question_candidate",
        "alternative_strategic_question_candidates",
        "pass1_central_tension_candidate",
        "pass1_broader_question_candidate",
        "central_tension_read",
        "tradeoff_review",
        "Design Confidence",
        "Total Scenario Score",
        "design_confidence",
        "total_scenario_score",
    ):
        if obsolete_token in frontend_source:
            errors.append(f"frontend trial simulator should not contain obsolete debug/display token {obsolete_token!r}.")

    captured_markdown: list[str] = []
    original_markdown = ts.st.markdown
    original_get_snapshot = ts.get_latest_prediction_snapshot
    original_pending = ts.has_pending_simulation_changes
    original_prompt_debug = ts._render_prompt_review_context_debug
    original_quality_diagnostics = ts._quality_review_diagnostics
    invalid_pass2_trace = _trace(3, xgb_pillars=_pillars())
    invalid_pass2_trace.update({
        "participant_narrative_status": "invalid",
        "participant_narrative_warning": (
            "Participant narrative failed after the Pass 2 repair retry: "
            "pillar_reading must include 2-4 material bullets"
        ),
        "validated_participant_narrative": {},
        "trial_score_narrative": {},
        "participant_pillar_reading": [],
        "participant_central_tension": {},
        "participant_broader_strategic_question": {},
        "central_tension": "",
        "validated_review": {
            "review_metadata": {"review_mode": "later_visible_iteration", "visible": True},
            "completion_outlook_analysis": {"summary": "This old fallback summary should not render."},
            "operational_fit": {
                "combined_operational_fit": {
                    "central_reason": "This old Operational Fit fallback should not render."
                }
            },
            "reality_check": {"central_reason": "This old Reality Check fallback should not render."},
            "key_questions": {
                "strategic_development_question": "This old Strategic Development Question should not render?"
            },
        },
    })
    try:
        ts.st.session_state["global_edit_mode"] = True
        ts.st.markdown = lambda body, *args, **kwargs: captured_markdown.append(str(body))
        ts.get_latest_prediction_snapshot = lambda nct_id: {
            "snapshot_id": "invalid-pass2-snapshot",
            "source": "simulation_ptc",
            "score": 64.7,
        }
        ts.has_pending_simulation_changes = lambda row: False
        ts._render_prompt_review_context_debug = lambda trace: None
        ts._quality_review_diagnostics = lambda trace, row=None, snapshot=None: None
        ts.render_scenario_review_report(pd.Series({"nct_id": NCT_ID}), trace=invalid_pass2_trace)
    finally:
        ts.st.markdown = original_markdown
        ts.get_latest_prediction_snapshot = original_get_snapshot
        ts.has_pending_simulation_changes = original_pending
        ts._render_prompt_review_context_debug = original_prompt_debug
        ts._quality_review_diagnostics = original_quality_diagnostics
    rendered = "\n".join(captured_markdown)
    if "Participant Narrative Unavailable" not in rendered:
        errors.append("Invalid Pass 2 UI fallback should render Participant Narrative Unavailable.")
    if "Reason:" not in rendered or "pillar_reading must include 2-4 material bullets" not in rendered:
        errors.append("Invalid Pass 2 UI fallback should render the participant narrative warning reason.")
    for hidden_old_title in (
        "Completion Outlook Analysis",
        "Operational Fit",
        "Strategic Development Question",
    ):
        if hidden_old_title in rendered:
            errors.append(f"Invalid Pass 2 UI fallback should not render old fallback title {hidden_old_title!r}.")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated Trial Score visual data composition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
