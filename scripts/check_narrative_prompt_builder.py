#!/usr/bin/env python
"""Validate active Trial Score three-pass prompt helpers."""

from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.mock_reviewer import review_packet_with_mock  # noqa: E402
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402
from src.narratives.prompt_builder import (  # noqa: E402
    PASS3_SCHEMA_VERSION,
    RESPONSE_SCHEMA_VERSION,
    SCORE_TRACE_PROMPT_RECENT_LIMIT,
    SCORING_RESPONSE_SCHEMA_VERSION,
    build_pass2_input,
    build_pass2_provider_prompt,
    build_provider_prompt,
    build_scoring_input,
    build_scoring_provider_prompt,
    gemini_response_schema,
    pass2_response_contract,
    provider_response_contract,
    scoring_gemini_response_schema,
    scoring_response_contract,
)
from src.narratives.trial_score_contract import (  # noqa: E402
    PASS1_SCHEMA_VERSION,
    PASS2_SCHEMA_VERSION,
    PASS3_SCHEMA_VERSION as CONTRACT_PASS3_SCHEMA_VERSION,
    xgboost_structured_state_hash,
)


def main() -> int:
    errors: list[str] = []
    fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "operational_only_ambitious_enrollment_v2"
    )
    packet = build_review_packet_from_fixture(fixture)
    result = review_packet_with_mock(packet)
    pass1 = result.get("validated_review") or {}
    scoring = result.get("scoring") or {}

    evidence_contract = provider_response_contract()
    if RESPONSE_SCHEMA_VERSION != PASS1_SCHEMA_VERSION:
        errors.append("Pass 1 response schema should use evidence-pass version")
    if set(evidence_contract.get("required_top_level_objects") or []) != {
        "review_metadata",
        "completion_outlook_analysis",
        "strategy_shift_check",
        "evolution_evidence",
        "continuity_update",
        "analytical_narrative_draft",
    }:
        errors.append("Pass 1 contract should require evidence/evolution objects only")
    evidence_prompt = build_provider_prompt(packet)
    for term in (
        "Pass 1 Evolution and Evidence",
        "Do not score Operational Fit or Reality Check in Pass 1",
        "bullet-first Pass 1 evidence formatting",
        "2-5 concise bullets",
        "Pass 3 writes the polished participant narrative",
        "exactly one development_discussion_options item",
        "Structured features and operational_assumptions are the authoritative current scenario state.",
        "text_context is descriptive context only",
        packet["input_hash"],
    ):
        if term not in evidence_prompt:
            errors.append(f"Pass 1 prompt missing term: {term}")
    bullet_rules = (evidence_contract.get("pass1_instructions") or []) + [
        (evidence_contract.get("analytical_narrative_draft_shape") or {}).get("field_guidance", "")
    ]
    if not any("bullet-first Pass 1 evidence formatting" in str(rule) for rule in bullet_rules):
        errors.append("Pass 1 contract should document bullet-first evidence formatting")
    hidden_packet = build_review_packet_from_fixture(
        next(item for item in get_contract_fixtures() if item["fixture_id"] == "baseline_hidden_review_v2")
    )
    hidden_prompt = build_provider_prompt(hidden_packet, prompt_mode="hidden_baseline")
    for term in (
        "compact hidden baseline context only",
        "Do not create a long baseline essay",
        "keep each analytical_narrative_draft field concise",
    ):
        if term not in hidden_prompt:
            errors.append(f"Hidden baseline prompt missing compactness term: {term}")
    evidence_schema = gemini_response_schema()
    if set(evidence_schema.get("required") or []) != set(evidence_contract.get("required_top_level_objects") or []):
        errors.append("Pass 1 Gemini schema required fields drifted from contract")
    if "operational_fit" in (evidence_schema.get("properties") or {}):
        errors.append("Pass 1 Gemini schema should not require Operational Fit scoring")
    if "reality_check" in (evidence_schema.get("properties") or {}):
        errors.append("Pass 1 Gemini schema should not require Reality Check scoring")

    scoring_input = build_scoring_input(packet, pass1)
    text_consistency = scoring_input.get("text_consistency_context") or {}
    if text_consistency.get("structured_and_operational_fields_are_authoritative") is not True:
        errors.append("Pass 2 scoring input should expose authoritative structured/operational text policy")
    if not isinstance(text_consistency.get("text_change_evidence"), list):
        errors.append("Pass 2 scoring input should expose text_change_evidence for changed descriptions")
    text_consistency_instruction = text_consistency.get("instruction") or ""
    if "incremental Reality Check coherence problem" not in text_consistency_instruction:
        errors.append("Pass 2 scoring text consistency context should tell Reality Check to penalize new contradictions")
    previous_score_trace = scoring_input.get("previous_score_trace") or {}
    for required_key in (
        "previous_operational_fit_assessment",
        "previous_reality_check_assessment",
        "previous_score_evolution_read",
        "recent_score_traces",
    ):
        if required_key not in previous_score_trace:
            errors.append(f"Pass 2 scoring input missing previous_score_trace.{required_key}")
    long_history_packet = {
        **packet,
        "iteration_context": {
            **(packet.get("iteration_context") or {}),
            "trial_score_continuity": {
                **((packet.get("iteration_context") or {}).get("trial_score_continuity") or {}),
                "previous_trial_score": 72.0,
                "previous_reality_check_points": -3.0,
                "recent_score_traces": [
                    {"iteration_id": index, "input_hash": f"trace-{index}", "trial_score": 50 + index}
                    for index in range(SCORE_TRACE_PROMPT_RECENT_LIMIT + 4)
                ],
            },
        },
    }
    long_history_scoring_input = build_scoring_input(long_history_packet, pass1)
    long_history_previous = long_history_scoring_input.get("previous_score_trace") or {}
    if len(long_history_previous.get("recent_score_traces") or []) != SCORE_TRACE_PROMPT_RECENT_LIMIT:
        errors.append("Pass 2 scoring input should cap general recent_score_traces to the prompt limit")
    if long_history_previous.get("available_recent_score_trace_count") != SCORE_TRACE_PROMPT_RECENT_LIMIT:
        errors.append("Pass 2 scoring input should expose only the capped retained score trace count")
    if long_history_previous.get("previous_trial_score") != 72.0:
        errors.append("Pass 2 scoring input should preserve immediate previous trajectory score separately")
    if long_history_previous.get("previous_reality_check_points") != -3.0:
        errors.append("Pass 2 scoring input should preserve immediate previous Reality Check separately")

    current_operational_hash = (long_history_scoring_input.get("operational_fit_continuity") or {}).get(
        "current_operational_fit_state_hash"
    )
    current_structured_hash = xgboost_structured_state_hash(long_history_packet)
    matching_history_packet = deepcopy(long_history_packet)
    traces = []
    for index in range(SCORE_TRACE_PROMPT_RECENT_LIMIT + 4):
        trace = {"iteration_id": index, "input_hash": f"trace-{index}", "trial_score": 50 + index}
        if index in {1, 3, SCORE_TRACE_PROMPT_RECENT_LIMIT + 3}:
            trace["operational_fit_state_hash"] = current_operational_hash
            trace["operational_fit_points"] = float(index)
            trace["xgboost_structured_state_hash"] = current_structured_hash
            trace["reality_check_points"] = -float(index)
            trace["reality_check_assessment"] = {
                "central_reason": "Prior matching feature state interpreted as shortcut-driven.",
                "evidence_fields": ["has_dmc_ml"],
                "incremental_check": "Prior feature interpretation should remain visible.",
            }
        traces.append(trace)
    matching_history_packet["iteration_context"]["trial_score_continuity"]["recent_score_traces"] = traces
    matching_scoring_input = build_scoring_input(matching_history_packet, pass1)
    operational_continuity = matching_scoring_input.get("operational_fit_continuity") or {}
    if "current_operational_fit_state_payload" in operational_continuity:
        errors.append("Pass 2 scoring input should not send duplicated full Operational Fit state payload")
    if operational_continuity.get("hash_scope") != "operational assumptions, operational movement context, and structured features":
        errors.append("Pass 2 scoring input should keep Operational Fit hash scope without duplicating payload")
    matching_traces = operational_continuity.get("previous_matching_score_traces") or []
    if len(matching_traces) != 1:
        errors.append("Pass 2 scoring input should send only the latest matching Operational Fit trace")
    elif matching_traces[0].get("iteration_id") != SCORE_TRACE_PROMPT_RECENT_LIMIT + 3:
        errors.append("Pass 2 scoring input should send the latest matching Operational Fit trace")
    if operational_continuity.get("matching_score_trace_count") != 1:
        errors.append("Pass 2 scoring input should expose capped retained Operational Fit matches only")
    structured_continuity = matching_scoring_input.get("structured_feature_continuity") or {}
    if structured_continuity.get("matching_feature_state_trace_count") != 1:
        errors.append("Pass 2 scoring input should expose capped retained structured feature matches only")
    latest_feature_match = structured_continuity.get("latest_matching_feature_state_trace") or {}
    if latest_feature_match.get("iteration_id") != SCORE_TRACE_PROMPT_RECENT_LIMIT + 3:
        errors.append("Pass 2 scoring input should send the latest matching structured feature trace")
    reality_memory = matching_scoring_input.get("reality_check_memory") or {}
    material_memory = reality_memory.get("material_recent_interpretations") or []
    if not material_memory:
        errors.append("Pass 2 scoring input should include material recent Reality Check memory")
    elif material_memory[-1].get("iteration_id") != SCORE_TRACE_PROMPT_RECENT_LIMIT + 3:
        errors.append("Pass 2 Reality Check memory should include retained material interpretations")
    dmc_removed_packet = deepcopy(packet)
    dmc_removed_packet.setdefault("iteration_context", {})["changed_fields"] = ["has_dmc_ml"]
    dmc_removed_packet["iteration_context"]["field_changes"] = [{
        "field": "has_dmc_ml",
        "change_type": "structured_feature",
        "previous_value": 1,
        "current_value": 0,
        "previous_label": "Yes",
        "current_label": "No",
    }]
    dmc_removed_scoring_input = build_scoring_input(dmc_removed_packet, pass1)
    governance_shortcut = dmc_removed_scoring_input.get("governance_shortcut_context") or {}
    if governance_shortcut.get("active") is not True:
        errors.append("Pass 2 scoring input should flag DMC present-to-absent as active governance shortcut context")
    if "strongly challenge" not in str(governance_shortcut.get("reality_check_calibration") or ""):
        errors.append("DMC governance shortcut context should calibrate Reality Check to strongly challenge favorable gains")
    scoring_contract = scoring_response_contract()
    if SCORING_RESPONSE_SCHEMA_VERSION != PASS2_SCHEMA_VERSION:
        errors.append("Pass 2 scoring schema version mismatch")
    if scoring_contract.get("schema_version") != PASS2_SCHEMA_VERSION:
        errors.append("Pass 2 scoring contract should expose scoring schema version")
    scoring_prompt = build_scoring_provider_prompt(scoring_input)
    for term in (
        "Pass 2 Score Adjudication",
        "assign points directly",
        "previous score trace",
        "carryover candidate",
        "Operational Fit points must be between -5 and +5",
        "Reality Check points must be between -15 and +15",
        "Reality Check calibration",
        "structured-feature continuity",
        "newly changed Trial description fields",
        "DMC/oversight downgrade rule",
        "governance_shortcut_context.active",
        "text_context evidence refs",
        "incremental scenario-coherence issue",
        "50-120%",
        "Reality Check must be 0 or negative",
        "subcategory named Reality Check",
        "not_touched",
        "preserve interpretation continuity",
    ):
        if term not in scoring_prompt:
            errors.append(f"Pass 2 scoring prompt missing term: {term}")
    scoring_schema = scoring_gemini_response_schema()
    if set(scoring_schema.get("required") or []) != {
        "review_metadata",
        "operational_fit",
        "reality_check",
        "score_evolution_read",
    }:
        errors.append("Pass 2 scoring Gemini schema required fields drifted")

    narrative_input = build_pass2_input(packet, pass1, scoring)
    if "model_evidence_context" in narrative_input:
        errors.append("Final narrative input should not send broad raw model_evidence_context")
    source_policy = narrative_input.get("source_of_truth_policy") or {}
    if source_policy.get("structured_and_operational_fields_are_authoritative") is not True:
        errors.append("Final narrative input should expose source-of-truth policy")
    previous_mismatch_note = (
        "In case of misalignment across Trial description fields and structured fields, "
        "the value in the structured fields drives the analysis, while the Trial description fields are used as supporting context."
    )
    if source_policy.get("participant_default_preface_note") != previous_mismatch_note:
        errors.append("Final narrative source-of-truth policy should use the default participant preface note")
    if source_policy.get("rendered_by_app_before_trial_score") is not True:
        errors.append("Final narrative source-of-truth policy should state the app renders the preface")
    selected_evidence = narrative_input.get("selected_model_evidence_context") or {}
    if "main_model_signals" not in selected_evidence:
        errors.append("Final narrative input should expose selected Pass 1 model signal list")
    if "current_model_state_evidence" in selected_evidence:
        errors.append("Final narrative input should not send broad current_model_state_evidence")
    negative_reality_scoring = {
        **scoring,
        "reality_check_points": -6.0,
        "reality_check_assessment": {
            **(scoring.get("reality_check_assessment") or {}),
            "points": -6.0,
        },
    }
    negative_reality_input = build_pass2_input(packet, pass1, negative_reality_scoring)
    negative_reality_notes = negative_reality_input.get("score_alignment_notes") or {}
    negative_reality_summary = negative_reality_notes.get("participant_safe_summary") or {}
    negative_reality_alignment = negative_reality_notes.get("reality_check_alignment") or {}
    if negative_reality_summary.get("reality_check_direction") != "negative_adjustment":
        errors.append("Pass 3 score alignment should derive negative Reality Check direction from accepted points")
    if negative_reality_alignment.get("scored_direction") != "negative_adjustment":
        errors.append("Pass 3 Reality Check alignment should expose negative direction from accepted points")
    narrative_contract = pass2_response_contract()
    if PASS3_SCHEMA_VERSION != CONTRACT_PASS3_SCHEMA_VERSION:
        errors.append("Prompt builder Pass 3 schema version mismatch")
    if narrative_contract.get("schema_version") != PASS3_SCHEMA_VERSION:
        errors.append("Final narrative contract should expose Pass 3 schema version")
    if narrative_contract.get("optional_top_level_objects"):
        errors.append("Pass 3 contract should not ask the model to return optional app-rendered preface fields")
    narrative_prompt = build_pass2_provider_prompt(narrative_input)
    for term in (
        "Pass 2 Participant Narrative",
        "Do not reanalyze",
        "one integrated Trial Score narrative",
        "app_calculated_scores",
        "development_discussion_options",
        "selected_model_evidence_context",
        "UI independently renders source_of_truth_policy.participant_default_preface_note",
        "before the Trial Score title",
        "do not repeat that sentence inside the returned narrative fields",
    ):
        if term not in narrative_prompt:
            errors.append(f"Final narrative prompt missing term: {term}")
    if len((narrative_input.get("pass1_analysis") or {}).get("development_discussion_options") or []) != 1:
        errors.append("Final narrative input should expose exactly one development discussion option")
    if scoring.get("trial_score") is None:
        errors.append("Mock scoring should provide accepted Trial Score for narrative input")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Validated Trial Score three-pass prompt helpers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
