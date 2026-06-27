#!/usr/bin/env python
"""Validate the simplified Trial Score three-pass contract."""

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
from src.narratives.scoring import validate_and_score_adjudication, validate_and_score_review  # noqa: E402
from src.narratives.trial_score_contract import (  # noqa: E402
    PASS1_SCHEMA_VERSION,
    PASS2_SCHEMA_VERSION,
    PASS3_SCHEMA_VERSION,
    PROMPT_TEMPLATE_VERSION,
    operational_fit_state_hash,
    validate_pass2_review,
    xgboost_structured_state_hash,
)


def _fixture(fixture_id: str) -> dict:
    return next(item for item in get_contract_fixtures() if item["fixture_id"] == fixture_id)


def _valid_pass1(packet: dict) -> dict:
    review = {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "completion_outlook_analysis": {
            "summary": "Completion Outlook changed on protected model-visible inputs.",
            "main_model_signals": ["model_interpretation.score_delta"],
            "model_boundary_note": "Completion Outlook remains model-owned.",
        },
        "strategy_shift_check": {"status": "not_applicable", "rationale": "No premise-sensitive shift."},
        "evolution_evidence": {
            "latest_meaningful_changes": ["operational_assumptions.planned_sites"],
            "model_movement_evidence": ["model_interpretation.score_delta"],
            "operational_movement_evidence": ["operational_assumptions.planned_sites"],
            "new_issues": ["The operational footprint changed."],
            "persistent_issues": [],
            "resolved_or_mitigated_issues": [],
            "strongest_current_development_tension": {
                "topic": "Execution scale versus evidence ambition.",
                "why_this_is_strongest_now": "The latest site-count movement makes execution scale the clearest current tension.",
                "relationship_to_previous_scenario": "new_issue",
                "relationship_to_original_baseline": "The current state differs from the original execution footprint.",
                "evidence_fields": ["operational_assumptions.planned_sites"],
            },
        },
        "development_discussion_options": [
            {
                "topic": "Execution scale versus evidence ambition.",
                "why_it_matters": "This is the strongest current development tension.",
                "supporting_evidence": ["operational_assumptions.planned_sites"],
                "relationship_to_previous_scenario": "new_issue",
                "relationship_to_original_baseline": "The execution footprint changed versus baseline.",
                "participant_wider_question": {
                    "question": "When does a larger execution footprint make a development case more credible, and when does it only make an unresolved evidence question bigger?",
                    "supporting_evidence": ["operational_assumptions.planned_sites"],
                },
            }
        ],
        "continuity_update": {
            "active_tension": "Execution scale versus evidence ambition.",
            "what_changed": "Site count changed.",
            "watch_next": "Whether evidence ambition remains proportionate.",
        },
        "analytical_narrative_draft": {
            "current_state_read": "The current scenario remains anchored in Completion Outlook but needs a broader development read across population, endpoint, comparator, safety, operations, and decision readiness. This draft is intentionally long enough to support later scoring. It treats the trial state as a clinical development scenario rather than a raw score output, so the next pass can weigh whether the evidence package still supports the intended decision. It also preserves the distinction between protected model evidence and qualitative scenario interpretation.",
            "movement_read": "The latest movement should be compared with the previous visible scenario before using the original baseline as context. It changes the operational footprint and may affect how credible the execution plan feels. The movement does not by itself prove the trial is better; it identifies evidence that the scoring adjudicator should compare against prior score history, unresolved issues, and any carryover concern. That comparison is the reason this draft remains descriptive instead of assigning points.",
            "operational_fit_read": "The operational evidence concerns planned site count, patient burden, and whether the assumed footprint is proportionate to the study question. It does not itself assign points in Pass 1. The later scoring pass should decide whether the operational change is large enough, coherent enough, and sufficiently supported by packet evidence to deserve positive or negative Operational Fit. This draft only identifies the relevant operational dimensions and how they changed.",
            "reality_check_read": "The realism evidence asks whether the pre-score evolution is coherent, under-supported, shortcut-driven, or appropriately conservative compared with the previous score history. It also records whether any prior issue appears persistent, mitigated, resolved, or replaced by a new issue. The scoring adjudicator can then decide whether Reality Check should stay neutral or move the Trial Score because the current evolution does not fully make sense.",
            "development_landscape_read": "The strongest current tension is execution scale versus evidence ambition because the latest movement makes operational delivery more central than older unresolved issues. This should guide scoring and narrative without generating multiple candidate questions. The tension is phrased as a development debate rather than a recommendation, because the participant should be asked to defend the scenario logic, not told which field to edit. It remains tied to the latest change while still acknowledging the original baseline as background context.",
        },
    }
    return review


def main() -> int:
    errors: list[str] = []
    if not PASS1_SCHEMA_VERSION.startswith("trial_score_evidence_pass"):
        errors.append("Pass 1 schema should be the evidence-pass contract")
    if not PASS2_SCHEMA_VERSION.startswith("trial_score_scoring_pass"):
        errors.append("Pass 2 schema should be the scoring-pass contract")
    if not PASS3_SCHEMA_VERSION.startswith("trial_score_narrative_pass"):
        errors.append("Pass 3 schema should be the narrative-pass contract")
    if "three_pass" not in PROMPT_TEMPLATE_VERSION:
        errors.append("Prompt version should identify the three-pass workflow")

    packet = build_review_packet_from_fixture(_fixture("operational_only_ambitious_enrollment_v2"))
    pass1_review = _valid_pass1(packet)
    pass1 = validate_and_score_review(packet, pass1_review)
    if pass1["validated_review"].get("validation_status") != "valid":
        errors.append(f"Pass 1 evidence review should validate: {pass1['validated_review'].get('validation_errors')}")
    if pass1["scoring"].get("trial_score") is not None:
        errors.append("Pass 1 should not adjudicate Trial Score")
    bare_discussion_option_review = deepcopy(pass1_review)
    bare_discussion_option_review["development_discussion_options"] = deepcopy(
        pass1_review["development_discussion_options"][0]
    )
    bare_discussion_option = validate_and_score_review(packet, bare_discussion_option_review)
    if bare_discussion_option["validated_review"].get("validation_status") != "valid":
        errors.append(
            "Pass 1 should normalize a bare visible development_discussion_options object: "
            f"{bare_discussion_option['validated_review'].get('validation_errors')}"
        )
    normalized_options = bare_discussion_option["validated_review"].get("development_discussion_options")
    if not isinstance(normalized_options, list) or len(normalized_options) != 1:
        errors.append("Bare visible development_discussion_options object should become one-item array")
    missing_evolution_review = deepcopy(pass1_review)
    missing_evolution_review.pop("evolution_evidence", None)
    missing_evolution = validate_and_score_review(packet, missing_evolution_review)
    if missing_evolution["validated_review"].get("validation_status") != "invalid":
        errors.append("Pass 1 should fail validation without evolution_evidence")

    hidden_packet = build_review_packet_from_fixture(_fixture("baseline_hidden_review_v2"))
    compact_hidden_review = {
        "review_metadata": {"review_mode": "hidden_baseline", "visible": False},
        "completion_outlook_analysis": {
            "summary": "Compact hidden baseline context.",
            "main_model_signals": ["completion_score"],
            "model_boundary_note": "Hidden baseline is context only.",
        },
        "strategy_shift_check": {"status": "not_applicable", "rationale": "No visible edit."},
        "evolution_evidence": {
            "latest_meaningful_changes": [],
            "model_movement_evidence": [],
            "operational_movement_evidence": ["Opening operational assumptions are neutral context."],
            "new_issues": [],
            "persistent_issues": ["Baseline watch item."],
            "resolved_or_mitigated_issues": [],
            "strongest_current_development_tension": {
                "topic": "Baseline context only",
                "why_this_is_strongest_now": "Hidden baseline establishes neutral context.",
                "relationship_to_previous_scenario": "No previous visible scenario exists.",
                "relationship_to_original_baseline": "This is the original baseline.",
                "evidence_fields": ["baseline_is_neutral_reference"],
            },
        },
        "continuity_update": {
            "active_tension": "",
            "what_changed": "No participant-visible change.",
            "watch_next": "Watch later edits.",
        },
        "analytical_narrative_draft": {
            "current_state_read": "Short baseline state.",
            "movement_read": "No movement.",
            "operational_fit_read": "Neutral operational reference.",
            "reality_check_read": "No visible Reality Check.",
            "development_landscape_read": "Compact context only.",
        },
    }
    compact_hidden = validate_and_score_review(hidden_packet, compact_hidden_review)
    if compact_hidden["validated_review"].get("validation_status") != "valid":
        errors.append(
            "Hidden baseline should validate compact analytical context without word-count repair: "
            f"{compact_hidden['validated_review'].get('validation_errors')}"
        )

    scoring_review = {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "operational_fit": {
            "points": 2.0,
            "relationship_to_previous": "improved",
            "reason": "The changed site footprint supports the enrollment burden.",
            "evidence_fields": ["operational_assumptions.planned_sites"],
            "boundary_check": "Operational assumptions changed, so Operational Fit can move.",
        },
        "reality_check": {
            "points": -1.0,
            "relationship_to_previous": "new_issue",
            "carryover_status": "none",
            "new_issue_status": "new_independent_issue",
            "reason": "The operational gain still leaves a realism concern.",
            "incremental_check": "This is incremental beyond Completion Outlook and Operational Fit.",
            "evidence_fields": ["operational_assumptions.planned_sites"],
            "allocations": [
                {
                    "allocation_target_id": "execution_framework.operational_fit",
                    "share": 1.0,
                    "movement_label": "Reality Check: execution realism",
                    "rationale": "The realism concern lands in execution proportionality.",
                    "incremental_check": "This checks residual realism after Operational Fit rather than duplicating it.",
                }
            ],
        },
        "score_evolution_read": {
            "direction": "improved_but_moderated",
            "main_reason": "Operational support improved but realism moderates the movement.",
            "active_issue_to_carry_forward": "Execution scale versus evidence ambition.",
        },
    }
    scoring = validate_and_score_adjudication(packet, pass1["validated_review"], scoring_review)
    if scoring.get("validation_status") != "valid":
        errors.append(f"Pass 2 scoring should validate: {scoring.get('validation_errors')}")
    if scoring.get("operational_fit_points") != 2.0:
        errors.append("Pass 2 scoring should preserve LLM-owned Operational Fit points")
    if scoring.get("reality_check_points") != -1.0:
        errors.append("Pass 2 scoring should preserve LLM-owned Reality Check points")
    expected_trial_score = scoring.get("pre_reality_score") + scoring.get("reality_check_points")
    if scoring.get("trial_score") != expected_trial_score:
        errors.append("App should calculate Trial Score arithmetic from accepted scoring points")
    large_reality_review = deepcopy(scoring_review)
    large_reality_review["reality_check"]["points"] = -12.0
    large_reality = validate_and_score_adjudication(packet, pass1["validated_review"], large_reality_review)
    if large_reality.get("validation_status") != "valid":
        errors.append(f"Pass 2 scoring should allow large in-range Reality Check points: {large_reality.get('validation_errors')}")
    if large_reality.get("reality_check_points") != -12.0:
        errors.append("Pass 2 scoring should preserve large LLM-owned Reality Check points inside -15/+15")
    large_positive_on_positive_move_review = deepcopy(scoring_review)
    large_positive_on_positive_move_review["reality_check"]["points"] = 5.0
    large_positive_on_positive_move_review["reality_check"]["allocations"][0]["movement_label"] = "positive"
    large_positive_on_positive_move = validate_and_score_adjudication(
        packet,
        pass1["validated_review"],
        large_positive_on_positive_move_review,
    )
    if large_positive_on_positive_move.get("validation_status") != "invalid":
        errors.append("Reality Check should reject large positive credit when pre-reality check already improved")
    one_point_positive_on_positive_move_review = deepcopy(scoring_review)
    one_point_positive_on_positive_move_review["reality_check"]["points"] = 1.0
    one_point_positive_on_positive_move_review["reality_check"]["allocations"][0]["movement_label"] = "positive"
    one_point_positive_on_positive_move = validate_and_score_adjudication(
        packet,
        pass1["validated_review"],
        one_point_positive_on_positive_move_review,
    )
    if one_point_positive_on_positive_move.get("validation_status") != "invalid":
        errors.append("Reality Check should reject any positive credit when pre-reality check already improved")
    zero_reality_on_positive_move_review = deepcopy(scoring_review)
    zero_reality_on_positive_move_review["reality_check"]["points"] = 0.0
    zero_reality_on_positive_move_review["reality_check"]["allocations"] = []
    zero_reality_on_positive_move = validate_and_score_adjudication(
        packet,
        pass1["validated_review"],
        zero_reality_on_positive_move_review,
    )
    if zero_reality_on_positive_move.get("validation_status") != "valid":
        errors.append(
            "Reality Check should allow neutral acceptance when pre-reality check already improved: "
            f"{zero_reality_on_positive_move.get('validation_errors')}"
        )
    three_point_positive_on_positive_move_review = deepcopy(scoring_review)
    three_point_positive_on_positive_move_review["reality_check"]["points"] = 3.0
    three_point_positive_on_positive_move_review["reality_check"]["allocations"][0]["movement_label"] = "positive"
    three_point_positive_on_positive_move = validate_and_score_adjudication(
        packet,
        pass1["validated_review"],
        three_point_positive_on_positive_move_review,
    )
    if three_point_positive_on_positive_move.get("validation_status") != "invalid":
        errors.append("Reality Check should reject +3 credit when pre-reality check already improved")
    negative_reality_on_positive_move_review = deepcopy(scoring_review)
    negative_reality_on_positive_move_review["reality_check"]["points"] = -1.0
    negative_reality_on_positive_move = validate_and_score_adjudication(
        packet,
        pass1["validated_review"],
        negative_reality_on_positive_move_review,
    )
    if negative_reality_on_positive_move.get("validation_status") != "valid":
        errors.append(
            "Reality Check should allow negative challenge when pre-reality check already improved: "
            f"{negative_reality_on_positive_move.get('validation_errors')}"
        )
    allocation_explanation = (
        negative_reality_on_positive_move.get("reality_check_allocation_points") or [{}]
    )[0].get("short_explanation")
    if allocation_explanation != "Operational support gap":
        errors.append(
            "Reality Check display explanation should be deterministic and concise, not rationale truncation: "
            f"{allocation_explanation!r}"
        )
    negative_label_on_positive_review = deepcopy(scoring_review)
    negative_label_on_positive_review["reality_check"]["points"] = 1.0
    negative_label_on_positive_review["reality_check"]["allocations"][0]["movement_label"] = "negative trade-off"
    negative_label_on_positive = validate_and_score_adjudication(
        packet,
        pass1["validated_review"],
        negative_label_on_positive_review,
    )
    if negative_label_on_positive.get("validation_status") != "invalid":
        errors.append("Reality Check should reject negative allocation labels when points are positive")
    missing_scoring_metadata_review = deepcopy(scoring_review)
    missing_scoring_metadata_review.pop("review_metadata", None)
    missing_scoring_metadata = validate_and_score_adjudication(
        packet,
        pass1["validated_review"],
        missing_scoring_metadata_review,
    )
    if missing_scoring_metadata.get("validation_status") != "invalid":
        errors.append("Pass 2 scoring should fail without review_metadata")
    missing_score_read_review = deepcopy(scoring_review)
    missing_score_read_review.pop("score_evolution_read", None)
    missing_score_read = validate_and_score_adjudication(packet, pass1["validated_review"], missing_score_read_review)
    if missing_score_read.get("validation_status") != "invalid":
        errors.append("Pass 2 scoring should fail without score_evolution_read")
    missing_operational_reasoning_review = deepcopy(scoring_review)
    missing_operational_reasoning_review["operational_fit"].pop("relationship_to_previous", None)
    missing_operational_reasoning = validate_and_score_adjudication(
        packet,
        pass1["validated_review"],
        missing_operational_reasoning_review,
    )
    if missing_operational_reasoning.get("validation_status") != "invalid":
        errors.append("Pass 2 scoring should fail without required Operational Fit reasoning fields")
    missing_reality_reasoning_review = deepcopy(scoring_review)
    missing_reality_reasoning_review["reality_check"].pop("carryover_status", None)
    missing_reality_reasoning = validate_and_score_adjudication(
        packet,
        pass1["validated_review"],
        missing_reality_reasoning_review,
    )
    if missing_reality_reasoning.get("validation_status") != "invalid":
        errors.append("Pass 2 scoring should fail without required Reality Check reasoning fields")
    zero_reality_review = deepcopy(scoring_review)
    zero_reality_review["reality_check"]["points"] = 0.0
    zero_reality_review["reality_check"]["allocations"] = []
    zero_reality = validate_and_score_adjudication(packet, pass1["validated_review"], zero_reality_review)
    if zero_reality.get("validation_status") != "valid":
        errors.append(f"Pass 2 scoring should allow empty allocations for zero Reality Check: {zero_reality.get('validation_errors')}")
    missing_allocations_review = deepcopy(zero_reality_review)
    missing_allocations_review["reality_check"].pop("allocations", None)
    missing_allocations = validate_and_score_adjudication(packet, pass1["validated_review"], missing_allocations_review)
    if missing_allocations.get("validation_status") != "invalid":
        errors.append("Pass 2 scoring should fail without canonical reality_check.allocations array")

    no_operational_packet = build_review_packet_from_fixture(_fixture("score_improves_design_neutral_v2"))
    matching_operational_hash = operational_fit_state_hash(no_operational_packet)
    changed_operational_context_packet = deepcopy(no_operational_packet)
    changed_operational_context_packet.setdefault("structured_features", {})["therapeutic_modality_ml"] = (
        "GENE_THERAPY"
    )
    if operational_fit_state_hash(changed_operational_context_packet) == matching_operational_hash:
        errors.append("Operational Fit state hash should change when relevant structured operational context changes")
    changed_indication_packet = deepcopy(no_operational_packet)
    changed_indication_packet.setdefault("structured_features", {})["gbd_cause_id_3_ml"] = "DIFFERENT_INDICATION"
    if operational_fit_state_hash(changed_indication_packet) == matching_operational_hash:
        errors.append("Operational Fit state hash should change when indication context changes")
    matching_structured_hash = xgboost_structured_state_hash(no_operational_packet)
    changed_structured_packet = deepcopy(no_operational_packet)
    changed_structured_packet.setdefault("structured_features", {})["has_dmc_ml"] = "__changed_for_hash_check__"
    if xgboost_structured_state_hash(changed_structured_packet) == matching_structured_hash:
        errors.append("XGBoost structured state hash should change when structured feature context changes")
    no_operational_packet.setdefault("iteration_context", {}).setdefault("trial_score_continuity", {})[
        "recent_score_traces"
    ] = [
        {
            "iteration_id": 1,
            "operational_fit_state_hash": matching_operational_hash,
            "operational_fit_points": 2.0,
            "operational_fit_assessment": {"central_reason": "Prior matching operational state."},
        }
    ]
    no_operational_pass1 = validate_and_score_review(no_operational_packet, _valid_pass1(no_operational_packet))
    matching_operational_review = deepcopy(scoring_review)
    matching_operational_review["operational_fit"]["points"] = 2.0
    matching_operational_review["operational_fit"]["boundary_check"] = (
        "Operational state matches a previous accepted trace; preserving prior Operational Fit."
    )
    matching_operational = validate_and_score_adjudication(
        no_operational_packet,
        no_operational_pass1["validated_review"],
        matching_operational_review,
    )
    if matching_operational.get("validation_status") != "valid":
        errors.append(
            "Operational Fit should validate when unchanged operational state matches previous accepted points: "
            f"{matching_operational.get('validation_errors')}"
        )
    drift_operational_review = deepcopy(matching_operational_review)
    drift_operational_review["operational_fit"]["points"] = 0.0
    drift_operational = validate_and_score_adjudication(
        no_operational_packet,
        no_operational_pass1["validated_review"],
        drift_operational_review,
    )
    if drift_operational.get("validation_status") != "invalid":
        errors.append("Operational Fit should fail when matching operational state does not preserve previous points")

    carryover_packet = deepcopy(no_operational_packet)
    carryover_packet.setdefault("iteration_context", {})["reality_check_carryover_candidate"] = {
        "active": True,
        "previous_reality_check_points": -6.0,
        "previous_reality_check_assessment": {
            "central_reason": "Prior DMC removal was a governance shortcut.",
            "evidence_fields": ["has_dmc_ml"],
        },
        "app_state_precheck": {
            "status": "not_touched",
            "evidence_fields": ["has_dmc_ml"],
        },
    }
    silent_neutral_review = deepcopy(matching_operational_review)
    silent_neutral_review["reality_check"]["points"] = 0.0
    silent_neutral_review["reality_check"]["allocations"] = []
    silent_neutral_review["reality_check"]["carryover_status"] = "unchanged"
    silent_neutral = validate_and_score_adjudication(
        carryover_packet,
        no_operational_pass1["validated_review"],
        silent_neutral_review,
    )
    if silent_neutral.get("validation_status") != "invalid":
        errors.append("Reality Check should fail when untouched material negative carryover silently becomes neutral")
    unresolved_neutral_review = deepcopy(silent_neutral_review)
    unresolved_neutral_review["reality_check"]["carryover_status"] = "unresolved"
    unresolved_neutral = validate_and_score_adjudication(
        carryover_packet,
        no_operational_pass1["validated_review"],
        unresolved_neutral_review,
    )
    if unresolved_neutral.get("validation_status") != "invalid":
        errors.append("Reality Check should not treat unresolved carryover as resolved")
    resolved_neutral_review = deepcopy(silent_neutral_review)
    resolved_neutral_review["reality_check"]["carryover_status"] = "resolved"
    resolved_neutral = validate_and_score_adjudication(
        carryover_packet,
        no_operational_pass1["validated_review"],
        resolved_neutral_review,
    )
    if resolved_neutral.get("validation_status") != "valid":
        errors.append(
            "Reality Check should allow neutralization only when untouched carryover is explicitly resolved: "
            f"{resolved_neutral.get('validation_errors')}"
        )

    baseline_packet = build_review_packet_from_fixture(_fixture("baseline_hidden_review_v2"))
    baseline_result = review_packet_with_mock(baseline_packet)
    if baseline_result.get("status") != "reviewed":
        errors.append("Hidden baseline mock review should validate")
    if baseline_result.get("scoring", {}).get("trial_score") is not None:
        errors.append("Hidden baseline should not expose visible Trial Score")

    narrative = {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "trial_score_narrative": {
            "summary": "The final score is improved but moderated.",
            "movement_reading": "Completion Outlook and execution evidence move favorably.",
            "score_interpretation": "Reality Check keeps the read cautious.",
        },
        "pillar_reading": [
            {"pillar": "Execution Framework", "reading": "Execution scale is now the main read."},
            {"pillar": "Scientific Challenge", "reading": "Evidence ambition remains relevant."},
        ],
        "central_tension": {
            "summary": "Execution scale versus evidence ambition.",
            "why_it_matters": "This is the accepted current development tension.",
        },
        "broader_strategic_question": {
            "mapped_tension": "Execution scale versus evidence ambition.",
            "question": "When does execution scale make evidence more credible rather than just larger?",
        },
    }
    if validate_pass2_review(narrative).get("validation_status") != "valid":
        errors.append("Final narrative shape should validate")

    visible_result = review_packet_with_mock(packet)
    if visible_result.get("status") != "reviewed":
        errors.append(f"Visible mock review should validate: {visible_result.get('failure_reason')}")
    if visible_result.get("scoring", {}).get("trial_score") is None:
        errors.append("Visible mock review should include accepted Trial Score")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Validated simplified Trial Score three-pass contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
