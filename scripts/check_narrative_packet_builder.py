#!/usr/bin/env python
"""Validate deterministic narrative review packet assembly."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.packet_builder import (  # noqa: E402
    ACTIVE_OPERATIONAL_ASSUMPTION_KEYS,
    DIRECT_XGBOOST_SHAP_FIELDS,
    FIELD_DICTIONARY_VERSION,
    STRUCTURED_FEATURE_KEYS,
    build_review_packet,
    build_review_packet_from_fixture,
    stable_packet_hash,
)


def _check_packet(fixture: dict, errors: list[str]) -> None:
    fixture_id = fixture["fixture_id"]
    source_packet = fixture["input_packet"]
    packet = build_review_packet_from_fixture(fixture)

    for key in (
        "prompt_version",
        "rubric_version",
        "field_dictionary_version",
        "mode",
        "trial_identity",
        "text_context",
        "structured_features",
        "structured_feature_display_values",
        "structured_feature_meanings",
        "text_context_field_meanings",
        "reference_packs",
        "operational_assumptions",
        "model_interpretation",
        "review_context",
        "clarification_context",
        "iteration_context",
        "input_hash",
    ):
        if key not in packet:
            errors.append(f"{fixture_id}: missing packet.{key}")

    if packet.get("prompt_version") != source_packet.get("prompt_version"):
        errors.append(f"{fixture_id}: prompt_version changed")
    if packet.get("rubric_version") != source_packet.get("rubric_version"):
        errors.append(f"{fixture_id}: rubric_version changed")
    if packet.get("field_dictionary_version") != FIELD_DICTIONARY_VERSION:
        errors.append(f"{fixture_id}: field_dictionary_version changed")
    if packet.get("mode") != source_packet.get("mode"):
        errors.append(f"{fixture_id}: mode changed")

    text_context = packet.get("text_context") or {}
    if "criteria_ui" in text_context:
        errors.append(f"{fixture_id}: criteria_ui should not be sent by default in V1 narrative packets")

    missing_features = set(STRUCTURED_FEATURE_KEYS).difference(packet.get("structured_features", {}))
    if missing_features:
        errors.append(f"{fixture_id}: missing structured features {sorted(missing_features)}")

    missing_operational = set(ACTIVE_OPERATIONAL_ASSUMPTION_KEYS).difference(packet.get("operational_assumptions", {}))
    if missing_operational:
        errors.append(f"{fixture_id}: missing operational assumptions {sorted(missing_operational)}")

    model = packet.get("model_interpretation", {})
    source_model = source_packet.get("model_interpretation", {})
    if model.get("completion_score") != source_model.get("completion_score"):
        errors.append(f"{fixture_id}: completion_score changed")
    if model.get("score_delta") != source_model.get("score_delta"):
        errors.append(f"{fixture_id}: score_delta changed")
    if model.get("direct_xgboost_shap_fields") != list(DIRECT_XGBOOST_SHAP_FIELDS):
        errors.append(f"{fixture_id}: direct_xgboost_shap_fields mismatch")
    guidance = model.get("model_signal_guidance") or {}
    if "model_signal_guidance" not in model:
        errors.append(f"{fixture_id}: model_interpretation missing model_signal_guidance")
    elif (
        "feature-level evidence" not in str(guidance.get("granularity_rule") or "")
        or "Movement explains what changed" not in str(guidance.get("main_model_signals_rule") or "")
        or "Scientific Challenge alignment" not in (guidance.get("avoid") or [])
    ):
        errors.append(f"{fixture_id}: model_signal_guidance should constrain main_model_signals granularity")

    iteration = packet.get("iteration_context", {})
    source_iteration = source_packet.get("iteration_context", {})
    if iteration.get("changed_fields") != source_iteration.get("changed_fields"):
        errors.append(f"{fixture_id}: changed_fields changed")
    if iteration.get("compact_storyline_memory") != source_iteration.get("compact_storyline_memory"):
        errors.append(f"{fixture_id}: compact_storyline_memory changed")

    review_context = packet.get("review_context")
    if not isinstance(review_context, dict):
        errors.append(f"{fixture_id}: review_context must be an object")
    elif set(review_context) != {"baseline_review", "previous_review"}:
        errors.append(f"{fixture_id}: review_context keys changed")

    clarification_context = packet.get("clarification_context")
    if not isinstance(clarification_context, dict):
        errors.append(f"{fixture_id}: clarification_context must be an object")
    elif "user_clarifications" not in clarification_context:
        errors.append(f"{fixture_id}: clarification_context missing user_clarifications")

    packet_without_hash = dict(packet)
    input_hash = packet_without_hash.pop("input_hash", None)
    if stable_packet_hash(packet_without_hash) != input_hash:
        errors.append(f"{fixture_id}: input_hash is not stable over packet content")


def _check_review_continuity_context(errors: list[str]) -> None:
    fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "score_improves_evidence_weakens_v2"
    )
    baseline_trace = {
        "input_hash": "baseline-input-hash",
        "iteration_id": 0,
        "status": "reviewed",
        "validation_status": "valid",
        "reality_check_points": 0,
        "trial_score": 68,
        "changed_fields": [],
        "score_delta": 0,
        "central_tension": "Baseline central discussion topic.",
        "development_discussion_options": [
            {
                "topic": "Hidden baseline option must be scrubbed.",
                "why_it_matters": "Participants never saw this.",
                "supporting_evidence": ["phase_ml"],
                "participant_wider_question": {
                    "question": "Hidden baseline question must be scrubbed?",
                    "supporting_evidence": ["phase_ml"],
                },
            },
        ],
        "participant_central_tension": {
            "summary": "Hidden selected discussion topic must be scrubbed.",
            "why_it_matters": "Participants never saw this.",
        },
        "participant_broader_strategic_question": {
            "question": "Hidden selected question must be scrubbed?",
            "mapped_tension": "Hidden selected discussion topic must be scrubbed.",
        },
        "recent_participant_visible_questions": [
            {
                "question": "Hidden baseline should not consume this participant topic.",
                "mapped_tension": "Hidden baseline seed development issue.",
            },
        ],
        "validated_review": {
            "review_metadata": {"review_mode": "hidden_baseline", "participant_visible": False},
            "main_tension": "Baseline main development issue from dedicated field.",
            "completion_outlook_analysis": {
                "risk_pattern_summary": "Baseline score reflects an acceptable original design profile.",
            },
            "reality_check": {
                "central_reason": "Baseline design comment.",
                "effect": "neutral",
                "strength": "none",
                "evidence_fields": [],
                "allocations": [],
            },
            "reality_check_assessment": {
                "effect": "neutral",
                "strength": "none",
                "points": 0,
            },
            "key_questions": {
                "medical_development_question": "What evidence standard matters most?",
                "clinical_operations_question": "What operational burden is proportionate?",
                "strategic_field_question": "What broader field challenge does this scenario expose?",
            },
            "continuity_update": {
                "active_tension": "Hidden baseline active discussion topic must be scrubbed.",
                "watch_next": "Watch whether later edits preserve the baseline evidence-operational balance.",
            },
            "continuity": {
                "new_concerns": [],
                "storyline_update": "Baseline memory",
            },
        },
        "storyline_state": {
            "active_tension": "Hidden baseline stored storyline development issue must be scrubbed.",
            "active_tension_status": "active",
            "next_consideration": "Watch whether later edits preserve the baseline evidence-operational balance.",
        },
        "compact_storyline_memory": "Baseline memory",
    }
    previous_trace = {
        **baseline_trace,
        "input_hash": "previous-input-hash",
        "iteration_id": 1,
        "trial_score": 66,
        "pre_reality_score": 68,
        "operational_fit_points": 0,
        "reality_check_points": -2,
        "reality_check_assessment": {
            "effect": "offset_gain",
            "strength": "moderate",
            "points": -2,
            "central_reason": "The prior move simplified evidence.",
            "allocation_points": [
                {
                    "allocation_target_id": "scientific_challenge.protocol_architecture",
                    "pillar": "Scientific Challenge",
                    "subpillar": "Protocol Architecture",
                    "points": -2,
                }
            ],
        },
        "reality_check_allocation_points": [
            {
                "allocation_target_id": "scientific_challenge.protocol_architecture",
                "pillar": "Scientific Challenge",
                "subpillar": "Protocol Architecture",
                "points": -2,
            }
        ],
        "changed_fields": ["operational_assumptions.planned_enrollment"],
        "score_delta": 0,
        "central_tension": "",
        "recent_participant_visible_questions": [
            {
                "question": "How should teams debate feasibility gains when evidence strength remains uncertain?",
                "mapped_tension": "Feasibility vs Evidence Strength",
            },
        ],
        "participant_central_tension": {
            "summary": "Feasibility vs Evidence Strength",
            "why_it_matters": "This was shown to the participant.",
        },
        "participant_broader_strategic_question": {
            "question": "How should teams debate feasibility gains when evidence strength remains uncertain?",
        },
        "validated_review": {
            **baseline_trace["validated_review"],
            "main_tension": "Previous main development issue from dedicated field.",
            "reality_check": {
                "effect": "offset_gain",
                "strength": "moderate",
                "central_reason": "The prior move simplified evidence.",
                "evidence_fields": ["endpoint_rigor_ml"],
                "allocations": [],
            },
            "development_discussion_options": [
                {
                    "topic": "Feasibility vs Evidence Strength",
                    "why_it_matters": "Endpoint credibility remains partly active.",
                    "supporting_evidence": ["endpoint_rigor_ml"],
                    "participant_wider_question": {
                        "question": "How should teams debate feasibility gains when evidence strength remains uncertain?",
                        "supporting_evidence": ["endpoint_rigor_ml"],
                    },
                },
                {
                    "topic": "Evidence completeness vs execution support",
                    "why_it_matters": "The prior move may change feasibility faster than interpretability.",
                    "supporting_evidence": ["operational_assumptions.planned_enrollment"],
                    "participant_wider_question": {
                        "question": "When does feasibility strengthen evidence confidence, and when does it mainly expose what remains uncertain?",
                        "supporting_evidence": ["operational_assumptions.planned_enrollment"],
                    },
                },
                {
                    "topic": "Population focus vs broader development confidence",
                    "why_it_matters": "A narrower population can clarify one question while limiting program-level inference.",
                    "supporting_evidence": ["patient_severity_ml"],
                    "participant_wider_question": {
                        "question": "When does a focused population create stronger evidence, and when does it narrow the strategic value of the result?",
                        "supporting_evidence": ["patient_severity_ml"],
                    },
                },
            ],
            "continuity_update": {
                "active_tension": "Feasibility vs Evidence Strength",
                "what_changed": "Previous strategic storyline memory.",
                "watch_next": "Restore evidence credibility without returning fully to baseline burden.",
            },
            "continuity": {
                "prior_concerns_resolved": ["reduced execution burden"],
                "prior_concerns_worsened": ["endpoint credibility"],
                "prior_concerns_unchanged": ["evidence burden"],
                "new_concerns": ["population focus"],
                "storyline_update": "Previous strategic storyline memory.",
            },
        },
        "storyline_state": {
            "active_tension": "Feasibility vs Evidence Strength",
            "active_tension_status": "partially_active",
            "last_effect_label": "partly_offsets_score_gain",
            "last_move_classification": ["oversimplification"],
            "protected_gains": ["reduced execution burden"],
            "regression_watch": ["endpoint credibility"],
            "active_carryover": ["evidence burden"],
            "new_concerns": ["population focus"],
            "next_consideration": "Restore evidence credibility without returning fully to baseline burden.",
            "storyline_update": "Previous strategic storyline memory.",
        },
        "compact_storyline_memory": "Previous iteration memory",
    }
    packet = fixture["input_packet"]
    built = build_review_packet_from_fixture(fixture)
    continuity_packet = build_review_packet(
        current_snapshot={
            "snapshot_id": packet["iteration_context"].get("current_snapshot_id"),
            "structured_features": packet.get("structured_features", {}),
            "operational_assumptions": packet.get("operational_assumptions", {}),
            "model_interpretation": packet.get("model_interpretation", {}),
            "changed_fields": [],
            "changed_operational_assumptions": ["planned_enrollment"],
        },
        previous_snapshot={"snapshot_id": packet["iteration_context"].get("previous_snapshot_id"), "score": 68},
        baseline_snapshot={"snapshot_id": packet["iteration_context"].get("baseline_snapshot_id")},
        baseline_review_trace=baseline_trace,
        previous_review_trace=previous_trace,
        compact_storyline_memory="Previous iteration memory",
    )

    context = continuity_packet.get("review_context") or {}
    if context.get("baseline_review", {}).get("input_hash") != "baseline-input-hash":
        errors.append("continuity packet missing baseline review context")
    if context.get("previous_review", {}).get("input_hash") != "previous-input-hash":
        errors.append("continuity packet missing previous review context")
    if context.get("baseline_review", {}).get("design_numeric_context") != "hidden_baseline_qualitative_only":
        errors.append("hidden baseline review context should be marked qualitative-only")
    if (
        context.get("baseline_review", {}).get("baseline_completion_outlook_summary")
        != "Baseline score reflects an acceptable original design profile."
    ):
        errors.append("hidden baseline review context should preserve completion outlook summary")
    if context.get("previous_review", {}).get("reality_check_points") != -2:
        errors.append("previous visible review context should preserve reality_check_points")
    if context.get("previous_review", {}).get("trial_score") != 66:
        errors.append("previous visible review context should preserve trial_score")
    if context.get("baseline_review", {}).get("central_tension"):
        errors.append("hidden baseline review context should not expose an active central_tension")
    if context.get("baseline_review", {}).get("development_discussion_options") != []:
        errors.append("hidden baseline review context should scrub development_discussion_options")
    if context.get("baseline_review", {}).get("participant_central_tension") != {}:
        errors.append("hidden baseline review context should scrub participant_central_tension")
    if context.get("baseline_review", {}).get("participant_broader_strategic_question") != {}:
        errors.append("hidden baseline review context should scrub participant_broader_strategic_question")
    baseline_storyline = context.get("baseline_review", {}).get("storyline_state") or {}
    if baseline_storyline.get("active_tension"):
        errors.append("hidden baseline review context should scrub storyline_state.active_tension")
    if baseline_storyline.get("active_tension_status") != "not_applicable":
        errors.append("hidden baseline review context should reset storyline_state.active_tension_status")
    if context.get("baseline_review", {}).get("recent_participant_visible_questions") != []:
        errors.append("hidden baseline review context must not mark baseline questions as participant-visible history")
    baseline_memory = context.get("baseline_review", {}).get("compact_storyline_memory") or ""
    if "Baseline watch:" not in baseline_memory:
        errors.append("hidden baseline review context should compact useful baseline watch memory")
    if context.get("previous_review", {}).get("central_tension") != "Feasibility vs Evidence Strength":
        errors.append("previous visible review context should preserve current participant-selected central discussion topic")
    previous_questions = context.get("previous_review", {}).get("key_questions") or {}
    if previous_questions.get("medical_clinical_development_question") != "What evidence standard matters most?":
        errors.append("previous visible review context should expose new medical/clinical-development question field")
    if previous_questions.get("strategic_development_question") != "What broader field challenge does this scenario expose?":
        errors.append("previous visible review context should expose new strategic development question field")
    if previous_questions.get("medical_development_question") != "What evidence standard matters most?":
        errors.append("previous visible review context should preserve legacy medical question alias")
    if (
        continuity_packet.get("iteration_context", {}).get("compact_storyline_memory")
        != "Previous iteration memory"
    ):
        errors.append("continuity packet missing compact storyline memory")
    visible_history = context.get("previous_review", {}).get("recent_participant_visible_questions") or []
    latest_visible = visible_history[-1] if visible_history else {}
    if latest_visible.get("question") != "How should teams debate feasibility gains when evidence strength remains uncertain?":
        errors.append("previous visible review context should preserve participant-visible strategic question")
    if latest_visible.get("mapped_tension") != "Feasibility vs Evidence Strength":
        errors.append("previous visible review context should preserve participant-visible mapped development issue")
    trial_score_continuity = continuity_packet.get("iteration_context", {}).get("trial_score_continuity") or {}
    if trial_score_continuity.get("available") is not True:
        errors.append("later visible continuity packet should include Trial Score continuity anchors")
    if trial_score_continuity.get("active_tension") != "Feasibility vs Evidence Strength":
        errors.append("Trial Score continuity should carry active_tension")
    if trial_score_continuity.get("previous_trial_score") != 66:
        errors.append("Trial Score continuity should carry previous_trial_score")
    if trial_score_continuity.get("previous_reality_check_points") != -2:
        errors.append("Trial Score continuity should carry previous_reality_check_points")
    if trial_score_continuity.get("protected_gains") != ["reduced execution burden"]:
        errors.append("Trial Score continuity should carry protected gains")
    if trial_score_continuity.get("regression_watch") != ["endpoint credibility"]:
        errors.append("Trial Score continuity should carry regression watch")
    if trial_score_continuity.get("next_consideration") != "Restore evidence credibility without returning fully to baseline burden.":
        errors.append("Trial Score continuity should carry next consideration")
    carryover_candidate = continuity_packet.get("iteration_context", {}).get("reality_check_carryover_candidate") or {}
    if carryover_candidate.get("active") is not True:
        errors.append("material previous negative Reality Check should create an active carryover candidate")
    if carryover_candidate.get("previous_reality_check_points") != -2.0:
        errors.append("carryover candidate should preserve previous negative Reality Check points")
    if (
        (carryover_candidate.get("previous_reality_check_assessment") or {}).get("central_reason")
        != "The prior move simplified evidence."
    ):
        errors.append("carryover candidate should preserve previous Reality Check reason")
    if not carryover_candidate.get("previous_reality_check_allocation_points"):
        errors.append("carryover candidate should preserve previous Reality Check allocations")
    if "strategic_review_continuity" in (continuity_packet.get("iteration_context") or {}):
        errors.append("packet should not send legacy Strategic Review continuity to the provider")
    if "design_confidence_continuity" in (continuity_packet.get("iteration_context") or {}):
        errors.append("packet should not send legacy Design Confidence continuity to the provider")
    operational_change_labels = {
        change.get("field"): change.get("display_label")
        for change in continuity_packet.get("iteration_context", {}).get("field_changes") or []
        if change.get("change_type") == "operational_assumption"
    }
    if operational_change_labels.get("operational_assumptions.planned_duration_months") not in {
        None,
        "Planned Total Timeline",
    }:
        errors.append("planned duration operational change should use Planned Total Timeline display label")
    if built.get("review_context", {}).get("previous_review") is not None:
        errors.append("fixture packet without review traces should not invent previous review context")
    if "design_confidence_continuity" in (built.get("iteration_context") or {}):
        errors.append("fixture packet should not include legacy Design Confidence continuity")
    hidden_baseline_packet = build_review_packet(
        current_snapshot={
            "snapshot_id": "baseline-snapshot",
            "source": "prerecorded_baseline",
            "structured_features": packet.get("structured_features", {}),
            "operational_assumptions": packet.get("operational_assumptions", {}),
            "model_interpretation": packet.get("model_interpretation", {}),
            "changed_fields": [],
        },
        previous_snapshot=None,
        baseline_snapshot={"snapshot_id": "baseline-snapshot"},
        baseline_review_trace=baseline_trace,
        previous_review_trace=None,
    )
    if "design_confidence_continuity" in (hidden_baseline_packet.get("iteration_context") or {}):
        errors.append("hidden baseline packet should not include legacy Design Confidence continuity")

    baseline_return_packet = build_review_packet(
        current_snapshot={
            "snapshot_id": "returned-baseline",
            "structured_features": packet.get("structured_features", {}),
            "operational_assumptions": packet.get("operational_assumptions", {}),
            "model_interpretation": packet.get("model_interpretation", {}),
            "changed_fields": [],
        },
        previous_snapshot={
            "snapshot_id": "previous-visible",
            "iteration_context": {"iteration_number": 1},
            "score": 68,
        },
        baseline_snapshot={
            "snapshot_id": "baseline-snapshot",
            "structured_features": packet.get("structured_features", {}),
            "operational_assumptions": packet.get("operational_assumptions", {}),
            "model_interpretation": packet.get("model_interpretation", {}),
        },
        baseline_review_trace=baseline_trace,
        previous_review_trace=previous_trace,
    )
    if baseline_return_packet.get("iteration_context", {}).get("returned_to_hidden_baseline_state") is not True:
        errors.append("baseline-return packet should identify the hidden-baseline scenario state")
    if (baseline_return_packet.get("iteration_context", {}).get("reality_check_carryover_candidate") or {}).get("active"):
        errors.append("baseline-return packet should suppress Reality Check carryover candidate")


def _check_canonical_values_prefer_compare_values(errors: list[str]) -> None:
    packet = build_review_packet(
        current_snapshot={
            "snapshot_id": "canonical-check",
            "submitted_values": {
                "endpoint_structure_ml": 1,
                "has_placebo_ml": 1,
            },
            "compare_values": {
                "endpoint_structure_ml": "MULTI_COMPOSITE",
                "has_placebo_ml": "1",
            },
            "display_values": {
                "endpoint_structure_ml": "Multi/Composite",
                "has_placebo_ml": "Yes",
            },
            "model_interpretation": {"completion_score": 70},
            "text_context": {"criteria_ui": "Long eligibility criteria should be deferred by default."},
            "operational_assumptions": {
                "planned_enrollment": {},
                "planned_sites": {},
                "planned_duration_months": {},
            },
        },
    )
    structured = packet.get("structured_features") or {}
    display = packet.get("structured_feature_display_values") or {}
    if structured.get("endpoint_structure_ml") != "MULTI_COMPOSITE":
        errors.append("packet should prefer taxonomy option key compare_values over model-facing submitted_values")
    if display.get("endpoint_structure_ml") != "Multi/Composite":
        errors.append("packet should preserve human-readable display values")
    meanings = packet.get("structured_feature_meanings") or {}
    if "evidence ambition" not in str(meanings.get("phase_ml")):
        errors.append("packet should include clinical meanings for structured features")
    if "decision interpretability" not in str(meanings.get("endpoint_rigor_ml")):
        errors.append("packet should include endpoint-rigor field meaning")
    text_meanings = packet.get("text_context_field_meanings") or {}
    if "summary" not in str(text_meanings.get("summary_ui")).lower():
        errors.append("packet should include meanings for text-context fields")
    if "criteria_ui" in (packet.get("text_context") or {}):
        errors.append("packet should not include criteria_ui by default")

    reference_packs = packet.get("reference_packs") or []
    pack_ids = {pack.get("pack_id") for pack in reference_packs if isinstance(pack, dict)}
    for required_pack in (
        "core_clinical_development_v1",
        "strategic_context_2026_v1",
        "ich_e8_quality_by_design_v1",
    ):
        if required_pack not in pack_ids:
            errors.append(f"packet should include default reference pack {required_pack}")
    if not all((pack.get("prompt_safe_summary") or "").strip() for pack in reference_packs if isinstance(pack, dict)):
        errors.append("reference packs should include prompt-safe summaries")


def _check_field_and_impact_changes(errors: list[str]) -> None:
    baseline_snapshot = {
        "snapshot_id": "baseline",
        "compare_values": {
            "endpoint_rigor_ml": "HARD_CLINICAL",
            "comparator_benchmark_ml": "ACTIVE_MODERN_STANDARD",
        },
        "display_values": {
            "endpoint_rigor_ml": "Hard Clinical (Survival/Death)",
            "comparator_benchmark_ml": "Active (Modern Standard)",
        },
        "text_context": {"summary_ui": "Original summary."},
        "score": 68,
        "pillar_impacts": [
            {"Pillar": "Scientific Challenge", "Impact": -1.0},
            {"Pillar": "Execution Framework", "Impact": 1.0},
        ],
        "feature_impacts": [
            {"Pillar": "Scientific Challenge", "Subcategory": "Protocol Architecture", "Impact": -1.0},
            {"Pillar": "Execution Framework", "Subcategory": "Methodological Setup", "Impact": 1.0},
        ],
        "feature_level_impacts": [
            {
                "Feature": "endpoint_rigor_ml",
                "Label": "Endpoint Rigor",
                "Value": "Clinical",
                "Pillar": "Scientific Challenge",
                "Subcategory": "Protocol Architecture",
                "Impact": -1.0,
            },
            {
                "Feature": "comparator_benchmark_ml",
                "Label": "Comparator",
                "Value": "Active (Modern Standard)",
                "Pillar": "Execution Framework",
                "Subcategory": "Methodological Setup",
                "Impact": 1.0,
            },
            {
                "Feature": "phase_ml",
                "Label": "Phase",
                "Value": "Phase 2",
                "Pillar": "Therapeutic Context",
                "Subcategory": "Development Phase",
                "Impact": -2.0,
            },
            {
                "Feature": "endpoint_structure_ml",
                "Label": "Endpoint Structure",
                "Value": "Single primary endpoint",
                "Pillar": "Scientific Challenge",
                "Subcategory": "Protocol Architecture",
                "Impact": -5.0,
            },
            {
                "Feature": "healthy_volunteers_ml",
                "Label": "Healthy Volunteers",
                "Value": "No",
                "Pillar": "Patient Profile",
                "Subcategory": "Population Scope",
                "Impact": -1.5,
            },
        ],
    }
    previous_snapshot = {
        **baseline_snapshot,
        "snapshot_id": "previous",
        "score": 70,
        "compare_values": {
            "endpoint_rigor_ml": "SURROGATE",
            "comparator_benchmark_ml": "ACTIVE_MODERN_STANDARD",
        },
        "display_values": {
            "endpoint_rigor_ml": "Surrogate / Biomarker",
            "comparator_benchmark_ml": "Active (Modern Standard)",
        },
        "pillar_impacts": [
            {"Pillar": "Scientific Challenge", "Impact": 0.5},
            {"Pillar": "Execution Framework", "Impact": 1.0},
        ],
        "feature_impacts": [
            {"Pillar": "Scientific Challenge", "Subcategory": "Protocol Architecture", "Impact": 0.5},
            {"Pillar": "Execution Framework", "Subcategory": "Methodological Setup", "Impact": 1.0},
        ],
        "feature_level_impacts": [
            {
                "Feature": "endpoint_rigor_ml",
                "Label": "Endpoint Rigor",
                "Value": "Surrogate / Biomarker",
                "Pillar": "Scientific Challenge",
                "Subcategory": "Protocol Architecture",
                "Impact": 0.5,
            },
            {
                "Feature": "comparator_benchmark_ml",
                "Label": "Comparator",
                "Value": "Active (Modern Standard)",
                "Pillar": "Execution Framework",
                "Subcategory": "Methodological Setup",
                "Impact": 1.0,
            },
            {
                "Feature": "phase_ml",
                "Label": "Phase",
                "Value": "Phase 2",
                "Pillar": "Therapeutic Context",
                "Subcategory": "Development Phase",
                "Impact": -2.0,
            },
            {
                "Feature": "endpoint_structure_ml",
                "Label": "Endpoint Structure",
                "Value": "Single primary endpoint",
                "Pillar": "Scientific Challenge",
                "Subcategory": "Protocol Architecture",
                "Impact": -3.0,
            },
            {
                "Feature": "healthy_volunteers_ml",
                "Label": "Healthy Volunteers",
                "Value": "No",
                "Pillar": "Patient Profile",
                "Subcategory": "Population Scope",
                "Impact": -1.5,
            },
        ],
    }
    current_snapshot = {
        **previous_snapshot,
        "snapshot_id": "current",
        "score": 74,
        "compare_values": {
            "endpoint_rigor_ml": "SURROGATE",
            "comparator_benchmark_ml": "PLACEBO",
        },
        "display_values": {
            "endpoint_rigor_ml": "Surrogate / Biomarker",
            "comparator_benchmark_ml": "Placebo Control",
        },
        "text_context": {"summary_ui": "Revised summary."},
        "changed_fields": ["comparator_benchmark_ml"],
        "changed_text_context_fields": ["summary_ui"],
        "pillar_impacts": [
            {"Pillar": "Scientific Challenge", "Impact": 0.5},
            {"Pillar": "Execution Framework", "Impact": 3.0},
        ],
        "feature_impacts": [
            {"Pillar": "Scientific Challenge", "Subcategory": "Protocol Architecture", "Impact": 0.8},
            {"Pillar": "Execution Framework", "Subcategory": "Methodological Setup", "Impact": 3.0},
        ],
        "feature_level_impacts": [
            {
                "Feature": "endpoint_rigor_ml",
                "Label": "Endpoint Rigor",
                "Value": "Surrogate / Biomarker",
                "Pillar": "Scientific Challenge",
                "Subcategory": "Protocol Architecture",
                "Impact": 0.5,
            },
            {
                "Feature": "comparator_benchmark_ml",
                "Label": "Comparator",
                "Value": "Placebo Control",
                "Pillar": "Execution Framework",
                "Subcategory": "Methodological Setup",
                "Impact": 3.0,
            },
            {
                "Feature": "allocation_ml",
                "Label": "Allocation",
                "Value": "Randomized",
                "Pillar": "Execution Framework",
                "Subcategory": "Methodological Setup",
                "Impact": 2.0,
            },
            {
                "Feature": "sponsor_tier_ml",
                "Label": "Sponsor Tier",
                "Value": "Large pharma",
                "Pillar": "Execution Framework",
                "Subcategory": "Sponsor Capability",
                "Impact": 1.0,
            },
            {
                "Feature": "phase_ml",
                "Label": "Phase",
                "Value": "Phase 2",
                "Pillar": "Therapeutic Context",
                "Subcategory": "Development Phase",
                "Impact": -2.0,
            },
            {
                "Feature": "administration_complexity_ml",
                "Label": "Administration Complexity",
                "Value": "Complex",
                "Pillar": "Execution Framework",
                "Subcategory": "Site Burden",
                "Impact": -0.7,
            },
            {
                "Feature": "line_of_therapy_ml",
                "Label": "Line of Therapy",
                "Value": "Later line",
                "Pillar": "Patient Profile",
                "Subcategory": "Treatment Context",
                "Impact": -0.2,
            },
            {
                "Feature": "endpoint_structure_ml",
                "Label": "Endpoint Structure",
                "Value": "Single primary endpoint",
                "Pillar": "Scientific Challenge",
                "Subcategory": "Protocol Architecture",
                "Impact": -3.0,
            },
        ],
        "operational_assumptions": {
            "planned_enrollment": {},
            "planned_sites": {},
            "planned_duration_months": {},
        },
    }
    packet = build_review_packet(
        current_snapshot=current_snapshot,
        previous_snapshot=previous_snapshot,
        baseline_snapshot=baseline_snapshot,
    )
    field_changes = packet.get("iteration_context", {}).get("field_changes") or []
    comparator_change = next(
        (item for item in field_changes if item.get("field") == "comparator_benchmark_ml"),
        None,
    )
    if not comparator_change:
        errors.append("packet should include structured field_changes for edited fields")
    elif (
        comparator_change.get("previous_value") != "ACTIVE_MODERN_STANDARD"
        or comparator_change.get("current_value") != "PLACEBO"
        or comparator_change.get("baseline_value") != "ACTIVE_MODERN_STANDARD"
    ):
        errors.append("structured field_changes should include baseline, previous, and current values")

    text_change = next(
        (item for item in field_changes if item.get("field") == "text_context.summary_ui"),
        None,
    )
    if not text_change or text_change.get("change_type") != "text_context":
        errors.append("packet should include text field_changes for changed text context")

    impact_changes = packet.get("model_interpretation", {}).get("xgboost_impact_changes") or []
    execution_change = next(
        (
            item for item in impact_changes
            if item.get("impact_level") == "pillar" and item.get("name") == "Execution Framework"
        ),
        None,
    )
    if not execution_change:
        errors.append("packet should include XGBoost pillar impact changes")
    elif (
        execution_change.get("baseline_impact") != 1.0
        or execution_change.get("previous_impact") != 1.0
        or execution_change.get("current_impact") != 3.0
        or execution_change.get("delta_from_previous") != 2.0
        or execution_change.get("delta_from_baseline") != 2.0
        or execution_change.get("changed_since_previous") is not True
        or execution_change.get("changed_from_baseline") is not True
    ):
        errors.append("XGBoost impact changes should include baseline, previous, current, and deltas")

    model = packet.get("model_interpretation", {})
    state = model.get("current_model_state_evidence") or {}
    movement = model.get("model_movement_evidence") or {}
    positive_pillars = state.get("top_positive_pillar_impacts") or []
    negative_pillars = state.get("top_negative_pillar_impacts") or []
    if not any(item.get("pillar") == "Execution Framework" and item.get("impact") == 3.0 for item in positive_pillars):
        errors.append("current model state should expose top positive pillar impacts")
    if not any(item.get("pillar") == "Scientific Challenge" and item.get("impact") == 0.5 for item in positive_pillars):
        errors.append("current model state should classify positive signed impacts as favorable state")
    if negative_pillars:
        errors.append("current model state should not report negative pillars when all current pillar impacts are positive")
    positive_subpillars = state.get("top_positive_subpillar_impacts") or []
    if not any(
        item.get("subpillar") == "Methodological Setup" and item.get("impact") == 3.0
        for item in positive_subpillars
    ):
        errors.append("current model state should expose top positive subpillar impacts")
    if state.get("feature_impact_availability") != "direct_xgboost_feature_impacts_available":
        errors.append("feature impact availability should report direct XGBoost feature impacts")
    positive_features = state.get("top_positive_feature_impacts") or []
    negative_features = state.get("top_negative_feature_impacts") or []
    if [item.get("feature") for item in positive_features] != [
        "comparator_benchmark_ml",
        "allocation_ml",
        "sponsor_tier_ml",
    ]:
        errors.append("current model state should expose only the top three positive feature impacts")
    if [item.get("feature") for item in negative_features] != [
        "endpoint_structure_ml",
        "phase_ml",
        "administration_complexity_ml",
    ]:
        errors.append("current model state should expose only the top three negative feature impacts")
    if any(item.get("feature") == "Protocol Architecture" for item in positive_features + negative_features):
        errors.append("feature impact state should not treat subpillar rows as feature rows")

    positive_movements = movement.get("top_positive_pillar_movements") or []
    if not any(
        item.get("pillar") == "Execution Framework"
        and item.get("delta_from_previous") == 2.0
        and item.get("movement_from_previous") == "still_positive_and_improved"
        and item.get("crossed_zero_from_previous") is False
        for item in positive_movements
    ):
        errors.append("model movement should expose positive pillar movement from previous iteration")
    positive_subpillar_movements = movement.get("top_positive_subpillar_movements") or []
    if not any(
        item.get("subpillar") == "Protocol Architecture"
        and item.get("movement_from_baseline") == "negative_to_positive"
        and item.get("crossed_zero_from_baseline") is True
        for item in positive_subpillar_movements
    ):
        errors.append("model movement should expose subpillar zero-crossing from baseline")
    positive_feature_movements = movement.get("top_positive_feature_movements") or []
    if not any(
        item.get("feature") == "comparator_benchmark_ml"
        and item.get("delta_from_previous") == 2.0
        and item.get("delta_from_baseline") == 2.0
        for item in positive_feature_movements
    ):
        errors.append("model movement should expose direct feature impact movement")
    if not any(
        item.get("feature") == "healthy_volunteers_ml"
        and item.get("current_impact") == 0.0
        and item.get("delta_from_previous") == 1.5
        and item.get("movement_from_previous") == "improved"
        for item in positive_feature_movements
    ):
        errors.append("model movement should treat missing current feature impact as zero for movement")
    if any(item.get("feature") == "endpoint_structure_ml" for item in positive_feature_movements):
        errors.append("model movement should not rank stale baseline movement ahead of previous-iteration movement")
    negative_feature_movements = movement.get("top_negative_feature_movements") or []
    if any(item.get("feature") == "endpoint_structure_ml" for item in negative_feature_movements):
        errors.append("model movement should not rank stale negative baseline movement ahead of previous-iteration movement")


def _check_operational_movement_context(errors: list[str]) -> None:
    baseline_snapshot = {
        "snapshot_id": "baseline-operational",
        "source": "prerecorded_baseline",
        "operational_assumptions": {
            "planned_enrollment": {
                "value": 40,
                "source": "final_observed_value",
                "benchmark_snapshot_id": "baseline-enrollment-cohort",
                "benchmark_p50": 120,
                "enrollment_status": "below_benchmark",
            },
            "planned_sites": {
                "value": 4,
                "source": "completed_registry_facility_count",
                "site_default_basis": "completed_registry_facility_count",
                "benchmark_p50": 12,
                "site_count_status": "below_benchmark",
                "patients_per_site_value": 10,
                "patients_per_site_p50": 30,
                "patients_per_site_status": "below_benchmark",
            },
            "planned_duration_months": {
                "value": 24,
                "source": "completed_actual_primary_completion",
                "benchmark_p50": 30,
                "duration_status": "typical",
            },
        },
    }
    current_snapshot = {
        "snapshot_id": "current-operational",
        "source": "simulation_ptc",
        "changed_fields": ["operational_assumptions.planned_enrollment", "operational_assumptions.planned_sites"],
        "changed_operational_assumptions": ["planned_enrollment", "planned_sites"],
        "operational_assumptions": {
            "planned_enrollment": {
                "value": 100,
                "source": "user_scenario",
                "benchmark_snapshot_id": "current-enrollment-cohort",
                "benchmark_p50": 120,
                "enrollment_status": "typical",
            },
            "planned_sites": {
                "value": 16,
                "source": "user_scenario",
                "benchmark_p50": 12,
                "site_count_status": "ambitious",
                "patients_per_site_value": 6.25,
                "patients_per_site_p50": 30,
                "patients_per_site_status": "below_benchmark",
            },
            "planned_duration_months": {
                "value": 30,
                "source": "user_scenario",
                "benchmark_p50": 30,
                "duration_status": "typical",
            },
        },
    }
    packet = build_review_packet(current_snapshot=current_snapshot, baseline_snapshot=baseline_snapshot)
    movement = packet.get("operational_movement_context") or {}
    fields = movement.get("fields") or {}
    if movement.get("baseline_is_neutral_reference") is not True:
        errors.append("operational_movement_context should mark the baseline as neutral reference")

    enrollment = fields.get("planned_enrollment") or {}
    if ((enrollment.get("movement_from_baseline") or {}).get("relative_to_p50")) != "toward_p50":
        errors.append("enrollment movement should identify movement toward P50 when current value is closer than baseline")
    if ((enrollment.get("baseline") or {}).get("confidence")) != "high":
        errors.append("completed observed enrollment baseline should have high confidence")
    if ((enrollment.get("benchmark_context") or {}).get("changed_from_baseline")) is not True:
        errors.append("enrollment movement should explicitly flag changed benchmark context")

    sites = fields.get("planned_sites") or {}
    if sites.get("value_origin") != "direct_operational_assumption":
        errors.append("planned-sites movement context should mark site count as a direct operational assumption")
    if (sites.get("baseline") or {}).get("value") != 4 or (sites.get("current") or {}).get("value") != 16:
        errors.append("planned-sites movement context should include baseline and current site values")
    if ((sites.get("movement_from_baseline") or {}).get("relative_to_p50")) != "toward_p50":
        errors.append("planned-sites movement should identify movement toward P50 when current site count is closer than baseline")
    if ((sites.get("movement_from_baseline") or {}).get("magnitude")) != "extreme":
        errors.append("planned-sites movement should classify 4-to-16 site movement as extreme")

    patients_per_site = fields.get("patients_per_site") or {}
    if patients_per_site.get("value_origin") != "calculated_from_enrollment_and_sites":
        errors.append("patients-per-site movement context should mark the value as calculated from enrollment and sites")
    if (patients_per_site.get("current") or {}).get("value") != 6.25:
        errors.append("patients-per-site movement context should use current planned enrollment/current planned sites")
    if ((patients_per_site.get("movement_from_baseline") or {}).get("magnitude")) != "major":
        errors.append("patients-per-site movement should classify 10-to-6.25 movement as major")
    if "counterbalance" not in str(patients_per_site.get("interpretation_rule") or ""):
        errors.append("patients-per-site movement context should state percentile counterbalance rule")

    duration = fields.get("planned_duration_months") or {}
    if duration.get("value_origin") != "direct_operational_assumption":
        errors.append("duration movement context should mark duration as a direct operational assumption")
    if (duration.get("baseline") or {}).get("value") != 24 or (duration.get("current") or {}).get("value") != 30:
        errors.append("duration movement context should include baseline and current duration values")
    if ((duration.get("movement_from_baseline") or {}).get("relative_to_p50")) != "toward_p50":
        errors.append("duration movement should identify movement toward P50 when current duration equals benchmark median")
    if ((duration.get("current") or {}).get("benchmark_position") or {}).get("status") != "typical":
        errors.append("duration movement context should include residual benchmark status")


def _check_storyline_report_compatibility(errors: list[str]) -> None:
    exporter = (ROOT / "scripts" / "export_storyline_review_pack.py").read_text(encoding="utf-8")
    if "Clinical operations:" in exporter or "Strategic/field:" in exporter:
        errors.append("storyline review pack exporter should use the new two-question framing")
    if "Medical / clinical development:" not in exporter or "Strategic development:" not in exporter:
        errors.append("storyline review pack exporter should label the two visible questions")

    temperature_compare = (ROOT / "scripts" / "compare_narrative_temperature_reports.py").read_text(
        encoding="utf-8"
    )
    signature_start = temperature_compare.find("def _narrative_signature")
    subcategory_start = temperature_compare.find("def _subcategory_signature")
    signature_body = temperature_compare[signature_start:subcategory_start]
    if '"operations_question"' in signature_body:
        errors.append("temperature narrative signature should ignore the retired operations question")


def main() -> int:
    errors: list[str] = []
    fixtures = get_contract_fixtures()
    for fixture in fixtures:
        _check_packet(fixture, errors)
    _check_review_continuity_context(errors)
    _check_canonical_values_prefer_compare_values(errors)
    _check_field_and_impact_changes(errors)
    _check_operational_movement_context(errors)
    _check_storyline_report_compatibility(errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Validated deterministic packet assembly for {len(fixtures)} narrative fixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
