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
    design_confidence_relevant_changed_fields,
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
        "design_confidence": 0,
        "total_scenario_score": 68,
        "changed_fields": [],
        "score_delta": 0,
        "central_tension": "Baseline central tension.",
        "validated_review": {
            "review_metadata": {"review_mode": "hidden_baseline", "participant_visible": False},
            "main_tension": "Baseline main tension from dedicated field.",
            "completion_outlook_analysis": {
                "risk_pattern_summary": "Baseline score reflects an acceptable original design profile.",
            },
            "design_confidence_subcategories": {
                "endpoint_evidence_strength": {
                    "rating": "supportive",
                    "rationale": "Baseline endpoint and allocation preserve conventional rigor.",
                    "evidence_fields": ["endpoint_rigor_ml", "allocation_ml"],
                },
            },
            "design_confidence_analysis": {
                "summary": "Baseline design comment.",
                "confidence_rationale": "Baseline central tension.",
                "supporting_evidence": [],
                "limiting_evidence": [],
            },
            "key_questions": {
                "medical_development_question": "What evidence standard matters most?",
                "clinical_operations_question": "What operational burden is proportionate?",
                "strategic_field_question": "What broader field challenge does this scenario expose?",
            },
            "continuity": {
                "new_concerns": [],
                "storyline_update": "Baseline memory",
            },
        },
        "compact_storyline_memory": "Baseline memory",
    }
    previous_trace = {
        **baseline_trace,
        "input_hash": "previous-input-hash",
        "iteration_id": 1,
        "design_confidence": -2,
        "total_scenario_score": 66,
        "changed_fields": ["operational_assumptions.planned_enrollment"],
        "score_delta": 0,
        "central_tension": "",
        "design_confidence_assessment": {
            "subcategories": {
                "endpoint_evidence_strength": {
                    "points": -2,
                    "raw_points": -2,
                },
                "operational_burden_balance": {
                    "points": 0,
                    "raw_points": 0,
                },
            },
        },
        "validated_review": {
            **baseline_trace["validated_review"],
            "main_tension": "Previous main tension from dedicated field.",
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
    if context.get("baseline_review", {}).get("design_confidence") is not None:
        errors.append("hidden baseline review context should not expose design_confidence")
    if context.get("baseline_review", {}).get("total_scenario_score") is not None:
        errors.append("hidden baseline review context should not expose total_scenario_score")
    if context.get("baseline_review", {}).get("design_numeric_context") != "hidden_baseline_qualitative_only":
        errors.append("hidden baseline review context should be marked qualitative-only")
    if (
        context.get("baseline_review", {}).get("baseline_completion_outlook_summary")
        != "Baseline score reflects an acceptable original design profile."
    ):
        errors.append("hidden baseline review context should preserve completion outlook summary")
    baseline_subcategories = context.get("baseline_review", {}).get("baseline_design_subcategory_ratings") or {}
    if baseline_subcategories.get("endpoint_evidence_strength", {}).get("rating") != "supportive":
        errors.append("hidden baseline review context should preserve design subcategory ratings")
    if not context.get("baseline_review", {}).get("baseline_strengths"):
        errors.append("hidden baseline review context should include compact baseline strengths")
    if context.get("previous_review", {}).get("design_confidence") != -2:
        errors.append("previous visible review context should preserve design_confidence")
    if context.get("previous_review", {}).get("total_scenario_score") != 66:
        errors.append("previous visible review context should preserve total_scenario_score")
    if context.get("baseline_review", {}).get("central_tension") != "Baseline central tension.":
        errors.append("hidden baseline review context should preserve trace central_tension when present")
    if context.get("previous_review", {}).get("central_tension") != "Previous main tension from dedicated field.":
        errors.append("previous visible review context should fall back to validated_review.main_tension")
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
    design_continuity = continuity_packet.get("iteration_context", {}).get("design_confidence_continuity") or {}
    if design_continuity.get("available") is not True:
        errors.append("later visible continuity packet should include Design Confidence continuity anchors")
    continuity_subcategories = design_continuity.get("subcategories") or {}
    endpoint_continuity = continuity_subcategories.get("endpoint_evidence_strength") or {}
    if endpoint_continuity.get("previous_rating") != "supportive":
        errors.append("Design Confidence continuity should carry previous subcategory rating")
    if endpoint_continuity.get("previous_points") != -2:
        errors.append("Design Confidence continuity should carry previous app-calculated points")
    if endpoint_continuity.get("current_relevant_changed_fields"):
        errors.append("endpoint continuity should not mark unrelated operational changes as relevant")
    operational_continuity = continuity_subcategories.get("operational_burden_balance") or {}
    if operational_continuity.get("current_relevant_changed_fields") != ["operational_assumptions.planned_enrollment"]:
        errors.append("operational continuity should identify relevant operational-assumption changes")
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
    population_fields = ["is_rare_disease_ml", "line_of_therapy_ml", "patient_severity_ml"]
    if design_confidence_relevant_changed_fields("operational_burden_balance", population_fields) != population_fields:
        errors.append("population changes should be valid operational-burden continuity evidence")
    if design_confidence_relevant_changed_fields("phase_intent_alignment", population_fields) != population_fields:
        errors.append("population changes should be valid phase/intent continuity evidence")
    if design_confidence_relevant_changed_fields(
        "endpoint_evidence_strength",
        ["operational_assumptions.planned_enrollment", "operational_assumptions.planned_sites"],
    ):
        errors.append("planned enrollment/sites should not explain Endpoint Evidence continuity flips")
    if design_confidence_relevant_changed_fields(
        "endpoint_evidence_strength",
        ["operational_assumptions.planned_duration_months"],
    ):
        errors.append("planned total timeline should not explain Endpoint Evidence continuity flips")
    if built.get("review_context", {}).get("previous_review") is not None:
        errors.append("fixture packet without review traces should not invent previous review context")
    fixture_continuity = built.get("iteration_context", {}).get("design_confidence_continuity") or {}
    if fixture_continuity.get("available") is not False:
        errors.append("fixture packet without previous visible trace should mark Design Confidence continuity unavailable")
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
    hidden_continuity = hidden_baseline_packet.get("iteration_context", {}).get("design_confidence_continuity") or {}
    if hidden_continuity.get("available") is not False:
        errors.append("hidden baseline packet should mark Design Confidence continuity unavailable")
    if hidden_continuity.get("subcategories") != {}:
        errors.append("hidden baseline packet should not include visible Design Confidence continuity subcategory anchors")


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
            {"Pillar": "Scientific Challenge", "Subcategory": "Protocol Architecture", "Impact": 0.5},
            {"Pillar": "Execution Framework", "Subcategory": "Methodological Setup", "Impact": 3.0},
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
    _check_storyline_report_compatibility(errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Validated deterministic packet assembly for {len(fixtures)} narrative fixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
