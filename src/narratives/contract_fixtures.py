"""Static contract fixtures for the serious-game narrative V1 layer.

These fixtures are intentionally provider-free. They define the scenarios,
input packet shape, expected structured review shape, and expected behavior
that later packet-builder, validation, scoring, mock-reviewer, and UI work
must preserve.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

PROMPT_VERSION = "narratives_v4"
RUBRIC_VERSION = "design_confidence_v2"

REQUIRED_SCENARIO_TYPES = {
    "baseline",
    "score_improves_evidence_weakens",
    "score_improves_design_neutral",
    "score_improves_design_improves",
    "score_declines_design_improves",
    "operational_only_edit",
    "material_text_only_edit",
    "endpoint_text_contradiction",
    "biomarker_population_mismatch",
    "phase_intent_weak_evidence",
    "modality_governance_mismatch",
    "no_adjustment_large_completion_movement",
    "no_op_minor_text_edit",
}

REQUIRED_DESIGN_SUBCATEGORIES = {
    "phase_intent_alignment",
    "endpoint_evidence_strength",
    "target_population_alignment",
    "operational_burden_balance",
}

SCORE_MATERIALITY_LEVELS = {
    "minimal",
    "low",
    "moderate",
    "high",
    "very_high",
}

# Temporary compatibility alias for old callers during the schema migration.
REQUIRED_REVIEW_DOMAINS = REQUIRED_DESIGN_SUBCATEGORIES

DESIGN_PILLAR_KEYS = {
    "therapeutic_context",
    "scientific_challenge",
    "patient_profile",
    "execution_framework",
}
BASELINE_STRUCTURED_FEATURES: dict[str, Any] = {
    "therapeutic_area_ml": "ONCOLOGY",
    "gbd_cause_id_3_ml": 429,
    "is_rare_disease_ml": 0,
    "phase_ml": "PHASE3",
    "strategic_ambition_ml": "PIVOTAL_INTENT",
    "target_precedent_ml": "PRECEDENT_IN_INDICATION",
    "target_pathway_class_ml": "KINASE_INHIBITOR",
    "therapeutic_modality_ml": "SMALL_MOLECULE",
    "innovation_tier_ml": "NEXT_GEN_OPTIMIZED",
    "intervention_model_ml": "PARALLEL",
    "primary_purpose_ml": "TREATMENT",
    "adaptive_design_ml": "STATIC",
    "endpoint_rigor_ml": "HARD_CLINICAL",
    "endpoint_structure_ml": "SINGLE_GOAL",
    "biomarker_stratification_ml": "1",
    "patient_severity_ml": "ADVANCED_METASTATIC",
    "line_of_therapy_ml": "LATER_LINE",
    "gender_ml": "ALL",
    "healthy_volunteers_ml": "0",
    "adult_ml": "1",
    "child_ml": "0",
    "older_adult_ml": "1",
    "masking_ml": "DOUBLE",
    "allocation_ml": "RANDOMIZED",
    "has_dmc_ml": 1,
    "has_placebo_ml": 0,
    "comparator_benchmark_ml": "ACTIVE_MODERN_STANDARD",
    "administration_complexity_ml": "ROUTINE_INFUSION",
    "number_of_arms_ml": 2,
    "sponsor_tier_ml": "TIER 1",
    "primary_duration_months_ml": 18.0,
}

BASELINE_STRUCTURED_FEATURE_DISPLAY_VALUES: dict[str, Any] = {
    "therapeutic_area_ml": "Oncology",
    "gbd_cause_id_3_ml": "Breast cancer",
    "is_rare_disease_ml": "Unlikely",
    "phase_ml": "Phase 3",
    "strategic_ambition_ml": "Confirmatory / Registration",
    "target_precedent_ml": "Established in Indication",
    "target_pathway_class_ml": "Kinase Inhibitor",
    "therapeutic_modality_ml": "Small Molecule",
    "innovation_tier_ml": "Next-Gen / Optimized",
    "intervention_model_ml": "Parallel",
    "primary_purpose_ml": "Treatment",
    "adaptive_design_ml": "Static Design",
    "endpoint_rigor_ml": "Hard Clinical (Survival/Death)",
    "endpoint_structure_ml": "Single Goal",
    "biomarker_stratification_ml": "Yes",
    "patient_severity_ml": "Advanced / Metastatic",
    "line_of_therapy_ml": "Later-Line (2nd+)",
    "gender_ml": "All (Male & Female)",
    "healthy_volunteers_ml": "Patients Only",
    "adult_ml": "Included",
    "child_ml": "Excluded",
    "older_adult_ml": "Included",
    "masking_ml": "Double Blind",
    "allocation_ml": "Randomized",
    "has_dmc_ml": "Yes",
    "has_placebo_ml": "No",
    "comparator_benchmark_ml": "Active (Modern Standard)",
    "administration_complexity_ml": "Routine (Injection/IV)",
    "number_of_arms_ml": "2",
    "sponsor_tier_ml": "Top-Tier Pharma",
    "primary_duration_months_ml": "18.0",
}

BASELINE_OPERATIONAL_ASSUMPTIONS: dict[str, Any] = {
    "planned_enrollment": {
        "value": 620,
        "source": "planned_value",
        "benchmark_level_used": "phase_indication_rare",
        "benchmark_n": 118,
        "benchmark_p25": 340,
        "benchmark_p50": 560,
        "benchmark_p75": 820,
        "benchmark_p90": 1120,
        "enrollment_status": "typical",
        "support_level": "supported_by_current_design",
        "supporting_signals": ["common_disease", "phase_3_confirmatory"],
        "conflicting_signals": [],
        "interpretation_hint": "Enrollment is typical versus similar trials.",
    },
    "planned_sites": {
        "value": 75,
        "source": "current_registry_facility_count_proxy",
        "benchmark_level_used": "phase_ta_rare",
        "benchmark_n": 220,
        "site_count_p50": 70,
        "patients_per_site_p50": 8.0,
        "site_count_status": "typical",
        "interpretation_hint": "Site count is typical versus similar trials.",
    },
    "planned_duration_months": {
        "value": 42.0,
        "source": "estimated_planned_total_duration",
        "duration_definition": "start_date_to_completion_date_months",
        "benchmark_level_used": "phase_ta_rare_endpoint_bin",
        "benchmark_n": 159,
        "benchmark_p25": 28.0,
        "benchmark_p50": 40.0,
        "benchmark_p75": 58.0,
        "benchmark_p90": 82.0,
        "duration_status": "typical",
        "planned_primary_completion_months": 18.0,
        "primary_completion_source": "estimated_primary_completion",
        "primary_completion_n": 121,
        "interpretation_hint": "Duration is typical versus similar trials.",
    },
}

BASELINE_PILLAR_IMPACTS = {
    "Therapeutic Context": 4.2,
    "Scientific Challenge": -1.6,
    "Patient Profile": 2.4,
    "Execution Framework": 1.0,
}


def _base_packet() -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "mode": "existing_study",
        "trial_identity": {
            "nct_id": "NCT-NARRATIVE-FIXTURE",
            "trial_label": "Fixture oncology confirmatory trial",
            "lead_sponsor_canonical": "Fixture Pharma",
            "start_year": "2024",
        },
        "text_context": {
            "title": "A randomized Phase 3 study of targeted therapy in advanced breast cancer",
            "summary_ui": (
                "Confirmatory study evaluating targeted therapy plus standard care "
                "in adults with advanced biomarker-positive breast cancer."
            ),
            "conditions_ui": "Advanced biomarker-positive breast cancer.",
            "primary_outcomes_ui": "Progression-free survival by blinded independent review.",
            "interventions_ui": "Targeted therapy plus standard care versus standard care.",
        },
        "structured_features": deepcopy(BASELINE_STRUCTURED_FEATURES),
        "structured_feature_display_values": deepcopy(BASELINE_STRUCTURED_FEATURE_DISPLAY_VALUES),
        "operational_assumptions": deepcopy(BASELINE_OPERATIONAL_ASSUMPTIONS),
        "model_interpretation": {
            "completion_score": 68,
            "previous_completion_score": None,
            "score_delta": 0,
            "direct_xgboost_shap_fields": [],
            "pillar_impacts": deepcopy(BASELINE_PILLAR_IMPACTS),
            "pillar_deltas": {},
            "top_positive_feature_drivers": ["phase_ml", "allocation_ml"],
            "top_negative_feature_drivers": ["primary_duration_months_ml"],
            "top_feature_impact_changes": [],
        },
        "iteration_context": {
            "baseline_snapshot_id": "fixture-baseline",
            "previous_snapshot_id": None,
            "current_snapshot_id": "fixture-baseline",
            "iteration_number": 0,
            "changed_fields": [],
            "compact_storyline_memory": "",
        },
    }


def _default_score_materiality(rating: str) -> str:
    return {
        "strong": "minimal",
        "supportive": "minimal",
        "balanced": "minimal",
        "weak": "moderate",
        "conflicting": "minimal",
    }.get(rating, "minimal")


def _domain(
    rating: str,
    rationale: str,
    evidence_fields: list[str],
    score_materiality: str | None = None,
    movement_direction: str | None = None,
    movement_materiality: str | None = None,
    effect_role: str | None = None,
) -> dict[str, Any]:
    materiality = score_materiality or _default_score_materiality(rating)
    if movement_direction is None:
        movement_direction = {
            "strong": "improved",
            "supportive": "improved",
            "balanced": "unchanged",
            "weak": "weakened",
            "conflicting": "worsened",
        }.get(rating, "unchanged")
    if movement_materiality is None:
        movement_materiality = {
            "minimal": "minor",
            "low": "minor",
            "moderate": "moderate",
            "high": "major",
            "very_high": "major",
        }.get(materiality, "none")
        if movement_direction == "unchanged":
            movement_materiality = "none"
    if effect_role is None:
        effect_role = "unchanged" if movement_direction == "unchanged" else "independent"
    return {
        "current_state": rating,
        "movement_direction": movement_direction,
        "movement_materiality": movement_materiality,
        "effect_role": effect_role,
        "rating": rating,
        "score_materiality": materiality,
        "rationale": rationale,
        "evidence_fields": evidence_fields,
        "short_rationale": rationale.split(".", 1)[0][:80],
        "optional_lenses_used": [],
        "regulatory_or_finance_note": "",
    }


def _visible_review(
    what_changed: str,
    moved: str,
    signal: str,
    tradeoff: str,
    medical_question: str,
    clinops_question: str,
    strategic_question: str = "What broader development tension does this scenario expose for similar trials in this field?",
) -> dict[str, str]:
    return {
        "completion_outlook_summary": " ".join(part for part in [what_changed, moved] if part).strip(),
        "design_confidence_summary": " ".join(part for part in [signal, tradeoff] if part).strip(),
        "medical_clinical_development_question": medical_question,
        "strategic_development_question": strategic_question or clinops_question,
        "medical_development_question": medical_question,
        "clinical_operations_question": clinops_question,
        "strategic_field_question": strategic_question or clinops_question,
    }


def _review(
    *,
    movement_summary: str,
    design_subcategories: dict[str, dict[str, Any]],
    visible_review: dict[str, str],
    storyline_update: str,
    review_mode: str = "first_visible_iteration",
    new_concerns: list[str] | None = None,
    operational_statuses: list[str] | None = None,
    design_gain: str = "",
    design_sacrifice: str = "",
) -> dict[str, Any]:
    consistency_note = {
        "has_clear_mismatch": False,
        "message": "",
        "fields_in_tension": [],
    }
    if any(
        token in " ".join(
            str(field)
            for subcategory in design_subcategories.values()
            for field in subcategory.get("evidence_fields", [])
        )
        for token in ("primary_outcomes_ui", "conditions_ui", "interventions_ui")
    ):
        consistency_note = {
            "has_clear_mismatch": True,
            "message": (
                "Some scenario details are not fully aligned across Trial description fields and structured fields. "
                "In this case the value in the structured fields drives the analysis, while the Trial description fields are used as supporting context."
            ),
            "fields_in_tension": ["structured fields", "Trial description fields"],
        }
    main_tension = "The main tension is whether completion favorability and design defensibility move in the same direction."
    return {
        "review_metadata": {
            "review_mode": review_mode,
            "visible": review_mode != "hidden_baseline",
        },
        "completion_outlook_analysis": {
            "risk_pattern_summary": movement_summary,
            "driver_summary": movement_summary,
            "main_model_signals": [],
            "interpretive_hypotheses": [
                {
                    "signal": "Completion Score movement",
                    "possible_pattern": movement_summary,
                    "context_modifiers": [],
                    "boundary": "This is a historical risk-pattern interpretation, not proof that a field caused completion.",
                }
            ],
            "movement_explanation": movement_summary,
            "model_boundary_note": "Completion Outlook reflects resemblance to completed versus early-terminated historical patterns.",
        },
        "design_confidence_subcategories": design_subcategories,
        "design_confidence_analysis": {
            "summary": visible_review.get("design_confidence_summary", ""),
            "confidence_rationale": main_tension,
            "supporting_evidence": [design_gain] if design_gain else [],
            "limiting_evidence": [design_sacrifice] if design_sacrifice else [],
        },
        "main_tension": main_tension,
        "key_questions": {
            "medical_clinical_development_question": visible_review.get(
                "medical_clinical_development_question",
                visible_review.get("medical_development_question", ""),
            ),
            "strategic_development_question": visible_review.get(
                "strategic_development_question",
                visible_review.get("strategic_field_question", ""),
            ),
        },
        "scenario_consistency_note": consistency_note,
        "continuity": {
            "prior_concerns_resolved": [],
            "prior_concerns_worsened": [],
            "prior_concerns_unchanged": [],
            "new_concerns": new_concerns or [],
            "storyline_update": storyline_update,
        },
        "trace": {
            "main_features_considered": sorted(
                {
                    field
                    for subcategory in design_subcategories.values()
                    for field in subcategory.get("evidence_fields", [])
                    if "." not in field
                }
            ),
            "main_completion_drivers_considered": [],
            "main_design_subcategories_considered": sorted(design_subcategories),
            "operational_statuses_considered": operational_statuses or [],
            "reference_pack_ids_used": [],
            "therapeutic_area_pack_used": "",
            "compared_against": "previous_prediction",
            "should_repeat_prior_warning": False,
        },
    }


def _expected(
    *,
    design_confidence: float,
    total_scenario_score: float,
    subcategories: dict[str, float],
    review_needed: bool = True,
    visible_initially: bool = True,
    storyline_behavior: str,
    reuse_previous_review: bool = False,
) -> dict[str, Any]:
    return {
        "review_needed": review_needed,
        "visible_initially": visible_initially,
        "reuse_previous_review": reuse_previous_review,
        "expected_design_confidence": design_confidence,
        "expected_total_scenario_score": total_scenario_score,
        "expected_design_subcategories": subcategories,
        "score_rule": "completion_score + design_confidence, clamped to 0..100",
        "storyline_behavior": storyline_behavior,
    }


def _packet(
    *,
    completion_score: float = 68,
    previous_completion_score: float | None = 68,
    score_delta: float = 0,
    changed_fields: list[str] | None = None,
    structured_updates: dict[str, Any] | None = None,
    display_updates: dict[str, Any] | None = None,
    text_updates: dict[str, str] | None = None,
    operational_updates: dict[str, Any] | None = None,
    pillar_deltas: dict[str, float] | None = None,
    top_feature_impact_changes: list[str] | None = None,
) -> dict[str, Any]:
    base = _base_packet()
    base["structured_features"].update(structured_updates or {})
    base["structured_feature_display_values"].update(display_updates or {})
    base["text_context"].update(text_updates or {})
    for key, value in (operational_updates or {}).items():
        base["operational_assumptions"][key] = value
    base["model_interpretation"].update(
        {
            "completion_score": completion_score,
            "previous_completion_score": previous_completion_score,
            "score_delta": score_delta,
            "pillar_deltas": pillar_deltas or {},
            "top_feature_impact_changes": top_feature_impact_changes or [],
        }
    )
    base["iteration_context"].update(
        {
            "previous_snapshot_id": "fixture-baseline" if previous_completion_score is not None else None,
            "current_snapshot_id": "fixture-current",
            "iteration_number": 1 if previous_completion_score is not None else 0,
            "changed_fields": changed_fields or [],
            "compact_storyline_memory": "Baseline was balanced; no prior visible concern.",
        }
    )
    return base


def _neutral_design() -> dict[str, dict[str, Any]]:
    return {
        "phase_intent_alignment": _domain("balanced", "Phase and intent are coherent for the scenario.", ["phase_ml", "strategic_ambition_ml"]),
        "endpoint_evidence_strength": _domain("balanced", "Endpoint and comparator evidence are not materially changed.", ["endpoint_rigor_ml", "comparator_benchmark_ml"]),
        "target_population_alignment": _domain("balanced", "Population scope remains aligned with the indication.", ["adult_ml", "older_adult_ml", "line_of_therapy_ml"]),
        "operational_burden_balance": _domain("balanced", "Operational assumptions are proportionate to the design.", ["operational_assumptions.planned_enrollment.enrollment_status"]),
    }


def _fixture(
    *,
    fixture_id: str,
    scenario_type: str,
    description: str,
    packet: dict[str, Any],
    design_subcategories: dict[str, dict[str, Any]] | None,
    visible_review: dict[str, str] | None,
    expected: dict[str, Any],
    movement_summary: str,
    storyline_update: str = "",
    new_concerns: list[str] | None = None,
    operational_statuses: list[str] | None = None,
    review_mode: str = "first_visible_iteration",
) -> dict[str, Any]:
    review = None
    if design_subcategories is not None and visible_review is not None:
        review = _review(
            movement_summary=movement_summary,
            design_subcategories=design_subcategories,
            visible_review=visible_review,
            storyline_update=storyline_update,
            new_concerns=new_concerns,
            operational_statuses=operational_statuses,
            review_mode=review_mode,
        )
    return {
        "fixture_id": fixture_id,
        "scenario_type": scenario_type,
        "description": description,
        "input_packet": packet,
        "mock_review": review,
        "expected_behavior": expected,
    }


CONTRACT_FIXTURES: list[dict[str, Any]] = [
    _fixture(
        fixture_id="baseline_hidden_review_v2",
        scenario_type="baseline",
        description="Hidden existing-study baseline review generated once from the original selected trial.",
        packet=_packet(previous_completion_score=None, changed_fields=[]),
        design_subcategories=_neutral_design(),
        visible_review=_visible_review(
            "This is the original selected-trial baseline.",
            "No model-score movement is reviewed at baseline.",
            "The baseline is balanced and creates qualitative memory only.",
            "No visible trade-off has been introduced yet.",
            "Which baseline design concern is most likely to matter as the scenario evolves?",
            "Which operational assumption would be most fragile if design ambition increases?",
        ),
        expected=_expected(
            design_confidence=0,
            total_scenario_score=68,
            subcategories={
                "phase_intent_alignment": 0,
                "endpoint_evidence_strength": 0,
                "target_population_alignment": 0,
                "operational_burden_balance": 0,
            },
            visible_initially=False,
            storyline_behavior="create_hidden_baseline_memory_only",
        ),
        movement_summary="Baseline review anchors later comparisons; no visible scenario edit has occurred.",
        storyline_update="Baseline anchored with balanced Design Confidence and no visible edit concern.",
        operational_statuses=["typical"],
        review_mode="hidden_baseline",
    ),
    _fixture(
        fixture_id="score_improves_evidence_weakens_v2",
        scenario_type="score_improves_evidence_weakens",
        description="Completion improves while endpoint rigor, comparator strength, and duration weaken.",
        packet=_packet(
            completion_score=74,
            score_delta=6,
            changed_fields=["endpoint_rigor_ml", "has_placebo_ml", "comparator_benchmark_ml", "primary_duration_months_ml"],
            structured_updates={
                "endpoint_rigor_ml": "SURROGATE",
                "has_placebo_ml": 1,
                "comparator_benchmark_ml": "PLACEBO",
                "primary_duration_months_ml": 9.0,
            },
            display_updates={
                "endpoint_rigor_ml": "Surrogate / Biomarker",
                "has_placebo_ml": "Yes",
                "comparator_benchmark_ml": "Placebo Control",
                "primary_duration_months_ml": "9.0",
            },
            pillar_deltas={"Scientific Challenge": 3.2, "Execution Framework": 1.4},
            top_feature_impact_changes=["endpoint_rigor_ml", "comparator_benchmark_ml", "primary_duration_months_ml"],
        ),
        design_subcategories={
            **_neutral_design(),
            "phase_intent_alignment": _domain("weak", "Confirmatory intent is less well matched to weakened evidence choices.", ["strategic_ambition_ml", "endpoint_rigor_ml"]),
            "endpoint_evidence_strength": _domain("conflicting", "Endpoint rigor, comparator strength, and endpoint timing weaken together.", ["endpoint_rigor_ml", "comparator_benchmark_ml", "primary_duration_months_ml"], "moderate"),
        },
        visible_review=_visible_review(
            "Endpoint rigor, comparator framing, and primary endpoint duration changed.",
            "The score could have moved upward because the design may be simpler and shorter.",
            "Design Confidence is challenged by weaker endpoint and comparator evidence.",
            "Completion improved, but evidence interpretability may have been sacrificed.",
            "Does the easier design still answer the same confirmatory question?",
            "Would the shorter, simpler design still justify the same operational commitment?",
        ),
        expected=_expected(
            design_confidence=-2,
            total_scenario_score=72,
            subcategories={
                "phase_intent_alignment": -1,
                "endpoint_evidence_strength": -1,
                "target_population_alignment": 0,
                "operational_burden_balance": 0,
            },
            storyline_behavior="append_new_iteration_with_shortcut_concern",
        ),
        movement_summary="The Completion Score may have improved because the revised design appears easier and shorter.",
        storyline_update="Raised shortcut concern: completion improved while evidence strength weakened.",
        new_concerns=["endpoint_comparator_shortcut"],
        operational_statuses=["typical"],
    ),
    _fixture(
        fixture_id="score_improves_design_neutral_v2",
        scenario_type="score_improves_design_neutral",
        description="Completion improves through an operationally easier pattern, but no supported Design Confidence change exists.",
        packet=_packet(
            completion_score=73,
            score_delta=5,
            changed_fields=["number_of_arms_ml"],
            structured_updates={"number_of_arms_ml": 1},
            display_updates={"number_of_arms_ml": "1"},
            pillar_deltas={"Execution Framework": 4.0},
            top_feature_impact_changes=["number_of_arms_ml"],
        ),
        design_subcategories=_neutral_design(),
        visible_review=_visible_review(
            "The number of arms changed.",
            "The score may have improved because the execution footprint is simpler.",
            "No supported Design Confidence adjustment is triggered by simplicity alone.",
            "A simpler footprint is not automatically better or worse without evidence of design impact.",
            "What decision would the simplified arm structure still support?",
            "Does the simpler footprint preserve the minimum operational information needed for the intended decision?",
        ),
        expected=_expected(
            design_confidence=0,
            total_scenario_score=73,
            subcategories={
                "phase_intent_alignment": 0,
                "endpoint_evidence_strength": 0,
                "target_population_alignment": 0,
                "operational_burden_balance": 0,
            },
            storyline_behavior="append_iteration_with_neutral_design_adjustment",
        ),
        movement_summary="Completion Outlook improves, but no supported design-strengthening or design-weakening evidence is present.",
    ),
    _fixture(
        fixture_id="score_improves_design_improves_v2",
        scenario_type="score_improves_design_improves",
        description="Completion improves while comparator, masking, and oversight become more coherent.",
        packet=_packet(
            completion_score=72,
            score_delta=4,
            changed_fields=["comparator_benchmark_ml", "masking_ml", "has_dmc_ml"],
            structured_updates={
                "comparator_benchmark_ml": "ACTIVE_MODERN_STANDARD",
                "masking_ml": "DOUBLE",
                "has_dmc_ml": 1,
            },
            pillar_deltas={"Scientific Challenge": 1.5, "Execution Framework": 2.0},
            top_feature_impact_changes=["comparator_benchmark_ml", "masking_ml", "has_dmc_ml"],
        ),
        design_subcategories={
            **_neutral_design(),
            "endpoint_evidence_strength": _domain("supportive", "Comparator and masking support interpretability.", ["comparator_benchmark_ml", "masking_ml"]),
            "operational_burden_balance": _domain("supportive", "Oversight is proportionate to the confirmatory setting.", ["has_dmc_ml", "patient_severity_ml"]),
        },
        visible_review=_visible_review(
            "Comparator, masking, and oversight changed.",
            "The score may have improved through a more coherent execution and evidence pattern.",
            "Design Confidence improves modestly because evidence controls and governance align.",
            "The gain is not large because the baseline was already relatively coherent.",
            "Which evidence risk is most reduced by the revised comparator and masking?",
            "Is the oversight level proportionate without adding avoidable burden?",
        ),
        expected=_expected(
            design_confidence=1.0,
            total_scenario_score=73,
            subcategories={
                "phase_intent_alignment": 0,
                "endpoint_evidence_strength": 0.5,
                "target_population_alignment": 0,
                "operational_burden_balance": 0.5,
            },
            storyline_behavior="append_iteration_with_modest_supported_design_gain",
        ),
        movement_summary="Completion Outlook improves while evidence controls and governance also strengthen.",
    ),
    _fixture(
        fixture_id="score_declines_design_improves_v2",
        scenario_type="score_declines_design_improves",
        description="Completion declines because the trial becomes harder, but Design Confidence improves through rigor and patient relevance.",
        packet=_packet(
            completion_score=62,
            score_delta=-6,
            changed_fields=["biomarker_stratification_ml", "older_adult_ml", "has_dmc_ml", "primary_duration_months_ml"],
            structured_updates={
                "biomarker_stratification_ml": "1",
                "older_adult_ml": "1",
                "has_dmc_ml": 1,
                "primary_duration_months_ml": 24.0,
            },
            pillar_deltas={"Patient Profile": -2.2, "Execution Framework": -2.6, "Scientific Challenge": -1.2},
            top_feature_impact_changes=["biomarker_stratification_ml", "older_adult_ml", "primary_duration_months_ml"],
        ),
        design_subcategories={
            **_neutral_design(),
            "endpoint_evidence_strength": _domain("supportive", "Longer endpoint timing supports the stated clinical outcome.", ["primary_duration_months_ml", "primary_outcomes_ui"], "moderate"),
            "target_population_alignment": _domain("supportive", "Older adults and biomarker strategy improve relevance to the intended population.", ["older_adult_ml", "biomarker_stratification_ml"], "moderate"),
            "operational_burden_balance": _domain("supportive", "Oversight is proportionate to added population and duration complexity.", ["has_dmc_ml", "patient_severity_ml"], "moderate"),
        },
        visible_review=_visible_review(
            "Population, biomarker, oversight, and duration choices became more demanding.",
            "The score may have declined because the design is harder to execute.",
            "Design Confidence improves because the added difficulty is tied to relevance, endpoint maturity, and governance.",
            "The scenario trades completion ease for a more defensible clinical-development question.",
            "Does the harder design produce evidence that is meaningfully more decision-useful?",
            "Which execution controls would be needed to make the added burden credible?",
        ),
        expected=_expected(
            design_confidence=3,
            total_scenario_score=65,
            subcategories={
                "phase_intent_alignment": 0,
                "endpoint_evidence_strength": 1,
                "target_population_alignment": 1,
                "operational_burden_balance": 1,
            },
            storyline_behavior="append_iteration_where_design_confidence_moderates_score_decline",
        ),
        movement_summary="Completion Outlook declines, but the added risk is plausibly linked to rigor and patient relevance.",
    ),
    _fixture(
        fixture_id="operational_only_ambitious_enrollment_v2",
        scenario_type="operational_only_edit",
        description="Model fields stay fixed, but enrollment and sites are above benchmark for the current design.",
        packet=_packet(
            changed_fields=["operational_assumptions.planned_enrollment", "operational_assumptions.planned_sites"],
            operational_updates={
                "planned_enrollment": {
                    **BASELINE_OPERATIONAL_ASSUMPTIONS["planned_enrollment"],
                    "value": 1400,
                    "source": "user_scenario",
                    "enrollment_status": "above_benchmark_high",
                    "support_level": "partly_supported_by_current_design",
                    "conflicting_signals": ["large_sample_for_biomarker_subset"],
                },
                "planned_sites": {
                    **BASELINE_OPERATIONAL_ASSUMPTIONS["planned_sites"],
                    "value": 90,
                    "source": "user_scenario",
                    "site_count_status": "ambitious",
                },
            },
        ),
        design_subcategories={
            **_neutral_design(),
            "operational_burden_balance": _domain("weak", "Enrollment is above benchmark high and only partly supported by the biomarker-defined design.", ["operational_assumptions.planned_enrollment.enrollment_status", "operational_assumptions.planned_enrollment.support_level"], "high"),
        },
        visible_review=_visible_review(
            "Only enrollment and site assumptions changed.",
            "The Completion Outlook did not move because operational assumptions are outside the score-input fields.",
            "Design Confidence decreases because the operational footprint is only partly supported.",
            "The scenario stress-tests feasibility without changing Completion Outlook score-input fields.",
            "What clinical rationale makes this enrollment target necessary?",
            "What site activation or recruitment evidence would make this footprint credible?",
        ),
        expected=_expected(
            design_confidence=-2.0,
            total_scenario_score=66,
            subcategories={
                "phase_intent_alignment": 0,
                "endpoint_evidence_strength": 0,
                "target_population_alignment": 0,
                "operational_burden_balance": -2.0,
            },
            storyline_behavior="append_iteration_without_model_score_delta",
        ),
        movement_summary="The Completion Score did not move because Completion Outlook score-input fields did not change.",
        new_concerns=["ambitious_enrollment_support"],
        operational_statuses=["above_benchmark_high", "ambitious"],
    ),
    _fixture(
        fixture_id="material_text_only_endpoint_conflict_v2",
        scenario_type="material_text_only_edit",
        description="Endpoint text changes materially while structured model fields and Completion Score stay fixed.",
        packet=_packet(
            changed_fields=["text_context.primary_outcomes_ui"],
            text_updates={
                "primary_outcomes_ui": "Short-term symptom response at 4 weeks.",
                "summary_ui": "Confirmatory registration study intended to establish durable disease control.",
            },
        ),
        design_subcategories={
            **_neutral_design(),
            "phase_intent_alignment": _domain("weak", "Confirmatory durable-control intent conflicts with short-term endpoint text.", ["summary_ui", "primary_outcomes_ui", "strategic_ambition_ml"]),
            "endpoint_evidence_strength": _domain("weak", "Endpoint text and structured endpoint duration are misaligned.", ["primary_outcomes_ui", "primary_duration_months_ml"]),
        },
        visible_review=_visible_review(
            "The endpoint text changed while structured Trial Features stayed the same.",
            "The Completion Score did not move because Completion Outlook score-input fields did not change.",
            "Design Confidence decreases because text now weakens endpoint and intent coherence.",
            "The trade-off is narrative clarity versus contradiction with the structured scenario.",
            "Is the endpoint text intended to replace or only clarify the endpoint strategy?",
            "Should the operational duration still be interpreted against the original endpoint maturity?",
        ),
        expected=_expected(
            design_confidence=-2,
            total_scenario_score=66,
            subcategories={
                "phase_intent_alignment": -1,
                "endpoint_evidence_strength": -1,
                "target_population_alignment": 0,
                "operational_burden_balance": 0,
            },
            storyline_behavior="append_text_only_iteration_without_model_score_delta",
        ),
        movement_summary="The Completion Score did not move because structured model fields did not change.",
        new_concerns=["endpoint_text_material_tension"],
        operational_statuses=["typical"],
    ),
    _fixture(
        fixture_id="endpoint_text_contradiction_v2",
        scenario_type="endpoint_text_contradiction",
        description="Structured endpoint says multi/composite, but text says single endpoint only.",
        packet=_packet(
            completion_score=70,
            score_delta=2,
            changed_fields=["endpoint_structure_ml", "text_context.primary_outcomes_ui"],
            structured_updates={"endpoint_structure_ml": "MULTI_COMPOSITE"},
            display_updates={"endpoint_structure_ml": "Multi/Composite"},
            text_updates={"primary_outcomes_ui": "The study has a single primary endpoint: progression-free survival."},
            top_feature_impact_changes=["endpoint_structure_ml"],
        ),
        design_subcategories={
            **_neutral_design(),
            "endpoint_evidence_strength": _domain("weak", "Structured endpoint complexity and endpoint text disagree.", ["endpoint_structure_ml", "primary_outcomes_ui"]),
        },
        visible_review=_visible_review(
            "Endpoint structure and endpoint text changed together.",
            "The score may have moved because the structured endpoint setting changed.",
            "Design Confidence decreases because the evidence hierarchy is unclear.",
            "The design may be technically richer but harder to interpret from the submitted text.",
            "Which endpoint hierarchy should reviewers believe?",
            "Would operational planning change if the endpoint structure is composite rather than single?",
        ),
        expected=_expected(
            design_confidence=-1,
            total_scenario_score=69,
            subcategories={
                "phase_intent_alignment": 0,
                "endpoint_evidence_strength": -1,
                "target_population_alignment": 0,
                "operational_burden_balance": 0,
            },
            storyline_behavior="append_structured_text_context_iteration",
        ),
        movement_summary="The Completion Score may have changed because endpoint structure changed.",
    ),
    _fixture(
        fixture_id="biomarker_population_mismatch_v2",
        scenario_type="biomarker_population_mismatch",
        description="Biomarker restriction is added while conditions text remains broad.",
        packet=_packet(
            completion_score=65,
            score_delta=-3,
            changed_fields=["biomarker_stratification_ml", "text_context.conditions_ui"],
            structured_updates={"biomarker_stratification_ml": "1"},
            text_updates={"conditions_ui": "All-comer advanced breast cancer without biomarker restriction."},
            top_feature_impact_changes=["biomarker_stratification_ml"],
        ),
        design_subcategories={
            **_neutral_design(),
            "target_population_alignment": _domain("conflicting", "Structured biomarker restriction conflicts with all-comer conditions text.", ["biomarker_stratification_ml", "conditions_ui"]),
        },
        visible_review=_visible_review(
            "Biomarker strategy and indication text no longer match.",
            "The score may have declined because the population is more selective.",
            "Design Confidence decreases because the intended population is ambiguous.",
            "A targeted design can be defensible, but the scenario must state the same population consistently.",
            "Is the intended population biomarker-positive or all-comer?",
            "How would recruitment assumptions change if the biomarker restriction is real?",
        ),
        expected=_expected(
            design_confidence=-0.5,
            total_scenario_score=64.5,
            subcategories={
                "phase_intent_alignment": 0,
                "endpoint_evidence_strength": 0,
                "target_population_alignment": -0.5,
                "operational_burden_balance": 0,
            },
            storyline_behavior="append_population_text_mismatch_concern",
        ),
        movement_summary="Completion Outlook declines as population selectivity increases.",
        new_concerns=["biomarker_population_mismatch"],
    ),
    _fixture(
        fixture_id="phase_intent_weak_evidence_v2",
        scenario_type="phase_intent_weak_evidence",
        description="Registration intent is paired with exploratory endpoint and weak comparator choices.",
        packet=_packet(
            completion_score=71,
            score_delta=3,
            changed_fields=["strategic_ambition_ml", "endpoint_rigor_ml", "comparator_benchmark_ml"],
            structured_updates={
                "strategic_ambition_ml": "PIVOTAL_INTENT",
                "endpoint_rigor_ml": "UNKNOWN",
                "comparator_benchmark_ml": "NO_CONTROL_GROUP",
            },
            top_feature_impact_changes=["strategic_ambition_ml", "endpoint_rigor_ml", "comparator_benchmark_ml"],
        ),
        design_subcategories={
            **_neutral_design(),
            "phase_intent_alignment": _domain("conflicting", "Pivotal ambition is not supported by exploratory endpoint and weak comparator choices.", ["strategic_ambition_ml", "endpoint_rigor_ml", "comparator_benchmark_ml"]),
            "endpoint_evidence_strength": _domain("conflicting", "Endpoint and comparator choices weaken decision strength.", ["endpoint_rigor_ml", "comparator_benchmark_ml"]),
        },
        visible_review=_visible_review(
            "Development intent, endpoint rigor, and comparator choices changed.",
            "The score may have improved despite a weaker evidence posture.",
            "Design Confidence decreases because ambition and evidence support diverge.",
            "The scenario may be easier to complete but less defensible for the stated decision.",
            "What decision can this evidence package credibly support?",
            "Would the same operational plan remain proportionate for an exploratory rather than pivotal question?",
        ),
        expected=_expected(
            design_confidence=-1,
            total_scenario_score=70,
            subcategories={
                "phase_intent_alignment": -0.5,
                "endpoint_evidence_strength": -0.5,
                "target_population_alignment": 0,
                "operational_burden_balance": 0,
            },
            storyline_behavior="append_phase_intent_evidence_mismatch",
        ),
        movement_summary="Completion Outlook improves, but pivotal intent is less supported by the evidence design.",
    ),
    _fixture(
        fixture_id="modality_governance_mismatch_v2",
        scenario_type="modality_governance_mismatch",
        description="Complex modality is introduced without proportional oversight.",
        packet=_packet(
            completion_score=64,
            score_delta=-4,
            changed_fields=["therapeutic_modality_ml", "administration_complexity_ml", "has_dmc_ml"],
            structured_updates={
                "therapeutic_modality_ml": "CELL_GENE_THERAPY",
                "administration_complexity_ml": "INTENSIVE_MANAGEMENT",
                "has_dmc_ml": 0,
            },
            top_feature_impact_changes=["therapeutic_modality_ml", "administration_complexity_ml", "has_dmc_ml"],
        ),
        design_subcategories={
            **_neutral_design(),
            "phase_intent_alignment": _domain("weak", "Complex modality increases the need for explicit development rationale.", ["therapeutic_modality_ml", "phase_ml"]),
            "operational_burden_balance": _domain("conflicting", "Complex administration without DMC creates a governance mismatch.", ["therapeutic_modality_ml", "administration_complexity_ml", "has_dmc_ml"]),
        },
        visible_review=_visible_review(
            "Modality, administration complexity, and DMC status changed.",
            "The score may have declined because modality and execution complexity increased.",
            "Design Confidence decreases because governance is not proportionate to the complexity.",
            "The trial may be scientifically ambitious but under-governed operationally.",
            "What safety or modality-specific uncertainty should this design explicitly manage?",
            "What governance structure would make the operational burden proportionate?",
        ),
        expected=_expected(
            design_confidence=-1.5,
            total_scenario_score=62.5,
            subcategories={
                "phase_intent_alignment": -1,
                "endpoint_evidence_strength": 0,
                "target_population_alignment": 0,
                "operational_burden_balance": -0.5,
            },
            storyline_behavior="append_modality_governance_mismatch",
        ),
        movement_summary="Completion Outlook declines as modality and execution complexity increase.",
    ),
    _fixture(
        fixture_id="no_adjustment_large_completion_movement_v2",
        scenario_type="no_adjustment_large_completion_movement",
        description="Completion moves materially, but packet evidence does not justify any Design Confidence points.",
        packet=_packet(
            completion_score=80,
            score_delta=12,
            changed_fields=["sponsor_tier_ml"],
            structured_updates={"sponsor_tier_ml": "TIER 1"},
            top_feature_impact_changes=["sponsor_tier_ml"],
        ),
        design_subcategories=_neutral_design(),
        visible_review=_visible_review(
            "Organization tier changed and Completion Outlook moved materially.",
            "The score pattern may associate the revised organization tier with stronger completion patterns.",
            "Design Confidence remains neutral because organization tier alone is not a design-strengthening action.",
            "A large Completion Outlook movement does not automatically justify a design adjustment.",
            "What design feature, separate from organization capability, actually changes the evidence value?",
            "What execution assumption changed, if any, beyond organizational capability?",
        ),
        expected=_expected(
            design_confidence=0,
            total_scenario_score=80,
            subcategories={
                "phase_intent_alignment": 0,
                "endpoint_evidence_strength": 0,
                "target_population_alignment": 0,
                "operational_burden_balance": 0,
            },
            storyline_behavior="append_large_completion_movement_without_design_adjustment",
        ),
        movement_summary="Completion Outlook moved materially, but no supported design adjustment is present.",
    ),
    _fixture(
        fixture_id="noop_minor_text_cleanup_v2",
        scenario_type="no_op_minor_text_edit",
        description="Participant makes only casing/punctuation cleanup; review should be reused.",
        packet=_packet(
            changed_fields=["text_context.summary_ui"],
            text_updates={
                "summary_ui": (
                    "Confirmatory study evaluating targeted therapy plus standard care "
                    "in adults with advanced biomarker-positive breast cancer"
                )
            },
        ),
        design_subcategories=None,
        visible_review=None,
        expected=_expected(
            design_confidence=0,
            total_scenario_score=68,
            subcategories={
                "phase_intent_alignment": 0,
                "endpoint_evidence_strength": 0,
                "target_population_alignment": 0,
                "operational_burden_balance": 0,
            },
            review_needed=False,
            reuse_previous_review=True,
            storyline_behavior="reuse_latest_validated_review_no_new_storyline_step",
        ),
        movement_summary="No review should be generated for a minor text cleanup.",
    ),
]


def get_contract_fixtures() -> list[dict[str, Any]]:
    """Return a deep copy so callers cannot mutate the canonical fixtures."""
    return deepcopy(CONTRACT_FIXTURES)


def validate_contract_fixtures(fixtures: list[dict[str, Any]] | None = None) -> list[str]:
    """Return structural validation errors for the static V1 fixtures."""
    fixtures = fixtures if fixtures is not None else CONTRACT_FIXTURES
    errors: list[str] = []
    seen_types: set[str] = set()
    seen_ids: set[str] = set()

    for index, fixture in enumerate(fixtures):
        fixture_id = fixture.get("fixture_id", f"<missing-{index}>")
        if fixture_id in seen_ids:
            errors.append(f"{fixture_id}: duplicate fixture_id")
        seen_ids.add(fixture_id)

        scenario_type = fixture.get("scenario_type")
        seen_types.add(str(scenario_type))
        if scenario_type not in REQUIRED_SCENARIO_TYPES:
            errors.append(f"{fixture_id}: unsupported scenario_type {scenario_type!r}")

        packet = fixture.get("input_packet")
        if not isinstance(packet, dict):
            errors.append(f"{fixture_id}: input_packet must be a dict")
            continue

        if packet.get("prompt_version") != PROMPT_VERSION:
            errors.append(f"{fixture_id}: prompt_version must be {PROMPT_VERSION}")
        if packet.get("rubric_version") != RUBRIC_VERSION:
            errors.append(f"{fixture_id}: rubric_version must be {RUBRIC_VERSION}")
        if packet.get("mode") != "existing_study":
            errors.append(f"{fixture_id}: mode must be existing_study")

        for key in (
            "trial_identity",
            "text_context",
            "structured_features",
            "operational_assumptions",
            "model_interpretation",
            "iteration_context",
        ):
            if key not in packet:
                errors.append(f"{fixture_id}: missing input_packet.{key}")

        operational = packet.get("operational_assumptions", {})
        for key in ("planned_enrollment", "planned_sites", "planned_duration_months"):
            if key not in operational:
                errors.append(f"{fixture_id}: missing operational_assumptions.{key}")

        expected = fixture.get("expected_behavior")
        if not isinstance(expected, dict):
            errors.append(f"{fixture_id}: expected_behavior must be a dict")
            continue

        if "expected_design_confidence" not in expected:
            errors.append(f"{fixture_id}: missing expected_design_confidence")
        else:
            adjustment = expected["expected_design_confidence"]
            if not isinstance(adjustment, (int, float)):
                errors.append(f"{fixture_id}: expected_design_confidence must be numeric")
            elif adjustment * 2 != int(adjustment * 2):
                errors.append(f"{fixture_id}: expected_design_confidence must use 0.5 increments")

        subcategory_points = expected.get("expected_design_subcategories")
        if not isinstance(subcategory_points, dict):
            errors.append(f"{fixture_id}: missing expected_design_subcategories")
        else:
            missing_subpoints = REQUIRED_DESIGN_SUBCATEGORIES.difference(subcategory_points)
            extra_subpoints = set(subcategory_points).difference(REQUIRED_DESIGN_SUBCATEGORIES)
            if missing_subpoints:
                errors.append(f"{fixture_id}: missing expected design subcategories {sorted(missing_subpoints)}")
            if extra_subpoints:
                errors.append(f"{fixture_id}: unexpected expected design subcategories {sorted(extra_subpoints)}")
            for subcategory_name, points in subcategory_points.items():
                if not isinstance(points, (int, float)):
                    errors.append(f"{fixture_id}: {subcategory_name} expected points must be numeric")
                elif points < -5 or points > 5:
                    errors.append(f"{fixture_id}: {subcategory_name} expected points must be between -5 and +5")
                elif points * 2 != int(points * 2):
                    errors.append(f"{fixture_id}: {subcategory_name} expected points must use 0.5 increments")
            if isinstance(expected.get("expected_design_confidence"), (int, float)):
                total = sum(value for value in subcategory_points.values() if isinstance(value, (int, float)))
                if total != expected["expected_design_confidence"]:
                    errors.append(
                        f"{fixture_id}: expected_design_confidence should equal sum of expected_design_subcategories"
                    )

        if "expected_total_scenario_score" not in expected:
            errors.append(f"{fixture_id}: missing expected_total_scenario_score")
        else:
            final_score = expected["expected_total_scenario_score"]
            if not isinstance(final_score, (int, float)) or final_score < 0 or final_score > 100:
                errors.append(f"{fixture_id}: expected_total_scenario_score must be numeric between 0 and 100")

        if (
            "expected_design_confidence" in expected
            and "expected_total_scenario_score" in expected
        ):
            completion_score = packet.get("model_interpretation", {}).get("completion_score")
            if isinstance(completion_score, (int, float)):
                calculated = max(0, min(100, completion_score + expected["expected_design_confidence"]))
                if calculated != expected["expected_total_scenario_score"]:
                    errors.append(
                        f"{fixture_id}: expected_total_scenario_score should be {calculated} "
                        "from completion_score + expected_design_confidence"
                    )
            else:
                errors.append(f"{fixture_id}: model_interpretation.completion_score must be numeric")

        review_needed = expected.get("review_needed")
        review = fixture.get("mock_review")
        if review_needed is False:
            if review is not None:
                errors.append(f"{fixture_id}: no-op fixture must not define mock_review")
            if expected.get("reuse_previous_review") is not True:
                errors.append(f"{fixture_id}: no-op fixture must set reuse_previous_review")
            continue

        if not isinstance(review, dict):
            errors.append(f"{fixture_id}: review-needed fixture must define mock_review")
            continue

        metadata = review.get("review_metadata") or {}
        expected_mode = "hidden_baseline" if scenario_type == "baseline" else "first_visible_iteration"
        if metadata.get("review_mode") != expected_mode:
            errors.append(f"{fixture_id}: review_metadata.review_mode should be {expected_mode}")
        if metadata.get("visible") is not (expected_mode != "hidden_baseline"):
            errors.append(f"{fixture_id}: review_metadata.visible mismatch for {expected_mode}")

        subcategories = review.get("design_confidence_subcategories")
        if not isinstance(subcategories, dict):
            errors.append(f"{fixture_id}: missing design_confidence_subcategories")
            continue

        missing_subcategories = REQUIRED_DESIGN_SUBCATEGORIES.difference(subcategories)
        extra_subcategories = set(subcategories).difference(REQUIRED_DESIGN_SUBCATEGORIES)
        if missing_subcategories:
            errors.append(f"{fixture_id}: missing design subcategories {sorted(missing_subcategories)}")
        if extra_subcategories:
            errors.append(f"{fixture_id}: unexpected design subcategories {sorted(extra_subcategories)}")

        for subcategory_name, subcategory in subcategories.items():
            if "current_state" not in subcategory:
                errors.append(f"{fixture_id}: {subcategory_name} missing current_state")
            elif subcategory.get("current_state") not in {"strong", "supportive", "balanced", "weak", "conflicting"}:
                errors.append(f"{fixture_id}: {subcategory_name} invalid current_state")
            if "movement_direction" not in subcategory:
                errors.append(f"{fixture_id}: {subcategory_name} missing movement_direction")
            elif subcategory.get("movement_direction") not in {
                "resolved",
                "improved",
                "partially_resolved",
                "unchanged",
                "offset",
                "weakened",
                "worsened",
                "newly_introduced",
            }:
                errors.append(f"{fixture_id}: {subcategory_name} invalid movement_direction")
            if "movement_materiality" not in subcategory:
                errors.append(f"{fixture_id}: {subcategory_name} missing movement_materiality")
            elif subcategory.get("movement_materiality") not in {"none", "minor", "moderate", "major"}:
                errors.append(f"{fixture_id}: {subcategory_name} invalid movement_materiality")
            if "effect_role" not in subcategory:
                errors.append(f"{fixture_id}: {subcategory_name} missing effect_role")
            elif subcategory.get("effect_role") not in {"counterweight", "confirming", "independent", "unchanged"}:
                errors.append(f"{fixture_id}: {subcategory_name} invalid effect_role")
            if "rating" in subcategory and subcategory.get("rating") not in {"strong", "supportive", "balanced", "weak", "conflicting"}:
                errors.append(f"{fixture_id}: {subcategory_name} invalid legacy rating")
            if "score_materiality" in subcategory and subcategory.get("score_materiality") not in SCORE_MATERIALITY_LEVELS:
                errors.append(f"{fixture_id}: {subcategory_name} invalid legacy score_materiality")
            if "rationale" not in subcategory:
                errors.append(f"{fixture_id}: {subcategory_name} missing rationale")
            if "short_rationale" not in subcategory:
                errors.append(f"{fixture_id}: {subcategory_name} missing short_rationale")
            if "optional_lenses_used" not in subcategory:
                errors.append(f"{fixture_id}: {subcategory_name} missing optional_lenses_used")
            if "regulatory_or_finance_note" not in subcategory:
                errors.append(f"{fixture_id}: {subcategory_name} missing regulatory_or_finance_note")
            evidence_fields = subcategory.get("evidence_fields")
            if not isinstance(evidence_fields, list):
                errors.append(f"{fixture_id}: {subcategory_name} evidence_fields must be a list")

        for key in (
            "review_metadata",
            "completion_outlook_analysis",
            "design_confidence_subcategories",
            "design_confidence_analysis",
            "main_tension",
            "key_questions",
            "scenario_consistency_note",
            "continuity",
            "trace",
        ):
            if key not in review:
                errors.append(f"{fixture_id}: missing mock_review.{key}")

    missing_types = REQUIRED_SCENARIO_TYPES.difference(seen_types)
    if missing_types:
        errors.append(f"missing required scenario types: {sorted(missing_types)}")

    return errors
