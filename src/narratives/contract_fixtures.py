"""Static contract fixtures for the serious-game narrative V1 layer.

These fixtures are intentionally provider-free. They define the scenarios,
input packet shape, expected structured review shape, and expected behavior
that later packet-builder, validation, scoring, mock-reviewer, and UI work
must preserve.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

PROMPT_VERSION = "narratives_v1"
RUBRIC_VERSION = "design_coherence_v1"

REQUIRED_SCENARIO_TYPES = {
    "baseline",
    "model_facing_edit",
    "operational_only_edit",
    "material_text_only_edit",
    "text_structured_alignment_clarification",
    "clarified_text_structured_alignment_review",
    "no_op_minor_text_edit",
}

REQUIRED_REVIEW_DOMAINS = {
    "development_question_fit",
    "scientific_rigor",
    "population_relevance",
    "endpoint_and_comparator_logic",
    "operational_scale_fit",
    "change_integrity",
    "text_consistency",
}

QUALITY_PILLAR_KEYS = {
    "evidence_coherence",
    "population_strategy_fit",
    "execution_plausibility",
}

BASELINE_STRUCTURED_FEATURES: dict[str, Any] = {
    "therapeutic_area_ml": "Oncology",
    "gbd_cause_id_3_ml": "Breast cancer",
    "is_rare_disease_ml": 0,
    "phase_ml": "Phase 3",
    "strategic_ambition_ml": "Confirmatory / registration-enabling",
    "target_precedent_ml": "Validated target",
    "target_pathway_class_ml": "Established pathway",
    "therapeutic_modality_ml": "Targeted small molecule",
    "innovation_tier_ml": "Next-generation",
    "intervention_model_ml": "Parallel assignment",
    "primary_purpose_ml": "Treatment",
    "adaptive_design_ml": "No adaptive design",
    "endpoint_rigor_ml": "Clinical outcome",
    "endpoint_structure_ml": "Single primary endpoint",
    "biomarker_stratification_ml": "Biomarker-stratified",
    "patient_severity_ml": "Advanced / high burden",
    "line_of_therapy_ml": "Second line",
    "gender_ml": "All sexes",
    "healthy_volunteers_ml": "Patients only",
    "adult_ml": "Adults included",
    "child_ml": "Children excluded",
    "older_adult_ml": "Older adults included",
    "masking_ml": "Double blind",
    "allocation_ml": "Randomized",
    "has_dmc_ml": 1,
    "has_placebo_ml": 0,
    "comparator_benchmark_ml": "Active comparator / standard of care",
    "administration_complexity_ml": "Moderate",
    "number_of_arms_ml": 2,
    "sponsor_tier_ml": "Large pharmaceutical",
    "primary_duration_months_ml": 18.0,
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
            "primary_outcomes_ui": "Progression-free survival by blinded independent review.",
            "criteria_ui": "Adults with advanced biomarker-positive breast cancer.",
        },
        "structured_features": deepcopy(BASELINE_STRUCTURED_FEATURES),
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


def _domain(rating: str, rationale: str, evidence_fields: list[str]) -> dict[str, Any]:
    return {
        "rating": rating,
        "rationale": rationale,
        "evidence_fields": evidence_fields,
    }


def _participant_review(
    what_changed: str,
    moved: str,
    gained: str,
    sacrificed: str,
    operational_note: str,
    text_note: str,
    question: str,
) -> dict[str, str]:
    return {
        "what_changed": what_changed,
        "why_completion_score_may_have_moved": moved,
        "what_the_design_gained": gained,
        "what_the_design_may_have_sacrificed": sacrificed,
        "operational_feasibility_note": operational_note,
        "text_consistency_note": text_note,
        "challenge_question": question,
    }


def _review(
    *,
    movement_summary: str,
    domains: dict[str, dict[str, Any]],
    participant_review: dict[str, str],
    storyline_update: str,
    new_concerns: list[str] | None = None,
    operational_statuses: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "score_movement_review": {
            "summary": movement_summary,
            "model_supported_reasons": [],
            "cautions": [],
        },
        "quality_review_domains": domains,
        "participant_review": participant_review,
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
                    for domain in domains.values()
                    for field in domain.get("evidence_fields", [])
                    if "." not in field
                }
            ),
            "main_pillars_considered": [],
            "operational_statuses_considered": operational_statuses or [],
            "compared_against": "previous_prediction",
            "should_repeat_prior_warning": False,
        },
    }


CONTRACT_FIXTURES: list[dict[str, Any]] = [
    {
        "fixture_id": "baseline_hidden_review_v1",
        "scenario_type": "baseline",
        "description": "Hidden existing-study baseline review generated once from the original selected trial.",
        "input_packet": _base_packet(),
        "mock_review": _review(
            movement_summary="Baseline review anchors later comparisons; no participant change has occurred.",
            domains={
                "development_question_fit": _domain("acceptable", "Phase, purpose, and intent are aligned.", ["phase_ml", "primary_purpose_ml", "strategic_ambition_ml"]),
                "scientific_rigor": _domain("acceptable", "Design preserves conventional confirmatory rigor.", ["endpoint_rigor_ml", "allocation_ml", "masking_ml"]),
                "population_relevance": _domain("acceptable", "Population appears relevant to the stated setting.", ["adult_ml", "older_adult_ml", "line_of_therapy_ml"]),
                "endpoint_and_comparator_logic": _domain("acceptable", "Endpoint and comparator are broadly coherent.", ["endpoint_rigor_ml", "comparator_benchmark_ml"]),
                "operational_scale_fit": _domain("acceptable", "Operational assumptions are typical for the cohort.", ["operational_assumptions.planned_enrollment.enrollment_status", "operational_assumptions.planned_sites.site_count_status", "operational_assumptions.planned_duration_months.duration_status"]),
                "change_integrity": _domain("neutral", "No participant change has occurred.", []),
                "text_consistency": _domain("consistent", "Text and structured fields are aligned.", ["summary_ui", "primary_outcomes_ui"]),
            },
            participant_review=_participant_review(
                "This is the original selected-trial baseline.",
                "No model-score movement is reviewed at baseline.",
                "The baseline preserves the original evidence and operational profile.",
                "No participant trade-off has been introduced yet.",
                "Operational assumptions are typical versus similar trials.",
                "The available text is consistent with the structured design.",
                "Which baseline concern should the team watch as the scenario evolves?",
            ),
            storyline_update="Baseline anchored with acceptable design coherence and no participant-introduced concern.",
            operational_statuses=["typical"],
        ),
        "expected_behavior": {
            "review_needed": True,
            "visible_to_participant_initially": False,
            "expected_quality_adjustment": 0,
            "expected_final_candidate_score": 68,
            "final_score_rule": "completion_score + quality_adjustment, clamped to 0..100",
            "storyline_behavior": "create_hidden_baseline_memory_only",
        },
    },
    {
        "fixture_id": "model_facing_endpoint_shortcut_v1",
        "scenario_type": "model_facing_edit",
        "description": "Participant edit raises Completion Score while weakening endpoint rigor and comparator logic.",
        "input_packet": {
            **_base_packet(),
            "structured_features": {
                **BASELINE_STRUCTURED_FEATURES,
                "endpoint_rigor_ml": "Surrogate / procedural endpoint",
                "has_placebo_ml": 1,
                "comparator_benchmark_ml": "Placebo / weak external comparator",
                "primary_duration_months_ml": 9.0,
            },
            "model_interpretation": {
                **_base_packet()["model_interpretation"],
                "completion_score": 74,
                "previous_completion_score": 68,
                "score_delta": 6,
                "pillar_deltas": {"Scientific Challenge": 3.2, "Execution Framework": 1.4},
                "top_feature_impact_changes": ["endpoint_rigor_ml", "comparator_benchmark_ml", "primary_duration_months_ml"],
            },
            "iteration_context": {
                **_base_packet()["iteration_context"],
                "previous_snapshot_id": "fixture-baseline",
                "current_snapshot_id": "fixture-model-facing-edit",
                "iteration_number": 1,
                "changed_fields": ["endpoint_rigor_ml", "has_placebo_ml", "comparator_benchmark_ml", "primary_duration_months_ml"],
                "compact_storyline_memory": "Baseline was acceptable; no prior participant concern.",
            },
        },
        "mock_review": _review(
            movement_summary="The Completion Score may have improved because the revised design appears easier to execute.",
            domains={
                "development_question_fit": _domain("weak", "Confirmatory intent is less well matched to weakened evidence choices.", ["strategic_ambition_ml", "endpoint_rigor_ml"]),
                "scientific_rigor": _domain("conflicting", "Endpoint rigor and comparator strength appear reduced together.", ["endpoint_rigor_ml", "comparator_benchmark_ml", "has_placebo_ml"]),
                "population_relevance": _domain("acceptable", "Population scope did not materially change.", ["adult_ml", "older_adult_ml"]),
                "endpoint_and_comparator_logic": _domain("conflicting", "Shorter endpoint timing and weaker comparator may reduce interpretability.", ["primary_duration_months_ml", "endpoint_rigor_ml", "comparator_benchmark_ml"]),
                "operational_scale_fit": _domain("acceptable", "Operational assumptions did not introduce a new scale concern.", ["operational_assumptions.planned_enrollment.enrollment_status"]),
                "change_integrity": _domain("potential_shortcut", "The path increases completion likelihood while reducing evidence value.", ["endpoint_rigor_ml", "comparator_benchmark_ml", "primary_duration_months_ml"]),
                "text_consistency": _domain("minor_tension", "The summary still reads like a confirmatory evidence trial.", ["summary_ui", "endpoint_rigor_ml"]),
            },
            participant_review=_participant_review(
                "Endpoint rigor, comparator framing, and primary endpoint duration changed.",
                "The score could have moved upward because the design may be simpler and shorter.",
                "The design may have gained execution feasibility.",
                "It may have sacrificed evidentiary strength and endpoint interpretability.",
                "Operational assumptions remain typical, so the main concern is not scale.",
                "The written summary still sounds confirmatory, creating some tension with the revised endpoint choices.",
                "Does the easier design still answer the same confirmatory question?",
            ),
            storyline_update="Raised potential shortcut concern: easier execution came with weaker endpoint/comparator evidence.",
            new_concerns=["endpoint_comparator_shortcut"],
            operational_statuses=["typical"],
        ),
        "expected_behavior": {
            "review_needed": True,
            "visible_to_participant_initially": True,
            "expected_quality_adjustment": -9,
            "expected_final_candidate_score": 65,
            "expected_quality_pillars": {
                "evidence_coherence": "negative",
                "population_strategy_fit": "negative_or_neutral",
                "execution_plausibility": "negative",
            },
            "storyline_behavior": "append_new_iteration_with_shortcut_concern",
        },
    },
    {
        "fixture_id": "operational_only_ambitious_enrollment_v1",
        "scenario_type": "operational_only_edit",
        "description": "Participant keeps model fields fixed but sets enrollment and sites above benchmark for the current design.",
        "input_packet": {
            **_base_packet(),
            "operational_assumptions": {
                **BASELINE_OPERATIONAL_ASSUMPTIONS,
                "planned_enrollment": {
                    **BASELINE_OPERATIONAL_ASSUMPTIONS["planned_enrollment"],
                    "value": 1400,
                    "source": "user_scenario",
                    "enrollment_status": "above_benchmark_high",
                    "support_level": "partly_supported_by_current_design",
                    "conflicting_signals": ["large_sample_for_biomarker_subset"],
                    "interpretation_hint": "Enrollment is high versus similar trials and only partly supported by the current design.",
                },
                "planned_sites": {
                    **BASELINE_OPERATIONAL_ASSUMPTIONS["planned_sites"],
                    "value": 90,
                    "source": "user_scenario",
                    "site_count_status": "ambitious",
                },
            },
            "model_interpretation": {
                **_base_packet()["model_interpretation"],
                "previous_completion_score": 68,
                "score_delta": 0,
                "pillar_deltas": {},
                "top_feature_impact_changes": [],
            },
            "iteration_context": {
                **_base_packet()["iteration_context"],
                "previous_snapshot_id": "fixture-baseline",
                "current_snapshot_id": "fixture-operational-only",
                "iteration_number": 1,
                "changed_fields": ["operational_assumptions.planned_enrollment", "operational_assumptions.planned_sites"],
                "compact_storyline_memory": "Baseline was acceptable; no prior participant concern.",
            },
        },
        "mock_review": _review(
            movement_summary="The Completion Score did not move because model-facing fields did not change.",
            domains={
                "development_question_fit": _domain("acceptable", "The development question remains unchanged.", ["phase_ml", "strategic_ambition_ml"]),
                "scientific_rigor": _domain("acceptable", "Evidence-generating fields remain intact.", ["endpoint_rigor_ml", "allocation_ml"]),
                "population_relevance": _domain("acceptable", "Population fields remain unchanged.", ["adult_ml", "older_adult_ml"]),
                "endpoint_and_comparator_logic": _domain("acceptable", "Endpoint and comparator fields remain unchanged.", ["endpoint_rigor_ml", "comparator_benchmark_ml"]),
                "operational_scale_fit": _domain("weak", "Enrollment is above benchmark high and only partly supported by the current biomarker-defined design.", ["operational_assumptions.planned_enrollment.enrollment_status", "operational_assumptions.planned_enrollment.support_level"]),
                "change_integrity": _domain("neutral", "The change stress-tests feasibility without weakening evidence fields.", ["operational_assumptions.planned_enrollment", "operational_assumptions.planned_sites"]),
                "text_consistency": _domain("consistent", "No text contradiction is introduced.", ["summary_ui"]),
            },
            participant_review=_participant_review(
                "Only enrollment and site assumptions changed.",
                "The model score did not move because operational assumptions are outside XGBoost.",
                "The scenario may test whether a larger study footprint is feasible.",
                "It may have sacrificed operational credibility if the biomarker-defined population is hard to recruit.",
                "Enrollment is above benchmark high and only partly supported by the current design.",
                "Text remains consistent with the structured design.",
                "What evidence would make this larger enrollment assumption credible?",
            ),
            storyline_update="Operational-scale concern added without changing Completion Score.",
            new_concerns=["ambitious_enrollment_support"],
            operational_statuses=["above_benchmark_high", "ambitious"],
        ),
        "expected_behavior": {
            "review_needed": True,
            "visible_to_participant_initially": True,
            "expected_quality_adjustment": -2,
            "expected_final_candidate_score": 66,
            "expected_quality_pillars": {
                "evidence_coherence": "neutral",
                "population_strategy_fit": "neutral",
                "execution_plausibility": "negative",
            },
            "storyline_behavior": "append_iteration_without_model_score_delta",
        },
    },
    {
        "fixture_id": "material_text_only_endpoint_conflict_v1",
        "scenario_type": "material_text_only_edit",
        "description": "Participant changes endpoint text materially while structured model fields and Completion Score stay fixed.",
        "input_packet": {
            **_base_packet(),
            "text_context": {
                **_base_packet()["text_context"],
                "primary_outcomes_ui": "Short-term symptom response at 4 weeks.",
                "summary_ui": "Confirmatory registration study intended to establish durable disease control.",
            },
            "model_interpretation": {
                **_base_packet()["model_interpretation"],
                "previous_completion_score": 68,
                "score_delta": 0,
            },
            "iteration_context": {
                **_base_packet()["iteration_context"],
                "previous_snapshot_id": "fixture-baseline",
                "current_snapshot_id": "fixture-material-text-only",
                "iteration_number": 1,
                "changed_fields": ["text_context.primary_outcomes_ui"],
                "compact_storyline_memory": "Baseline was acceptable; no prior participant concern.",
            },
        },
        "mock_review": _review(
            movement_summary="The Completion Score did not move because structured model fields did not change.",
            domains={
                "development_question_fit": _domain("weak", "Confirmatory durable-control intent may conflict with short-term symptom endpoint text.", ["summary_ui", "primary_outcomes_ui", "strategic_ambition_ml"]),
                "scientific_rigor": _domain("weak", "Endpoint text may weaken decision usefulness for the stated intent.", ["primary_outcomes_ui", "endpoint_rigor_ml"]),
                "population_relevance": _domain("acceptable", "Population scope did not change.", ["adult_ml", "older_adult_ml"]),
                "endpoint_and_comparator_logic": _domain("weak", "Endpoint text and structured endpoint duration appear misaligned.", ["primary_outcomes_ui", "primary_duration_months_ml"]),
                "operational_scale_fit": _domain("acceptable", "Operational assumptions remain typical.", ["operational_assumptions.planned_duration_months.duration_status"]),
                "change_integrity": _domain("neutral", "Only text changed; the path needs interpretation rather than shortcut attribution.", ["text_context.primary_outcomes_ui"]),
                "text_consistency": _domain("material_tension", "Text creates a material endpoint-intent tension.", ["summary_ui", "primary_outcomes_ui", "primary_duration_months_ml"]),
            },
            participant_review=_participant_review(
                "The endpoint text changed while structured Trial Features stayed the same.",
                "The Completion Score did not move because the XGBoost input fields did not change.",
                "The edit may clarify the endpoint being discussed.",
                "It may create tension between short-term response and durable confirmatory intent.",
                "Operational assumptions remain typical, but duration interpretation may need discussion.",
                "The text now materially differs from the structured endpoint-duration context.",
                "Is the endpoint text intended to replace or only clarify the original endpoint strategy?",
            ),
            storyline_update="Material text-only endpoint concern added; no model-score movement.",
            new_concerns=["endpoint_text_material_tension"],
            operational_statuses=["typical"],
        ),
        "expected_behavior": {
            "review_needed": True,
            "visible_to_participant_initially": True,
            "expected_quality_adjustment": -6,
            "expected_final_candidate_score": 62,
            "expected_quality_pillars": {
                "evidence_coherence": "negative",
                "population_strategy_fit": "negative",
                "execution_plausibility": "neutral",
            },
            "storyline_behavior": "append_text_only_iteration_without_model_score_delta",
        },
    },
    {
        "fixture_id": "endpoint_structure_text_alignment_requires_clarification_v1",
        "scenario_type": "text_structured_alignment_clarification",
        "description": "Structured endpoint setting and text appear materially misaligned; Quality Review should pause for clarification.",
        "input_packet": {
            **_base_packet(),
            "structured_features": {
                **BASELINE_STRUCTURED_FEATURES,
                "endpoint_structure_ml": "MULTI_COMPOSITE",
            },
            "text_context": {
                **_base_packet()["text_context"],
                "primary_outcomes_ui": "The study has a single primary endpoint: progression-free survival.",
            },
            "model_interpretation": {
                **_base_packet()["model_interpretation"],
                "completion_score": 70,
                "previous_completion_score": 68,
                "score_delta": 2,
                "top_feature_impact_changes": ["endpoint_structure_ml"],
            },
            "iteration_context": {
                **_base_packet()["iteration_context"],
                "previous_snapshot_id": "fixture-baseline",
                "current_snapshot_id": "fixture-alignment-clarification",
                "iteration_number": 1,
                "changed_fields": ["endpoint_structure_ml", "text_context.primary_outcomes_ui"],
                "compact_storyline_memory": "Baseline was acceptable; no prior participant concern.",
            },
        },
        "mock_review": None,
        "expected_behavior": {
            "review_needed": True,
            "clarification_needed": True,
            "visible_to_participant_initially": True,
            "expected_clarification_issues": ["endpoint_structure_text_mismatch"],
            "expected_quality_adjustment": None,
            "expected_final_candidate_score": None,
            "storyline_behavior": "pause_quality_review_until_user_corrects_or_explains_alignment_issue",
        },
    },
    {
        "fixture_id": "endpoint_structure_text_alignment_explained_v1",
        "scenario_type": "clarified_text_structured_alignment_review",
        "description": "The same apparent endpoint mismatch has a user explanation, so Quality Review can proceed using clarified context.",
        "input_packet": {
            **_base_packet(),
            "structured_features": {
                **BASELINE_STRUCTURED_FEATURES,
                "endpoint_structure_ml": "MULTI_COMPOSITE",
            },
            "text_context": {
                **_base_packet()["text_context"],
                "primary_outcomes_ui": "The study has a single primary endpoint: progression-free survival.",
            },
            "clarification_context": {
                "user_clarifications": [
                    {
                        "issue_id": "endpoint_structure_text_mismatch",
                        "field_id": "endpoint_structure_ml",
                        "explanation": (
                            "The visible endpoint text names the main clinical endpoint; "
                            "a second co-primary biomarker endpoint is handled in the scenario design."
                        ),
                    }
                ],
            },
            "model_interpretation": {
                **_base_packet()["model_interpretation"],
                "completion_score": 70,
                "previous_completion_score": 68,
                "score_delta": 2,
                "top_feature_impact_changes": ["endpoint_structure_ml"],
            },
            "iteration_context": {
                **_base_packet()["iteration_context"],
                "previous_snapshot_id": "fixture-baseline",
                "current_snapshot_id": "fixture-alignment-explained",
                "iteration_number": 1,
                "changed_fields": ["endpoint_structure_ml", "text_context.primary_outcomes_ui"],
                "compact_storyline_memory": "Baseline was acceptable; no prior participant concern.",
            },
        },
        "mock_review": _review(
            movement_summary="The Completion Score may have changed because endpoint structure changed.",
            domains={
                "development_question_fit": _domain("acceptable", "The explanation keeps the confirmatory intent interpretable.", ["strategic_ambition_ml", "endpoint_structure_ml"]),
                "scientific_rigor": _domain("acceptable", "The clarified co-primary structure can remain defensible if both endpoints are prospectively specified.", ["endpoint_structure_ml", "primary_outcomes_ui"]),
                "population_relevance": _domain("acceptable", "Population scope did not materially change.", ["adult_ml", "older_adult_ml"]),
                "endpoint_and_comparator_logic": _domain("weak", "The text still names only one endpoint, so endpoint hierarchy should be clearer.", ["endpoint_structure_ml", "primary_outcomes_ui"]),
                "operational_scale_fit": _domain("acceptable", "Operational assumptions remain typical.", ["operational_assumptions.planned_duration_months.duration_status"]),
                "change_integrity": _domain("neutral", "The user clarified an apparent mismatch rather than changing the model-facing field.", ["endpoint_structure_ml", "clarification_context.user_clarifications"]),
                "text_consistency": _domain("minor_tension", "The explanation reduces the apparent contradiction but the visible endpoint text remains incomplete.", ["endpoint_structure_ml", "primary_outcomes_ui", "clarification_context.user_clarifications"]),
            },
            participant_review=_participant_review(
                "Endpoint structure and endpoint text appeared misaligned, and the team added clarification.",
                "The Completion Score may have moved because the structured endpoint setting changed.",
                "The explanation may preserve the intended co-primary endpoint strategy.",
                "The design may still sacrifice clarity if the endpoint text names only one endpoint.",
                "Operational assumptions remain typical.",
                "The added explanation reduces the apparent mismatch but should remain visible in the scenario rationale.",
                "Is the endpoint hierarchy clear enough for another team to understand the scenario without the clarification note?",
            ),
            storyline_update="Endpoint-structure mismatch was explained; minor text clarity concern remains.",
            new_concerns=["endpoint_text_clarity_after_clarification"],
            operational_statuses=["typical"],
        ),
        "expected_behavior": {
            "review_needed": True,
            "visible_to_participant_initially": True,
            "expected_quality_adjustment": -3,
            "expected_final_candidate_score": 67,
            "expected_quality_pillars": {
                "evidence_coherence": "negative",
                "population_strategy_fit": "neutral",
                "execution_plausibility": "neutral",
            },
            "storyline_behavior": "append_clarified_alignment_iteration",
        },
    },
    {
        "fixture_id": "noop_minor_text_cleanup_v1",
        "scenario_type": "no_op_minor_text_edit",
        "description": "Participant makes only casing/punctuation cleanup; review should be reused.",
        "input_packet": {
            **_base_packet(),
            "text_context": {
                **_base_packet()["text_context"],
                "summary_ui": (
                    "Confirmatory study evaluating targeted therapy plus standard care "
                    "in adults with advanced biomarker-positive breast cancer"
                ),
            },
            "model_interpretation": {
                **_base_packet()["model_interpretation"],
                "previous_completion_score": 68,
                "score_delta": 0,
            },
            "iteration_context": {
                **_base_packet()["iteration_context"],
                "previous_snapshot_id": "fixture-baseline",
                "current_snapshot_id": "fixture-noop-minor-text",
                "iteration_number": 1,
                "changed_fields": ["text_context.summary_ui"],
                "compact_storyline_memory": "Baseline was acceptable; no prior participant concern.",
            },
        },
        "mock_review": None,
        "expected_behavior": {
            "review_needed": False,
            "visible_to_participant_initially": True,
            "reuse_previous_review": True,
            "expected_quality_adjustment": 0,
            "expected_final_candidate_score": 68,
            "expected_quality_pillars": {
                "evidence_coherence": "unchanged",
                "population_strategy_fit": "unchanged",
                "execution_plausibility": "unchanged",
            },
            "storyline_behavior": "reuse_latest_validated_review_no_new_storyline_step",
        },
    },
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

        clarification_needed = expected.get("clarification_needed") is True

        if "expected_quality_adjustment" not in expected:
            errors.append(f"{fixture_id}: missing expected_quality_adjustment")
        else:
            adjustment = expected["expected_quality_adjustment"]
            if clarification_needed and adjustment is None:
                pass
            elif not isinstance(adjustment, int) or adjustment < -10 or adjustment > 10:
                errors.append(f"{fixture_id}: expected_quality_adjustment must be an int between -10 and 10")

        if "expected_final_candidate_score" not in expected:
            errors.append(f"{fixture_id}: missing expected_final_candidate_score")
        else:
            final_score = expected["expected_final_candidate_score"]
            if clarification_needed and final_score is None:
                pass
            elif not isinstance(final_score, int) or final_score < 0 or final_score > 100:
                errors.append(f"{fixture_id}: expected_final_candidate_score must be an int between 0 and 100")

        if (
            not clarification_needed
            and "expected_quality_adjustment" in expected
            and "expected_final_candidate_score" in expected
        ):
            completion_score = packet.get("model_interpretation", {}).get("completion_score")
            if isinstance(completion_score, int):
                calculated = max(0, min(100, completion_score + expected["expected_quality_adjustment"]))
                if calculated != expected["expected_final_candidate_score"]:
                    errors.append(
                        f"{fixture_id}: expected_final_candidate_score should be {calculated} "
                        "from completion_score + expected_quality_adjustment"
                    )
            else:
                errors.append(f"{fixture_id}: model_interpretation.completion_score must be an int")

        review_needed = expected.get("review_needed")
        review = fixture.get("mock_review")
        if clarification_needed:
            if review is not None:
                errors.append(f"{fixture_id}: clarification fixture must not define mock_review before user explanation")
            if expected.get("expected_clarification_issues") is None:
                errors.append(f"{fixture_id}: clarification fixture must define expected_clarification_issues")
            continue
        if review_needed is False:
            if review is not None:
                errors.append(f"{fixture_id}: no-op fixture must not define mock_review")
            if expected.get("reuse_previous_review") is not True:
                errors.append(f"{fixture_id}: no-op fixture must set reuse_previous_review")
            continue

        if not isinstance(review, dict):
            errors.append(f"{fixture_id}: review-needed fixture must define mock_review")
            continue

        domains = review.get("quality_review_domains")
        if not isinstance(domains, dict):
            errors.append(f"{fixture_id}: missing quality_review_domains")
            continue

        missing_domains = REQUIRED_REVIEW_DOMAINS.difference(domains)
        extra_domains = set(domains).difference(REQUIRED_REVIEW_DOMAINS)
        if missing_domains:
            errors.append(f"{fixture_id}: missing domains {sorted(missing_domains)}")
        if extra_domains:
            errors.append(f"{fixture_id}: unexpected domains {sorted(extra_domains)}")

        for domain_name, domain in domains.items():
            if "rating" not in domain:
                errors.append(f"{fixture_id}: {domain_name} missing rating")
            if "rationale" not in domain:
                errors.append(f"{fixture_id}: {domain_name} missing rationale")
            evidence_fields = domain.get("evidence_fields")
            if not isinstance(evidence_fields, list):
                errors.append(f"{fixture_id}: {domain_name} evidence_fields must be a list")

        for key in ("score_movement_review", "participant_review", "continuity", "trace"):
            if key not in review:
                errors.append(f"{fixture_id}: missing mock_review.{key}")

    missing_types = REQUIRED_SCENARIO_TYPES.difference(seen_types)
    if missing_types:
        errors.append(f"missing required scenario types: {sorted(missing_types)}")

    return errors
