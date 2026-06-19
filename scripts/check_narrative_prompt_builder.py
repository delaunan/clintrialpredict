#!/usr/bin/env python
"""Validate active Trial Score provider prompt and response-contract helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narratives.contract_fixtures import get_contract_fixtures  # noqa: E402
from src.narratives.packet_builder import build_review_packet_from_fixture  # noqa: E402
from src.narratives.prompt_builder import (  # noqa: E402
    PROMPT_MODE_FIRST_VISIBLE_ITERATION,
    PROMPT_MODE_HIDDEN_BASELINE,
    PROMPT_MODE_LATER_VISIBLE_ITERATION,
    PROMPT_TEMPLATE_VERSION,
    RESPONSE_SCHEMA_VERSION,
    build_pass2_input,
    build_pass2_provider_prompt,
    build_provider_prompt,
    gemini_response_schema,
    infer_prompt_mode,
    pass2_gemini_response_schema,
    pass2_response_contract,
    provider_response_contract,
)
from src.narratives.trial_score_contract import (  # noqa: E402
    APP_OWNED_TRIAL_SCORE_FIELDS,
    OPERATIONAL_FIT_MATERIALITIES,
    OPERATIONAL_FIT_RATINGS,
    OPERATIONAL_INTERACTION_LABELS,
    PASS1_SCHEMA_VERSION,
    PASS2_SCHEMA_VERSION,
    REALITY_CHECK_ALLOCATION_TARGETS,
    REALITY_CHECK_EFFECTS,
    REALITY_CHECK_STRENGTHS,
    score_pass1_review,
)


def main() -> int:
    errors: list[str] = []
    fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "operational_only_ambitious_enrollment_v2"
    )
    baseline_fixture = next(
        item for item in get_contract_fixtures()
        if item["fixture_id"] == "baseline_hidden_review_v2"
    )
    packet = build_review_packet_from_fixture(fixture)
    baseline_packet = build_review_packet_from_fixture(baseline_fixture)
    prompt = build_provider_prompt(packet)
    baseline_prompt = build_provider_prompt(baseline_packet)
    contract = provider_response_contract()
    narrative_doc = (ROOT / "docs" / "trial_score_narrative_direction.md").read_text()

    if RESPONSE_SCHEMA_VERSION != PASS1_SCHEMA_VERSION:
        errors.append("response schema version should use the active Pass 1 Trial Score schema")
    if contract.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        errors.append("response contract should expose stable schema version")
    if set(contract.get("required_top_level_objects") or []) != {
        "review_metadata",
        "completion_outlook_analysis",
        "strategy_shift_check",
        "operational_fit",
        "reality_check",
        "tension_question_options",
        "continuity_update",
        "analytical_narrative_draft",
    }:
        errors.append("response contract should require the Pass 1 Trial Score object model")
    if set(contract.get("allowed_operational_fit_ratings") or []) != OPERATIONAL_FIT_RATINGS:
        errors.append("response contract should include all Operational Fit ratings")
    if set(contract.get("allowed_operational_fit_materiality") or []) != OPERATIONAL_FIT_MATERIALITIES:
        errors.append("response contract should include all Operational Fit materiality labels")
    if set(contract.get("allowed_operational_interaction_labels") or []) != OPERATIONAL_INTERACTION_LABELS:
        errors.append("response contract should include all Operational Fit interaction labels")
    if set(contract.get("allowed_reality_check_effects") or []) != REALITY_CHECK_EFFECTS:
        errors.append("response contract should include all Reality Check effects")
    if set(contract.get("allowed_reality_check_strengths") or []) != REALITY_CHECK_STRENGTHS:
        errors.append("response contract should include all Reality Check strengths")
    if set(contract.get("allowed_reality_check_allocation_targets") or {}) != set(REALITY_CHECK_ALLOCATION_TARGETS):
        errors.append("response contract should include all Reality Check allocation target IDs")
    for target_id, target in REALITY_CHECK_ALLOCATION_TARGETS.items():
        contract_target = (contract.get("allowed_reality_check_allocation_targets") or {}).get(target_id) or {}
        if contract_target.get("pillar") != target.get("pillar") or contract_target.get("subpillar") != target.get("subpillar"):
            errors.append(f"response contract target mapping drifted for {target_id}")
        if not contract_target.get("description"):
            errors.append(f"response contract target is missing description for {target_id}")
        if target_id not in narrative_doc:
            errors.append(f"narrative direction doc missing allocation target ID: {target_id}")
    if set(contract.get("forbidden_provider_fields") or []) != APP_OWNED_TRIAL_SCORE_FIELDS:
        errors.append("response contract should declare app-owned Trial Score fields")

    scoring_ownership = str(contract.get("scoring_ownership") or "")
    for term in (
        "Operational Fit points",
        "Reality Check points",
        "Trial Score",
        "application calculates",
    ):
        if term not in scoring_ownership:
            errors.append(f"scoring ownership missing term: {term}")

    pass1_rules = " ".join(contract.get("pass1_instructions") or [])
    for term in (
        "structured analytical judgments",
        "XGBoost Completion Outlook",
        "model_signal_guidance",
        "latest movement signals first",
        "feature label/value with parent pillar/subpillar",
        "combined_operational_fit",
        "Reality Check allocations",
        "analytical_narrative_draft",
        "extensive rough analytical draft",
        "score-aware wording is allowed here",
        "Do not stop at model-signal recap",
        "population/setting/clinical context",
        "endpoint interpretability",
        "safety governance",
        "comparator or standard-of-care context",
        "development decision supported",
        "evidence completeness risk",
        "program-level meaning",
        "immune markers, disease-control measures, clinically confirmed events",
        "tension_question_options",
        "Do not split them into one main option plus alternatives",
        "participant_wider_question",
        "population-specific clinical meaning",
        "clinically confirmed event follow-up",
        "next development decision",
        "development issue",
        "trial evidence behind it",
        "associated wider-perspective strategic question topic Pass 2 can shape",
        "Prefer analytically specific tension summaries",
        "evidence-confidence or evidence-completeness trade-offs",
        "review the evidence package",
        "observe scenario dynamics",
        "identify weak assumptions",
        "Do not return app-owned point values",
    ):
        if term not in pass1_rules:
            errors.append(f"Pass 1 instructions missing term: {term}")

    schema = gemini_response_schema()
    schema_properties = schema.get("properties") or {}
    operational_schema = schema_properties.get("operational_fit") or {}
    combined_schema = ((operational_schema.get("properties") or {}).get("combined_operational_fit") or {})
    combined_properties = combined_schema.get("properties") or {}
    reality_schema = schema_properties.get("reality_check") or {}
    reality_properties = reality_schema.get("properties") or {}
    allocation_schema = ((((reality_properties.get("allocations") or {}).get("items") or {}).get("properties") or {}))
    metadata_schema = schema_properties.get("review_metadata") or {}
    draft_schema = schema_properties.get("analytical_narrative_draft") or {}
    tension_options_schema = schema_properties.get("tension_question_options") or {}
    tension_options_item_schema = (tension_options_schema.get("items") or {}).get("properties") or {}
    if schema.get("type") != "OBJECT":
        errors.append("Gemini response schema should require a top-level object")
    if set(schema.get("required") or []) != set(contract.get("required_top_level_objects") or []):
        errors.append("Gemini response schema should require all Pass 1 objects")
    if set(combined_schema.get("required") or []) != {
        "rating",
        "materiality",
        "interaction_with_completion_outlook",
        "central_reason",
        "evidence_fields",
    }:
        errors.append("Gemini schema should require combined Operational Fit fields")
    if set(combined_properties.get("rating", {}).get("enum") or []) != OPERATIONAL_FIT_RATINGS:
        errors.append("Gemini schema should enumerate Operational Fit ratings")
    if set(combined_properties.get("materiality", {}).get("enum") or []) != OPERATIONAL_FIT_MATERIALITIES:
        errors.append("Gemini schema should enumerate Operational Fit materiality")
    if set(reality_properties.get("effect", {}).get("enum") or []) != REALITY_CHECK_EFFECTS:
        errors.append("Gemini schema should enumerate Reality Check effects")
    if set(reality_properties.get("strength", {}).get("enum") or []) != REALITY_CHECK_STRENGTHS:
        errors.append("Gemini schema should enumerate Reality Check strengths")
    if set(allocation_schema.get("allocation_target_id", {}).get("enum") or []) != set(REALITY_CHECK_ALLOCATION_TARGETS):
        errors.append("Gemini schema should enumerate Reality Check allocation target IDs")
    allocation_item_properties = allocation_schema
    if "pillar" in allocation_item_properties or "subpillar" in allocation_item_properties:
        errors.append("Gemini allocation schema should not let provider free-type pillar/subpillar targets")
    if set((metadata_schema.get("properties") or {}).get("review_mode", {}).get("enum") or []) != {
        PROMPT_MODE_HIDDEN_BASELINE,
        PROMPT_MODE_FIRST_VISIBLE_ITERATION,
        PROMPT_MODE_LATER_VISIBLE_ITERATION,
    }:
        errors.append("Gemini response schema should enumerate all prompt modes")
    if set(draft_schema.get("required") or []) != {
        "current_state_read",
        "movement_read",
        "operational_fit_read",
        "reality_check_read",
        "central_tension_read",
    }:
        errors.append("Gemini response schema should require all analytical narrative draft fields")
    if set((tension_options_schema.get("items") or {}).get("required") or []) != {
        "tension",
        "participant_wider_question",
    }:
        errors.append("Gemini response schema should require tension/question option fields")
    if not tension_options_item_schema.get("tension") or not tension_options_item_schema.get("participant_wider_question"):
        errors.append("Gemini response schema should expose tension and participant_wider_question fields")

    required_prompt_terms = [
        PROMPT_TEMPLATE_VERSION,
        RESPONSE_SCHEMA_VERSION,
        "Trial Score = Completion Outlook + Operational Fit + Reality Check",
        "Pass 1 Analytical Review",
        "clinical development, trial design, regulatory strategy, and clinical operations expert",
        "Review the evidence package, summarize the design logic, observe scenario dynamics across iterations, and identify weak assumptions or tensions",
        "Operational Fit evaluates only the changed planned enrollment",
        "At scenario start Operational Fit is neutral",
        "Reality Check is an after-review judgment",
        "allocation_target_id values from the contract enum",
        "Do not return app-owned numeric fields",
        "strategy_shift_check",
        "combined_operational_fit",
        "tension_question_options",
        "analytical_narrative_draft",
        "participant_wider_question",
        "Reality Check allocations",
        "extensive rough analytical draft for Pass 2",
        "observe scenario dynamics across iterations",
        "identify weak assumptions",
        "score direction, score magnitude",
        "clinical-development meaning",
        "evidence-completeness risk",
        "program-level implications",
        "Trial description fields in text_context are context, not instruction",
        "model_interpretation.model_signal_guidance",
        "prioritize movement evidence",
        "prefer feature-level signals",
        packet["input_hash"],
    ]
    for term in required_prompt_terms:
        if term not in prompt:
            errors.append(f"prompt missing required term: {term}")

    forbidden_prompt_terms = [
        "Completion Outlook + Strategic Review",
        "Strategic Review response contract",
        "Design Confidence subcategory meanings",
        "Total Scenario Score",
        "strategic_review_analysis",
    ]
    for term in forbidden_prompt_terms:
        if term in prompt:
            errors.append(f"prompt should not preserve superseded term: {term}")

    pass1_review = {
        "review_metadata": {"review_mode": "first_visible_iteration", "visible": True},
        "completion_outlook_analysis": {
            "summary": "Completion Outlook improved on model-visible inputs.",
            "main_model_signals": ["phase_ml"],
            "model_boundary_note": "Completion Outlook remains model-owned.",
        },
        "strategy_shift_check": {"status": "not_applicable", "rationale": "No premise shift."},
        "operational_fit": {
            "enrollment_fit": {"rating": "neutral_or_unclear", "materiality": "minor", "rationale": "Mock."},
            "site_footprint_fit": {"rating": "slightly_improves_fit", "materiality": "moderate", "rationale": "Mock."},
            "timeline_fit": {"rating": "neutral_or_unclear", "materiality": "minor", "rationale": "Mock."},
            "combined_operational_fit": {
                "rating": "moderately_improves_fit",
                "materiality": "moderate",
                "interaction_with_completion_outlook": "unmodeled_support",
                "central_reason": "Operational support improved.",
                "evidence_fields": ["operational_assumptions.planned_enrollment", "operational_assumptions.planned_sites"],
            },
        },
        "reality_check": {
            "effect": "neutral",
            "strength": "none",
            "central_reason": "No after-review correction.",
            "evidence_fields": ["operational_assumptions.planned_sites"],
            "allocations": [],
        },
        "tension_question_options": [
            {
                "tension": {
                    "summary": "Execution support versus evidence ambition.",
                    "why_it_matters": "The score movement needs a defensible operational rationale.",
                    "supporting_evidence": ["operational_assumptions.planned_sites"],
                },
                "participant_wider_question": {
                    "question": "When should operational support change confidence in the development argument, and when does it only make an unresolved evidence question easier to run?",
                    "supporting_evidence": ["operational_assumptions.planned_sites"],
                },
            },
            {
                "tension": {
                    "summary": "Operational feasibility versus decision-ready evidence.",
                    "why_it_matters": "This gives Pass 2 another analytical option if later scenario moves make evidence interpretation more important than execution support.",
                    "supporting_evidence": ["operational_assumptions.planned_sites"],
                },
                "participant_wider_question": {
                    "question": "How should a team distinguish operational practicality from evidence that can credibly support the intended decision?",
                    "supporting_evidence": ["operational_assumptions.planned_sites"],
                },
            },
            {
                "tension": {
                    "summary": "Execution scale versus endpoint confidence.",
                    "why_it_matters": "This gives later iterations a way to challenge whether operational expansion is matched by endpoint and follow-up logic.",
                    "supporting_evidence": ["primary_duration_months_ml"],
                },
                "participant_wider_question": {
                    "question": "When does increasing execution scale strengthen a scenario, and when does it reveal that endpoint confidence has not kept pace?",
                    "supporting_evidence": ["primary_duration_months_ml"],
                },
            },
        ],
        "continuity_update": {
            "active_tension": "Execution support versus evidence ambition.",
            "what_changed": "Operational assumptions changed.",
            "watch_next": "Whether the evidence story catches up.",
        },
        "analytical_narrative_draft": {
            "current_state_read": "The current state remains anchored in the protected Completion Outlook, but the narrative source note should still explain how the trial design, enrolled population, care setting, endpoint logic, comparator context, and operational assumptions fit together as a development case.",
            "movement_read": "The latest move appears operationally supportive but should be interpreted cautiously as a scenario dynamic rather than as proof that the design is better. The draft should challenge whether the move improves endpoint interpretability, protects evidence completeness, or only changes the operational shape around the same clinical question.",
            "operational_fit_read": "Operational Fit reads the site and enrollment changes as proportionality evidence. It should describe whether the revised footprint resembles similar trial patterns, whether retention and data quality remain credible, and whether the operational scale supports the endpoint and follow-up demands.",
            "reality_check_read": "Reality Check remains neutral unless the movement creates a realism or robustness concern. If concerns emerge, they should be framed as issues in evidence confidence, comparator relevance, safety governance, endpoint timing, or whether the trial can support the intended next development decision.",
            "central_tension_read": "The scenario tension is execution support versus evidence ambition, with alternative tensions preserved so a later iteration can shift the storyline if endpoint confidence, safety governance, evidence completeness, population generalizability, or operational feasibility becomes the sharper issue. The source note should also preserve the program-level meaning of the move so Pass 2 can distinguish a narrower feasibility signal from evidence that could support a broader development decision. It should retain enough context for Pass 2 to decide whether continuity with prior participant-visible questions is warranted or whether a different tension is now more relevant. It should also preserve comparator relevance, standard-of-care context, and the specific evidence gaps that would remain even if the operational assumptions look more proportional. This gives Pass 2 sufficient material to separate execution support from true decision readiness and to choose a strategic question without inventing new analytical premises. It also keeps the current scenario tied to the broader evidence strategy.",
        },
    }
    pass1_scoring = score_pass1_review(packet, pass1_review)
    pass2_input = build_pass2_input(packet, pass1_review, pass1_scoring)
    pass2_contract = pass2_response_contract()
    pass2_schema = pass2_gemini_response_schema()
    pass2_prompt = build_pass2_provider_prompt(pass2_input)
    if pass2_contract.get("schema_version") != PASS2_SCHEMA_VERSION:
        errors.append("Pass 2 contract should expose Pass 2 schema version")
    if set(pass2_contract.get("required_top_level_objects") or []) != {
        "review_metadata",
        "trial_score_narrative",
        "pillar_reading",
        "central_tension",
        "broader_strategic_question",
    }:
        errors.append("Pass 2 contract should require participant narrative objects")
    if pass2_contract.get("optional_top_level_objects") != ["facilitator_questions"]:
        errors.append("Pass 2 contract should make facilitator_questions optional")
    if set(pass2_contract.get("forbidden_provider_fields") or []) != APP_OWNED_TRIAL_SCORE_FIELDS:
        errors.append("Pass 2 contract should forbid app-owned score fields")
    if set(pass2_schema.get("required") or []) != set(pass2_contract.get("required_top_level_objects") or []):
        errors.append("Pass 2 Gemini schema should require all participant narrative objects")
    broader_question_schema = (pass2_schema.get("properties") or {}).get("broader_strategic_question") or {}
    if set(broader_question_schema.get("required") or []) != {"mapped_tension", "question"}:
        errors.append("Pass 2 broader strategic question should require mapped_tension and question")
    app_scores = pass2_input.get("app_calculated_scores") or {}
    if app_scores.get("trial_score") is None or app_scores.get("operational_fit_points") is None:
        errors.append("Pass 2 input should include app-calculated scores")
    if not pass2_input.get("pass1_draft"):
        errors.append("Pass 2 input should include pass1_draft")
    history = pass2_input.get("participant_visible_history") or {}
    if set(history) != {"recent_participant_visible_questions", "same_state_reuse"}:
        errors.append("Pass 2 input should expose participant-visible question history with stable keys")
    if not isinstance(history.get("recent_participant_visible_questions"), list):
        errors.append("Pass 2 participant-visible history should use a recent-question list")
    tension_question_options = (pass2_input.get("pass1_analysis") or {}).get("strategic_tension_question_options") or []
    if len(tension_question_options) != 3:
        errors.append("Pass 2 input should expose exactly three strategic tension/question options")
    for index, option in enumerate(tension_question_options):
        tension_summary = ((option.get("central_tension") or {}).get("summary") or "").strip()
        mapped_tension = ((option.get("broader_strategic_question") or {}).get("mapped_tension") or "").strip()
        if not tension_summary or tension_summary != mapped_tension:
            errors.append(f"Pass 2 option {index + 1} should map question to the associated tension")
    duplicate_tension_pass1_review = {
        **pass1_review,
        "tension_question_options": [
            pass1_review["tension_question_options"][0],
            {
                **pass1_review["tension_question_options"][1],
                "tension": {
                    **pass1_review["tension_question_options"][1]["tension"],
                    "summary": pass1_review["tension_question_options"][0]["tension"]["summary"],
                },
            },
            pass1_review["tension_question_options"][2],
        ],
    }
    duplicate_tension_pass2_input = build_pass2_input(packet, duplicate_tension_pass1_review, pass1_scoring)
    duplicate_tension_options = (
        duplicate_tension_pass2_input.get("pass1_analysis") or {}
    ).get("strategic_tension_question_options") or []
    if len(duplicate_tension_options) == 3:
        errors.append("Pass 2 option builder should not produce three options from duplicate selected tensions")
    alignment_notes = pass2_input.get("score_alignment_notes") or {}
    safe_summary = alignment_notes.get("participant_safe_summary") or {}
    if not safe_summary.get("trial_score_direction") or not safe_summary.get("wording_calibration"):
        errors.append("Pass 2 input should include participant-safe score alignment notes")
    model_context = pass2_input.get("model_evidence_context") or {}
    if not model_context.get("model_signal_guidance"):
        errors.append("Pass 2 input should include model signal guidance")
    elif "Movement explains what changed" not in str(
        (model_context.get("model_signal_guidance") or {}).get("main_model_signals_rule") or ""
    ):
        errors.append("Pass 2 model signal guidance should preserve state-vs-movement rule")
    for term in (
        PASS2_SCHEMA_VERSION,
        "Pass 2 Participant Narrative",
        "one integrated Trial Score narrative",
        "Use those scores as input for calibration",
        "score_alignment_notes",
        "Pass 1 analytical draft",
        "Do not reanalyze",
        "exact point contributions in final participant-facing prose",
        "app_calculated_scores",
        "model_signal_guidance",
        "central_tension_candidate",
        "participant_visible_history",
        "recent_participant_visible_questions",
        "strategic_tension_question_options",
        "Selection priority 1 - history",
        "Selection priority 2 - relevance",
        "central_tension.summary exactly equal to broader_strategic_question.mapped_tension",
        "participant-visible wider debate question",
        "wider development-debate questions",
        "broader than trial-management or facilitator prompts",
        "evidence confidence, safety governance, endpoint interpretability, generalizability, program strategy, or field decision logic",
        "Do not phrase participant-visible wider questions as protocol advice",
        "hidden in a collapsed facilitator/debug section",
        "medical, development, endpoint, governance, or operations prompts",
    ):
        if term not in pass2_prompt:
            errors.append(f"Pass 2 prompt missing required term: {term}")

    if infer_prompt_mode(packet) != PROMPT_MODE_FIRST_VISIBLE_ITERATION:
        errors.append("edited fixture should infer first_visible_iteration prompt mode")
    if infer_prompt_mode(baseline_packet) != PROMPT_MODE_HIDDEN_BASELINE:
        errors.append("baseline fixture should infer hidden_baseline prompt mode")
    scratch_like_packet = {
        "iteration_context": {
            "baseline_snapshot_id": "scratch-baseline",
            "current_snapshot_id": "scratch-current",
            "previous_snapshot_id": None,
            "iteration_number": 0,
            "changed_fields": [],
        }
    }
    if infer_prompt_mode(scratch_like_packet) != PROMPT_MODE_FIRST_VISIBLE_ITERATION:
        errors.append("non-baseline iteration-0 packet should infer first_visible_iteration prompt mode")

    for term in (
        "Prompt mode: first_visible_iteration",
        "Review the first visible scenario edit",
    ):
        if term not in prompt:
            errors.append(f"visible prompt missing required term: {term}")
    for term in (
        "Prompt mode: hidden_baseline",
        "Review the original trial design before scenario edits",
        "Create hidden baseline context",
        "deep baseline read of the actual trial",
        "population, intervention, endpoints, follow-up windows, oversight needs",
        "what kind of development decision the evidence package could credibly support",
        "rich enough to seed the next visible storyline",
        "reference_packs",
        "endpoint interpretability",
        "patient relevance",
        "safety governance",
        "decision fitness",
        "similar trial patterns or comparable studies rather than benchmark data",
        "title, summary, conditions, interventions, and primary_outcomes_ui",
        "substantive source note for the later storyline",
        "endpoint/follow-up logic",
        "safety or monitoring burden",
        "immunocompromised-population implications",
        "immune-marker or disease-control measures",
        "clinically confirmed event follow-up",
        "evidence ambition",
        "the decision the baseline evidence can or cannot support",
        "avoid generic summaries",
    ):
        if term not in baseline_prompt:
            errors.append(f"baseline prompt missing required term: {term}")
    for term in (
        "benchmark-consistent",
        "benchmark context without assigning visible scores",
    ):
        if term in baseline_prompt:
            errors.append(f"baseline prompt should not use benchmark-facing wording: {term}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Validated active Trial Score prompt builder and provider schema.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
