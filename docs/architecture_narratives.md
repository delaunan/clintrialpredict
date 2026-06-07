# Serious-Game Narrative Architecture

## Document Role

This file owns the future serious-game narrative layer: LLM commentary, Quality Review, Quality Adjustment, Final Candidate Score, facilitator/participant outputs, and narrative payload contracts.

It should not own the existing XGBoost Completion Score, SHAP impact mechanics, or simulation UI state; use `docs/architecture_edit.md` for those. It should consume operational benchmark metadata from `docs/architecture_estimation.md` rather than redefining benchmark construction.

Efficient update rule: change this file when narrative inputs/outputs, LLM contracts, Quality Review scoring, or participant/facilitator interpretation rules change. Do not implement or imply changes to XGBoost, SHAP, calibration, or operational benchmark construction here.

## 1. Purpose Of The Narrative Architecture

This document defines the staged design for adding a serious-game narrative layer around single-trial simulation in ClinTrialPredict. The first implementation artifact is limited to static contract fixtures; no API, UI, model, taxonomy, prediction, or parity behavior is changed by this document.

The current edit/simulation workflow remains the foundation. A facilitator selects an existing trial, participants adjust structured Trial Features, and the application calls the existing prediction flow to produce a completion score with SHAP-derived impact decomposition.

The narrative layer exists because completion likelihood alone is not enough for a serious-game discussion. Some changes may raise completion likelihood by making a trial easier to complete while reducing scientific rigor, evidence value, endpoint interpretability, population relevance, governance quality, or strategic defensibility. Other changes may lower completion likelihood while making the design more robust or more relevant.

The future layer should help participants reason about this trade-off without giving direct optimization instructions. It should interpret score movement, surface design trade-offs, and challenge teams to defend their choices.

## 2. Core Scoring Boundary

The LLM layer is separate from the existing prediction system. The serious-game score stack has three layers:

1. `Completion Score`: the existing XGBoost, SHAP, therapeutic-area calibrated score from `/predict`, shown in points from 0 to 100.
2. `Quality Review`: a constrained LLM structured reviewer that evaluates coherence, scientific rigor, operational feasibility, text consistency, and change integrity.
3. `Final Candidate Score`: a deterministic application calculation: `Completion Score + Quality Adjustment`.

The LLM must not generate the final score. The LLM returns structured ratings, evidence fields, narrative, and continuity fields. The application then performs two deterministic calculations:

1. Convert validated Quality Review ratings into a bounded `Quality Adjustment`.
2. Add the Quality Adjustment to the XGBoost `Completion Score` to calculate `Final Candidate Score`.

```text
quality_adjustment = clamp(app_mapped_review_points, -10, +10)

final_candidate_score = clamp(
    completion_score + quality_adjustment,
    0,
    100
)
```

The initial V1 range should not use `-20` to `+20`, because that would allow the LLM-derived layer to dominate the XGBoost Completion Score. Future calibration can revisit the mapping after playtesting, but the total adjustment should remain modest.

Example:

- Completion Score: `72`
- Quality Adjustment: `+4`
- Final Candidate Score: `76`

Interpretation:

- The Completion Score remains the modelled likelihood of completion.
- The Quality Adjustment is a small serious-game modifier for design defensibility and quality of choices.
- The Final Candidate Score can recognize a risky but well-strengthened design without replacing or rewriting XGBoost.
- A trial below `50` on Completion Score can improve modestly if the design choices meaningfully mitigate risk, but the adjustment must remain bounded.

Terminology:

- `Completion Score` = modelled likelihood of completion.
- `Quality Review` = participant-facing narrative explanation and structured LLM review of design coherence, rigor, operational fit, text consistency, and change integrity.
- `Quality Adjustment` = bounded application-calculated point bonus or penalty.
- `Final Candidate Score` = Completion Score plus Quality Adjustment.

In plain scoring terms, `Final Candidate Score = Completion Score + Quality Adjustment`, with application-level bounds applied.

The application, not the LLM, calculates the Quality Adjustment and Final Candidate Score. The LLM must never modify the XGBoost completion score, SHAP values, pillar impacts, therapeutic-area calibration, prediction pipeline, or audit/demo parity behavior.

Core boundary:

- The LLM never modifies XGBoost.
- The LLM never modifies SHAP values.
- The LLM never modifies therapeutic-area calibration.
- The LLM never rewrites the prediction score.
- The LLM returns structured Quality Review ratings, evidence fields, explanation, and continuity fields.
- The application maps validated review ratings into Quality Adjustment and Final Candidate Score.

## 3. Current Technical Foundation

The current architecture provides the data needed for a later narrative layer:

- `frontend/app.py` routes `APP_VARIANT=trial_simulator` into the isolated simulation view.
- `frontend/views/trial_simulator.py` owns the Simulation Mode UI, structured Trial Features, pending-change tracking, latest prediction snapshots, prediction history, and score/charts rendering.
- `api/main.py` keeps `/predict` backward compatible for audit mode and adds `simulation_mode: true` live scoring through the production pipeline.
- `api/main.py` returns `score`, `pillar_impacts`, `subcat_impacts`, `mode`, and live probability for simulation calls.
- `models/taxonomy_01.json` defines the structured feature labels, options, mappings, pillars, subgroups, and encodings used by the UI and API.
- `src/prep/pipeline.py` defines the model-facing preprocessing registry and ColumnTransformer behavior for ordinal, target-encoded, and numeric features.
- `docs/architecture_edit.md` records the current simulation contract, including baseline snapshot behavior, pending-change behavior, and parity requirements.

The future narrative layer should consume these outputs and snapshots. It should not duplicate or reinterpret the model pipeline.

## 4. Existing-Study Mode User Experience

Existing-study mode is the current implementation priority.

Intended flow:

1. Facilitator selects an existing trial.
2. User opens Simulation Mode.
3. The system loads or creates a hidden baseline review object from the original selected-trial state and available baseline context during Simulation Mode initialization.
4. Participants do not immediately see a detailed LLM narrative. Showing rich interpretation before the exercise could reveal too much guidance.
5. Participants change structured dropdown fields and, later, optional short text fields.
6. Participants click `Predict Trial Completion`.
7. XGBoost returns the new completion score, pillar impacts, and impact decomposition through the existing prediction path.
8. The narrative layer receives baseline context, previous prediction, current prediction, changed fields, score deltas, SHAP/pillar movement, operational benchmark metadata, text context, and prior storyline memory.
9. The application validates the structured Quality Review, calculates Quality Adjustment and Final Candidate Score, and stores a compact storyline update.
10. The UI displays the Quality Review below or near the score and charts.
11. Participants iterate.

The visible narrative should usually compare the current prediction against the previous prediction. Internally, the LLM should also receive enough baseline and path memory to avoid contradicting prior feedback.

### Baseline Review Object

For existing-study mode, generate a baseline review once per selected study/version and store it durably, keyed by stable trial identity plus baseline input hash, prompt version, rubric version, and provider/model namespace. The first team opening Simulation Mode for a trial may create this baseline if it does not exist. Later teams opening the same trial should load the same baseline review and compact memory, so all teams start from a consistent original-trial interpretation.

The baseline review is initialized when Simulation Mode opens, not delayed until the first participant prediction. It remains hidden from participants at the start of the exercise, but it should be passed into later review calls so the LLM can evaluate how the design path evolved from the original trial.

The baseline review context passed to later LLM calls is qualitative-only. It may include baseline strengths, concerns, consistency flags, participant-review text, continuity fields, and compact memory. It must not expose hidden baseline `Quality Adjustment`, `Final Candidate Score`, or a hidden numeric quality score to later prompt logic as a prior visible score.

The hidden baseline review should include two qualitative analyses. First, it should interpret the prerecorded Completion Score when baseline decomposition is available. The provider should use the baseline structured features, text context, Completion Score, pillar impacts, and feature/subcategory impacts to summarize why the original trial appears completion-like or risky in clinical trial / pharma development language. This is baseline reasoning context, not a visible participant score and not a new XGBoost calculation. If only a registry score is available without pillar or feature decomposition, the baseline review should explicitly treat the score interpretation as lower-detail and avoid inventing missing driver analysis. Second, it should produce a hidden baseline Quality Review analysis using the same quality rubric as later visible reviews: design coherence, scientific rigor, population/strategy fit, operational plausibility, text consistency, and baseline concerns. This hidden quality analysis should create baseline strengths, baseline concerns, consistency flags, and compact memory, but it must not expose a visible baseline Quality Adjustment, Final Candidate Score, or hidden numeric quality score to the participant.

The stored baseline review should include:

- Baseline score/model snapshot when available. For existing audit trials, this should include precomputed Completion Score decomposition from saved SHAP artifacts, not a live XGBoost prediction rerun.
- Baseline structured Trial Features.
- Baseline operational assumptions and source metadata.
- Baseline study summary and endpoint text, when available.
- Baseline conditions, interventions, and primary outcomes text, when available.
- Baseline strengths.
- Baseline concerns.
- Baseline text/structured consistency flags.
- Baseline compact memory summary.

Participant-facing rule: do not show the baseline Quality Review, Quality Adjustment, Final Candidate Score, or baseline narrative comments by default. Participants may see the original XGBoost Completion Score and model drivers if that is part of the trial-opening experience. The hidden baseline review exists to enrich the first and later visible reviews, not to expose a quality score before teams make their first scenario choice. If hidden baseline review initialization fails, Simulation Mode should still open; later visible Quality Review calls can proceed without baseline review context or retry when a durable provider/store exists.

Later prediction reviews should receive:

- Baseline review.
- Previous iteration review.
- Compact storyline memory.
- Current delta packet.
- Current snapshot.

For the first visible review after a team edit, the narrative may compare qualitatively against the original trial baseline, but it must not refer to hidden baseline quality numbers as if participants had already seen them. For example, it may say that the model Completion Score decreased while the current design appears less penalized on coherence than the original baseline context. It should not say that the team "improved the score" when the visible Completion Score or visible Final Candidate Score declined.

## 5. Scratch Mode User Experience, Future Version

Scratch mode is a future variant and is not the current implementation priority.

In scratch mode, users start without an existing selected study:

- The first prediction is visible.
- The first narrative becomes the baseline review.
- Later predictions compare mainly against the previous prediction.
- The system still retains memory of the full design path so the narrative can distinguish newly introduced issues from recurring concerns.

Scratch mode may require additional field completeness rules because there is no original trial record to anchor interpretation.

## 6. Participant-Facing Outputs

After each prediction, the participant view should display:

- `Final Candidate Score`, shown in the main gauge when adjusted view is enabled.
- `Completion Score` as a component score.
- `Quality Adjustment` as a component value.
- A short `Operational Assumptions` note.
- Concise `Quality Review`.

Participant narrative sections:

- `What changed`
- `Why the completion score may have moved`
- `What the design may have gained`
- `What the design may have sacrificed`
- `Operational coherence note`
- `One question for the team to debate`

Suggested participant UI wording:

```text
Operational assumptions:
Enrollment: 600 patients, ambitious versus similar trials
Sites: 30 sites, typical versus similar trials
Duration: 42.0 months, long versus similar trials
```

Planned Enrollment, Planned Sites, and Planned Duration are active now. They use the same operational-assumption visual pattern and metadata behavior.

For opening UI labels, use the same compact source convention across operational fields:

- Direct AACT-backed opening values have no suffix.
- System-filled benchmark/default values show `(est.)`.
- Participant edits remove the suffix, while metadata still stores `source = user_scenario`.

For duration specifically, direct AACT-backed values include usable completed actual total duration and usable active/non-stopped estimated total duration. Stopped/interrupted dates are lower-bound or floor context, not trusted planned-duration targets by themselves. If benchmark/default logic supplies the editable opening value, the UI label should show `(est.)`.

The main participant UI should not overexpose benchmark percentiles unless needed. The benchmark can be shown lightly as `below benchmark`, `typical`, `ambitious`, or `above benchmark high`. Detailed benchmark statistics can be reserved for development or facilitator view.

The narrative must use conditional and analytical language:

- "may suggest"
- "could indicate"
- "might have"
- "would need support from the selected trial profile"
- "one possible interpretation is"

The participant narrative must avoid direct instructions such as:

- "change this field"
- "add a DMC"
- "remove elderly patients"
- "this is the best choice"

The goal is to challenge the team, not solve the exercise for them.

## 7. Facilitator-Facing Outputs

A future optional facilitator view may expose more explicit analysis than the participant view.

Potential facilitator fields:

- Shortcut risk level.
- Main suspected shortcut, if any.
- Main gain.
- Main potential sacrifice.
- Coherence concern.
- Suggested facilitator probe.
- Whether the change appears legitimate, shortcut-driven, unclear, or strategically defensible.
- Whether a previously raised concern was resolved, worsened, or unchanged.

The facilitator view may be more direct, but it must still not override the model score or present itself as clinical truth. It should support discussion facilitation, not adjudicate trial validity.

## 8. Quality Review Rubric

The LLM should apply an internal rubric across at least these dimensions:

- Development-question fit: whether phase, regulatory intent, population, intervention model, comparator, and endpoints appear aligned to the implied development question.
- Population relevance: whether the population remains clinically and strategically relevant rather than merely easier to enroll or complete.
- Endpoint and estimand coherence: whether endpoint rigor, endpoint structure, duration, comparator, and trial architecture appear mutually consistent.
- Scientific rigor: whether the design preserves interpretability, biological plausibility, and evidentiary value.
- Operational feasibility: whether the design appears proportionate and executable without becoming trivially easy at the expense of value.
- Change integrity: whether participants genuinely improved the design or mainly gamed completion likelihood.
- Text consistency: whether editable text fields such as study summary and primary endpoint description remain consistent with the structured design.
- Cross-pillar coherence: whether choices that are reasonable in isolation remain coherent in combination across therapeutic context, scientific challenge, patient profile, execution framework, operational assumptions, and text.

The Quality Review should reflect both current design coherence and change integrity:

- Current design coherence: whether the revised design is coherent and defensible now.
- Change integrity: whether the path from baseline to current design appears like meaningful improvement, acceptable simplification, or score-seeking shortcut behavior.

For V1, do not make a numeric `Coherence Score` or `Quality Score` the primary user-facing concept. These can sound falsely precise and may be mistaken for competing prediction models. The user-facing concepts should be `Quality Review`, `Quality Adjustment`, and `Final Candidate Score`.

The Quality Review is bidirectional. It should recognize:

- Strong endpoint and comparator logic.
- Coherent population definition.
- Proportional safety oversight.
- Appropriate biomarker strategy.
- Enrollment assumptions supported by the design.
- Operational choices that mitigate risk without trivializing the trial.
- Difficult but strategically defensible designs.

It should penalize:

- Score-seeking simplification.
- Weakened endpoint rigor.
- Weakened comparator logic.
- Population narrowing that reduces relevance.
- Unsupported enrollment assumptions.
- Design changes that make completion easier but reduce evidence value.

Principle:

```text
One weak feature creates a discussion point.
Several weak or conflicting features create a Quality Adjustment penalty.
A difficult design can receive a positive Quality Adjustment if the participant strengthens it in a coherent and defensible way.
```

Recommended V1 review ratings:

- `strong`: coherent, rigorous, and strategically defensible in the current context.
- `acceptable`: balanced, with strengths outweighing trade-offs.
- `weak`: unresolved weakness or simplification that needs discussion.
- `conflicting`: meaningful evidence, feasibility, text-consistency, or change-integrity concern.

For change integrity, use:

- `improved`: the path appears to strengthen the design.
- `neutral`: the change appears broadly neutral for quality.
- `simplified`: the change simplifies execution but may reduce evidence value.
- `potential_shortcut`: the change appears score-seeking or weakens defensibility.

For text consistency, use:

- `consistent`.
- `minor_tension`.
- `material_tension`.
- `contradiction`.

The application should map these validated ratings to a bounded Quality Adjustment. A concern should affect the Quality Adjustment only when the LLM provides supporting `evidence_fields`; otherwise it can appear in the narrative but should not move the score.

### Quality Assessment Pillars For Visuals

The internal Quality Review rubric can be more detailed than the user-facing plot structure. For visual display, V1 should use three Quality Assessment pillars. These pillars are organized around the narrative jobs the review must perform:

1. `Evidence Coherence`: does the revised design still produce interpretable, decision-useful evidence?
2. `Population & Strategy Fit`: does the selected population and strategic intent still match the disease setting and study purpose?
3. `Execution Plausibility`: are the operational assumptions and implementation choices credible for this design, and does the change path look defensible rather than shortcut-driven?

Recommended V1 Quality Assessment hierarchy:

```text
Quality Assessment
├── Evidence Coherence
│   ├── Endpoint & Comparator Fit
│   └── Scientific Rigor
├── Population & Strategy Fit
│   ├── Population Relevance
│   └── Development Fit
└── Execution Plausibility
    ├── Operational Scale Fit
    └── Change Integrity
```

Internal rubric-to-visual mapping:

```text
Evidence Coherence =
    endpoint_and_comparator_logic
  + scientific_rigor
  + endpoint-related text consistency

Population & Strategy Fit =
    development_question_fit
  + population_relevance
  + study-summary / criteria consistency

Execution Plausibility =
    operational_scale_fit
  + change_integrity
  + operational / intervention consistency
```

`Text consistency` should not be a standalone visual pillar in V1. It should be routed to the Quality Assessment pillar affected by the inconsistency:

- Endpoint text conflict -> `Evidence Coherence`.
- Study-summary or strategic-intent conflict -> `Population & Strategy Fit`.
- Intervention complexity or operational text conflict -> `Execution Plausibility` or `Evidence Coherence`, depending on the evidence fields.

Operational assumptions should remain outside the XGBoost `Execution Framework` contribution. They may appear in the Simulation Mode editing area near Execution Framework for workflow reasons, but in adjusted-score plots they belong under:

```text
Quality Assessment -> Execution Plausibility -> Operational Scale Fit
```

This keeps model-facing `Execution Framework` impacts distinct from non-XGBoost operational quality review.

For reproducibility, the application should derive these three plotted Quality Assessment pillars from validated `quality_review_domains`. The LLM may provide suggested grouping language, but the app owns the final pillar/subcategory point mapping used in charts and score math.

Recommended V1 point caps before total clamping:

```text
Each Quality Assessment pillar: normally -4 to +3 points.
Each Quality Assessment subcategory: normally -3 to +2 points.
Total Quality Adjustment: clamp to -10 to +10.
```

This prevents one quality dimension from dominating the Final Candidate Score and keeps the Quality Adjustment modest relative to the XGBoost Completion Score.

## 9. Shortcut Detection Concept

A shortcut is not simply a change that increases the completion score. A higher Completion Score can reflect a more robust, better-governed, better-aligned, or lower-risk development pattern, but it can also reflect simplification or loss of evidence value when scientific challenge, endpoint rigor, comparator credibility, population relevance, or interpretability is reduced. For example, a Data Monitoring Committee may increase operational oversight and completion confidence in one context while adding complexity or risk in another. A shortcut is a change that increases completion likelihood while potentially weakening evidence value, scientific rigor, population relevance, endpoint interpretability, or strategic defensibility.

Examples:

- Narrowing the population in a way that may improve recruitment or completion but reduce relevance.
- Reducing endpoint rigor in a way that may simplify execution but weaken interpretability.
- Removing biomarker stratification where it may be mechanistically important.
- Simplifying comparator or control structure in a way that may reduce evidentiary value.
- Adding oversight features only because they improve the score, without proportional clinical or operational justification.
- Shortening trial duration in a way that may make completion easier but may not fit the endpoint or disease course.

The LLM should use changed fields, SHAP/feature deltas, pillar deltas, and previous narrative memory to detect potential shortcut patterns. The output should frame shortcut concerns as hypotheses for discussion, not definitive findings.

## 10. Operational Assumptions MVP

For estimation v1, `Planned Enrollment`, `Planned Sites`, and `Planned Duration` are active operational assumptions. Countries, cost, market, and downstream commitment layers remain postponed.

Purpose:

```text
Operational assumptions are deterministic scenario stress tests.
They help the narrative judge whether selected enrollment, site count, and total duration assumptions are coherent with the trial profile.
They do not enter the XGBoost model.
They do not directly change the Completion Score.
They feed the future Quality Review only.
```

Active fields:

```text
operational_assumptions.planned_enrollment
operational_assumptions.planned_sites
operational_assumptions.planned_duration_months
```

Source priorities, cohort hierarchy, defaulting logic, and benchmark status calculation are owned by `docs/architecture_estimation.md`. The narrative layer consumes that metadata; it does not rebuild or reinterpret benchmark construction.

The user does not write a free-text justification for planned enrollment, planned sites, or planned duration in the platform. Instead, the Quality Review assesses whether the operational assumptions are coherent with the current structured design, text context, and benchmark metadata.

System-filled benchmark/default values are neutral. They should not create a positive Quality Adjustment simply because they sit inside a benchmark range. They become evaluated scenario assumptions when the user keeps them as the current assumption for a prediction snapshot or actively edits them.

Generic operational interpretation rule for enrollment, sites, and duration:

```text
Operational benchmark status alone is context, not a penalty.
Operational benchmark status plus conflicting design context can affect Execution Plausibility.
System-filled benchmark/default values are neutral unless retained as the current scenario or edited by the participant.
Participant-edited values are scenario choices and can be evaluated against both benchmark metadata and the current structured design.
```

### Operational Benchmark Context

For V1, the narrative layer should consume operational benchmark metadata, not require a separate operational support/conflict rule engine. The LLM should receive benchmark status, source metadata, confidence flags, and relevant structured design fields, then evaluate whether the operational assumption is coherent with the current design.

Examples:

```text
Common adult Phase III disease + 1,200 patients:
high but potentially supported.

Rare pediatric Phase II gene therapy + 1,200 patients:
above benchmark high and weakly supported by the design.

Short duration + short-term endpoint:
potentially coherent.

Short duration + endpoint text implying long-term clinical outcome:
possible Evidence Coherence concern.
```

Explicit support/conflict signals are optional future derived fields. If implemented later, their methodology belongs in `docs/architecture_estimation.md` or a future calibration note, and the signals must be deterministic, auditable, and not invented by the LLM. Until then, benchmark metadata plus structured design context is sufficient for V1 Quality Review.

### Bounded Operational Effect

Operational assumptions should not dominate the Quality Adjustment. Planned Enrollment, Planned Sites, and Planned Duration together should normally affect only:

```text
Quality Assessment -> Execution Plausibility -> Operational Scale Fit
```

They remain bounded by the Quality Assessment pillar/subcategory caps. Operational benchmark status alone is context, not a penalty; it becomes score-relevant only when combined with coherent supporting or conflicting design evidence.

## 11. Input Payload Architecture

The future LLM input object should be assembled after the existing prediction response has been received and after the application has created the latest prediction snapshot.

The payload must preserve the separation between the existing prediction system and the LLM narrative layer. XGBoost/TreeSHAP outputs explain completion-score movement; structured serious-game fields, operational assumptions, and text context define the broader design-reasoning space for the Quality Review.

This is conceptual JSON for planning only, not an implementation contract yet:

```json
{
  "prompt_version": "narratives_v1",
  "rubric_version": "design_coherence_v1",
  "field_dictionary_version": "taxonomy_01_narrative_v1",
  "mode": "existing_study",
  "trial_identity": {
    "nct_id": "...",
    "trial_label": "...",
    "lead_sponsor_canonical": "...",
    "start_year": "..."
  },
  "text_context": {
    "title": "...",
    "summary_ui": "...",
    "conditions_ui": "...",
    "primary_outcomes_ui": "...",
    "interventions_ui": "..."
  },
  "structured_features": {
    "therapeutic_area_ml": "...",
    "gbd_cause_id_3_ml": "...",
    "is_rare_disease_ml": "...",
    "phase_ml": "...",
    "strategic_ambition_ml": "...",
    "target_precedent_ml": "...",
    "target_pathway_class_ml": "...",
    "therapeutic_modality_ml": "...",
    "innovation_tier_ml": "...",
    "intervention_model_ml": "...",
    "primary_purpose_ml": "...",
    "adaptive_design_ml": "...",
    "endpoint_rigor_ml": "...",
    "endpoint_structure_ml": "...",
    "biomarker_stratification_ml": "...",
    "patient_severity_ml": "...",
    "line_of_therapy_ml": "...",
    "gender_ml": "...",
    "healthy_volunteers_ml": "...",
    "adult_ml": "...",
    "child_ml": "...",
    "older_adult_ml": "...",
    "masking_ml": "...",
    "allocation_ml": "...",
    "has_dmc_ml": "...",
    "has_placebo_ml": "...",
    "comparator_benchmark_ml": "...",
    "administration_complexity_ml": "...",
    "number_of_arms_ml": "...",
    "sponsor_tier_ml": "...",
    "primary_duration_months_ml": "..."
  },
  "structured_feature_display_values": {
    "phase_ml": "Phase 3",
    "endpoint_rigor_ml": "Clinical outcome"
  },
  "operational_assumptions": {
    "planned_enrollment": {
      "value": 600,
      "source": "planned_value | final_observed_value | observed_lower_bound | model_default | user_scenario",
      "benchmark_level_used": "phase_indication_rare | phase_ta_rare | phase_ta | phase_only",
      "benchmark_n": 123,
      "benchmark_p25": 120,
      "benchmark_p50": 220,
      "benchmark_p75": 420,
      "benchmark_p90": 750,
      "enrollment_status": "below_benchmark | typical | ambitious | above_benchmark_high",
      "support_level": "supported_by_current_design | partly_supported_by_current_design | weakly_supported_by_current_design",
      "supporting_signals": [],
      "conflicting_signals": [],
      "interpretation_hint": "Enrollment is above the usual benchmark and is only partly supported by the current design choices."
    },
    "planned_sites": {
      "value": 30,
      "source": "completed_registry_facility_count | current_registry_facility_count_proxy | benchmark_default | enrollment_coherent_benchmark_default | user_scenario",
      "benchmark_level_used": "phase_ta_rare | phase_ta | phase_only",
      "benchmark_n": 123,
      "site_count_p50": 20,
      "patients_per_site_p50": 8.5,
      "site_count_status": "below_benchmark | typical | ambitious | above_benchmark_high | not_available",
      "interpretation_hint": "Site count is an operational scenario assumption and does not enter the XGBoost Completion Score."
    },
    "planned_duration_months": {
      "value": 42.0,
      "source": "final_observed_total_duration | estimated_planned_total_duration | actual_completion_noncompleted_status_lag | benchmark_default_with_floors | user_scenario",
      "duration_definition": "start_date_to_completion_date_months",
      "benchmark_level_used": "phase_ta_rare_endpoint_bin",
      "benchmark_n": 159,
      "benchmark_p25": 15.0,
      "benchmark_p50": 30.0,
      "benchmark_p75": 55.0,
      "benchmark_p90": 80.0,
      "duration_status": "shorter_than_benchmark | typical | long | very_long | not_available",
      "planned_primary_completion_months": 18.0,
      "primary_completion_source": "actual_primary_completion | estimated_primary_completion | completed_actual_primary_completion | completed_missing_primary_date_type_duration | same_cohort_benchmark | not_available",
      "primary_completion_n": 120,
      "interpretation_hint": "Duration is an operational scenario assumption and does not enter the XGBoost Completion Score."
    }
  },
  "model_interpretation": {
    "completion_score": 72,
    "previous_completion_score": 65,
    "score_delta": 7,
    "direct_xgboost_shap_fields": [],
    "pillar_impacts": {},
    "pillar_deltas": {},
    "xgboost_impact_changes": [
      {
        "impact_level": "pillar | subcategory",
        "name": "Execution Framework",
        "pillar": "Execution Framework",
        "subcategory": null,
        "baseline_impact": 1.0,
        "previous_impact": 1.0,
        "current_impact": 3.0,
        "delta_from_previous": 2.0,
        "delta_from_baseline": 2.0,
        "changed_since_previous": true,
        "changed_from_baseline": true,
        "direction_from_previous": "increased",
        "direction_from_baseline": "increased"
      }
    ],
    "top_positive_feature_drivers": [],
    "top_negative_feature_drivers": [],
    "top_feature_impact_changes": []
  },
  "review_context": {
    "baseline_review": {
      "input_hash": "...",
      "iteration_id": 0,
      "status": "reviewed",
      "quality_adjustment": null,
      "final_candidate_score": null,
      "quality_numeric_context": "hidden_baseline_qualitative_only",
      "compact_storyline_memory": "..."
    },
    "previous_review": {
      "input_hash": "...",
      "iteration_id": 1,
      "status": "reviewed",
      "quality_adjustment": -2,
      "final_candidate_score": 70,
      "compact_storyline_memory": "..."
    }
  },
  "clarification_context": {
    "user_clarifications": [
      {
        "issue_id": "...",
        "field_id": "...",
        "structured_value": "...",
        "text_signal": "...",
        "explanation": "..."
      }
    ]
  },
  "iteration_context": {
    "baseline_snapshot_id": "...",
    "previous_snapshot_id": "...",
    "current_snapshot_id": "...",
    "iteration_number": 2,
    "changed_fields": [],
    "field_changes": [
      {
        "field": "comparator_benchmark_ml",
        "change_type": "structured_feature",
        "baseline_value": "ACTIVE_MODERN_STANDARD",
        "baseline_label": "Active (Modern Standard)",
        "previous_value": "ACTIVE_MODERN_STANDARD",
        "previous_label": "Active (Modern Standard)",
        "current_value": "PLACEBO",
        "current_label": "Placebo Control",
        "changed_by_user": true
      }
    ],
    "compact_storyline_memory": "..."
  }
}
```

The payload should include all active operational assumptions: `planned_enrollment`, `planned_sites`, and `planned_duration_months`. These assumptions remain outside XGBoost and Completion Score; they feed narrative / Quality Review only.

Operational-assumption values are assembled after the latest prediction snapshot. If the user changes fields that define an operational benchmark cohort, the affected benchmark becomes stale until the next `Predict Trial Completion` action. Enrollment and sites react to the implemented benchmark cohort fields. Duration reacts only to phase, indication, therapeutic area, rare-disease status, and primary endpoint duration bin. Duration does not use therapeutic modality in v1.

The LLM should use operational source metadata, not the compact UI label, to distinguish direct AACT-backed values, system-filled benchmark defaults, and participant scenarios. Direct AACT-backed values are source facts for the selected trial. System-filled values are benchmark assumptions. `user_scenario` values are participant scenario choices even when the visible label has no suffix.

Structured dropdown fields are the primary source of truth. Short text fields are secondary and should be used for coherence checking, contradiction detection, and narrative context rather than as the main source of scoring. Narrative packets should send canonical submitted structured values first and display labels separately, so future providers can reason from readable labels without losing model-facing value provenance.

Missing or brief free-text fields should not be heavily penalized unless they directly contradict structured trial features or make an otherwise important design claim impossible to interpret.

For structured Trial Features, narrative packets should use taxonomy option keys as the canonical value where an option key exists, and should include human-readable labels separately in `structured_feature_display_values`. For example, `endpoint_structure_ml = MULTI_COMPOSITE` should be paired with display label `Multi/Composite`. Numeric fields keep numeric values. The packet should include `field_dictionary_version` so prompts and providers know which taxonomy meanings and option definitions apply without repeating the full field dictionary in every packet. Narrative field meanings must be generated from the production taxonomy source path in `src/prep/pipeline.py`, not patched only into `models/taxonomy_01.json`, so rerunning `notebooks/production_01.ipynb` preserves them.

Narrative packets should keep scenario edit facts separate from model explanation facts:

- `iteration_context.field_changes` records what the participant changed, with baseline, previous, and current values/labels. This covers structured Trial Features, changed text-context fields, and operational assumptions when available.
- `model_interpretation.xgboost_impact_changes` records what moved in the model explanation, with baseline, previous, and current impact values plus deltas. `changed_since_previous` marks local movement since the last prediction; `changed_from_baseline` marks accumulated drift from the original trial. These entries are XGBoost/SHAP explanation facts at pillar or subcategory level, not proof of clinical causality and not necessarily limited to fields the participant directly edited.

The LLM should use `field_changes` to explain what changed in the scenario and `xgboost_impact_changes` to weight which model-explanation movements were material. It should not infer that every model impact movement was directly caused by a single changed field.

Before a new prediction result is committed or displayed, the application may run a conservative structured/text alignment gate. This gate is deterministic and intentionally small in V1. It looks only for a few obvious material mismatches, starting with endpoint-structure and placebo/control signals that appear inconsistent between Trial Features and editable text. When a material mismatch is detected, the prediction workflow pauses before new scoring. The participant either corrects the structured field/text or adds a short explanation. That explanation becomes `clarification_context` in the review packet and is available to the LLM as participant scenario context.

The alignment gate should not become a broad NLP adjudication engine in V1. Its job is to prevent the LLM from silently deciding which field is true when the scenario may simply be asynchronous. Ambiguous or low-confidence tensions should continue into the Quality Review as narrative context rather than blocking review. Additional deterministic mismatch checks, such as endpoint timing, comparator, biomarker, or population-age checks, should be added only after they have low false-positive risk and fixture coverage. A future LLM alignment pass may provide broader semantic comparison, but its findings should be treated as clarification suggestions with confidence/severity, not automatic truth or direct scoring.

## 12. Output JSON Contract

The LLM should return structured JSON. The application should validate the response, reject or downgrade malformed scoring fields, and calculate the Quality Adjustment and Final Candidate Score deterministically.

The LLM should not return the final score as an authority. It should return review ratings, evidence fields, narrative, and continuity fields.

Proposed contract:

```json
{
  "score_movement_review": {
    "summary": "short explanation of the observed Completion Score movement",
    "clinical_design_interpretation": "participant-facing explanation in clinical trial / pharma development language",
    "model_supported_reasons": [],
    "cautions": []
  },
  "quality_review_domains": {
    "development_question_fit": {
      "rating": "strong | acceptable | weak | conflicting",
      "rationale": "...",
      "evidence_fields": []
    },
    "scientific_rigor": {
      "rating": "strong | acceptable | weak | conflicting",
      "rationale": "...",
      "evidence_fields": []
    },
    "population_relevance": {
      "rating": "strong | acceptable | weak | conflicting",
      "rationale": "...",
      "evidence_fields": []
    },
    "endpoint_and_comparator_logic": {
      "rating": "strong | acceptable | weak | conflicting",
      "rationale": "...",
      "evidence_fields": []
    },
    "operational_scale_fit": {
      "rating": "strong | acceptable | weak | conflicting",
      "rationale": "...",
      "evidence_fields": []
    },
    "change_integrity": {
      "rating": "improved | neutral | simplified | potential_shortcut",
      "rationale": "...",
      "evidence_fields": []
    },
    "text_consistency": {
      "rating": "consistent | minor_tension | material_tension | contradiction",
      "rationale": "...",
      "evidence_fields": []
    }
  },
  "participant_review": {
    "what_changed": "...",
    "why_completion_score_may_have_moved": "...",
    "what_the_design_gained": "...",
    "what_the_design_may_have_sacrificed": "...",
    "operational_feasibility_note": "...",
    "text_consistency_note": "...",
    "challenge_question": "..."
  },
  "facilitator_view_optional": {
    "shortcut_risk": "low | moderate | high",
    "change_integrity": "improved | neutral | simplified | potential_shortcut | unclear",
    "main_tradeoff": "...",
    "coherence_concern": "...",
    "suggested_facilitator_probe": "...",
    "memory_update": "..."
  },
  "continuity": {
    "prior_concerns_resolved": [],
    "prior_concerns_worsened": [],
    "prior_concerns_unchanged": [],
    "new_concerns": [],
    "storyline_update": "..."
  },
  "trace": {
    "main_features_considered": [],
    "main_pillars_considered": [],
    "operational_statuses_considered": [],
    "compared_against": "previous_prediction",
    "should_repeat_prior_warning": false
  }
}
```

`facilitator_view_optional` may be omitted in the first V1 implementation. The minimum provider contract is the participant review, quality review domains, continuity fields, and trace fields needed for validation and replay.

Participant-facing narrative should translate model evidence into clinical trial / pharma development language. The provider may use `field_changes`, `xgboost_impact_changes`, `score_delta`, and pillar/subcategory movement internally, but visible explanations should avoid technical model terms such as SHAP, feature impact, XGBoost movement, or pillar delta unless the facilitator view explicitly asks for model diagnostics. Preferred language should discuss endpoint maturity, evidence strength, comparator credibility, blinding/control implications, recruitment burden, trial duration, patient population fit, operational complexity, execution feasibility, development strategy, design shortcut risk, and regulatory persuasiveness.

The LLM does not return `Quality Adjustment`, `Final Candidate Score`, or final Quality Assessment pillar/subcategory point contributions. The application derives them from validated `quality_review_domains`, evidence fields, and the documented deterministic mapping. If a provider returns app-owned score fields, validation should mark them as ignored and the application should still calculate its own values. This keeps plotted Quality Assessment contributions reproducible.

The application calculates:

```text
evidence_coherence_points = deterministic_map(validated_evidence_coherence_ratings)
population_strategy_points = deterministic_map(validated_population_strategy_ratings)
execution_plausibility_points = deterministic_map(validated_execution_plausibility_ratings)

rating_points =
    evidence_coherence_points
  + population_strategy_points
  + execution_plausibility_points

quality_adjustment = clamp(rating_points, -10, +10)

final_candidate_score = clamp(
    completion_score + quality_adjustment,
    0,
    100
)
```

Suggested initial V1 mapping:

```text
strong = +1
acceptable = 0
weak = -2
conflicting = -4

change_integrity:
improved = +1
neutral = 0
simplified = -2
potential_shortcut = -4

text_consistency:
consistent = 0
minor_tension = -1
material_tension = -2
contradiction = -4
```

The final score should be rounded by application logic using a documented UI rule. A domain rating should affect the Quality Adjustment only when the LLM provides supporting `evidence_fields`; otherwise the point effect should be zero and the issue can remain narrative-only. Evidence fields must also reference evidence available in the review packet. Unsupported evidence references are preserved for auditability but do not move the Quality Adjustment.

Allowed evidence references may cite:

- direct structured/text fields, such as `endpoint_rigor_ml`, `primary_outcomes_ui`, or `text_context.primary_outcomes_ui`
- operational fields and nested metadata, such as `operational_assumptions.planned_enrollment.support_level`
- clarification context, such as `clarification_context.user_clarifications`
- Step-2 delta evidence, such as `field_changes.endpoint_rigor_ml`
- model explanation sections or names, such as `xgboost_impact_changes`, `Execution Framework`, or `xgboost_impact_changes.Execution Framework`

Guardrails:

- If all validated Quality Review domains are `acceptable`, `neutral`, or otherwise non-concerning, `Quality Adjustment = 0`.
- A positive Quality Adjustment requires evidence that the participant strengthened design quality, not merely that the design avoided obvious concerns.
- Unsupported `evidence_fields` have zero scoring effect even when the rating is non-neutral.
- Benchmark-typical operational assumptions are neutral by default; they do not create a positive Quality Adjustment unless supported by broader design improvements.
- Low-confidence operational benchmark metadata should be narrative-first. It should affect points only when multiple conflict signals agree.

## 13. Plot Integration Guidance

Plot integration should preserve source clarity while allowing an adjusted-score view.

Use a toggle:

```text
Completion Score View
Final Candidate Score View
```

### Completion Score View

This is the existing model-first view:

- Gauge: `Completion Score`.
- Bar chart: four XGBoost / SHAP-derived completion pillars.
- Treemap: existing XGBoost / SHAP-derived completion hierarchy.
- Labels should remain tied to Completion Score drivers.

### Final Candidate Score View

This is the adjusted serious-game view:

- Gauge: `Final Candidate Score`.
- Component cards:
  - `Completion Score`.
  - `Quality Adjustment`.
  - `Final Candidate Score`.
- Bar chart may show seven bars:
  - Four Completion Score pillars:
    - `Therapeutic Context`.
    - `Scientific Challenge`.
    - `Patient Profile`.
    - `Execution Framework`.
  - Three Quality Assessment pillars:
    - `Evidence Coherence`.
    - `Population & Strategy Fit`.
    - `Execution Plausibility`.
- Use visual separation or a distinct color family so users can see that the first four bars are XGBoost / SHAP-derived and the last three are structured Quality Review contributions.

Recommended adjusted treemap structure:

```text
Final Candidate Score
├── Completion Score
│   ├── Therapeutic Context
│   ├── Scientific Challenge
│   ├── Patient Profile
│   └── Execution Framework
└── Quality Assessment
    ├── Evidence Coherence
    │   ├── Endpoint & Comparator Fit
    │   └── Scientific Rigor
    ├── Population & Strategy Fit
    │   ├── Population Relevance
    │   └── Development Fit
    └── Execution Plausibility
        ├── Operational Scale Fit
        └── Change Integrity
```

The treemap should make source boundaries explicit:

```text
Completion Score = XGBoost / SHAP-derived
Quality Assessment = structured LLM review, app-scored
```

Do not create fake SHAP attribution. Quality Assessment values are not SHAP values and should not be described as model drivers. Use terms such as `Quality Review Contributions` or `Quality Assessment` rather than `SHAP drivers`.

Operational assumptions should not be redistributed into the XGBoost `Execution Framework` branch. In adjusted view, Planned Enrollment, Planned Sites, and Planned Duration should contribute only through `Quality Assessment -> Execution Plausibility -> Operational Scale Fit`.

The minimal participant panel does not need to expose all validation/debug fields. It may show Completion Score, Quality Adjustment, Final Candidate Score, the three Quality Assessment pillars, and concise participant-review narrative. Future facilitator or debug views should consider exposing packet-supported versus unsupported `evidence_fields` from `quality_assessment` so facilitators can audit whether a review rating was grounded in packet evidence. Future real-provider UI work should also decide whether `score_movement_review.clinical_design_interpretation` becomes visible in the participant panel, since it is intended to translate model movement into clinical trial / pharma development language.

Treemap signed-value rule:

- Tile labels show signed point contribution.
- Color indicates positive versus negative contribution.
- Tile size should use absolute magnitude or a fixed group sizing rule, because treemap area cannot directly represent negative values.
- The root label should show the Final Candidate Score, but the chart should not imply that negative tile areas add arithmetically as positive area.
- Completion Score and Quality Assessment branches should be visually separated so users do not confuse app-scored quality contributions with SHAP-derived model impacts.

## 14. Narrative Tone Rules

Participant-facing writing rules:

- Use conditional, analytical language.
- Avoid definitive causal claims.
- Avoid overexposing optimization logic.
- Avoid telling users exactly which feature to change next.
- Keep the participant narrative concise.
- Encourage discussion rather than provide the answer.
- Maintain continuity with previous iterations.
- Do not contradict previous feedback unless the current change resolves or changes the issue.

The narrative should say what a pattern may suggest, what trade-off may be present, and what question the team should debate. It should not claim clinical truth or prescribe the next design edit.

## 15. Memory And Iteration Policy

The visible narrative should compare mainly against the previous prediction, because that is how participants experience iteration. The LLM must still receive enough memory to avoid contradictions across the full case.

Recommended stored state per serious-game session:

- Baseline snapshot.
- Each prediction snapshot.
- Changed fields per iteration.
- Completion score per iteration.
- Planned enrollment assumption per iteration.
- Enrollment source per iteration.
- Benchmark level used per iteration.
- Benchmark percentiles per iteration.
- Enrollment status per iteration.
- Support level per iteration.
- Supporting signals per iteration.
- Conflicting signals per iteration.
- Quality Review ratings per iteration.
- Quality Adjustment per iteration.
- Final Candidate Score per iteration.
- Pillar impacts and pillar deltas.
- Feature drivers and deltas.
- Participant review.
- Facilitator view.
- Compact memory update.

The storyline should be application-owned, not implicit LLM memory. After each review, store:

- Iteration number.
- Changed fields.
- Changed operational assumptions.
- Score movement.
- Quality Adjustment.
- Main gain.
- Main trade-off.
- Resolved concerns.
- Persistent concerns.
- New concerns.
- Storyline update.

For the next prediction, pass baseline review, previous review, compact storyline memory, current delta packet, and current snapshot. After several iterations, the system should pass a compact case memory summary rather than the full raw history every time. This avoids long context, repeated warnings, and drift in the narrative. Raw history can still be stored for audit, export, or facilitator debrief.

### Review Regeneration And No-Op Policy

The application should decide whether a new Quality Review is needed before calling the LLM. This prevents the narrative from changing when the user has not materially changed the scenario.

Do not call the LLM and do not create a new storyline step when:

- No model-facing Trial Features changed.
- No active operational assumptions changed.
- No editable text fields changed materially.
- Completion Score and operational metadata are unchanged.

For no-op predictions, reuse the latest validated review and leave Quality Adjustment, Final Candidate Score, and storyline memory unchanged.

For minor text-only edits, use a materiality gate before triggering a full review. The first implemented gate normalizes case, whitespace, and punctuation before comparing text snapshots; richer semantic materiality can be added later:

```text
Normalize text -> compare to previous text -> classify as no-op, minor wording, or material meaning change.
```

Examples:

- Typo, punctuation, casing, whitespace, or a single wording cleanup = no new review.
- A short clarification that does not alter endpoint, population, intervention, or operational meaning = no full review; optionally update displayed text only.
- A text edit that changes endpoint intent, population scope, intervention description, rationale, or creates a structured-field contradiction = material text change and may trigger a new Quality Review.

If a text-only material change triggers review, the narrative should state that the design variables did not change and the review changed only because the textual rationale/context changed. The application should avoid presenting this as a new model-score movement.

For obvious structured/text mismatches, run a clarification gate before committing new scoring rather than immediately creating a prediction snapshot or full review. If the participant corrects the structured field or text, the next prediction uses the corrected scenario. If the participant keeps the structured value and adds an explanation, the explanation is stored with the snapshot and included in `clarification_context`. The Quality Review can then discuss the clarified scenario and may still flag residual clarity concerns. The clarification step should not create a new Completion Score, Quality Adjustment, Final Candidate Score, or storyline entry until a full validated prediction/review workflow is generated.

## 16. Reproducibility And Provider Fallback

The architecture should support OpenAI and Gemini provider calls later without binding product logic to one provider.

Provider selection and secret handling:

- Default provider should be OpenAI. Gemini should be configured as an explicit fallback provider so the app can switch when OpenAI is unavailable, rate-limited, or operationally unsuitable.
- Provider config code must read secrets from environment variables or deployment secret managers only. It must never store API keys in committed Python files, notebooks, docs, fixtures, or frontend state.
- Local development may use `.env` loaded by `python-dotenv`, following the existing project pattern. Deployment should use Cloud Run environment variables or Secret Manager-backed values.
- Recommended environment variables:
  - `NARRATIVE_LLM_PROVIDER=openai`
  - `NARRATIVE_LLM_FALLBACK_PROVIDER=gemini`
  - `OPENAI_API_KEY`
  - `OPENAI_NARRATIVE_MODEL`
  - `OPENAI_REASONING_EFFORT=high`
  - `GEMINI_API_KEY` or existing `GOOGLE_API_KEY`
  - `GEMINI_NARRATIVE_MODEL`
  - `NARRATIVE_LLM_TEMPERATURE`
  - `NARRATIVE_LLM_SEED`, only for providers/models that support seed-like reproducibility.
  - `NARRATIVE_LLM_MAX_OUTPUT_TOKENS=6000`
  - `NARRATIVE_LLM_TIMEOUT_SECONDS`
  - `NARRATIVE_LLM_MAX_RETRIES`
- Current setup status: local `.env` can hold these values, and `src/narratives/provider_config.py` reads and validates them without making any LLM API call. `scripts/check_narrative_openai_smoke.py` and `scripts/check_narrative_gemini_smoke.py` can run opt-in API smoke tests when `RUN_NARRATIVE_OPENAI_SMOKE=1` or `RUN_NARRATIVE_GEMINI_SMOKE=1` is set; they skip by default to avoid accidental network calls or API spend. `src/narratives/provider.py` now contains real OpenAI and Gemini invocation helpers behind the same normalized provider result shape, but the simulator still uses the deterministic mock provider by default until live review activation is explicitly enabled.
- After provider config and optional OpenAI/Gemini smoke testing are validated, add richer prompt/schema fixtures and then activate real OpenAI/Gemini review calls through the existing provider chain.
- As of the June 2026 planning decision, the preferred OpenAI default is a pinned GPT-5.5 snapshot, such as `gpt-5.5-2026-04-23`, rather than a floating alias. `gpt-5.5-pro-2026-04-23` can be considered for slower, high-quality hidden baseline generation or offline review, but not as the default live interactive model unless latency and cost are acceptable.
- Gemini fallback should be configured with an explicit model ID rather than hard-coded in product logic. Prefer a high-quality Gemini Pro model for fallback review quality; choose a stable Flash-family model only when latency, cost, or preview-model risk dominates.
- Real-provider implementation should use a provider chain: try the configured primary provider first, then the configured fallback only for provider/network/rate-limit/unavailable failure. Do not fallback when the primary provider returns valid but unfavorable clinical reasoning, or when the provider returns malformed/invalid review JSON; that would create provider-shopping behavior and hide prompt/contract problems that should be fixed.
- Cache and trace keys must include provider and model name so OpenAI and Gemini outputs for the same packet are not treated as interchangeable.
- Keep the implementation deliberately small: one provider config reader, one prompt/schema builder, and one normalized provider result shape shared by mock, OpenAI, and Gemini. Avoid separate scoring, packet-building, cache, or UI code paths per provider.
- Provider fallback should be bounded and auditable. Try at most the configured primary and one configured fallback for a given packet. Store which provider failed, why it failed, and which provider generated the accepted review. Do not silently retry multiple times or cascade across many models.
- If both providers fail, return an unavailable Quality Review state and show Completion Score only. Do not reuse a stale Quality Adjustment for the new packet.
- If the fallback provider succeeds, cache the fallback result under its own provider/model namespace. Do not overwrite or pretend it is the primary provider result.
- Full narrative reviews need enough output budget for reasoning plus seven domain ratings and participant-facing review lines. Treat incomplete provider responses caused by output-token limits as provider failures, not as valid reviews.
- Runtime controls such as temperature, seed, reasoning effort, and JSON-output controls are provider/model-specific. Config may read them, but real provider code should send only parameters supported by the selected model. For the pinned GPT-5.5 OpenAI snapshot, use `reasoning.effort` from `OPENAI_REASONING_EFFORT`, request JSON output through the Responses API text format, and do not send temperature or seed unless a future model-specific capability check proves they are accepted. For Gemini, send temperature and seed only through Gemini's supported generation config and request JSON through Gemini's response MIME/config path. Store configured versus applied generation controls in provider metadata for every real provider result.

Recommended trace fields to store for each narrative pass:

- Provider.
- Model name.
- Prompt version.
- Rubric version.
- Temperature.
- Seed, when supported.
- System fingerprint or equivalent provider metadata, when available.
- Input JSON.
- Output JSON.
- Input hash.
- Timestamp.
- Iteration ID.
- Baseline ID.
- Session ID.

Trace robustness staging:

- Current prototype trace should remain simple and session-state compatible. It should store `input_packet`, provider/mock `output_json`, `validated_review`, validation status/errors, app-owned Quality Adjustment, Final Candidate Score, Quality Assessment, clarification issues, user clarifications, changed fields, score movement, provider/model identity, and compact storyline memory.
- Worth doing before or alongside real-provider prompt work: add prompt identity fields once prompt files exist, such as prompt template hash, system prompt hash, and response schema version. Add an explicit pre-review gate status, such as `passed` or `clarification_needed`, if clarification debugging becomes ambiguous. Add a compact evidence-audit summary only if unsupported evidence fields become hard to inspect from `quality_assessment`.
- Defer until real LLM provider integration: raw provider response, parsed JSON response, token usage, latency, provider response ID, finish reason, temperature, seed, system fingerprint, and provider-specific safety/refusal metadata. These fields are not meaningful for the deterministic mock reviewer.
- Defer until durable storage: database/file persistence, shared trial-level baseline review records, cross-team replay, facilitator export, retention policy, privacy controls, and schema migration strategy.
- Do not expand the prompt packet just because the trace stores more audit data. Store enough for audit; send only curated current-context fields to the LLM.

The goal is to make repeated runs as consistent as possible while acknowledging that exact determinism is not guaranteed for LLM outputs.

Provider abstraction should be thin. The application should own payload construction, validation, Quality Adjustment calculation, persistence, cache lookup, and UI rendering. Provider-specific code should own only model invocation and response normalization. The V1 provider boundary starts with a deterministic mock provider and an explicit unsupported-provider failure path; real OpenAI/Gemini invocation can be added behind the same boundary later.

Future real-provider prompts must use a funnel instruction:

- Use `iteration_context.field_changes` to identify what the participant changed.
- Use `model_interpretation.xgboost_impact_changes` to understand model movement and materiality.
- Treat XGBoost/SHAP movement as model explanation evidence, not proof of clinical causality.
- Translate model evidence into clinical trial / pharma development language for participant-facing text. Explain why the revised scenario may look more or less completion-like, robust, feasible, governed, strategically aligned, risk-reduced, simplified, or less evidence-generating in terms of endpoint timing, comparator choice, population scope, oversight, operational burden, trial duration, scientific challenge, or development strategy rather than exposing raw model vocabulary. Do not equate a higher Completion Score with simplification by default, but do flag simplification or value loss when the evidence points that way.
- Produce domain ratings, rationale, evidence fields, participant-facing narrative, continuity, and trace fields.
- Do not calculate or return `Quality Adjustment`, `Final Candidate Score`, or Quality Assessment point values. The application calculates those from the validated domain ratings.

Use a deterministic input hash based on prompt version, rubric version, baseline snapshot, current snapshot, storyline memory, and text context. If the same input hash is reviewed again with the same provider/model cache namespace, reuse the stored validated review instead of calling the provider again. Generate the baseline review once per selected study and store it for the session. Hashable review context should avoid session-specific trace IDs; use stable input hashes and iteration IDs instead.

Validation and failure behavior:

- If the LLM provider call fails, show Completion Score only and mark Quality Adjustment as unavailable for the current snapshot.
- Do not reuse a stale Quality Adjustment for a new snapshot.
- If JSON is malformed or fails schema validation, discard scoring fields and either show no narrative or show only validated narrative fields.
- If a domain rating is valid but lacks required `evidence_fields`, set its point contribution to zero and keep the issue narrative-only.
- If partial JSON validates, the application may render validated narrative sections, but Final Candidate Score should be calculated only from validated scoring fields.
- Store validation status and failure reason with the review trace.

## 17. Fields And Source-Of-Truth Principle

The structured feature registry remains the primary design source of truth for the narrative layer.

The LLM narrative layer should treat structured dropdown and numeric fields as the primary source of truth. Short text fields are secondary. They should help detect contradiction, missing rationale, or narrative inconsistency. Missing or brief free-text fields should not be heavily penalized unless they directly contradict structured trial features.

If structured fields and text fields conflict, the LLM should flag the inconsistency rather than silently penalize the Quality Adjustment. For example, if the structured fields say `adult_ml` is adult-only but the summary says the intended treatment population includes elderly patients with high disease burden, the LLM may flag a population-relevance concern.

User-editable text is untrusted context. The provider prompt must instruct the model to ignore any instructions, scoring requests, or role changes embedded inside study summary, endpoint, intervention, eligibility, or other trial text fields. Text can provide rationale, context, or contradiction evidence, but it must not override structured fields unless a future UI explicitly marks it as participant rationale.

Text conflict handling:

- For obvious material mismatches, pause the prediction workflow before new scoring and ask the participant to correct the scenario or add an explanation.
- For softer tensions, continue Quality Review and flag the inconsistency in narrative context.
- Route it to the affected Quality Assessment pillar.
- Require `evidence_fields` before it can affect Quality Adjustment.
- Treat missing, brief, or noisy text as low-confidence context rather than a direct penalty.
- Do not let the LLM silently choose whether structured fields or text fields are true. For Completion Score, structured Trial Features are authoritative. For Quality Review, structured fields remain primary context, while text and user explanations provide clarification, contradiction evidence, or scenario rationale.

### Field Selection for LLM Narrative Layer

Two field lists are needed for the serious-game narrative architecture:

1. `Full serious-game structured field list`: the full set of structured Trial Features available to participants. These fields are the primary source of truth for coherence, rigor, quality, and shortcut analysis.
2. `Direct XGBoost/SHAP field list`: the subset of fields directly transformed into model inputs and associated with direct XGBoost/TreeSHAP feature contributions. These fields help explain movement in the Completion Score and should be interpreted together with score deltas, pillar deltas, feature SHAP deltas, and top positive/negative drivers.

The LLM narrative layer should receive all 31 structured serious-game fields, even if only 27 are direct transformed XGBoost/SHAP fields in the current preprocessing path.

#### Full Serious-Game Structured Field List

Recommended v1 structured input payload: 31 fields grouped by pillar.

Therapeutic Context:

- `therapeutic_area_ml`
  - Label: Therapeutic Area
  - Narrative use: essential context, therapeutic-area expectations, calibration context, disease-setting interpretation.
- `gbd_cause_id_3_ml`
  - Label: Indication
  - Narrative use: essential indication context, disease-setting expectations, population relevance, endpoint relevance.
- `is_rare_disease_ml`
  - Label: Rare Condition Status
  - Narrative use: recruitment feasibility, evidence tolerance, operational feasibility, endpoint expectations.
- `phase_ml`
  - Label: Clinical Phase
  - Narrative use: Phase II/III rigor expectations, level of evidence expected, development maturity.
- `strategic_ambition_ml`
  - Label: Regulatory Intent
  - Narrative use: development-question fit, whether the design remains aligned with exploratory, signal-seeking, or confirmatory intent.

Scientific Challenge:

- `target_precedent_ml`
  - Label: Target Precedent Status
  - Narrative use: scientific novelty, biological risk, evidence expectations.
- `target_pathway_class_ml`
  - Label: Pathway Profile
  - Narrative use: mechanistic coherence and biological plausibility.
- `therapeutic_modality_ml`
  - Label: Therapeutic Modality
  - Narrative use: operational complexity, delivery complexity, scientific complexity.
- `innovation_tier_ml`
  - Label: Innovation Rank
  - Narrative use: novelty, uncertainty, risk-adjusted interpretation.
- `intervention_model_ml`
  - Label: Intervention Model
  - Narrative use: trial structure, parallel/crossover/single-group implications, coherence with endpoint and comparator logic.
- `primary_purpose_ml`
  - Label: Primary Purpose
  - Narrative use: objective coherence, whether the design still answers the intended question.
- `adaptive_design_ml`
  - Label: Design Flexibility Level
  - Narrative use: design sophistication, adaptive complexity, execution feasibility.
- `endpoint_rigor_ml`
  - Label: Primary Endpoint Type
  - Narrative use: scientific rigor, evidence strength, shortcut detection, endpoint relevance.
- `endpoint_structure_ml`
  - Label: Primary Endpoints Number
  - Narrative use: interpretability, multiplicity, complexity, decision clarity.
- `biomarker_stratification_ml`
  - Label: Biomarker Patient Selection
  - Narrative use: mechanistic fit, targeted population, enrichment strategy, shortcut detection if removed or weakened.

Patient Profile:

- `patient_severity_ml`
  - Label: Patient Severity
  - Narrative use: feasibility, ethics, disease burden, endpoint relevance.
- `line_of_therapy_ml`
  - Label: Line of Therapy
  - Narrative use: clinical setting, comparator expectations, development strategy.
- `gender_ml`
  - Label: Patient Gender Eligibility Status
  - Narrative use: population relevance, inclusiveness, generalizability.
- `healthy_volunteers_ml`
  - Label: Population Type
  - Narrative use: coherence with phase, disease setting, intervention risk.
- `adult_ml`
  - Label: Adult Profiles
  - Narrative use: population inclusion, representativeness.
- `child_ml`
  - Label: Pediatric Profiles
  - Narrative use: population inclusion, ethical complexity, development relevance.
- `older_adult_ml`
  - Label: Geriatric Profiles
  - Narrative use: population relevance, generalizability, shortcut detection if elderly participants are excluded to simplify completion.

Execution Framework:

- `masking_ml`
  - Label: Bias Control
  - Narrative use: internal validity, evidence credibility, execution complexity.
- `allocation_ml`
  - Label: Allocation Method
  - Narrative use: internal validity, allocation rigor, interpretability.
- `has_dmc_ml`
  - Label: Data Monitoring Committee
  - Narrative use: oversight proportionality, safety governance, risk management.
- `has_placebo_ml`
  - Label: Placebo Control
  - Narrative use: comparator rigor, interpretability, evidence quality.
- `comparator_benchmark_ml`
  - Label: Benchmark Comparator
  - Narrative use: evidentiary strength, comparator relevance, shortcut detection.
- `administration_complexity_ml`
  - Label: Delivery Profile
  - Narrative use: operational feasibility, patient burden, site burden.
- `number_of_arms_ml`
  - Label: Number of Arms
  - Narrative use: complexity, interpretability, feasibility, statistical design burden.
- `sponsor_tier_ml`
  - Label: Sponsor Type
  - Narrative use: execution capacity context, operational maturity context.
- `primary_duration_months_ml`
  - Label: Max Primary Endpoint Duration
  - Narrative use: endpoint timing, feasibility, disease-course coherence, shortcut detection if shortened too aggressively.

#### Direct XGBoost/SHAP Field List

The direct model-interpretation list should be treated as the 31 structured serious-game fields minus these four non-direct fields:

- `therapeutic_area_ml`
- `strategic_ambition_ml`
- `intervention_model_ml`
- `masking_ml`

These four fields must still be sent to the LLM because they are clinically and narratively important, but they should not be treated as direct transformed XGBoost/SHAP fields while the current preprocessing path excludes them.

The 27 direct XGBoost/SHAP fields are:

- `gbd_cause_id_3_ml`
- `is_rare_disease_ml`
- `phase_ml`
- `target_precedent_ml`
- `target_pathway_class_ml`
- `therapeutic_modality_ml`
- `innovation_tier_ml`
- `primary_purpose_ml`
- `adaptive_design_ml`
- `endpoint_rigor_ml`
- `endpoint_structure_ml`
- `biomarker_stratification_ml`
- `patient_severity_ml`
- `line_of_therapy_ml`
- `gender_ml`
- `healthy_volunteers_ml`
- `adult_ml`
- `child_ml`
- `older_adult_ml`
- `allocation_ml`
- `has_dmc_ml`
- `has_placebo_ml`
- `comparator_benchmark_ml`
- `administration_complexity_ml`
- `number_of_arms_ml`
- `sponsor_tier_ml`
- `primary_duration_months_ml`

This list should help the LLM interpret why the Completion Score moved. It should be combined with score deltas, pillar deltas, feature SHAP deltas, and top positive/negative drivers. It should not define the full clinical reasoning space alone.

#### Non-Direct Fields That Still Matter

The following four fields are essential for the Quality Review even if they are not direct transformed XGBoost/SHAP fields:

- `therapeutic_area_ml`: essential for disease-setting context and therapeutic-area calibration.
- `strategic_ambition_ml`: essential for development-question fit.
- `intervention_model_ml`: essential for interpreting the structure and validity of the trial design.
- `masking_ml`: essential for evidence credibility and bias-control reasoning.

The LLM should use these fields for design coherence and scientific rigor even if they do not carry direct SHAP contribution in the current model path.

#### Default Text Fields For v1

Recommended v1 default text context:

- `title`
  - UI source: `top_title`
  - Use: trial identity, broad objective, basic interpretation of the development question.
- `summary_ui`
  - UI source: `study_summary`
  - Use: main design rationale, trial intent, coherence between structured fields and written study description.
- `conditions_ui`
  - UI source: `conditions`
  - Use: supporting clinical context for indication and population coherence.
- `primary_outcomes_ui`
  - UI source: `primary_outcomes`
  - Use: endpoint coherence, evidence value, endpoint timing, interpretability.
- `interventions_ui`
  - UI source: `interventions`
  - Use: modality, mechanism, operational complexity, and consistency with structured therapeutic modality.

These fields are sent when present. They support coherence review only; they do not enter XGBoost Completion Score.

`primary_endpoint_description` should be treated as strongly recommended when the UI supports it, because it materially improves endpoint and duration coherence review.

#### Optional Text Fields

Optional future text fields for better coherence checking:

- `primary_endpoint_description`
  - UI source: future editable short endpoint field, when present.
  - Use: endpoint coherence, duration fit, evidence value, and consistency with endpoint rigor / endpoint structure.
- `criteria_ui`
  - UI source: `eligibility_criteria`
  - Use: deferred optional context for population relevance, inclusion/exclusion coherence, and shortcut detection when a future compact eligibility summary is available.

These optional fields can improve coherence analysis, but they should not be required for the serious-game v1 experience.

#### Text Fields To Avoid Relying On Heavily In v1

The architecture should avoid relying heavily on long or noisy free-text fields during v1.

Fields that should not be core scoring inputs in v1:

- `conditions_ui`
- Long or noisy `interventions_ui`
- Long `eligibility_criteria`
- Long protocol-style descriptions

Reasons:

- Participants will have limited time during the serious game.
- Dropdown changes are easier to maintain.
- Long text can become noisy, inconsistent, and time-consuming.
- The purpose of v1 is to use structured trial design decisions as the primary decision surface.
- Text should support coherence checking, not become the main scoring source.

#### Recommended v1 LLM Input Field Set

Always send:

- `nct_id`
- `trial_label`
- `lead_sponsor_canonical`
- `start_year`
- `title`
- `summary_ui`
- `conditions_ui`, when present
- `primary_outcomes_ui`, when present
- `interventions_ui`, when present
- All 31 structured Trial Features
- `operational_assumptions.planned_enrollment` benchmark and support metadata
- `operational_assumptions.planned_sites` benchmark and source metadata
- `operational_assumptions.planned_duration_months` benchmark and source metadata
- `completion_score`
- `previous_completion_score`
- `score_delta`
- `pillar_impacts`
- `pillar_deltas`
- `changed_fields`
- `field_changes`
- `xgboost_impact_changes`
- `compact_storyline_memory`

Send when implemented or derivable from available SHAP/subcategory data:

- `top_positive_feature_drivers`
- `top_negative_feature_drivers`
- `top_feature_impact_changes`

Optionally send:

- `primary_endpoint_description`

Defer by default:

- `criteria_ui`, unless a future compact eligibility summary or explicit narrative-edit policy is added

Do not rely heavily on:

- `conditions_ui`
- Long or noisy `interventions_ui`
- Long eligibility text
- Long protocol-style descriptions

#### Role In The Quality Review

The field set should support the Quality Review rubric by evaluating:

- Development-question fit.
- Population relevance.
- Endpoint and estimand coherence.
- Scientific rigor.
- Operational feasibility.
- Change integrity.

Examples:

- A narrower population may improve completion likelihood but could lower population relevance.
- A shorter endpoint duration may improve feasibility but could weaken clinical interpretability.
- Adding a DMC may be appropriate if proportionate to risk, phase, population, and intervention, but should not be automatically treated as a quality improvement.
- Removing biomarker stratification may simplify operations but could weaken mechanistic coherence in a targeted development setting.
- Simplifying comparator or masking may make execution easier but could weaken interpretability or evidentiary value.

## 18. Storage And Persistence, Planning Only

Each narrative pass and its context should be saved so future LLM calls can continue the story and so facilitators can review the decision path.

Implementation layers:

- Streamlit session state for the initial prototype.
- Local JSON/session file for development.
- Durable storage for shared trial-level baseline reviews and serious-game sessions.
- Future export for facilitator debrief.

The durable baseline-review requirement is trial-level: if two teams choose the same trial/version, they should load the same hidden baseline review unless the baseline input hash, prompt version, rubric version, or provider/model namespace changes. Team/session-specific iteration traces remain separate from that shared baseline.

This document does not prescribe the concrete database engine yet. The implementation choice should be made based on deployment environment, facilitator workflow, session privacy requirements, and export needs.

The stored serious-game snapshot should include:

- Operational assumptions snapshot.
- Planned enrollment metadata.
- Planned sites metadata.
- Planned duration metadata.
- Source labels.
- Benchmark level used.
- Benchmark percentiles.
- Benchmark status labels.
- Support/conflict signals, when implemented.
- Quality Review ratings.
- Quality Assessment pillar/subcategory contributions.
- Quality Adjustment.
- Final Candidate Score.
- Input hash.
- Prompt version.
- Rubric version.
- Validation status.
- Failure reason, if any.
- Compact storyline memory.

Storage should keep `Quality Review` ratings, `Quality Adjustment`, and `Final Candidate Score` as explicit fields rather than storing only a derived narrative explanation.

## 19. Non-Goals For This Architecture Phase

- No production LLM implementation in the contract-fixture phase.
- No UI implementation in the contract-fixture phase.
- No new API endpoint in the contract-fixture phase.
- No model retraining.
- No retraining of XGBoost for v1.
- No change to XGBoost prediction logic.
- No change to audit mode.
- No change to SHAP computation.
- No change to therapeutic-area calibration.
- No change to taxonomy unless later required.
- No change to audit/demo parity behavior.
- No portfolio mode yet.
- No model-based site-count estimator in v1 beyond the implemented deterministic benchmark metadata.
- No country-count model in v1.
- No model-based total-duration estimator in v1 beyond the implemented deterministic benchmark metadata.
- No cost layer in v1.
- No market layer in v1.
- No full operational scale engine in v1.
- No full operational-estimation engine in v1.
- No redistribution of Quality Adjustment into XGBoost / SHAP Completion Score pillars.
- No feature-level LLM pseudo-SHAP in v1.
- No LLM-generated final score.
- No nearest-neighbor / similarity cohorts in v1.
- No full feature-norm benchmark table in v1.

## 20. V1 Roadmap Summary

V1 serious-game narrative layer:

- Keep XGBoost unchanged.
- Use Planned Enrollment, Planned Sites, and Planned Duration as deterministic operational assumptions.
- Classify operational assumptions against similar-trial benchmarks.
- Use operational assumptions as bounded inputs into the future Quality Review.
- Make Quality Review bidirectional.
- Map Quality Review into three user-facing Quality Assessment pillars: Evidence Coherence, Population & Strategy Fit, and Execution Plausibility.
- Calculate a bounded Quality Adjustment in application logic.
- Calculate Final Candidate Score additively.
- Keep Completion Score View XGBoost-first.
- In Final Candidate Score View, show Completion Score and Quality Assessment as separate branches or grouped contributions.
- Show a narrative panel explaining design trade-offs.
- Store compact storyline memory so later predictions build on earlier changes.

Implementation staging:

1. Contract fixtures: define a small set of static example scenarios before implementation. Include at least baseline, model-facing edit, operational-only edit, material text-only edit, generic structured/text clarification, clarified structured/text review, and no-op/minor text edit. For each fixture, record expected review ratings, Quality Adjustment, Final Candidate Score behavior, clarification behavior, and storyline behavior. Current implementation artifact: `src/narratives/contract_fixtures.py`, validated by `scripts/check_narrative_contract_fixtures.py`.
2. Deterministic review packet builder: assemble baseline/current/previous snapshots, changed fields, explicit field-change deltas, operational metadata, score deltas, XGBoost impact movements, text context, user clarification context, field dictionary version, canonical taxonomy option-key values, display labels, and compact storyline memory without calling an LLM. Current implementation artifact: `src/narratives/packet_builder.py`, validated by `scripts/check_narrative_packet_builder.py`.
3. Validation and scoring engine: validate review JSON, enforce packet-supported `evidence_fields`, derive Quality Assessment pillars/subcategories, apply subcategory/pillar/total caps, apply zero/positive-adjustment guardrails, clamp Quality Adjustment, and calculate Final Candidate Score. Current implementation artifact: `src/narratives/scoring.py`, validated by `scripts/check_narrative_scoring.py`.
4. Mock reviewer: use deterministic fake JSON responses based on the fixtures to test validation, scoring math, no-op behavior, text-materiality behavior, and failure handling. Current implementation artifact: `src/narratives/mock_reviewer.py`, validated by `scripts/check_narrative_mock_reviewer.py`.
5. Storage and replay: persist validated review traces in session state first, including input hash, validation status, Quality Adjustment, Final Candidate Score, and compact storyline memory. Reuse cached reviews for identical input hashes. Current implementation artifact: `src/narratives/review_store.py`, validated by `scripts/check_narrative_review_store.py`. This is prototype storage only and does not yet satisfy the durable cross-team baseline requirement.
6. Minimal UI panel: render Completion Score, Quality Adjustment, Final Candidate Score, Quality Review, and compact Quality Assessment rows. Do not build adjusted treemap yet. Do not expose supported/unsupported evidence fields in the participant panel by default; reserve them for future facilitator/debug views. Revisit whether `score_movement_review.clinical_design_interpretation` should be displayed when real-provider output is implemented. Current implementation artifact: `frontend/views/trial_simulator.py`, using the provider-free packet builder, mock reviewer, and session-state review store.
7. Hidden baseline continuity: generate/store the hidden baseline review and verify that later iteration reviews use baseline review, previous review, compact non-numeric baseline quality summary, and compact storyline memory consistently. Current implementation is session-level only; it does not yet provide cross-team durable baseline reuse. Current implementation artifacts: `src/narratives/packet_builder.py`, `frontend/views/trial_simulator.py`, and `scripts/check_narrative_packet_builder.py`.
8. Thin LLM provider wrapper: add provider config first, then opt-in config-path API smoke tests, then provider-chain invocation through the same normalized result shape for mock, OpenAI, and Gemini. Provider config reads API keys from environment variables or secret managers; it never stores keys in code. Current implementation artifacts: `src/narratives/provider.py` and `src/narratives/provider_config.py`, validated by `scripts/check_narrative_provider.py`, `scripts/check_narrative_provider_config.py`, and the skipped-by-default `scripts/check_narrative_openai_smoke.py` / `scripts/check_narrative_gemini_smoke.py`; the simulator still uses the deterministic mock provider by default. Provider code invokes the model and normalizes JSON only; it should not decide field truth when structured/text inputs are asynchronous. The application owns scoring and returns `clarification_needed` before new scoring/provider review for unresolved material mismatches. Prompt/schema fixtures and explicit live-review UI activation remain the next work before participant-facing real LLM use.
9. First adjusted-score visual: add Final Candidate Score View with component cards and the seven-bar grouped chart.
10. Durable baseline store: add a database-backed baseline review repository keyed by trial/version and input hash. It should use create-if-missing semantics so the first team creates the hidden baseline and later teams reuse it.
11. Two-branch adjusted treemap: add only after the simpler adjusted view is stable and understandable; defer to V1.1 if it slows the first implementation.
12. Calibration/playtesting: review examples and tune rating-to-point mapping within the `-10` to `+10` total Quality Adjustment range.

## 21. Open Questions

- Whether later versions add, remove, or reorder fields beyond the v1 field-selection policy above.
- Exact storage mechanism.
- Exact participant versus facilitator UI placement.
- Exact provider abstraction for OpenAI/Gemini.
- Exact number of previous iterations to keep raw before summarization.
- Exact Quality Adjustment calibration examples after v1 playtesting.
- Whether facilitator view is hidden behind an expander or separate mode.
- Whether final governance recommendation is generated by participants, LLM, or both.
