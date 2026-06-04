# Serious-Game Narrative Architecture

## 1. Purpose Of The Narrative Architecture

This document defines the future design for adding a serious-game narrative layer around single-trial simulation in ClinTrialPredict. It is planning-only: no code, API, UI, model, taxonomy, or parity behavior is changed by this document.

The current edit/simulation workflow remains the foundation. A facilitator selects an existing trial, participants adjust structured Trial Features, and the application calls the existing prediction flow to produce a completion score with SHAP-derived impact decomposition.

The narrative layer exists because completion likelihood alone is not enough for a serious-game discussion. Some changes may raise completion likelihood by making a trial easier to complete while reducing scientific rigor, evidence value, endpoint interpretability, population relevance, governance quality, or strategic defensibility. Other changes may lower completion likelihood while making the design more robust or more relevant.

The future layer should help participants reason about this trade-off without giving direct optimization instructions. It should interpret score movement, surface design trade-offs, and challenge teams to defend their choices.

## 2. Core Scoring Boundary

The LLM layer is separate from the existing prediction system.

- `Completion Score`: the existing XGBoost, SHAP, therapeutic-area calibrated score from `/predict`, shown in points from 0 to 100.
- `Coherence Score`: the user-facing second score, from 0 to 100, assessing design defensibility and risk-mitigation quality.
- `Coherence Adjustment`: a deterministic application calculation, bounded from `-20` to `+20` points.
- `Adjusted Trial Value Score`: a deterministic application calculation.

```text
coherence_adjustment = clamp(round((coherence_score - 70) * 0.67), -20, +20)

adjusted_trial_value_score = clamp(
    completion_score + coherence_adjustment,
    0,
    100
)
```

Example:

- Completion Score: `72`
- Coherence Score: `84`
- Coherence Adjustment: `+9`
- Adjusted Trial Value Score: `81`

Interpretation:

- Coherence Score around `70` is neutral.
- Coherence Score below `70` creates a negative adjustment.
- Coherence Score above `70` creates a positive adjustment.
- This allows the narrative layer to recognize trials that are inherently risky but well strengthened by the participant's design choices.
- A trial below `50` on Completion Score can be boosted if the clinical operations design meaningfully mitigates risk.

Terminology:

- `Completion Score` = modelled likelihood of completion.
- `Coherence Score` = design defensibility and risk-mitigation quality.
- `Coherence Adjustment` = bounded point bonus or penalty.
- `Adjusted Trial Value Score` = final serious-game score.
- `Design Coherence Review` = narrative explanation.

In plain scoring terms, `Adjusted Trial Value Score = Completion Score + Coherence Adjustment`, with application-level bounds applied.

The application, not the LLM, calculates the Coherence Adjustment and Adjusted Trial Value Score. The LLM must never modify the XGBoost completion score, SHAP values, pillar impacts, therapeutic-area calibration, prediction pipeline, or audit/demo parity behavior.

Core boundary:

- The LLM never modifies XGBoost.
- The LLM never modifies SHAP values.
- The LLM never modifies therapeutic-area calibration.
- The LLM never rewrites the prediction score.
- The LLM returns Coherence Score and explanation.
- The application calculates Coherence Adjustment and Adjusted Trial Value Score.

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
3. The system creates a hidden baseline enrichment object from the original selected-trial state and the baseline prediction snapshot.
4. Participants do not immediately see a detailed LLM narrative. Showing rich interpretation before the exercise could reveal too much guidance.
5. Participants change structured dropdown fields and, later, optional short text fields.
6. Participants click `Predict Trial Completion`.
7. XGBoost returns the new completion score, pillar impacts, and impact decomposition through the existing prediction path.
8. The narrative layer receives baseline context, previous prediction, current prediction, changed fields, score deltas, SHAP/pillar movement, and prior narrative memory.
9. The UI displays the Design Coherence Review below or near the score and charts.
10. Participants iterate.

The visible narrative should usually compare the current prediction against the previous prediction. Internally, the LLM should also receive enough baseline and path memory to avoid contradicting prior feedback.

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

- `Adjusted Trial Value Score`, shown in the main gauge.
- `Completion Score` as a component score.
- `Coherence Score` as a component score.
- A short `Operational Assumptions` note.
- Concise `Design Coherence Review`.

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

## 8. Coherence Score Rubric

The LLM should apply an internal rubric across at least these dimensions:

- Development-question fit: whether phase, regulatory intent, population, intervention model, comparator, and endpoints appear aligned to the implied development question.
- Population relevance: whether the population remains clinically and strategically relevant rather than merely easier to enroll or complete.
- Endpoint and estimand coherence: whether endpoint rigor, endpoint structure, duration, comparator, and trial architecture appear mutually consistent.
- Scientific rigor: whether the design preserves interpretability, biological plausibility, and evidentiary value.
- Operational feasibility: whether the design appears proportionate and executable without becoming trivially easy at the expense of value.
- Change integrity: whether participants genuinely improved the design or mainly gamed completion likelihood.

The Coherence Score should reflect both current design coherence and change integrity:

- Current design coherence: whether the revised design is coherent and defensible now.
- Change integrity: whether the path from baseline to current design appears like meaningful improvement, acceptable simplification, or score-seeking shortcut behavior.

For v1, one Coherence Score is shown to users. Internally, the LLM should still consider both final design coherence and quality of changes.

The Coherence Score is bidirectional. It should reward:

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
Several weak or conflicting features create a Coherence Score penalty.
A difficult design can receive a positive Coherence Adjustment if the participant strengthens it in a coherent and defensible way.
```

Recommended Coherence Score interpretation:

- `85` to `100`: difficult but highly coherent, rigorous, and strategically defensible.
- `70` to `84`: coherent and balanced, with strengths outweighing trade-offs.
- `55` to `69`: unresolved weaknesses or simplifications that need discussion.
- `40` to `54`: meaningful evidence, feasibility, or change-integrity concerns.
- `0` to `39`: serious coherence problem or shortcut-driven design.

## 9. Shortcut Detection Concept

A shortcut is not simply a change that increases the completion score. A shortcut is a change that increases completion likelihood while potentially weakening evidence value, scientific rigor, population relevance, endpoint interpretability, or strategic defensibility.

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
They feed the future Coherence Score only.
```

Active fields:

```text
operational_assumptions.planned_enrollment
operational_assumptions.planned_sites
operational_assumptions.planned_duration_months
```

Source priority:

```text
1. Use planned/estimated enrollment if available.
2. Use completed-trial actual enrollment only when it represents a final observed value.
3. If missing, use a simple benchmark default from similar historical trials.
4. If the user edits it, source becomes user scenario.
```

The user does not write a free-text justification for planned enrollment in the platform. Instead, the enrollment assumption is assessed against the current design choices. The enrollment assumption must be supported by the selected trial profile.

If planned enrollment is missing and the system uses a benchmark-derived default, that default is neutral. It should not create a positive Coherence Score effect simply because it sits inside the benchmark range. It becomes an evaluated scenario assumption only when the user keeps it as the current assumption for a prediction snapshot or actively edits it.

Practical behavior:

- `model_default` inside benchmark = neutral.
- `user_scenario` inside benchmark = usually neutral or lightly supportive if consistent with the design.
- `user_scenario` outside benchmark = discussion signal, not automatic penalty.
- `user_scenario` outside benchmark + conflicting design signals = possible Coherence Score penalty.

### Enrollment Benchmark Classification

The system should classify the enrollment assumption deterministically before sending it to the LLM.

Benchmark cohort hierarchy, pragmatic v1 version:

```text
Level 1: same phase + same indication + rare disease flag
Level 2: same phase + same therapeutic area + rare disease flag
Level 3: same phase + same therapeutic area
Level 4: same phase only
```

Additional features such as therapeutic modality, sponsor tier, administration complexity, line of therapy, or population subtype can be used as secondary support/conflict signals when sufficient historical coverage exists. However, v1 should avoid making the benchmark hierarchy overly granular, because excessive stratification can quickly reduce cohort sizes and create unstable percentile estimates. Sponsor tier is legitimate as a contextual support signal inside the Coherence Score logic, but it should not usually define the primary benchmark cohort unless enough comparable trials are available.

Use the strictest level with enough historical trials. A simple minimum sample size threshold, such as `n >= 50`, is appropriate for v1; if the cohort is smaller, relax one level.

For the first implementation, the benchmark can be based on the simple deterministic hierarchy:

```text
phase + indication / therapeutic area + rare disease flag
```

Calculate benchmark percentiles:

```text
P25 = low benchmark
P50 = typical benchmark
P75 = high benchmark
P90 = very high benchmark
```

Classify user/current enrollment:

```text
if enrollment < P25:
    enrollment_status = "below_benchmark"

if P25 <= enrollment <= P75:
    enrollment_status = "typical"

if P75 < enrollment <= P90:
    enrollment_status = "ambitious"

if enrollment > P90:
    enrollment_status = "above_benchmark_high"
```

The classification is deterministic. The LLM interprets the label but does not invent it.

The deterministic enrollment label is a benchmark position, not a clinical judgment. Clinical interpretation belongs to the Coherence Score layer. For example, below-benchmark enrollment may become an evidence concern depending on phase, endpoint rigor, indication, and development intent, but the benchmark layer itself only reports position relative to similar trials.

### Optional Calibration Analysis Before Hardening V1

Later calibration may investigate within the existing system which trial features most strongly influence enrollment burden and enrollment feasibility:

1. Start with exploratory analysis on historical trials.
2. Identify which structured trial features correlate most with higher or lower enrollment sizes.
3. Use deterministic/statistical logic first, not LLM inference.
4. Use those findings to define the enrollment support/conflict signals used in the Coherence Score.

Suggested investigation logic:

- Analyze enrollment distributions by phase, indication, therapeutic area, rare disease status, sponsor class, modality, line of therapy, age group, endpoint type, endpoint duration, administration complexity, and comparator type.
- Use feature importance analysis, an enrollment proxy model, SHAP on the enrollment proxy model, correlation analysis, percentile segmentation, and clustering of similar trials.
- Determine which features consistently explain very high enrollment expectations, recruitment difficulty, operational feasibility, and realistic versus unrealistic enrollment assumptions.
- Classify signals into supportive of large enrollment, neutral, and conflicting with large enrollment.

This can ground the enrollment coherence layer in observed historical patterns, avoid arbitrary LLM reasoning, keep the system deterministic and auditable, and help define bounded enrollment effects inside the Coherence Score.

V1 now works with deterministic benchmark metadata for enrollment, sites, and duration. Optional calibration analysis can improve future coherence interpretation, but should not block the first serious-game narrative implementation.

### Enrollment Support Signals

The enrollment benchmark label alone is not enough. The platform should assess whether the selected trial profile supports the enrollment assumption.

Therapeutic modality should influence whether an enrollment assumption is supported by the current design, but it should not usually define the primary benchmark cohort in v1. A complex cell or gene therapy may weaken support for very large enrollment, while an oral small-molecule therapy in a common adult condition may support a larger enrollment assumption.

Supportive signals include:

- Common indication.
- Non-rare disease.
- Adult population.
- Broader patient profile.
- Earlier line of therapy.
- Larger sponsor tier.
- Simpler administration.
- Simple endpoint structure.
- Sufficient primary endpoint duration.
- Phase III context when a large confirmatory trial is expected.

Conflicting signals include:

- Rare disease.
- Pediatric population.
- Severe or fragile population.
- Later-line population.
- Complex therapeutic modality.
- Complex administration.
- Strict endpoint or hard clinical endpoint.
- Short endpoint duration.
- Niche indication.

Examples:

```text
Common adult Phase III disease + 1,200 patients:
high but potentially supported.

Rare pediatric Phase II gene therapy + 1,200 patients:
above benchmark high and weakly supported by the design.
```

Support level values:

```text
support_level = "supported_by_current_design | partly_supported_by_current_design | weakly_supported_by_current_design"
```

### Bounded Enrollment Effect

Enrollment must not dominate the Coherence Score. It is one operational-feasibility signal among the broader rubric:

- Development-question fit.
- Population relevance.
- Endpoint / evidence coherence.
- Scientific rigor.
- Operational feasibility.
- Change integrity.

Enrollment mainly affects:

- Population relevance.
- Operational feasibility.
- Change integrity when the participant changes enrollment in a way that appears to game the score.

Recommended cap:

```text
Enrollment should normally have a maximum standalone effect of about -10 to +4 points inside the Coherence Score logic.
```

This means high enrollment alone should not destroy the trial, low enrollment alone should not destroy the trial, enrollment becomes more important when combined with other conflicting choices, and several inconsistent design choices together can materially reduce Coherence Score.

## 11. Input Payload Architecture

The future LLM input object should be assembled after the existing prediction response has been received and after the application has created the latest prediction snapshot.

The payload must preserve the separation between the existing prediction system and the LLM narrative layer. XGBoost/TreeSHAP outputs explain completion-score movement; structured serious-game fields and operational assumptions define the broader design-reasoning space for the Coherence Score.

This is conceptual JSON for planning only, not an implementation contract yet:

```json
{
  "prompt_version": "narratives_v1",
  "rubric_version": "design_coherence_v1",
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
    "primary_outcomes_ui": "...",
    "criteria_ui": "..."
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
    "top_positive_feature_drivers": [],
    "top_negative_feature_drivers": [],
    "top_feature_impact_changes": []
  },
  "iteration_context": {
    "baseline_snapshot_id": "...",
    "previous_snapshot_id": "...",
    "current_snapshot_id": "...",
    "iteration_number": 2,
    "changed_fields": [],
    "previous_narrative_memory": "..."
  }
}
```

The payload should include all active operational assumptions: `planned_enrollment`, `planned_sites`, and `planned_duration_months`. These assumptions remain outside XGBoost and Completion Score; they feed narrative / Coherence only.

Operational-assumption values are assembled after the latest prediction snapshot. If the user changes fields that define an operational benchmark cohort, the affected benchmark becomes stale until the next `Predict Trial Completion` action. Enrollment and sites react to the implemented benchmark cohort fields. Duration reacts only to phase, indication, therapeutic area, rare-disease status, and primary endpoint duration bin. Duration does not use therapeutic modality in v1.

The LLM should use operational source metadata, not the compact UI label, to distinguish direct AACT-backed values, system-filled benchmark defaults, and participant scenarios. Direct AACT-backed values are source facts for the selected trial. System-filled values are benchmark assumptions. `user_scenario` values are participant scenario choices even when the visible label has no suffix.

Structured dropdown fields are the primary source of truth. Short text fields are secondary and should be used for coherence checking, contradiction detection, and narrative context rather than as the main source of scoring.

Missing or brief free-text fields should not be heavily penalized unless they directly contradict structured trial features or make an otherwise important design claim impossible to interpret.

## 12. Output JSON Contract

The LLM should return structured JSON. The application should validate the response, apply bounds to numeric fields, and calculate the adjusted score deterministically.

Proposed contract:

```json
{
  "coherence_score": 84,
  "coherence_confidence": "low | medium | high",
  "coherence_summary": "short one-sentence explanation",
  "participant_narrative": {
    "what_changed": "...",
    "why_completion_score_may_have_moved": "...",
    "what_design_may_have_gained": "...",
    "what_design_may_have_sacrificed": "...",
    "enrollment_coherence_note": "...",
    "question_for_team": "..."
  },
  "facilitator_view": {
    "shortcut_risk": "low | moderate | high",
    "change_integrity": "legitimate_improvement | acceptable_simplification | potential_shortcut | high_risk_shortcut | unclear",
    "main_tradeoff": "...",
    "coherence_concern": "...",
    "suggested_facilitator_probe": "...",
    "memory_update": "..."
  },
  "trace": {
    "main_features_considered": [],
    "main_pillars_considered": [],
    "enrollment_status": "below_benchmark | typical | ambitious | above_benchmark_high | not_available",
    "compared_against": "previous_prediction",
    "should_repeat_prior_warning": false
  }
}
```

The application calculates:

```text
coherence_adjustment = clamp(round((coherence_score - 70) * 0.67), -20, +20)

adjusted_trial_value_score = clamp(
    completion_score + coherence_adjustment,
    0,
    100
)
```

The adjusted score should be rounded by application logic using a documented UI rule. The LLM should not return the adjusted score as an authority.

## 13. Plot Integration Guidance

Plot integration should be conservative for v1:

- Gauge: shows `Adjusted Trial Value Score`.
- Small score cards: show `Completion Score` and `Coherence Score`.
- For v1, the bar chart and treemap remain XGBoost-first.
- Add one visible enrollment coherence note below or near the charts.
- The enrollment coherence note should be shown as a separate narrative or small card, not redistributed into SHAP-style pillar or subcategory impacts.
- Do not attempt pillar-level attribution for the Coherence Score in v1.
- Do not create fake SHAP attribution.

If a future version mixes coherence into the bar chart or treemap, rename the chart to `Adjusted Design Drivers` and clearly distinguish XGBoost impacts from coherence adjustments.

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
- Coherence Score per iteration.
- Coherence Adjustment per iteration.
- Adjusted Trial Value Score per iteration.
- Pillar impacts and pillar deltas.
- Feature drivers and deltas.
- Participant narrative.
- Facilitator view.
- Compact memory update.

After several iterations, the system should pass a compact case memory summary rather than the full raw history every time. This avoids long context, repeated warnings, and drift in the narrative. Raw history can still be stored for audit, export, or facilitator debrief.

## 16. Reproducibility And Provider Fallback

The architecture should support OpenAI and Gemini provider calls later without binding product logic to one provider.

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
- Timestamp.
- Iteration ID.
- Baseline ID.
- Session ID.

The goal is to make repeated runs as consistent as possible while acknowledging that exact determinism is not guaranteed for LLM outputs.

Provider abstraction should be thin. The application should own payload construction, validation, adjusted-score calculation, persistence, and UI rendering. Provider-specific code should own only model invocation and response normalization.

## 17. Fields And Source-Of-Truth Principle

The structured feature registry remains the primary design source of truth for the narrative layer.

The LLM narrative layer should treat structured dropdown and numeric fields as the primary source of truth. Short text fields are secondary. They should help detect contradiction, missing rationale, or narrative inconsistency. Missing or brief free-text fields should not be heavily penalized unless they directly contradict structured trial features.

If structured fields and text fields conflict, the LLM should flag the inconsistency rather than silently penalize the Coherence Score. For example, if the structured fields say `adult_ml` is adult-only but the summary says the intended treatment population includes elderly patients with high disease burden, the LLM may flag a population-relevance concern.

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

The following four fields are essential for the Coherence Score even if they are not direct transformed XGBoost/SHAP fields:

- `therapeutic_area_ml`: essential for disease-setting context and therapeutic-area calibration.
- `strategic_ambition_ml`: essential for development-question fit.
- `intervention_model_ml`: essential for interpreting the structure and validity of the trial design.
- `masking_ml`: essential for evidence credibility and bias-control reasoning.

The LLM should use these fields for design coherence and scientific rigor even if they do not carry direct SHAP contribution in the current model path.

#### Core Text Fields for v1

Recommended v1 core text context:

- `title`
  - UI source: `top_title`
  - Use: trial identity, broad objective, basic interpretation of the development question.
- `summary_ui`
  - UI source: `study_summary`
  - Use: main design rationale, trial intent, coherence between structured fields and written study description.

These two text fields are the core text context for v1.

#### Optional Text Fields

Optional text fields for better coherence checking:

- `primary_outcomes_ui`
  - UI source: `primary_outcomes`
  - Use: endpoint coherence, evidence value, endpoint timing, interpretability.
- `criteria_ui`
  - UI source: `eligibility_criteria`
  - Use: population relevance, inclusion/exclusion coherence, shortcut detection when population restrictions appear inconsistent with the stated study objective.

These optional fields can improve coherence analysis, but they should not be required for the serious-game v1 experience.

#### Text Fields To Avoid Relying On Heavily In v1

The architecture should avoid relying heavily on long or noisy free-text fields during v1.

Fields that should not be core scoring inputs in v1:

- `conditions_ui`
- `interventions_ui`
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
- All 31 structured Trial Features
- `operational_assumptions.planned_enrollment` benchmark and support metadata
- `operational_assumptions.planned_sites` benchmark and source metadata
- `operational_assumptions.planned_duration_months` benchmark and source metadata
- `completion_score`
- `previous_completion_score`
- `score_delta`
- `pillar_impacts`
- `pillar_deltas`
- `top_positive_feature_drivers`
- `top_negative_feature_drivers`
- `top_feature_impact_changes`
- `changed_fields`
- `previous_narrative_memory`

Optionally send:

- `primary_outcomes_ui`
- `criteria_ui`

Do not rely heavily on:

- `conditions_ui`
- `interventions_ui`
- Long eligibility text
- Long protocol-style descriptions

#### Role In The Coherence Score

The field set should support the Coherence Score rubric by evaluating:

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

Implementation options to decide later:

- Streamlit session state for the initial prototype.
- Local JSON/session file for development.
- Future durable storage for serious-game sessions.
- Future export for facilitator debrief.

This document does not prescribe a database implementation. The implementation choice should be made later based on deployment environment, facilitator workflow, session privacy requirements, and export needs.

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
- Coherence Score.
- Coherence Adjustment.
- Adjusted Trial Value Score.

Storage should keep `Coherence Score`, `Coherence Adjustment`, and `Adjusted Trial Value Score` as explicit fields rather than storing only a derived narrative explanation.

## 19. Non-Goals For This Architecture Phase

- No code implementation now.
- No UI implementation now.
- No new API endpoint now.
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
- No pillar-level coherence redistribution in v1.
- No feature-level LLM pseudo-SHAP in v1.

## 20. V1 Roadmap Summary

V1 serious-game narrative layer:

- Keep XGBoost unchanged.
- Use Planned Enrollment, Planned Sites, and Planned Duration as deterministic operational assumptions.
- Classify operational assumptions against similar-trial benchmarks.
- Use operational assumptions as bounded inputs into the future Coherence Score.
- Make Coherence Score bidirectional.
- Calculate a bounded Coherence Adjustment.
- Calculate Adjusted Trial Value Score additively.
- Keep bar chart and treemap XGBoost-first.
- Show a narrative panel explaining design trade-offs.

## 21. Open Questions

- Whether later versions add, remove, or reorder fields beyond the v1 field-selection policy above.
- Exact storage mechanism.
- Exact participant versus facilitator UI placement.
- Exact provider abstraction for OpenAI/Gemini.
- Exact number of previous iterations to keep raw before summarization.
- Exact Coherence Score calibration examples after v1 playtesting.
- Whether facilitator view is hidden behind an expander or separate mode.
- Whether final governance recommendation is generated by participants, LLM, or both.
