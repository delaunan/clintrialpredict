# Serious-Game Narrative Architecture

## Document Role

This file owns the future serious-game narrative layer: LLM commentary, Scenario Review, Reality Check, Trial Score, facilitator/participant outputs, and narrative payload contracts.

Current active planning direction is defined in `docs/trial_score_narrative_direction.md`: `Trial Score = Completion Outlook + Operational Fit + Reality Check`. Older sections in this document that describe `Quality Review`, `Design Confidence`, `Total Scenario Score`, or the first `Strategic Review` migration are historical unless explicitly restated in the current direction document.

It should not own the existing XGBoost Completion Score, SHAP impact mechanics, or simulation UI state; use `docs/architecture_edit.md` for those. It should consume operational benchmark metadata from `docs/architecture_estimation.md` rather than redefining benchmark construction.

Efficient update rule: change this file when narrative inputs/outputs, LLM contracts, Scenario Review scoring, or participant/facilitator interpretation rules change. Do not implement or imply changes to XGBoost, SHAP, calibration, or operational benchmark construction here.

## 1. Purpose Of The Narrative Architecture

This document defines the serious-game narrative layer around single-trial simulation in ClinTrialPredict. The next product direction pauses further Strategic Review implementation and shifts toward a simpler final-score narrative: XGBoost Completion Outlook remains the protected model anchor, Operational Fit becomes an additive Execution Framework interpretation layer, and Reality Check gives the LLM more room to judge the total scenario constructively.

The active planning contract is `docs/trial_score_narrative_direction.md`. If future work needs live batch evaluation, rebuild the eval harness around the new Trial Score / Operational Fit / Reality Check contract rather than reusing Design Confidence or first-generation Strategic Review checks.

The current edit/simulation workflow remains the foundation. A facilitator selects an existing trial, participants adjust structured Trial Features, and the application calls the existing prediction flow to produce a completion score with SHAP-derived impact decomposition.

The narrative layer exists because the Completion Score alone is not enough for a serious-game discussion. Some changes may make the scenario look more similar to historically completed-trial patterns while reducing scientific rigor, evidence value, endpoint interpretability, population relevance, governance quality, or strategic defensibility. Other changes may increase apparent early-termination risk while making the design more robust or more relevant.

The narrative layer should help participants reason about this trade-off without giving direct optimization instructions. It should interpret score movement, surface design trade-offs, and challenge teams to defend their choices.

## 2. Core Scoring Boundary

The LLM layer is separate from the existing prediction system. The active planning target is:

```text
Trial Score = Completion Outlook + Operational Fit + Reality Check
```

`docs/trial_score_narrative_direction.md` owns the current contract details. This section only records the durable boundary:

Core boundary:

- The LLM never modifies XGBoost.
- The LLM never modifies SHAP values.
- The LLM never modifies therapeutic-area calibration.
- The LLM never rewrites the prediction score.
- `Completion Outlook` remains the existing XGBoost / SHAP / therapeutic-area calibrated model output from `/predict`.
- `Operational Fit` is planned as an additive `Execution Framework` interpretation layer, not a replacement for `/predict`.
- `Reality Check` is planned as a freer LLM judgment layer over the total scenario, with evidence and safety guardrails.
- The participant-facing narrative should assess the final `Trial Score` rather than repeat separate component narratives.
- The V1 provider schema, three-pass participant-narrative contract, and numeric validation rails for `Operational Fit` and `Reality Check` are implemented in `src/narratives/trial_score_contract.py` and the adjacent narrative provider/prompt modules.

### Current Trial Score V1 Narrative Status

As of 2026-06-18, the narrative-production direction has moved from the first-generation Strategic Review migration to the current Trial Score V1 stack:

```text
Trial Score = Completion Outlook + Operational Fit + Reality Check
```

Durable implementation decisions:

- `Completion Outlook` remains anchored to the protected XGBoost score and SHAP-derived model drivers.
- `Operational Fit` is LLM-scored in the targeted Pass 2 Score Adjudication call and appears as an additive `Execution Framework` subpillar.
- `Reality Check` is LLM-scored in Pass 2, validated by app rails, and allocated only to canonical existing pillar/subpillar targets.
- Pass 1 performs evolution/evidence generation; Pass 2 scores Operational Fit and Reality Check against previous score trace and carryover context; Pass 3 writes the participant-facing Trial Score narrative.
- Provider label drift is handled by strict schema validation and targeted repair retries. If retries still fail, the UI reports the failed level clearly rather than using unsafe labels.
- Same-state scoring reuses the prior validated score trace, while Pass 3 may regenerate narrative with reversion/path context.
- Locked premise fields are visually disabled in the simulator, with greyed controls and lighter text.
- Obsolete `Strategic Review`, `Design Confidence`, `Quality Review`, and `Total Scenario Score` active scoring/eval paths have been deleted rather than preserved as compatibility logic.
- The current verification gate is `bash scripts/check_trial_score_v1_migration.sh`.

## 3. Current Technical Foundation

The current architecture provides the data needed for a later narrative layer:

- `frontend/app.py` routes `APP_VARIANT=trial_simulator` into the isolated simulation view.
- `frontend/views/trial_simulator.py` owns the Simulation Mode UI, structured Trial Features, pending-change tracking, latest prediction snapshots, prediction history, and score/charts rendering.
- `api/main.py` keeps `/predict` backward compatible for audit mode and adds `simulation_mode: true` live scoring through the production pipeline.
- `api/main.py` returns `score`, `pillar_impacts`, `subcat_impacts`, `mode`, and live probability for simulation calls.
- `models/taxonomy_01.json` defines the structured feature labels, options, mappings, pillars, subgroups, and encodings used by the UI and API.
- `src/prep/pipeline.py` defines the scoring preprocessing registry and ColumnTransformer behavior for ordinal, target-encoded, and numeric features.
- `docs/architecture_edit.md` records the current simulation contract, including baseline snapshot behavior, pending-change behavior, and parity requirements.

The narrative layer consumes these outputs and snapshots. It should not duplicate or reinterpret the model pipeline.

## 4. Existing-Study Mode User Experience

Existing-study mode is the current implementation priority.

Intended flow:

1. Facilitator selects an existing trial.
2. User opens Simulation Mode.
3. The system currently ensures a hidden baseline review object from the original selected-trial state and available baseline context during Simulation Mode initialization. This timing remains a calibration point because deferring baseline generation until first prediction would make Simulation Mode faster to open but could move the wait into the first visible prediction.
4. Participants do not immediately see a detailed LLM narrative. Showing rich interpretation before the exercise could reveal too much guidance.
5. Participants change structured dropdown fields and editable short text fields.
6. Participants click `Predict Trial Completion`.
7. XGBoost returns the new completion score, pillar impacts, and impact decomposition through the existing prediction path.
8. The narrative layer receives baseline context, previous prediction, current prediction, changed fields, score deltas, feature/subcategory/pillar movement, operational benchmark metadata, `text_context` Trial description fields, clarification context, and prior storyline memory.
9. The application validates the structured review, runs the targeted scoring pass for visible scenarios, validates score rails/arithmetic, and stores compact storyline state.
10. The UI displays the Trial Score review below or near the score and charts.
11. Participants iterate.

The visible narrative should usually compare the current prediction against the previous prediction. Internally, the LLM should also receive enough baseline and path memory to avoid contradicting prior feedback.

### Baseline Review Object

For existing-study mode, generate a baseline review once per selected study/version and store it durably, keyed by stable trial identity plus baseline input hash, prompt version, rubric version, and provider/model namespace. The first team opening Simulation Mode for a trial may create this baseline if it does not exist. Later teams opening the same trial should load the same baseline review and compact memory, so all teams start from a consistent original-trial interpretation.

The current prototype initializes the baseline review when Simulation Mode opens, not delayed until the first participant prediction. It remains hidden from participants at the start of the exercise, but it should be passed into later review calls so the LLM can evaluate how the design path evolved from the original trial. The exact production timing remains open pending latency calibration.

The baseline review context passed to later LLM calls is qualitative-only. It may include baseline strengths, concerns, consistency flags, participant-review text, continuity fields, and compact memory. It must not expose hidden baseline component scores, hidden Trial Score values, or other hidden numeric quality scores to later prompt logic as prior visible scores.

The compacted hidden-baseline context should preserve a useful baseline Completion Outlook summary when available and a short storyline memory, preferably baseline orientation plus next-watch focus, so the first visible prompt receives qualitative orientation rather than empty or self-referential baseline text.

Temporary simulator debug output may expose the same structured review-context inspection for hidden baseline and visible iterations. In hidden-baseline mode, `baseline_context_shared_with_current_prompt` and `previous_review_context_shared_with_current_prompt` are expected to be empty; the compacted hidden-baseline payload is what should appear as baseline context in the first visible iteration. For visible iterations, the debug payload should show the current prompt context, Pass 1 provider output, Pass 2 scoring input/output, and Pass 3 participant-narrative input/output when available, without requiring the full rendered prompt text. The final-output score-language restriction applies only to participant-facing Pass 3 prose, not to Pass 2 scoring or Pass 1 draft.

The same debug output may expose `current_model_state_evidence_shared_with_prompt` and `model_movement_evidence_shared_with_prompt`. State evidence describes the signed current pillar/subpillar forces. Movement evidence describes baseline/current and previous/current deltas and sign reversals. Movement ranking is previous-first for visible iterations, with baseline retained as context and used for ranking only when no previous iteration exists. The decomposition also exposes direct model-backed `feature_level_impacts` for registry fields that match XGBoost feature columns. Prompt-facing feature evidence is capped to the top three positive and top three negative direct feature impacts; therapeutic-area threshold offsets, residual/clipping adjustments, unmapped internal factors, and registry fields absent from XGBoost are not treated as feature-level evidence.

`completion_outlook_analysis.main_model_signals` should use this model evidence at the most concrete useful level. Hidden baseline should use current state only. Visible iterations should list latest movement signals first, then current-state anchors that still matter. The preferred wording is `Feature Label: Value under Pillar / Subpillar (+/-impact)`; for example, `DMC Involvement Status: Yes under Execution Framework / Methodological Setup (-5.7)`. Bare values such as `Yes` or `38.0 months` are not readable enough without the feature label. Subpillar is the fallback, and pillar-only language is the last fallback. Generic entries such as `Scientific Challenge alignment`, `Patient Profile fit`, or `Execution Framework constraints` are discouraged unless no granular evidence exists.

The current narrative-production workflow is intentionally staged. Pass 1 is the evolution/evidence pass and rough narrative drafter: it returns `completion_outlook_analysis`, `evolution_evidence`, `strategy_shift_check`, `continuity_update`, one visible development discussion option, and `analytical_narrative_draft`. Pass 1 evidence arrays are bullet-first and compact, while `analytical_narrative_draft` is short source-note prose for scoring and narrative shaping rather than the final participant explanation. Hidden baseline provides orientation only and does not create participant-visible development discussion options. Pass 2 is the scoring adjudicator: it receives Pass 1 evidence, previous score trace, previous Operational Fit and Reality Check assessments, previous score-evolution read, current score evolution, compact operational context, Operational Fit hash/match continuity, and carryover candidate, then returns Operational Fit and Reality Check points with rationale. Pass 3 is an editor/storyline shaper: it receives the accepted score trace, Pass 1 analysis, Pass 2 scoring review, trajectory/reuse context, selected model evidence from Pass 1, and participant-visible history, then restructures the result into participant-facing sections without re-scoring. Full raw operational hash payloads and broad raw model evidence should remain available in audit/diagnostic material rather than being duplicated in model-facing Pass 2 or Pass 3 inputs.

This workflow uses the existing provider validation and repair system. Pass 1 validation checks evidence/evolution shape, gated-field strategy checks, exactly one visible discussion option, and substantive non-empty draft fields; it does not enforce a visible word-count minimum. Targeted Pass 1 repair fixes only schema/evidence/scaffold issues. Pass 2 validation checks scoring schema, point ranges, evidence refs, allocation targets, same-state/baseline rails, and arithmetic; targeted Pass 2 scoring repair fixes only invalid scoring JSON. Pass 3 validation mirrors the participant-narrative schema, rejects returned app-owned arithmetic fields, and requires two to four material pillar-reading bullets. The active versions are `trial_score_evidence_pass_schema_v4`, `trial_score_scoring_pass_schema_v1`, `trial_score_narrative_pass_schema_v1`, and `trial_score_three_pass_prompt_v2_2`.

The hidden baseline review should include compact qualitative baseline analysis. It should interpret the prerecorded Completion Score when baseline decomposition is available, but it is a bounded context setup step rather than a rich participant-facing review. The provider should use baseline `structured_features`, `text_context` Trial description fields, Completion Score, pillar impacts, and feature/subcategory impacts to summarize why the original trial appears completion-like or risky in clinical trial / pharma development language. This is baseline reasoning context, not a visible participant score and not a new XGBoost calculation. If only a registry score is available without pillar or feature decomposition, the baseline review should explicitly treat the score interpretation as lower-detail and avoid inventing missing driver analysis. It may describe baseline strengths, concerns, consistency flags, and development issues inside `analytical_narrative_draft.development_landscape_read`, but each hidden-baseline draft field should remain concise. It must not produce participant-visible development discussion options, an active participant storyline, a visible baseline component adjustment, Trial Score, or hidden numeric quality score.

If a later visible scenario returns exactly to the hidden baseline scenario state, the app still lets Pass 1/Pass 3 produce the visible narrative, but app validation neutralizes Operational Fit and Reality Check to prevent path-dependent score drift. In that case Trial Score equals the baseline Completion Outlook for the restored state, and the narrative should describe the return to baseline rather than reward or penalize the path taken to get there.

The stored baseline review should include:

- Baseline score/model snapshot when available. For existing audit trials, this should include precomputed Completion Score decomposition from saved SHAP artifacts, not a live XGBoost prediction rerun.
- Baseline structured Trial Features.
- Baseline operational assumptions and source metadata.
- Baseline `text_context.summary_ui` and `text_context.primary_outcomes_ui`, when available.
- Baseline conditions, interventions, and primary outcomes text, when available.
- Baseline strengths.
- Baseline concerns.
- Baseline text/structured consistency flags.
- Baseline compact memory summary.

Participant-facing rule: do not show the hidden baseline review, hidden Trial Score, hidden component adjustments, or baseline narrative comments by default. Participants may see the original XGBoost Completion Score and model drivers if that is part of the trial-opening experience. The hidden baseline review exists to enrich the first and later visible reviews, not to expose a design score before teams make their first scenario choice. If hidden baseline review initialization fails, Simulation Mode should still open; later visible review calls can proceed without baseline review context or retry when a durable provider/store exists.

Later prediction reviews should receive:

- Baseline review.
- Previous iteration review.
- Compact storyline memory.
- Current delta packet.
- Current snapshot.

For the first visible review after a team edit, the narrative may compare qualitatively against the original trial baseline, but it must not refer to hidden baseline design numbers as if participants had already seen them. For example, it may say that the model Completion Score decreased while the current design appears more defensible than the original baseline context. It should not say that the team "improved the score" when the visible Completion Outlook or Trial Score declined.

## 5. Scratch Mode User Experience, Future Version

Scratch mode is a future variant and is not the current implementation priority.

In scratch mode, users start without an existing selected study:

- The first prediction is visible.
- The first narrative becomes the baseline review.
- Later predictions compare mainly against the previous prediction.
- The system still retains memory of the full design path so the narrative can distinguish newly introduced issues from recurring concerns.

Scratch mode may require additional field completeness rules because there is no original trial record to anchor interpretation.

## 6. Participant-Facing Outputs

The active participant-facing output target is defined in `docs/trial_score_narrative_direction.md`.

Near-term target:

- show `Trial Score` as the assessed serious-game score;
- keep the development `Completion Outlook` view visible during transition, with exact UI semantics owned by `docs/trial_score_narrative_direction.md`;
- fold material `Operational Fit` into the relevant pillar/subpillar reading, usually under Execution Framework, rather than presenting it as a separate participant-visible score component;
- mention `Reality Check` only when it materially changes, qualifies, or conflicts with the score interpretation, and frame it as a realism/coherence qualifier rather than numeric points;
- use two to four selective pillar-reading bullets, combining related pillars/subpillars when clearer instead of listing every pillar mechanically;
- render the participant card with at most three main titles: `Trial Score`, `What Is Driving The Score`, and `Discussion Point`;
- surface one selected discussion topic as explanatory prose, using the title-like summary for history/validation and the `why_it_matters` sentence for display;
- end the discussion point with one broader strategic question;
- keep detailed component evidence available for facilitator/debug trace where useful.

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

## 8. Scenario Review And Design Confidence Rubric

This section is historical context for the superseded Design Confidence path. It is retained only to explain prior implementation provenance and should not guide new work unless a future decision explicitly revives it.

The historical Scenario Review had two analytical jobs:

1. Explain `Completion Outlook`: why the XGBoost Completion Score moved, using changed fields, feature movements, subcategory movements, pillar movements, baseline context, previous iteration context, and cross-pillar interaction hypotheses.
2. Evaluate `Design Confidence`: whether the scenario is strategically defensible, evidence-generating, patient-relevant, and proportionate to execute.

That historical participant-facing hierarchy kept the existing four Completion Outlook pillars and added one Design Confidence subcategory under each pillar:

```text
Therapeutic Context
├── Therapeutic Area Profile
├── Development Phase and Goal
└── Phase & Intent Alignment

Scientific Challenge
├── Biological Profile
├── Protocol Architecture
└── Endpoint & Evidence Strength

Patient Profile
├── Clinical Severity
├── Population Scope
└── Target Population Alignment

Execution Framework
├── Methodological Setup
├── Trial Complexity Footprint
└── Operational Burden Balance
```

`Completion Outlook` remains the model-derived Completion Score and its existing pillar/subcategory movements. `Design Confidence` is an app-owned review adjustment derived bottom-up from:

- `Phase & Intent Alignment`: whether phase, purpose, development ambition, modality, endpoint posture, comparator support, and aligned Trial description fields align with the implied decision.
- `Endpoint & Evidence Strength`: whether endpoints, comparator/control choices, masking/allocation, duration, adaptive design, biomarker use, and text description support interpretable evidence.
- `Target Population Alignment`: whether severity, line of therapy, rare-disease context, age/sex scope, biomarker strategy, `text_context.conditions_ui`, and `text_context.summary_ui` support the intended patient and indication question.
- `Operational Burden Balance`: whether enrollment, sites, duration, arms, administration complexity, oversight, intervention model, modality, and benchmark metadata are proportionate to the evidence ambition and patient context.

These design subcategories are not restricted to the same feature list as their parent Completion Outlook pillar. A design subcategory may use any packet field, XGBoost movement, text change, operational assumption, benchmark signal, baseline context, prior iteration context, curated reference summary, or local dataset statistic that is relevant and explicitly present. The parent pillar is a participant-facing location, not a hard evidence silo.

The Scenario Review should reflect both current design defensibility and change integrity:

- Current design defensibility: whether the revised design is coherent, interpretable, patient-relevant, and executable now.
- Change integrity: whether the path from baseline to current design appears like meaningful strengthening, acceptable simplification, or score-seeking shortcut behavior.
- Collateral impact: whether a move that improves one pillar plausibly weakens or strengthens another pillar.
- Score boundary: whether a Completion Score movement is supported by Completion Outlook score inputs, clinically plausible, or only a hypothesis.

### Design Confidence Continuity Target

Scenario edits are cumulative, but `Design Confidence` is not a running sum of prior bonuses and penalties. Each visible review should score the current full scenario state while preserving interpretation continuity for unchanged evidence. Previous visible review context is the primary iteration-to-iteration anchor; hidden baseline remains qualitative context for the original trial and should not become a hard numeric comparator.

The packet builder adds a compact deterministic `iteration_context.design_confidence_continuity` object for the four Design Confidence subcategories on later visible reviews. For each subcategory, it includes the previous visible current state, movement fields, previous visible app-calculated points, current relevant changed fields, previous rationale/evidence fields, and one short continuity instruction. First-visible or no-prior packets mark this object unavailable. Example shape:

```json
{
  "available": true,
  "source_iteration_id": 1,
  "source_input_hash": "previous-input-hash",
  "changed_fields": ["operational_assumptions.planned_enrollment"],
  "instruction": "Use this object as deterministic continuity context for Design Confidence subcategories. The current scenario is still scored fresh, but large subcategory shifts need current relevant evidence.",
  "subcategories": {
	    "phase_intent_alignment": {
	      "label": "Phase & Intent Alignment",
	      "previous_current_state": "strong",
	      "previous_movement_direction": "improved",
	      "previous_movement_materiality": "moderate",
	      "previous_effect_role": "counterweight",
	      "previous_points": 1,
	      "previous_raw_points": 1,
	      "previous_rationale": "Prior phase/intent rationale.",
	      "previous_evidence_fields": ["phase_ml", "strategic_ambition_ml"],
	      "current_relevant_changed_fields": []
    }
  }
}
```

The provider should use this object as a continuity anchor: if a subcategory's relevant evidence did not change, current_state should usually remain stable and movement should usually be unchanged unless current packet evidence justifies a change. The scoring engine still calculates Design Confidence from the current review output; continuity anchors guide the LLM's qualitative judgment but do not mechanically carry forward points.

This target addresses observed live-play drift where a persistent fact can be interpreted in opposite directions across adjacent iterations, such as an evidence-standard upgrade receiving strong `phase_intent_alignment` credit in one iteration and then flipping to a strong phase/intent penalty in the next even though phase and strategic-intent fields did not materially change.

The field-to-subcategory relevance map for `design_confidence_continuity` is many-to-many and diagnostic-only. It is not a scoring map and does not automatically create positive or negative points. A field can be relevant to several Design Confidence lenses when a clinical reviewer could plausibly use it to explain movement: population fields can affect Target Population, Operational Burden, and Phase/Intent; endpoint/comparator/masking/allocation fields can affect Endpoint Evidence and Operational Burden. Planned Total Timeline is operational-context evidence for proportionality and executability, not Endpoint Evidence continuity evidence.

Provider prompts use a continuity-resolution lock for Design Confidence. For each subcategory, the reviewer compares prior current_state, prior movement fields, prior points, prior evidence fields, prior rationale, current relevant changed fields, and current `field_changes` before selecting the new current_state and movement fields. The current effect is treated as unchanged, prior weakness unresolved, prior weakness resolved, prior weakness offset, prior weakness worsened, new strength, or new weakness. When a changed field appears in prior evidence fields, the reviewer should compare previous/current/baseline values and labels to recognize restoration or reversal instead of mechanically carrying forward a prior penalty. If a structured/text conflict is unchanged from the prior visible iteration, it should remain visible as a consistency warning and unresolved prior concern rather than becoming a new or expanded penalty. Persistent strengths should not be re-credited as new improvement merely because they remain true.

Completion Outlook has a lighter consistency lock. Movement follows `score_delta`, changed structured Completion Outlook score inputs, and `xgboost_impact_changes`; prior visible Completion Outlook summary is storyline continuity only. If `score_delta` is stable and no structured score input changed, the prior Completion Outlook storyline should remain stable and non-score-input implications belong in Design Confidence. Static `top_positive_feature_drivers` and `top_negative_feature_drivers` remain in the packet as current-state support/risk context, but they should not explain latest movement unless the same field also appears in `field_changes` or `top_feature_impact_changes`. `xgboost_impact_changes` remains pillar/subcategory movement context, not field-identity evidence.

Provider prompts define the top-level boundary positively: Completion Outlook explains the estimated likelihood that the scenario reaches completion or faces early termination, based on previously observed trial patterns. Design Confidence evaluates whether the scenario is a coherent, interpretable, patient-relevant, and operationally proportionate design for the intended development decision. Completion Outlook pillar definitions are kept compact: Therapeutic Context covers disease and treatment context; Scientific Challenge covers difficulty of generating clear evidence; Patient Profile covers population focus and patient-selection difficulty; Execution Framework covers trial structure and conduct burden.

Narrative validation reports warn when Completion Outlook direction conflicts with `score_delta`, when stable-score Completion Outlook uses unsupported movement language, or when a Design Confidence subcategory moves materially without citing current relevant changed evidence or explaining resolution, offset, worsening, restoration, reversal, new strength, or new weakness. These warnings are diagnostics, not deterministic score caps.

Provider-facing duration labels should keep endpoint timing separate from operational planning. Use `Max Endpoint Duration` for `primary_duration_months_ml` and `Planned Total Timeline` for `operational_assumptions.planned_duration_months`. These fields are related because endpoint timing can influence the total trial timeline, but they are not interchangeable. The internal operational key remains `planned_duration_months`; the label exists to prevent the LLM from treating endpoint-duration changes as changed operational timelines or operational timeline changes as changed primary endpoint duration.

For V1, do not make a numeric `Coherence Score` or `Quality Score` the primary user-facing concept. Use:

- `Completion Outlook` for model-derived score movement.
- `Scenario Review` for the narrative panel.
- `Design Confidence` for the app-owned design adjustment.
- `Total Scenario Score` only if the combined score view is activated.

The review is bidirectional. It should recognize:

- Strong endpoint and comparator logic.
- Coherent phase, purpose, and development intent.
- Coherent population definition and patient relevance.
- Proportional safety oversight and risk governance.
- Operational assumptions supported by the evidence ambition and patient context.
- Difficult but strategically defensible designs.

It should challenge:

- Score-seeking simplification.
- Weakened endpoint rigor, comparator logic, or interpretability.
- Population narrowing that reduces relevance without clear rationale.
- Unsupported enrollment, site, or duration assumptions.
- Operational burden that is disproportionate to the evidence generated.
- Design changes that make completion easier but reduce clinical-development usefulness.

Principle:

```text
One weak feature creates a discussion point.
Several weak or conflicting signals can create a Design Confidence penalty.
A difficult design can receive a positive Design Confidence adjustment when evidence shows that the risk reflects rigor, patient relevance, scientific ambition, or prudent governance rather than bad design.
```

Recommended review ratings:

- `strong`: coherent, rigorous, and strategically defensible in the current context.
- `supportive`: positive and defensible, but not enough to deserve the top positive rating.
- `balanced`: mixed or neutral; trade-offs are understandable and not clearly score-moving.
- `weak`: unresolved weakness or simplification that needs discussion.
- `conflicting`: meaningful evidence, feasibility, text-consistency, or change-integrity concern.

The application should map validated ratings to Design Confidence only when the review provides supported `evidence_fields`. Unsupported concerns or strengths may appear in the narrative, but they have zero scoring effect.

Recommended scoring discipline:

```text
Default design adjustment = 0.0
Non-zero adjustment requires supported packet evidence
Implemented design subcategory movement range = -2.0 to +2.0 in 0.5 increments, app-owned
Typical subcategory movement = -1.0 to +1.0
Total Design Confidence = sum of four design subcategories
Proportional net cap from Completion Score movement and changed-field materiality
```

The provider must not return numeric subcategory points. Instead, the provider separates current-state judgment from movement judgment for each subcategory after selecting supported evidence fields and writing the rationale. `current_state` describes the current full scenario; `movement_direction`, `movement_materiality`, and `effect_role` drive app-owned points.

Target movement mapping:

- `movement_materiality`: `none = 0.0`, `minor = 0.5`, `moderate = 1.0`, `major = 2.0`.
- Positive movement: `resolved`, `improved`, `partially_resolved`, `offset`.
- Negative movement: `weakened`, `worsened`, `newly_introduced`.
- Neutral movement: `unchanged`.
- `effect_role=confirming` halves the effect to reduce double counting with Completion Outlook.
- `effect_role=counterweight` and `effect_role=independent` keep full movement weight.
- `effect_role=unchanged` scores `0.0`.

The adjustment must not be fake balancing:

- If Design Confidence confirms the same field direction already reflected in Completion Outlook, use `effect_role=confirming` so the app reduces double counting.
- If Design Confidence challenges Completion Outlook movement, use `effect_role=counterweight` so the contradiction remains visible.
- If the Design Confidence movement is independent of Completion Outlook movement, use `effect_role=independent`.
- If a baseline or prior strength remains true but was not changed by the current edit, use `movement_direction=unchanged`, `movement_materiality=none`, and `effect_role=unchanged`.

Implemented scoring-calibration refinement:

Design Confidence is a qualitative critical lens on Completion Outlook, not a second completion predictor and not a bonus multiplier. Completion Outlook mainly describes resemblance to historical completion versus early-termination risk. That movement can reflect quality, simplicity, operational burden, or risk patterns; higher Completion Outlook does not necessarily mean better design, and lower Completion Outlook does not necessarily mean worse design. The scoring layer should therefore preserve the LLM's qualitative judgment while limiting numeric over-amplification of signals already captured by Completion Outlook.

This refinement supersedes the earlier absolute `rating + score_materiality` scoring and matching-pillar cap language. The governing model is movement-based scoring, confirming-role double-counting reduction, and proportional net caps.

> Preserve each Design Confidence subcategory's meaning. When a change improves one design dimension but worsens another, reflect both effects in their relevant subcategories. Cross-functional trade-offs may be justified in the overall Design Confidence judgment, but a subcategory should be positive only when that subcategory itself improved.

The deterministic layer remains simple and conservative:

- Map `movement_direction + movement_materiality + effect_role -> raw_points`.
- Add calibrated `points` and `calibration_notes` when proportional net scaling changes a subcategory point.
- Apply calibration to Design Confidence only, not to Completion Outlook score or Completion Outlook pillars.
- Allow Design Confidence to move when Completion Outlook movement is flat or small, especially for Trial description, scenario-readiness, operational-assumption, or cross-pillar quality changes that are not directly reflected in Completion Outlook, but keep the net movement bounded by changed-field materiality.
- Use Completion Score movement and changed-field materiality only to set the net cap, not to decide clinical direction.
- Preserve subcategory meaning. Positive points should stay in the subcategory that improved; negative or neutral counter-impact should remain visible in the relevant other subcategory. Compensation happens through the total Design Confidence sum, not by making unrelated subcategories positive.
- Operational assumptions and all other supported packet evidence follow the same Design Confidence scoring and calibration rules. Planning assumptions may improve or worsen Operational Burden Balance because they affect whether the scenario feels operationally proportionate and executable. They may also create counter-effects in other Design Confidence subcategories, such as endpoint maturity or evidence sufficiency, when supported by the rationale. The special rule for planning assumptions is only that they must not explain Completion Outlook movement; within Design Confidence they are handled like other supported current-scenario evidence. Preserve subcategory meaning through the LLM rationale rather than through field-family-specific numeric caps.
- Shortcut-driven ease should not create strong positive Operational Burden Balance or Design Confidence. If easier completion comes from weaker randomization, masking, comparator, endpoint rigor, arms, governance, or development ambition, cap positive feasibility credit and preserve evidence/phase critique.
- Negative critiques should generally remain visible when supported. Soften negative points only when they duplicate strong same-direction negative Completion Outlook or pillar movement; do not soften them merely because another subcategory improved.
- The deterministic layer should mostly cap or soften excessive same-pillar, same-direction amplification. It should not invent new positive or negative points and should not perform deep clinical interpretation beyond matching-pillar movement and provider evidence fields.

Participant-facing Design Confidence treemap display shows signed subcategory points and the short rationale only. Movement fields remain internal for scoring, validation, and audit; those labels are not shown as participant-facing treemap text.

Text consistency should not be a standalone visual pillar in V1. It should be routed to the affected Design Confidence subcategory:

- Endpoint text conflict -> `Endpoint & Evidence Strength`.
- Study-summary, phase, or strategic-intent conflict -> `Phase & Intent Alignment`.
- Population or indication conflict -> `Target Population Alignment`.
- Intervention complexity or operational text conflict -> `Operational Burden Balance` or `Endpoint & Evidence Strength`, depending on the evidence fields.

Operational assumptions remain outside XGBoost. They may appear near Execution Framework for workflow reasons, but in combined Scenario Review plots they contribute through `Operational Burden Balance`, not through the model-derived `Execution Framework` Completion Outlook value.

## 9. Shortcut Detection Concept

A shortcut is not simply a change that increases the completion score. A higher Completion Score can reflect a more robust, better-governed, better-aligned, or lower-risk development pattern, but it can also reflect simplification or loss of evidence value when scientific challenge, endpoint rigor, comparator credibility, population relevance, or interpretability is reduced. For example, a Data Monitoring Committee may increase operational oversight and reduce apparent early-termination risk in one context while adding complexity or risk in another. A shortcut is a change that improves the Completion Outlook while potentially weakening evidence value, scientific rigor, population relevance, endpoint interpretability, or strategic defensibility.

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
They feed Scenario Review and Design Confidence only.
```

Active fields:

```text
operational_assumptions.planned_enrollment
operational_assumptions.planned_sites
operational_assumptions.planned_duration_months
```

Source priorities, cohort hierarchy, defaulting logic, and benchmark status calculation are owned by `docs/architecture_estimation.md`. The narrative layer consumes that metadata; it does not rebuild or reinterpret benchmark construction.

The user does not write a Trial description justification for planned enrollment, planned sites, or planned duration in the platform. Instead, the Scenario Review assesses whether the operational assumptions are coherent with the current structured design, `text_context` Trial description fields, benchmark metadata, and evidence ambition.

System-filled benchmark/default values are neutral. They should not create a positive Design Confidence adjustment simply because they sit inside a benchmark range. They become evaluated scenario assumptions when the user keeps them as the current assumption for a prediction snapshot or actively edits them.

Generic operational interpretation rule for enrollment, sites, and duration:

```text
Operational benchmark status alone is context, not a penalty.
Operational benchmark status plus conflicting design context can affect Operational Burden Balance.
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

Short duration + `text_context.primary_outcomes_ui` implying long-term clinical outcome:
possible Endpoint & Evidence Strength concern.
```

Explicit support/conflict signals are optional future derived fields. If implemented later, their methodology belongs in `docs/architecture_estimation.md` or a future calibration note, and the signals must be deterministic, auditable, and not invented by the LLM. Until then, benchmark metadata plus structured design context is sufficient for Scenario Review.

### Bounded Operational Effect

Operational assumptions should not dominate Design Confidence. Planned Enrollment, Planned Sites, and Planned Duration together should normally affect only:

```text
Execution Framework -> Operational Burden Balance
```

They remain controlled by the Design Confidence subcategory mapping. Operational benchmark status alone is context, not a penalty; it becomes score-relevant only when combined with coherent supporting or conflicting design evidence.

## 11. Input Payload Architecture

The LLM input object is assembled after the existing prediction response has been received and after the application has created the latest prediction snapshot.

The payload must preserve the separation between the existing prediction system and the LLM narrative layer. XGBoost/TreeSHAP outputs explain Completion Outlook score movement; `structured_features`, operational assumptions, and `text_context` Trial description fields define the broader design-reasoning space for the Scenario Review.

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
  "structured_feature_meanings": {
    "phase_ml": "Clinical development phase, used to interpret evidence expectations, trial purpose, and operational scale.",
    "endpoint_rigor_ml": "Type and evidentiary rigor of the primary endpoint, such as hard clinical, subjective/PRO, surrogate, or unknown."
  },
  "text_context_field_meanings": {
    "summary_ui": "Brief study summary used to interpret scenario intent and possible structured/text development issues.",
    "primary_outcomes_ui": "Primary outcome text used to cross-check endpoint intent and evidence interpretation."
  },
  "reference_packs": [
    {
      "pack_id": "core_clinical_development_v1",
      "role": "always_on",
      "tags": ["clinical_development", "phase_intent"],
      "prompt_safe_summary": "..."
    },
    {
      "pack_id": "strategic_context_2026_v1",
      "role": "current_context",
      "tags": ["current_strategy", "access", "governance"],
      "prompt_safe_summary": "..."
    }
  ],
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
      "design_confidence": null,
      "total_scenario_score": null,
      "design_numeric_context": "hidden_baseline_qualitative_only",
      "compact_storyline_memory": "..."
    },
    "previous_review": {
      "input_hash": "...",
      "iteration_id": 1,
      "status": "reviewed",
      "design_confidence": -2,
      "total_scenario_score": 70,
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

The payload should include all active operational assumptions: `planned_enrollment`, `planned_sites`, and `planned_duration_months`. These assumptions remain outside XGBoost and Completion Score; they feed narrative / Scenario Review only.

Operational-assumption values are assembled after the latest prediction snapshot. If the user changes fields that define an operational benchmark cohort, the affected benchmark becomes stale until the next `Predict Trial Completion` action. Enrollment and sites react to the implemented benchmark cohort fields, including strategic intent / Regulatory Intent when it affects the selected enrollment or patients-per-site benchmark. Duration reacts only to phase, indication, therapeutic area, rare-disease status, and primary endpoint duration bin. Duration does not use therapeutic modality or strategic intent in v1.

The LLM should use operational source metadata, not the compact UI label, to distinguish direct AACT-backed values, system-filled benchmark defaults, and participant scenarios. Direct AACT-backed values are source facts for the selected trial. System-filled values are benchmark assumptions. `user_scenario` values are participant scenario choices even when the visible label has no suffix.

`structured_features` dropdown/numeric fields are the primary source of truth. Trial description fields in `text_context` are secondary and should be used for coherence checking, contradiction detection, and narrative context rather than as the main source of scoring. Newly changed Trial description fields may make Reality Check more negative when they introduce material inconsistency versus the authoritative structured or operational scenario state. Unchanged description text must not override, dilute, or negate canonical structured or operational changes. Narrative packets should send canonical submitted structured values first and display labels separately, so future providers can reason from readable labels without losing scoring value provenance.

Missing or brief Trial description fields should not be heavily penalized unless they directly contradict structured trial features or make an otherwise important design claim impossible to interpret.

For structured Trial Features, narrative packets should use taxonomy option keys as the canonical value where an option key exists, and should include human-readable labels separately in `structured_feature_display_values`. For example, `endpoint_structure_ml = MULTI_COMPOSITE` should be paired with display label `Multi/Composite`. Numeric fields keep numeric values. The packet should include `field_dictionary_version`, `structured_feature_meanings`, and `text_context_field_meanings` so prompts and providers know which taxonomy meanings and text fields apply without requiring an external lookup or repeating the full field dictionary in every packet. Narrative field meanings must be generated from the production taxonomy source path in `src/prep/pipeline.py`, not patched only into `models/taxonomy_01.json`, so rerunning `notebooks/production_01.ipynb` preserves them.

Reference-pack routing should be active but compact. The packet should include only selected pack IDs, tags, roles, and `Prompt-Safe Summary` text from `frontend/data/docs/narrative_reference_packs`, not full source documents. Default V1 inclusion is `core_clinical_development_v1`, `strategic_context_2026_v1`, and `ich_e8_quality_by_design_v1`. Operational/governance scenarios may add `ich_e6_r3_gcp_v1`; endpoint/statistical scenarios may add `ich_e9_r1_estimands_v1` and `ich_e9_statistical_principles_v1`. Reference packs are secondary context: the scenario packet remains authoritative, and the provider should record used pack IDs in `trace.reference_pack_ids_used`.

Narrative packets should keep scenario edit facts separate from model explanation facts:

- `iteration_context.field_changes` records what the participant changed, with baseline, previous, and current values/labels. This covers structured Trial Features, changed text-context fields, and operational assumptions when available.
- `model_interpretation.xgboost_impact_changes` records what moved in the model explanation, with baseline, previous, and current impact values plus deltas. `changed_since_previous` marks local movement since the last prediction; `changed_from_baseline` marks accumulated drift from the original trial. These entries are XGBoost/SHAP explanation facts at pillar or subcategory level, not proof of clinical causality and not necessarily limited to fields the participant directly edited.

The LLM should use `field_changes` to explain what changed in the scenario and `xgboost_impact_changes` to weight which model-explanation movements were material. It should not infer that every model impact movement was directly caused by a single changed field.

The active simulator does not run a pre-prediction structured/text consistency check. Editable text fields remain narrative context and are submitted with the scenario, but they do not create a correction gate before scoring.

## 12. Output JSON Contract

The active workflow is the three-pass Trial Score workflow. Pass 1 returns compact bullet-first evolution/evidence JSON and rough source-note narrative. Pass 2 is a targeted score-adjudication call where the LLM assigns Operational Fit and Reality Check points directly inside app validation rails. Pass 3 returns participant-facing narrative only.

Pass 1 should return:

```json
{
  "review_metadata": {"review_mode": "hidden_baseline | first_visible_iteration | later_visible_iteration", "visible": true},
  "completion_outlook_analysis": {"summary": "...", "main_model_signals": [], "model_boundary_note": "..."},
  "strategy_shift_check": {"status": "supported | partly_supported | unsupported_or_incoherent | not_applicable", "rationale": "..."},
  "evolution_evidence": {"latest_meaningful_changes": [], "model_movement_evidence": [], "operational_movement_evidence": [], "new_issues": [], "persistent_issues": [], "resolved_or_mitigated_issues": [], "strongest_current_development_tension": {"topic": "...", "why_this_is_strongest_now": "...", "relationship_to_previous_scenario": "...", "relationship_to_original_baseline": "...", "evidence_fields": []}},
  "development_discussion_options": [{"topic": "...", "why_it_matters": "...", "supporting_evidence": [], "participant_wider_question": {"question": "...", "supporting_evidence": []}}],
  "continuity_update": {"what_changed": "...", "watch_next": "..."},
  "analytical_narrative_draft": {"current_state_read": "...", "movement_read": "...", "operational_fit_read": "...", "reality_check_read": "...", "development_landscape_read": "..."}
}
```

Hidden baseline is qualitative context only. It must not return `development_discussion_options`, participant-visible questions, or an active selected discussion point. Baseline development issues belong only in `analytical_narrative_draft.development_landscape_read` and compact baseline orientation/watch context.

Visible iterations should return exactly one `development_discussion_options` item. The option pairs the strongest current development tension with one participant-visible wider debate question. The option should compare the current scenario with both the previous visible scenario and the original baseline when those contexts are available.

Pass 2 should return:

```json
{
  "review_metadata": {"review_mode": "first_visible_iteration | later_visible_iteration", "visible": true},
  "operational_fit": {"points": 0, "relationship_to_previous": "...", "reason": "...", "evidence_fields": [], "boundary_check": "..."},
  "reality_check": {"points": 0, "relationship_to_previous": "...", "carryover_status": "...", "new_issue_status": "...", "reason": "...", "incremental_check": "...", "evidence_fields": [], "allocations": []},
  "score_evolution_read": {"direction": "...", "main_reason": "...", "active_issue_to_carry_forward": "..."}
}
```

Pass 2 receives the Pass 1 evidence, previous score trace, carryover candidate, current Completion Outlook movement, operational context, and allowed Reality Check allocation targets. The LLM owns the Operational Fit and Reality Check judgment for new states. The app validates numeric ranges, evidence references, allocation target IDs, baseline-return neutralization, same-state replay, and arithmetic.

Pass 3 should return:

```json
{
  "review_metadata": {"review_mode": "first_visible_iteration | later_visible_iteration", "visible": true},
  "trial_score_narrative": {"summary": "...", "movement_reading": "...", "score_interpretation": "..."},
  "pillar_reading": [{"pillar": "...", "reading": "..."}],
  "central_tension": {"summary": "...", "why_it_matters": "..."},
  "broader_strategic_question": {"mapped_tension": "...", "question": "..."}
}
```

The stable JSON fields are rendered into three participant-facing sections:

- `Trial Score`: combines three labeled subparagraphs into one integrated read: `trial_score_narrative.summary` as `Overall Evolution`, `movement_reading` as `Completion Outlook`, and `score_interpretation` as `Reality Check`.
- `What Is Driving The Score`: renders `pillar_reading` as two to four material bullets. Each bullet may cover one pillar or combine related pillars/subpillars; it should not mechanically list every pillar or repeat the same central message from the Trial Score section.
- `Discussion Point: <topic>`: renders the selected concise `central_tension.summary` as the section title, followed by `central_tension.why_it_matters` and the paired wider `broader_strategic_question.question`.

Provider validation enforces that Pass 3 `central_tension.summary` matches the supplied Pass 1 development topic and that `broader_strategic_question.question` matches the paired Pass 1 question. Repetition avoidance is prompt-level guidance for Pass 1/Pass 3 rather than a hard validation blocker. Same-state reuse or direct storyline continuity may keep the same topic; otherwise, the single topic should reflect the strongest current tension.

Reality Check is a scoring correction / realism adjustment. It should explain whether pre-reality check score movement is coherent, realistic, and incrementally supported by scenario evidence. It must not select the participant-visible discussion point.

When the immediately previous visible review has a material negative Reality Check, the packet may include `iteration_context.reality_check_carryover_candidate`. Pass 2 receives that candidate and decides how the previous issue relates to the current score evolution through `reality_check.carryover_status`, `new_issue_status`, `relationship_to_previous`, and `score_evolution_read.active_issue_to_carry_forward`. The app validates the returned points and preserves same-state/baseline-return rails, but it does not run the old symbolic carryover formula.

Gated premise-sensitive field changes do not automatically reset carryover. They are strong context for Pass 1 evidence and Pass 2 scoring to decide whether the prior concern has been superseded by a new development premise. Exact same-state reuse remains the higher-priority path: if the scenario state matches a prior visible review, the app reuses that prior score trace instead of asking Pass 2 to rescore.

Participant-facing Reality Check wording should explain the direction of the adjustment in plain language when material: it may offset an apparent gain, reinforce a movement, rarely soften a decline when the accepted scoring adjustment supports it, or reverse a misleading pre-reality check movement. It should not expose points or exact score arithmetic.

Participant-facing operational wording should not expose Operational Fit as a separate component. When Pass 2-rated operational evidence matters, Pass 3 should absorb it into the relevant pillar or Completion Outlook read using plain language such as right scale, footprint, duration, size, or operational dimensions.

## 13. Plot Integration Guidance

The active score views are Completion Outlook, Reality Check, and Trial Score. Completion Outlook remains model-derived. Operational Fit and Reality Check are LLM-scored adjustments validated by app rails and rendered as part of the Trial Score view without being described as SHAP/model drivers.

Debug and audit JSON should expose current structures only:

- Pass 1 analytical basis: `completion_outlook_analysis`, `evolution_evidence`, `strategy_shift_check`, one `development_discussion_options` item for visible iterations, `continuity_update`, and `analytical_narrative_draft`.
- Pass 2 scoring result: `operational_fit`, `reality_check`, `score_evolution_read`, app-validated arithmetic fields, and validation notes.
- Pass 3 participant result: `trial_score_narrative`, `pillar_reading`, selected `central_tension`, and selected `broader_strategic_question`.
- Participant-visible history: only Pass 2 selected development discussion pairs in `recent_participant_visible_questions`.
- Hidden baseline compact context: completion outlook summary, baseline orientation/watch context, and `baseline_development_landscape`; no active discussion topic or participant question history.

Do not expose legacy Design Confidence, Total Scenario Score, main/alternative Pass 1 candidate development discussion fields, or old Strategic Review fields in active debug JSON.

### Active Three-Pass Scenario Review

As of 2026-06-22, the active Trial Score Scenario Review implementation uses three provider-facing stages:

```text
Pass 1: Evolution and Evidence
Pass 2: Score Adjudication
Pass 3: Participant Narrative
```

This supersedes older text in this document wherever it says Pass 1 returns Operational Fit ratings, Reality Check effect/strength, or multiple development discussion options for participant-narrative selection. The active contract is:

- Pass 1 produces evidence/evolution only, plus one strongest current development tension and paired wider question.
- Pass 2 lets the LLM assign Operational Fit and Reality Check points directly inside app validation rails, comparing current evolution with previous score trace, carryover, and new/resolved issues.
- Pass 3 shapes the accepted score trace into participant-facing narrative and must not re-score.

The app still owns XGBoost Completion Outlook, SHAP/calibration boundaries, score arithmetic, evidence-ref validation, score ranges, baseline-return neutralization, same-state replay/cache reuse, storage, and UI rendering. New-state scoring is intentionally less reproducible because the scoring judgment is LLM-owned; identical same-state replay remains deterministic.

## 14. Narrative Tone Rules

Pass 1 prioritizes analytical depth, not participant-facing style. It should use packet evidence, model evidence, operational context, trial text, relevant reference-pack summaries, and general clinical-development expertise to explain clinical-development meaning. Reference packs can support interpretation, but the provider must not imply a document supports a claim unless the pack actually provides that support.

Pass 3 owns final participant-facing wording. It should use the validated Pass 1 analysis and accepted Pass 2 score context without recalculating scores, re-rating Operational Fit, re-deciding Reality Check, or adding unsupported clinical/regulatory claims. It should write in conditional clinical-development language because the narrative is an interpretation of scenario evidence, not a claim of fact.

The participant-visible wider question should be a broad development-debate question mapped to the selected discussion topic. Facilitator questions are deferred out of the main participant-narrative contract and should be generated separately if reintroduced.

## 15. Memory And Iteration Policy

The storyline is application-owned, not implicit LLM memory. Hidden baseline provides context but does not consume or define participant-visible topics.

Store and pass forward:

- Baseline orientation/watch context from hidden baseline.
- Previous visible Trial Score trace and compact score movement context.
- Pass 2 selected `central_tension` and `broader_strategic_question`.
- `recent_participant_visible_questions`, limited to recent selected pairs.
- Changed fields, operational assumptions, model evidence, and score diagnostics needed for audit.

For visible iterations, Pass 2 uses participant-visible history to preserve continuity or avoid unnecessary repetition. Only selected Pass 2 pairs enter active history; raw Pass 1 options do not.

### Review Regeneration And No-Op Policy

The application should avoid creating a new storyline step when the scenario state has not materially changed. Same-state reuse may reuse scores and ask Pass 2 to describe the latest move as a return to a prior reviewed state. Text-only edits should trigger review only when they materially change endpoint intent, population scope, intervention description, rationale, or create a structured-field contradiction.

### Live Regression Targets

Keep a small named trial set for manual live Scenario Review calibration after schema or prompt changes:

- `NCT03386721` - Simlukafusp alfa (ROCHE), Oncology, 2018: review whether Execution Framework / Operational Fit and Reality Check narratives are specific, expert, auditable, and strategically useful without prescribing the next edit.
- `NCT03896581` - `[BE COMPLETE]` Bimekizumab (UCB), Musculoskeletal, 2019: change Pathway Profile from `Interleukin Cytokine` to `Kinase Inhibitor` and review whether Trial Score explains the clinical-development meaning of a pathway-class change without overclaiming mechanism, efficacy, or regulatory implications.

## 16. Reproducibility And Provider Fallback

The architecture supports OpenAI and Gemini provider calls without binding product logic to one provider.

Provider selection and secret handling:

- Current interactive development default should be Gemini-only with `gemini-3.1-flash-lite`, because the June 2026 live audit found it fast, low-cost, and valid after schema/thinking/output-budget hardening. Use `NARRATIVE_LLM_PROVIDER=gemini` and set `NARRATIVE_LLM_FALLBACK_PROVIDER=gemini` only to express a no-effective-fallback profile; the config loader normalizes same-provider fallback to `None`.
- Longer-term production resilience may re-enable a provider chain after latency, reliability, budget, and provider-output quality are calibrated. In that future mode, the configured primary provider and fallback provider should be explicit and auditable. OpenAI high-reasoning or Pro-class models should be treated as slower offline/validation candidates unless live-play latency and cost prove acceptable.
- Provider config code must read secrets from environment variables or deployment secret managers only. It must never store API keys in committed Python files, notebooks, docs, fixtures, or frontend state.
- Local development may use `.env` loaded by `python-dotenv`, following the existing project pattern. Deployment should use Cloud Run environment variables or Secret Manager-backed values.
- Recommended environment variables:
  - `NARRATIVE_LIVE_REVIEW_ENABLED=1`, only when the simulator should call the configured live provider chain instead of the mock reviewer.
  - `NARRATIVE_LLM_PROVIDER=gemini`
  - `NARRATIVE_LLM_FALLBACK_PROVIDER=gemini`, for a Gemini-only profile with no effective fallback after config normalization.
  - `OPENAI_API_KEY`, optional for the current Gemini-only profile and needed only for OpenAI validation or future provider-chain tests.
  - `OPENAI_NARRATIVE_MODEL`, optional for the current Gemini-only profile.
  - `OPENAI_REASONING_EFFORT=high`, optional for slower OpenAI validation paths.
  - `GEMINI_API_KEY` or existing `GOOGLE_API_KEY`
  - `GEMINI_NARRATIVE_MODEL=gemini-3.1-flash-lite`
  - `NARRATIVE_LLM_TEMPERATURE`
  - `NARRATIVE_LLM_SEED`, only for providers/models that support seed-like reproducibility.
  - `NARRATIVE_LLM_MAX_OUTPUT_TOKENS=25000`
  - `NARRATIVE_LLM_TIMEOUT_SECONDS`
  - `NARRATIVE_LLM_MAX_RETRIES`
- Current setup status: local `.env` can hold these values, and `src/narratives/provider_config.py` reads and validates them without making any LLM API call. `scripts/check_narrative_openai_smoke.py` and `scripts/check_narrative_gemini_smoke.py` can run opt-in API smoke tests when `RUN_NARRATIVE_OPENAI_SMOKE=1` or `RUN_NARRATIVE_GEMINI_SMOKE=1` is set; they skip by default to avoid accidental network calls or API spend. `src/narratives/provider.py` contains real OpenAI and Gemini invocation helpers behind the same normalized provider result shape. `frontend/views/trial_simulator.py` uses the deterministic mock provider by default and routes both hidden baseline and visible Scenario Review calls through the live provider chain only when `NARRATIVE_LIVE_REVIEW_ENABLED=1`. Live wrapper checks have validated full fixture reviews using the normalized provider boundary.
- Historical provider config, prompt/schema fixtures, opt-in OpenAI/Gemini smoke testing, and opt-in simulator UI routing originally targeted an earlier Scenario Review contract. Those artifacts remain useful only as provenance or stable scenario inputs. The active contract is Trial Score V1 in `docs/trial_score_narrative_direction.md` and `src/narratives/trial_score_contract.py`.
- Active provider prompts use packet-section names consistently. `Completion Outlook score` is `model_interpretation.completion_score`; readable score evidence is supplied through `model_interpretation`; `Trial description fields` live under `text_context` with keys such as `title`, `summary_ui`, `conditions_ui`, `interventions_ui`, and `primary_outcomes_ui`; planning assumptions are `operational_assumptions.planned_enrollment`, `planned_sites`, and `planned_duration_months`; Pass 1 output is `completion_outlook_analysis`, `evolution_evidence`, `strategy_shift_check`, `development_discussion_options`, `continuity_update`, and `analytical_narrative_draft`.
- Participant-facing Completion Outlook wording should avoid internal model vocabulary. Use plain phrases such as `Completion Outlook score inputs`, `score inputs`, `score pattern`, `score-driving fields`, or `early-termination risk pattern` when explaining the scoring boundary; avoid phrases such as `model-facing`, `model signal`, `model-score inputs`, `model suggests`, `model indicates`, `model registers`, `model-derived`, `model interpretation`, `in the model`, `model's...`, or `the model reflects`. The provider prompt asks the model to replace any remaining internal model-language phrase before finalizing participant-facing text.
- Planning-assumption fields are outside Completion Outlook: planned enrollment, planned site count, and planned total duration do not feed the XGBoost score. They feed Operational Fit and may inform Reality Check or the analytical narrative when they affect proportionality, retention, evidence completeness, or execution realism.
- Mixed structured-plus-planning changes should keep source boundaries clear. Completion Outlook explains score-input movement; Operational Fit explains planning-assumption proportionality; Reality Check explains whether the combined pre-reality check movement is coherent and incrementally supported.
- Visible-iteration narratives should privilege latest movement over persistent state. A field changed in an earlier iteration may remain an unresolved current-state constraint, but it should not be described as driving the latest score movement unless model movement evidence shows that field's impact changed again. A previously negative unchanged field should not become a positive argument in a later iteration unless the latest change demonstrably improves its fit or model impact; otherwise it remains an unresolved constraint or quality concern. If another latest field changes the weight or interpretation of a persistent field, the narrative should explain that interaction explicitly.
- Non-operational structured changes, such as rare-disease status, modality, indication, endpoint duration, governance, or administration burden, can change the context around unchanged enrollment, site count, or duration. Operational Fit is a current-state score, so previous Operational Fit points are preserved when the operational assumptions, operational benchmark/movement context, and structured scenario context match a previous accepted trace. If that Operational Fit state no longer matches, Pass 2 may reassess Operational Fit inside the app rails. Full same-state replay remains broader: when the whole scenario state matches a previous accepted scenario, the app reuses the whole score trace, including Operational Fit and Reality Check. Compact score continuity keeps the latest 5 accepted traces for component matching, structured-feature interpretation continuity, and compact Reality Check memory while avoiding raw prompt/narrative history in scoring input.
- `primary_duration_months_ml` alone is not Operational Fit scoring evidence. If an endpoint-duration change is already reflected in Completion Outlook as endpoint maturity or follow-up evidence movement, Reality Check should stay neutral unless there is an incremental contradiction, shortcut, unsupported assumption, or realism problem beyond that model movement. Non-neutral Reality Check allocation `incremental_check` text should explain this non-duplication explicitly.
- Reality Check is a conservative challenge layer. It should default to neutral unless there is a clear incremental reason not already captured by Completion Outlook or Operational Fit. It should be more willing to challenge favorable movements than to soften unfavorable movements. For negative pre-reality check movement, `soften_decline` should be rare and should require a material decline plus a newly changed, concrete compensating strength. Unchanged strengths can provide context, but should not be the main basis for a non-neutral adjustment.
- Reality Check can be aggressive when model-favorable simplification is clinically misleading. If a positive pre-reality check movement is mainly caused by weakening governance, oversight, evidence collection, or critical-to-quality design protections, `offset_gain` with strength `strong` may be appropriate for a strong offset. A true reversal requires both `effect: reversal` and `strength: reversal`; `effect: reversal` with `strength: strong` is only a strong offset and will not cross through neutral.
- Resource, staffing, and budget implications remain qualitative only unless explicit financial inputs exist. Added resource intensity should be discussed through operational proportionality and evidence-completeness risk, not as a cost model.
- Operational simplification may improve Operational Fit when it genuinely improves executability. When simplification is achieved mainly by weakening comparator, masking, allocation, endpoint rigor, or evidence ambition, Reality Check may offset the apparent gain if the movement looks shortcut-driven.
- For hard product-boundary cases, the app may pass narrow `review_controls` to the provider. These controls should define the Completion Outlook mode, latest-change focus, and forbidden latest fields for Completion Outlook; they should not turn Operational Fit, Reality Check, or Pass 2 narrative selection into templates.
- The deterministic Completion Outlook boundary is shared in `src/narratives/review_controls.py`. For operational-only and stable non-score-input modes, Completion Outlook wording may be normalized while Operational Fit, Reality Check, development discussion options, and Pass 2 narrative remain governed by the active Trial Score contract.
- When `review_controls` are present, participant-facing narratives should explain the latest change without re-labeling older cumulative issues as newly changed. Older issues may remain relevant to the current full scenario, but they should not be described as if they were introduced by the latest edit.
- For later visible iterations, Pass 1 proposes one development discussion option and Pass 3 uses that participant-visible wider debate question unless same-state/direct-continuity context requires preserving a prior topic. Facilitator questions are deferred out of the main participant-narrative contract.
- Trial description fields do not directly feed the Completion Outlook score. They may support the Completion Outlook narrative only when they align with, clarify, or add non-conflicting detail to selected Completion Outlook score inputs. This conflict rule applies across all Trial description fields in `text_context` and all relevant `structured_features`, not only intervention descriptions. Completion Outlook score inputs define the score-interpreted scenario when they directly conflict with Trial description fields. Only the conflicting Trial description field detail should be treated as stale scenario text superseded by the structured_features value; it should not be used as Completion Outlook evidence or as evidence that the selected structured design has the contradicted modality, delivery burden, endpoint, or population feature. Non-conflicting Trial description field details and latest `text_context` changes remain valid context when they clarify population, endpoints, intervention rationale, or trial context. In the participant warning, "text is used as supporting context" means aligned or non-conflicting Trial description field content; the directly conflicting detail remains stale scenario text superseded by the corresponding `structured_features` value.
- `structured_features` / `text_context` conflict is a scenario-readiness warning. It may affect the analytical draft or Reality Check when it weakens interpretability, and newly introduced material contradictions should usually increase negative Reality Check calibration. It should not be converted into a separate obsolete scoring layer.
- When only the three planning-assumption fields changed and the Completion Outlook score delta is `0.0`, the app may deterministically set the participant-facing Completion Outlook boundary sentence before storing/reporting the trace. This fixed sentence is exclusive to the planning-assumption-only boundary mode and must not be reused for `structured_features` / `text_context` consistency cases or intervention-modality changes. This is a product boundary, not a clinical judgment, and should not alter Operational Fit, Reality Check, development discussion options, or Pass 2 narrative selection.
- Do not add post-narrative deterministic cleanup for participant wording or question rewriting at this stage. Internal-language leaks and repeated/similar questions should be handled by prompt wording and eval findings only, except for the existing fixed planning-assumption Completion Outlook boundary sentence and provider-neutral unavailable-review error formatting.
- Latest three-trial live Gemini run `first_wave_operational_shortcut_cap_3trials_1` returned 12/12 reviewed visible iterations, 0 failed checks, and 3 warning checks. The operational shortcut cap behaved as intended, with shortcut-driven simplification receiving only limited operational credit. Remaining warnings were question opening-frame repetition and one scenario-readiness dominance review item, so no urgent prompt change is required before the next broader wave.
- Broader five-trial live run `first_wave_broader_trials_5_1` returned 20/20 reviewed visible iterations, 3 failed checks, and 10 warning checks. Follow-up adjustments are eval/prompt-boundary only: avoid `model signals` by using score-pattern wording, avoid over-crediting patient-relevance when the synthetic population edit conflicts with a prevention/vaccine-style trial objective, and require operational-only medical questions to reference planning burden, scale, or proportionality.
- The older three-question contract tested after `first_wave_broader_trials_5_2` was useful diagnostically but is no longer the target. The active simplified contract uses two questions: one medical/clinical-development question grounded in the current trial scenario and one strategic development question that raises a broader development-path or field-level challenge.
- Latest five-trial live run `first_wave_three_question_contract_5_1` returned 19/20 reviewed visible iterations, 4 failed checks, and 20 warning checks. One fail was a transient Gemini `ServerError`, not a prompt issue. Follow-up changes remain light: remove participant-facing `in the model` leakage by strengthening score-pattern replacement language, make the patient-relevance expectation skip broader prevention/vaccine contexts when refractory/metastatic edits conflict with the base objective, and vary the strategic/field question lens to reduce repeated `What evidence standard...` / `How should the field balance...` openings.
- Any OpenAI model used in later validation should be pinned to an explicit snapshot rather than a floating alias. OpenAI Pro/high-reasoning profiles can be considered for slower, high-quality hidden baseline generation or offline review, but they are not the default live interactive path after the June 2026 Gemini Flash-Lite decision.
- Any Gemini model used in production or fallback should be configured with an explicit model ID rather than hard-coded in product logic. The current live interactive candidate is `gemini-3.1-flash-lite`; a Pro-class Gemini model can be evaluated later for slower offline or fallback review quality.
- Future provider-chain mode should try the configured primary provider first, then the configured fallback only for provider/network/rate-limit/unavailable failure. Do not fallback when the primary provider returns valid but unfavorable clinical reasoning, or when the provider returns malformed/invalid review JSON; that would create provider-shopping behavior and hide prompt/contract problems that should be fixed.
- Cache and trace keys must include provider, model name, and live generation-control namespace so OpenAI and Gemini outputs, or outputs produced with different reproducibility settings, are not treated as interchangeable.
- For live provider-chain calls, if the same input packet and same live generation-control namespace already have a validated cached review, reuse it before calling any provider. This keeps provider fallback transparent to participants: the app does not regenerate a review just because the provider that answered last time differs from the provider that would answer now.
- Keep the implementation deliberately small: one provider config reader, one prompt/schema builder, and one normalized provider result shape shared by mock, OpenAI, and Gemini. Avoid separate scoring, packet-building, cache, or UI code paths per provider.
- In future provider-chain mode, provider fallback should be bounded and auditable. Try at most the configured primary and one configured fallback for a given packet. Store which provider failed, why it failed, and which provider generated the accepted review. Do not silently retry multiple times or cascade across many models.
- Provider identity should remain transparent to participants. The participant panel should not say whether OpenAI, Gemini, or another live provider produced a review. Provider and model names belong in trace/debug/facilitator metadata only.
- If all configured live providers fail, return an unavailable Scenario Review state and show Completion Outlook only. Do not reuse stale Operational Fit, Reality Check, Trial Score, or participant narrative for the new packet.
- If a future fallback provider succeeds, cache the fallback result under its own provider/model namespace. Do not overwrite or pretend it is the primary provider result.
- Full narrative reviews need enough output budget for reasoning plus seven domain ratings and participant-facing review lines. Treat incomplete provider responses caused by output-token limits as provider failures, not as valid reviews.
- Runtime controls such as temperature, seed, reasoning effort, and JSON-output controls are provider/model-specific. Config may read them, but real provider code should send only parameters supported by the selected model. For the pinned GPT-5.5 OpenAI snapshot, use `reasoning.effort` from `OPENAI_REASONING_EFFORT`, request JSON output through the Responses API text format, and do not send temperature or seed unless a future model-specific capability check proves they are accepted. For Gemini, send temperature and seed only through Gemini's supported generation config and request JSON through Gemini's response MIME/config path. Store configured versus applied generation controls in provider metadata for every real provider result.
- Live latency must be measured separately for provider smoke calls, hidden baseline generation, visible iteration review, provider-chain fallback, and cache replay. Current helper script: `scripts/benchmark_narrative_latency.py`. Use it to compare OpenAI and Gemini on the same baseline/iteration packets, estimate worst-case timeout budget from `timeout_seconds * (max_retries + 1) * provider_count`, and test interactive profiles such as lower timeout, no retry, lower reasoning effort, or smaller output budget.
- Current UI behavior creates or ensures hidden baseline review context when Simulation Mode initializes a baseline snapshot. With live review enabled, this can create an invisible wait on toggle. The hidden baseline trace is stored only in Streamlit session state and reusable only within the same running session/runtime/input hash; it is not durable across app restarts or separate teams. Deferring hidden baseline generation until the first Predict click would make toggle faster but move the baseline wait into the first prediction workflow, where the first visible review may require baseline generation followed by visible iteration generation.

Live-provider latency experiments run during the June 2026 implementation session:

- Minimal provider smoke checks were fast and only prove key/config/connectivity: OpenAI completed in about 2.5 seconds and Gemini completed in about 5.0 seconds. These smoke timings are not representative of the full Quality Review prompt.
- Full hidden-baseline generation with OpenAI, high reasoning effort, 6000 max output tokens, 60 second timeout, and one retry took about 120 seconds and succeeded after two attempts. This confirms that hidden baseline creation can be a substantial invisible wait if triggered when Simulation Mode opens.
- A full visible-iteration provider-chain call with the same high-quality settings took about 149 seconds: OpenAI timed out, then Gemini fallback succeeded in about 28 seconds. This shows that fallback can rescue the review, but the primary-provider timeout is paid first.
- A lower-latency interactive profile tested `OPENAI_REASONING_EFFORT=medium`, `NARRATIVE_LLM_MAX_OUTPUT_TOKENS=3500`, timeout 35 seconds, and zero retries. In that profile, baseline-chain calls were still about 56 seconds because OpenAI timed out first and Gemini then answered. Visible-iteration calls were about 62 seconds when Gemini returned non-JSON. Failed reviews are not cached as reusable validated reviews, so repeating the same failed scenario may call providers again.
- Direct provider comparison on the same baseline and visible-iteration packets, with timeout 90 seconds, zero retries, 3500 max output tokens, and medium OpenAI reasoning effort, showed OpenAI slower but valid on both packets: about 36 seconds for baseline and 50 seconds for visible iteration. Gemini was faster on baseline, about 22 seconds and valid, and faster on visible iteration, about 26 seconds, but returned malformed/non-JSON output for the visible-iteration contract.
- Earlier interpretation from the first provider benchmark: OpenAI was the more reliable full-contract provider in those tests; Gemini was faster when it returned valid JSON, but needed schema and generation-control hardening before it could be treated as the live interactive default.
- Historical OpenAI-only live-playtesting recommendation was `OPENAI_REASONING_EFFORT=medium`, `NARRATIVE_LLM_MAX_OUTPUT_TOKENS=3500`, `NARRATIVE_LLM_TIMEOUT_SECONDS=60`, and `NARRATIVE_LLM_MAX_RETRIES=0`. The current validated interactive recommendation is the Gemini-only Flash-Lite profile described below; OpenAI high-reasoning profiles remain candidates for slower offline validation, not the default live interactive path.
- The main artificial slowdown drivers are primary-provider timeout, retry count, reasoning effort, output budget, and hidden-baseline generation timing. With the current 100-second timeout, one retry, primary plus fallback, the worst-case provider wait can approach 400 seconds before normal generation overhead.
- Cache behavior should be interpreted carefully: validated successful reviews can be replayed for the same packet and same generation-control namespace; malformed, incomplete, provider-error, or validation-failed reviews should not be treated as reusable Scenario Reviews.
- Local Completion Score prediction API latency was not measured in these provider benchmarks because the local `/health` endpoint was unavailable during the shell check. The observed multi-minute UI delay was dominated by live LLM generation and provider fallback, not by known XGBoost scoring work.

Implementation-time cost-control decision after the first billing check:

- During active coding and UI iteration, keep live review disabled by default with `NARRATIVE_LIVE_REVIEW_ENABLED=0`. This makes the deterministic mock path the default and prevents accidental API spend.
- For many low-cost live tests per day, use a single cheap primary provider rather than a fallback chain. The current validated local development profile is Gemini-only with no effective fallback: `NARRATIVE_LLM_PROVIDER=gemini`, `NARRATIVE_LLM_FALLBACK_PROVIDER=gemini`, `GEMINI_NARRATIVE_MODEL=gemini-3.1-flash-lite`, visible Pass 1 `thinking_level=medium`, Pass 2 scoring and Pass 3 narrative `GEMINI_THINKING_LEVEL=medium`, `NARRATIVE_LLM_MAX_OUTPUT_TOKENS=25000`, `NARRATIVE_LLM_TIMEOUT_SECONDS=100`, and `NARRATIVE_LLM_MAX_RETRIES=0`. The config loader collapses same-provider fallback to `None`. Hidden baseline uses the same 25000 output-token ceiling while keeping Gemini `thinking_level=medium`, request timeout capped at 100 seconds, compact prompt wording, no hidden-baseline word-count minimum, and deterministic compact fallback context if the bounded provider pass fails.
- Keep the OpenAI model configured as a cheaper reserve option, such as `gpt-5.4-mini`, rather than `gpt-5.5` during implementation. `gpt-5.5` should be reserved for rare high-quality/offline validation, not repeated local development testing.
- This implementation-time profile is intentionally different from the later production resilience design. Production can re-enable a primary/fallback chain after latency, reliability, budget, and provider-output quality are calibrated.

Open live-play calibration items before rollout:

- Decide whether hidden baseline generation remains on Simulation Mode toggle or is deferred until first `Predict Trial Completion`. Toggle-time generation makes the first prediction faster after the wait; first-predict generation keeps toggle responsive but can make the first prediction require both baseline and visible-review calls.
- Continue monitoring Gemini's full visible-iteration JSON reliability during representative live play. The current hardening uses SDK response schema, omitted/default temperature, visible Pass 1 `thinking_level=medium`, Pass 2/Pass 3 primary `thinking_level=medium`, a 12000-token minimum schema output floor, and explicit malformed/MAX_TOKENS repairs that use the centralized 25000 output-token ceiling with lower thinking, while preserving app-owned validation and scoring.
- Measure the local `/predict` API separately once the API is running, so model scoring time is separated from provider time in latency budgets.
- Repeat timing tests with representative real trial scenarios, not only contract fixtures, before setting production timeout/retry defaults.
- Make a first qualitative assessment of live participant-review text across several representative real scenarios before deciding whether the prompt should become shorter, more structured, or model-specific.
- Keep the participant UI focused on Completion Outlook, Operational Fit, Reality Check, and Trial Score while preserving the distinction between XGBoost Completion Outlook drivers and app-owned review adjustments.
- During live testing, expose compact timing diagnostics for successful and failed Scenario Reviews. The diagnostics should separate hidden-baseline lookup/generation time, visible-review provider/store time, total visible workflow time, provider latency, attempts, cache hits, configured timeout, applied provider timeout, response length, and validation status. These diagnostics belong in an expander for calibration/debugging, not in the main participant narrative.

Historical mid-way development sanity check:

This review has been superseded by the four-pillar Design Confidence plan. It remains useful as historical context for why prompt size, provider reliability, and participant display need explicit calibration.

1. Prompt, token, output, and cost audit. First inspect exactly what the application sends to the LLM for `hidden_baseline`, `first_visible_iteration`, and `later_visible_iteration` prompts: prompt instructions, response contract, packet JSON, field-change evidence, XGBoost movement evidence, `text_context` Trial description fields, baseline/previous review context, operational assumptions, and clarification context. For each representative packet, record prompt character count, approximate input tokens, configured output budget, actual response length, parser/validation result, cache behavior, and estimated cost for the selected model. Then compare observed wall-clock time against this input/output volume and provider limitations. The audit should make the LLM input readable as if a facilitator had written the prompt manually, while keeping secrets and raw provider outputs out of participant UI.
2. Historical Design Confidence and participant display review. Reassess whether the four design subcategories, rating-to-point mapping, and participant wording were coherent, legitimate, and easy to explain.
3. Historical implementation plan for prompt/UI changes. Only after the first two reviews decide what to change in code: which packet fields should be removed, summarized, or added; whether hidden baseline generation should be deferred or cached durably; which provider/model profile should be used for interactive play; and how the old Total Scenario Score, Completion Outlook drivers, and Design Confidence contributions were rendered in the simulator.

Gemini JSON reliability finding from the NCT02741128 live audit:

- The real visible-iteration prompt was moderate in size, about 13.9k characters and 3.5k estimated input tokens. Prompt volume alone did not explain 45-second waits.
- The malformed JSON failures were caused by Gemini spending too much of the generation budget on hidden thinking and leaving too little visible budget to complete the JSON object. With `NARRATIVE_LLM_MAX_OUTPUT_TOKENS=2500`, one failed call used about 1.6k thinking tokens and cut the JSON mid-string.
- The Gemini provider path now uses the SDK response schema and the Gemini 3 `thinking_level` control. The current production-style setting is `gemini-3.1-flash-lite`, omitted/default temperature, visible Pass 1 `thinking_level=medium`, Pass 2/Pass 3 primary `thinking_level=medium`, and a 12000-token minimum schema output ceiling. The output ceiling is a completion-safety margin, not a quality knob; in earlier medium-thinking benchmarks, ceilings from 4000 through 12000 produced the same visible review quality and actual token use for the tested scenarios. The 12000-token default is retained to prepare for longer future reviews.
- The explicit metadata-visible repair path is reserved for malformed JSON or provider `MAX_TOKENS`; repairs lower thinking to `low` and use the centralized 25000 output-token ceiling because excessive thinking can consume output budget and harm JSON completion.
- Later temperature/thinking evals on five-trial Scenario Review waves showed omitted/default temperature gave better visible quality than explicit `0` or `0.3`, while explicit high thinking reduced failed/warning checks versus default medium thinking. Higher thinking did not make outputs deterministic: duplicate runs still drifted in scoring and wording. Therefore high thinking is a quality setting, not a reproducibility guarantee; reproducibility-sensitive evals should keep using duplicate traces and drift inspection.
- Real-provider diagnostics now record token usage metadata when available, including Gemini prompt, candidate, thought, cached-content, and total token counts; they also record finish metadata such as finish reason and safety-rating count when exposed by the SDK.

Current structured/text consistency-check status:

- Removed from the active simulator.
- `Predict Trial Completion` proceeds directly with edited structured fields, text fields, and operational assumptions.
- The deterministic alignment module and checker are not part of the active workflow.
- The active Scenario Review path should not return `clarification_needed` before scoring.

Implemented structured-field red flags:

These are UI coherence checks for impossible or internally incompatible structured Trial Feature combinations. They are not Trial Score scoring rules and should stay separate from Operational Fit / Reality Check calibration. Implementation should mirror the existing placebo consistency behavior: run immediately when fields change, highlight involved controls with red background, do not add amber states, do not add a compact warning card for now, and do not auto-correct except for the existing placebo sync behavior. Red-highlighted fields should remain highlighted until the incompatible combination is resolved, but they must not disable `Review Scenario` or block Scenario Review generation.

Implementation artifacts: `frontend/utils/structured_incompatibility.py`, `frontend/views/trial_simulator.py`, and `scripts/check_structured_incompatibility.py`.

- `Intervention Model = Parallel` with `Number of Arms <= 1`: highlight `Intervention Model` and `Number of Arms`.
- `Allocation Method = Randomized` with `Intervention Model = Single Group`: highlight `Allocation Method` and `Intervention Model`.
- `Allocation Method = Randomized` with `Number of Arms <= 1`: highlight `Allocation Method` and `Number of Arms`.
- `Benchmark Comparator = Placebo` with `Placebo Control = No`: highlight `Benchmark Comparator` and `Placebo Control`.
- `Placebo Control = Yes` with `Benchmark Comparator = No Control Group or Not Specified`: highlight `Placebo Control` and `Benchmark Comparator`.
- `Placebo Control = Yes` with `Number of Arms <= 1`: highlight `Placebo Control` and `Number of Arms`.
- `Intervention Model = Single Group` with `Placebo Control = Yes`: highlight `Intervention Model` and `Placebo Control`.
- `Benchmark Comparator = Active / Legacy Standard` or `Active / Modern Standard` with `Intervention Model = Single Group`: highlight `Benchmark Comparator` and `Intervention Model`.
- `Benchmark Comparator = Active / Legacy Standard` or `Active / Modern Standard` with `Number of Arms <= 1`: highlight `Benchmark Comparator` and `Number of Arms`.
- `Bias Control = Double Blind`, `Triple Blind`, or `Quadruple Blind` with `Intervention Model = Single Group`, `Benchmark Comparator = No Control Group or Not Specified`, and `Placebo Control = No`: highlight `Bias Control`, `Intervention Model`, `Benchmark Comparator`, and `Placebo Control`.
- `Bias Control = Double Blind`, `Triple Blind`, or `Quadruple Blind` with `Number of Arms <= 1`, `Benchmark Comparator = No Control Group or Not Specified`, and `Placebo Control = No`: highlight `Bias Control`, `Number of Arms`, `Benchmark Comparator`, and `Placebo Control`.
- `Intervention Model = Factorial` with `Number of Arms <= 1`: highlight `Intervention Model` and `Number of Arms`.
- `Intervention Model = Crossover` with `Number of Arms <= 1`: highlight `Intervention Model` and `Number of Arms`.
- `Intervention Model = Sequential` with `Number of Arms <= 1`: highlight `Intervention Model` and `Number of Arms`.

Do not include the following as red flags in this first pass because they can be unusual but not impossible: `Phase 3` with `Single Group`, `Confirmatory / Registration` with `Single Group`, `Hard Clinical (Survival/Death)` with short endpoint duration, `Advanced / Metastatic` with `Adjuvant / Neoadjuvant`, `Prevention` with patients only, `Healthy Volunteers = Yes` with non-healthy patient fields, or `Data Monitoring Committee = No` in high-risk or pivotal trials. These may remain analytical considerations or future amber warnings, but not hard red structured-field incompatibility flags.

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

- Current prototype trace should remain simple and session-state compatible. It should store `input_packet`, provider/mock Pass 1 output, validated Pass 1 review, Pass 2 scoring review, validated scoring diagnostics, Pass 3 participant narrative JSON, validation status/errors, accepted Operational Fit points, Reality Check points, Trial Score, changed fields, score movement, provider/model identity, selected participant development discussion, participant-visible question history, and compact storyline memory.
- Current real-provider traces store prompt template version, response schema version, configured/applied generation controls, attempts, latency, parse status, response text length, token usage when available, finish metadata when available, malformed-JSON retry metadata, Pass 2 validation metadata, and fallback-after metadata in provider metadata. The UI may expose these fields in a compact technical diagnostics expander when live Scenario Review is unavailable, without showing API keys, raw prompts, or raw provider output. Add prompt template hashes only if prompt version strings are not enough for audit.
- Defer until durable provider tracing: raw provider response, parsed JSON response, provider response ID, system fingerprint, and provider-specific safety/refusal metadata beyond compact finish/safety counts. These fields are not meaningful for the deterministic mock reviewer and are not required for the current mock-default simulator path.
- Defer until durable storage: database/file persistence, shared trial-level baseline review records, cross-team replay, facilitator export, retention policy, privacy controls, and schema migration strategy.
- Do not expand the prompt packet just because the trace stores more audit data. Store enough for audit; send only curated current-context fields to the LLM.

Gemini prompt-size guidance:

- Keep each live Gemini narrative input prompt under roughly `10k` to `20k` input tokens where practical.
- Treat this as an operating target, not a hard validator limit. A larger prompt may be acceptable for an exceptional baseline or offline review, but it should trigger prompt-size diagnostics and a review of what can be summarized.
- Do not send full raw iteration history, full source documents, raw reference PDFs, raw database tables, or verbose prior narratives. Send compact baseline memory, compact previous-review memory, selected reference-pack summaries, selected local context statistics, current field changes, and material XGBoost movement evidence.
- If representative visible-iteration prompts drift above this range, reduce prompt volume before increasing provider timeouts or output ceilings.

The goal is to make repeated runs as consistent as possible while acknowledging that exact determinism is not guaranteed for LLM outputs.

Provider abstraction should be thin. The application should own payload construction, validation rails, Trial Score arithmetic, persistence, cache lookup, and UI rendering. Pass 2 owns the judgmental Operational Fit and Reality Check point decisions inside app-validated hard ranges. Provider-specific code should own only model invocation and response normalization. The V1 provider boundary includes the deterministic mock provider, explicit unsupported-provider failure path, and real OpenAI/Gemini invocation behind the same normalized result shape.

Real-provider prompts use a funnel instruction, currently implemented in `src/narratives/prompt_builder.py` and validated by `scripts/check_narrative_prompt_builder.py`:

- Use prompt mode `hidden_baseline` for the original trial before participant changes. This mode creates hidden qualitative baseline context plus baseline orientation/watch context in a compact draft. It must not expose participant-facing baseline Trial Score, Operational Fit points, Reality Check points, development discussion options, or an active participant storyline.
- Use prompt mode `first_visible_iteration` for the first participant-modified scenario. This mode can compare Completion Outlook to the visible original Completion Score, but participant-visible development discussion history starts only after Pass 3 shapes the visible participant discussion pair.
- Use prompt mode `later_visible_iteration` for later participant-modified scenarios. This mode can use previous visible review context for continuity, but development discussion reuse should be driven by participant-visible history and same-state reuse, not by hidden baseline topics or raw Pass 1 options.
- In visible modes, use `iteration_context.field_changes` to identify what the participant changed.
- Use `model_interpretation.xgboost_impact_changes` to understand model movement and materiality. In `hidden_baseline` mode, do not invent participant edits when `field_changes` is empty.
- Treat XGBoost/SHAP movement as model explanation evidence, not proof of clinical causality.
- Translate score evidence into clinical trial / pharma development language for participant-facing text. Explain why the revised scenario may look more or less completion-like, robust, feasible, governed, strategically aligned, risk-reduced, simplified, or less evidence-generating in terms of supported evidence such as endpoint timing, comparator choice, population scope, oversight, operational burden, scientific challenge, or development strategy rather than exposing raw model vocabulary. Total duration, planned enrollment, planned site count, and operational benchmark assumptions must not be cited as Completion Outlook drivers; maximum primary endpoint duration may be used only when present as Completion Outlook score evidence. Do not equate a higher Completion Score with simplification by default, but do flag simplification or value loss when the evidence points that way.
- Participant-facing narratives should state unresolved concerns rather than prescribe exact redesign paths. It is acceptable to say a scenario has unresolved bias-control, interpretability, proportionality, or scenario-readiness concerns; it should not tell participants to switch to a specific comparator, randomization, blinding, endpoint, modality, or population.
- Pass 1 produces Completion Outlook analysis, evolution evidence, visible-iteration development discussion option, continuity update, and analytical draft for Pass 2 scoring and Pass 3 narrative shaping.
- Pass 2 returns `operational_fit.points`, `reality_check.points`, Reality Check allocations, and score-evolution rationale. Do not return app-owned arithmetic fields such as `operational_fit_points`, `reality_check_points`, `pre_reality_score`, or `trial_score`; the application validates Pass 2, calculates arithmetic fields, and passes the accepted score trace to Pass 3.

Use a deterministic input hash based on prompt version, rubric version, baseline snapshot, current snapshot, storyline memory, and `text_context` Trial description fields. If the same input hash is reviewed again with the same provider/model cache namespace, reuse the stored validated review instead of calling the provider again. Generate the baseline review once per selected study and store it for the session. Hashable review context should avoid session-specific trace IDs; use stable input hashes and iteration IDs instead.

Validation and failure behavior:

- If the LLM provider call fails, show Completion Outlook only and mark the Trial Score review unavailable for the current snapshot.
- Do not reuse stale Operational Fit, Reality Check, Trial Score, or participant narrative for a new scenario state unless same-state reuse explicitly applies.
- If Pass 1 JSON is malformed or fails schema validation after repair, do not calculate Trial Score for that review.
- If Pass 3 narrative JSON is malformed or fails validation after repair, keep the accepted Trial Score trace but mark participant narrative unavailable/warning.
- If partial JSON validates, render only validated narrative sections and keep app-owned score fields derived from validated Pass 1 scoring.
- Store validation status and failure reason with the review trace.

## 17. Fields And Source-Of-Truth Principle

The structured feature registry remains the primary design source of truth for the narrative layer.

The LLM narrative layer should treat `structured_features` dropdown and numeric fields as the primary source of truth. Trial description fields in `text_context` are secondary. They should help detect contradiction, missing rationale, or narrative inconsistency. Missing or brief Trial description fields should not be heavily penalized unless they directly contradict structured trial features.

If `structured_features` and Trial description fields conflict, the LLM should flag the inconsistency rather than silently converting it into a score penalty. For example, if `adult_ml` says the scenario is adult-only but `text_context.summary_ui` says the intended treatment population includes pediatric or adolescent participants, the LLM may flag a target-population concern.

Trial description fields in `text_context` are untrusted context. The provider prompt must instruct the model to ignore any instructions, scoring requests, or role changes embedded inside `text_context.title`, `text_context.summary_ui`, `text_context.interventions_ui`, `text_context.primary_outcomes_ui`, `text_context.conditions_ui`, or clarifications. Trial description fields can provide rationale, context, or contradiction evidence, but they must not override `structured_features` unless a future UI explicitly marks them as participant rationale.

Trial Description / Structured-Feature Conflict Handling:

- This rule applies across all Trial description fields in `text_context` and all relevant `structured_features`, not only intervention descriptions.
- For obvious material mismatches, pause the prediction workflow before new scoring and ask the participant to correct the scenario or add an explanation.
- For softer development issues, continue Scenario Review and flag the inconsistency in narrative context.
- Route it to the relevant analytical narrative, Reality Check rationale, hidden discussion prompt, or participant development issue when supported.
- Require supported evidence before it can affect accepted scoring.
- Treat missing, brief, or noisy text as low-confidence context rather than a direct penalty.
- Do not let the LLM silently choose whether structured fields or text fields are true. For Completion Score, structured Trial Features are authoritative. For Scenario Review, structured fields remain primary context, while text and user explanations provide clarification, contradiction evidence, or scenario rationale.

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

The following four fields are essential for Scenario Review even if they are not direct transformed XGBoost/SHAP fields:

- `therapeutic_area_ml`: essential for disease-setting context and therapeutic-area calibration.
- `strategic_ambition_ml`: essential for development-question fit.
- `intervention_model_ml`: essential for interpreting the structure and validity of the trial design.
- `masking_ml`: essential for evidence credibility and bias-control reasoning.

The LLM should use these fields for design coherence and scientific rigor even if they do not carry direct SHAP contribution in the current model path.

#### Default Text Fields For v1

Recommended v1 default Trial description field context:

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

The architecture should avoid relying heavily on long or noisy Trial description fields during v1.

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

#### Role In The Scenario Review

The field set should support the active Trial Score review by giving Pass 1 enough evidence to evaluate:

- Completion Outlook movement and current score-pattern context.
- Operational Fit proportionality.
- Reality Check coherence, realism, simplification risk, and incremental evidence support.
- Endpoint and evidence interpretability.
- Population, setting, comparator, governance, and follow-up implications.
- Visible-iteration development discussion options for Pass 2 selection.

Examples:

- A narrower population may reduce apparent early-termination risk but could lower population relevance.
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
- Completion Outlook context.
- Operational Fit assessment and points.
- Reality Check assessment, allocation, and points.
- Trial Score.
- Pass 1 development discussion options for visible iterations.
- Pass 2 selected participant discussion topic and broader strategic question.
- Participant-visible question history.
- Input hash.
- Prompt version.
- Rubric version.
- Validation status.
- Failure reason, if any.
- Compact storyline memory.

Storage should keep validated Pass 1 analysis, Pass 2 score diagnostics, Pass 3 participant narrative, selected development discussion pair, and participant-visible question history as explicit fields rather than storing only a derived narrative explanation.

## 19. Non-Goals And Boundaries

The earliest contract-fixture phase intentionally had no production LLM implementation, no UI implementation, and no new API endpoint. That historical constraint has been superseded by the current prototype: minimal UI, provider boundaries, opt-in live-provider routing, and session-state review storage now exist. The remaining non-goals below still apply to the current V1 narrative work.

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
- No redistribution of Operational Fit or Reality Check into XGBoost / SHAP Completion Outlook pillars as if they were model drivers.
- No feature-level LLM pseudo-SHAP in v1; feature-level evidence must come from direct model-exported `feature_level_impacts`.
- No LLM-generated final score.
- No nearest-neighbor / similarity cohorts in v1.
- No full feature-norm benchmark table in v1.

## 20. Active Roadmap Summary

The active next-step roadmap is now owned by `docs/trial_score_narrative_direction.md`.

Near-term narrative work should:

- keep first-generation Strategic Review code as compatibility-only unless it is explicitly migrated;
- preserve the existing XGBoost Completion Outlook and SHAP behavior unchanged;
- continue implementation from the V1 Operational Fit and Reality Check contract in `src/narratives/trial_score_contract.py`;
- make the participant narrative assess only the final `Trial Score`;
- avoid separate repetitive component narratives;
- surface one central discussion topic and one broader strategic question;
- keep only pieces compatible with the new direction when touching remaining legacy UI, eval, or fixture code.

Obsolete roadmap items tied to `Design Confidence`, `Total Scenario Score`, and first-generation rigid `Strategic Review` scoring should not guide new implementation.

## 21. Open Questions

- Whether later versions add, remove, or reorder fields beyond the v1 field-selection policy above.
- Exact storage mechanism.
- Exact participant versus facilitator UI placement.
- Exact number of previous iterations to keep raw before summarization.
- Whether V1 Operational Fit and Reality Check budgets need tuning after live testing.
- Whether future versions should keep the current direct LLM-scored fields under strict validation or add more deterministic rails.
- Whether continuity checks should block scoring, warn in eval reports, or remain facilitator-only diagnostics.
- Whether facilitator view is hidden behind an expander or separate mode.
- Whether final governance recommendation is generated by participants, LLM, or both.
