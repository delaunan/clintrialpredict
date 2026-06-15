# Serious-Game Narrative Architecture

## Document Role

This file owns the future serious-game narrative layer: LLM commentary, Scenario Review, Design Confidence, Total Scenario Score, facilitator/participant outputs, and narrative payload contracts.

It should not own the existing XGBoost Completion Score, SHAP impact mechanics, or simulation UI state; use `docs/architecture_edit.md` for those. It should consume operational benchmark metadata from `docs/architecture_estimation.md` rather than redefining benchmark construction.

Efficient update rule: change this file when narrative inputs/outputs, LLM contracts, Scenario Review scoring, or participant/facilitator interpretation rules change. Do not implement or imply changes to XGBoost, SHAP, calibration, or operational benchmark construction here.

## 1. Purpose Of The Narrative Architecture

This document defines the staged design for adding a serious-game narrative layer around single-trial simulation in ClinTrialPredict. Current implementation covers contract fixtures, deterministic packet building, validation/scoring, mock review, session-state storage/replay, minimal Quality Review UI, hidden baseline continuity, provider configuration, real OpenAI/Gemini provider boundaries, Gemini Flash-Lite live settings, prompt-mode scaffolding, and opt-in live-provider UI routing. The simulator still uses the deterministic mock provider by default unless `NARRATIVE_LIVE_REVIEW_ENABLED=1` explicitly enables the live provider chain. The active target architecture supersedes the earlier three-pillar Quality Assessment model with a four-pillar Design Confidence model aligned to the existing Completion Outlook pillars.

Current planning checkpoint: `prompt_enhancement_plan.md` is the working implementation plan for the next prompt/schema migration. Its accepted durable decisions should be promoted into this architecture document before code changes. The next target contract keeps app-owned scoring but reshapes participant output around Completion Outlook Analysis, Design Confidence Analysis, and Key Questions; adds `review_metadata`; separates `hidden_baseline`, `first_visible_iteration`, and `later_visible_iteration`; treats Completion Outlook as early-termination risk-pattern interpretation; keeps operational assumptions out of Completion Outlook; adds optional therapeutic-area `.md` context by XGBoost canonical `therapeutic_area_ml`; and adds `scenario_consistency_note` plus `text_change_evidence`.

The current edit/simulation workflow remains the foundation. A facilitator selects an existing trial, participants adjust structured Trial Features, and the application calls the existing prediction flow to produce a completion score with SHAP-derived impact decomposition.

The narrative layer exists because the Completion Score alone is not enough for a serious-game discussion. Some changes may make the scenario look more similar to historically completed-trial patterns while reducing scientific rigor, evidence value, endpoint interpretability, population relevance, governance quality, or strategic defensibility. Other changes may increase apparent early-termination risk while making the design more robust or more relevant.

The narrative layer should help participants reason about this trade-off without giving direct optimization instructions. It should interpret score movement, surface design trade-offs, and challenge teams to defend their choices.

## 2. Core Scoring Boundary

The LLM layer is separate from the existing prediction system. The serious-game score stack has three layers:

1. `Completion Score`: the existing XGBoost, SHAP, therapeutic-area calibrated score from `/predict`, shown in points from 0 to 100.
2. `Scenario Review`: a constrained LLM structured reviewer that explains Completion Outlook movement and evaluates Design Confidence without calculating final score values.
3. `Total Scenario Score`: a deterministic application calculation: `Completion Score + Design Confidence`.

The LLM must not generate the final score. The LLM returns structured ratings, evidence fields, narrative, and continuity fields. The application then performs two deterministic calculations:

1. Convert validated Design Confidence subcategory ratings into app-owned point adjustments.
2. Add the Design Confidence adjustment to the XGBoost `Completion Score` to calculate `Total Scenario Score` when the combined view is enabled.

```text
design_confidence = sum(app_mapped_design_subcategory_points)

total_scenario_score = clamp(
    completion_score + design_confidence,
    0,
    100
)
```

The V1 scoring display should reconcile exactly: Design Confidence subcategory contributions add up bottom-up into their associated participant-facing pillar, and the four design subcategories add up to total Design Confidence. There is no hidden subcategory, pillar, or total Design Confidence cap. The display score is still clamped to the 0-100 range only at the final Total Scenario Score layer.

Example:

- Completion Score: `72`
- Design Confidence: `+4`
- Total Scenario Score: `76`

Interpretation:

- The Completion Score remains the existing XGBoost score; participant-facing narrative should treat it as an early-termination risk-pattern signal, not as a promise of completion.
- Design Confidence is a serious-game modifier for design defensibility, decision usefulness, patient relevance, and proportionate execution choices.
- The Total Scenario Score can recognize a risky but well-strengthened design without replacing or rewriting XGBoost.
- A trial below `50` on Completion Score can improve if the design choices meaningfully explain why some completion risk reflects rigor, ambition, patient relevance, or prudent governance rather than poor design.
- A high Completion Score should not automatically receive positive Design Confidence. Positive adjustment requires specific evidence of design confidence, not model-favorable simplicity.

Terminology:

- `Completion Score` = existing XGBoost score shown from 0 to 100.
- `Completion Outlook` = explanation of model-derived score movement using feature, subcategory, pillar, and score movement evidence, framed as lower/higher early-termination risk or resemblance to historically completed/terminated-trial patterns.
- `Scenario Review` = participant-facing narrative explanation of Completion Outlook movement, Design Confidence evidence, trade-offs, and three expert questions.
- `Design Confidence` = application-calculated point adjustment derived from validated design subcategories with supported evidence.
- `Total Scenario Score` = Completion Score plus Design Confidence, when the combined view is enabled.

In plain scoring terms, `Total Scenario Score = Completion Score + Design Confidence`, with application-level bounds applied only to the final display score.

The application, not the LLM, calculates Design Confidence points and Total Scenario Score. The LLM must never modify the XGBoost completion score, SHAP values, pillar impacts, therapeutic-area calibration, prediction pipeline, or audit/demo parity behavior.

Core boundary:

- The LLM never modifies XGBoost.
- The LLM never modifies SHAP values.
- The LLM never modifies therapeutic-area calibration.
- The LLM never rewrites the prediction score.
- The LLM returns structured Scenario Review ratings, evidence fields, explanation, and continuity fields.
- The application maps validated design review ratings into Design Confidence and Total Scenario Score.

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
9. The application validates the structured Scenario Review, calculates Design Confidence and Total Scenario Score when enabled, and stores a compact storyline update.
10. The UI displays the Scenario Review below or near the score and charts.
11. Participants iterate.

The visible narrative should usually compare the current prediction against the previous prediction. Internally, the LLM should also receive enough baseline and path memory to avoid contradicting prior feedback.

### Baseline Review Object

For existing-study mode, generate a baseline review once per selected study/version and store it durably, keyed by stable trial identity plus baseline input hash, prompt version, rubric version, and provider/model namespace. The first team opening Simulation Mode for a trial may create this baseline if it does not exist. Later teams opening the same trial should load the same baseline review and compact memory, so all teams start from a consistent original-trial interpretation.

The current prototype initializes the baseline review when Simulation Mode opens, not delayed until the first participant prediction. It remains hidden from participants at the start of the exercise, but it should be passed into later review calls so the LLM can evaluate how the design path evolved from the original trial. The exact production timing remains open pending latency calibration.

The baseline review context passed to later LLM calls is qualitative-only. It may include baseline strengths, concerns, consistency flags, participant-review text, continuity fields, and compact memory. It must not expose hidden baseline `Design Confidence`, `Total Scenario Score`, or a hidden numeric quality score to later prompt logic as a prior visible score.

The hidden baseline review should include two qualitative analyses. First, it should interpret the prerecorded Completion Score when baseline decomposition is available. The provider should use baseline `structured_features`, `text_context` Trial description fields, Completion Score, pillar impacts, and feature/subcategory impacts to summarize why the original trial appears completion-like or risky in clinical trial / pharma development language. This is baseline reasoning context, not a visible participant score and not a new XGBoost calculation. If only a registry score is available without pillar or feature decomposition, the baseline review should explicitly treat the score interpretation as lower-detail and avoid inventing missing driver analysis. Second, it should produce a hidden baseline Scenario Review analysis using the same Design Confidence rubric as later visible reviews: phase and intent alignment, endpoint and evidence strength, target population alignment, operational burden balance, Trial description consistency, and baseline concerns. This hidden design analysis should create baseline strengths, baseline concerns, consistency flags, and compact memory, but it must not expose a visible baseline Design Confidence adjustment, Total Scenario Score, or hidden numeric quality score to the participant.

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

Participant-facing rule: do not show the baseline Scenario Review, Design Confidence, Total Scenario Score, or baseline narrative comments by default. Participants may see the original XGBoost Completion Score and model drivers if that is part of the trial-opening experience. The hidden baseline review exists to enrich the first and later visible reviews, not to expose a design score before teams make their first scenario choice. If hidden baseline review initialization fails, Simulation Mode should still open; later visible Scenario Review calls can proceed without baseline review context or retry when a durable provider/store exists.

Later prediction reviews should receive:

- Baseline review.
- Previous iteration review.
- Compact storyline memory.
- Current delta packet.
- Current snapshot.

For the first visible review after a team edit, the narrative may compare qualitatively against the original trial baseline, but it must not refer to hidden baseline design numbers as if participants had already seen them. For example, it may say that the model Completion Score decreased while the current design appears more defensible than the original baseline context. It should not say that the team "improved the score" when the visible Completion Score or visible Total Scenario Score declined.

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

- `Total Scenario Score`, shown in the main gauge when combined view is enabled.
- `Completion Score` as a component score.
- `Design Confidence` as a component value.
- A short `Operational Assumptions` note.
- Concise `Scenario Review`.

Participant narrative sections:

- `What changed`
- `Why the completion score may have moved`
- `What the design may have gained`
- `What the design may have sacrificed`
- `Operational coherence note`
- `Two questions for the team to debate`: one medical/development question and one clinops/execution question.

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

The Scenario Review has two analytical jobs:

1. Explain `Completion Outlook`: why the XGBoost Completion Score moved, using changed fields, feature movements, subcategory movements, pillar movements, baseline context, previous iteration context, and cross-pillar interaction hypotheses.
2. Evaluate `Design Confidence`: whether the scenario is strategically defensible, evidence-generating, patient-relevant, and proportionate to execute.

The active participant-facing hierarchy keeps the existing four Completion Outlook pillars and adds one Design Confidence subcategory under each pillar:

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
Implemented design subcategory range = -5.0 to +5.0 in 0.5 increments, app-owned
Typical subcategory movement = -2.5 to +2.5
Total Design Confidence = sum of four design subcategories
No hidden total cap
```

The implemented `-5.0..+5.0` range preserves app-owned scoring. The provider must not return numeric subcategory points. Instead, the provider returns qualitative `rating` plus qualitative `score_materiality` for each subcategory after selecting supported evidence fields and writing the rationale. The app maps `rating + score_materiality + context guardrails` into points, preserving 0.5 increments and supported-evidence gates.

Target qualitative `score_materiality` mapping:

- `strong`: minimal `+3.0`, low `+3.5`, moderate `+4.0`, high `+4.5`, very_high `+5.0`.
- `supportive`: minimal `+0.5`, low `+1.0`, moderate `+1.5`, high `+2.0`, very_high `+2.5`.
- `balanced`: always `0.0`.
- `weak`: minimal `-0.5`, low `-1.0`, moderate `-1.5`, high `-2.0`, very_high `-2.5`.
- `conflicting`: minimal `-3.0`, low `-3.5`, moderate `-4.0`, high `-4.5`, very_high `-5.0`.

The adjustment must not be fake balancing:

- If a Completion Outlook pillar is already strongly positive, positive Design Confidence for that same pillar should usually remain `0.0`, `+0.5`, or at most `+1.0` unless the packet shows a specific unresolved current-scenario weakness was improved or adds new design-quality evidence not already captured by Completion Outlook.
- If a Completion Outlook pillar is neutral or negative, positive Design Confidence can be larger only when supported evidence shows the risk is caused by rigor, patient relevance, scientific ambition, or prudent governance.
- If Completion Outlook rises sharply, negative Design Confidence can moderate the increase only when supported evidence suggests shortcut behavior or weakened design confidence.
- If Completion Outlook falls sharply, positive Design Confidence can moderate the decrease only when supported evidence suggests the added risk comes from better evidence, broader patient relevance, or proportionate governance.
- `high` positive `score_materiality` is rare and should require a clear critical narrative showing why the scenario is more defensible, interpretable, patient-relevant, governed, or proportionate. A favorable Completion Outlook, benchmark-typical operational assumption, or unresolved text/field mismatch is not enough.

Implemented scoring-calibration refinement:

Design Confidence is a qualitative critical lens on Completion Outlook, not a second completion predictor and not a bonus multiplier. Completion Outlook mainly describes resemblance to historical completion versus early-termination risk. That movement can reflect quality, simplicity, operational burden, or risk patterns; higher Completion Outlook does not necessarily mean better design, and lower Completion Outlook does not necessarily mean worse design. The scoring layer should therefore preserve the LLM's qualitative judgment while limiting numeric over-amplification of signals already captured by Completion Outlook.

This refinement supersedes the earlier stricter already-positive-pillar cap language. The governing model is matching-pillar same-direction double-counting control plus opposite-direction counterweight preservation.

The provider contract remains unchanged: the LLM returns `rating`, `score_materiality`, evidence fields, rationale, and `short_rationale`; the application owns numeric points. Add only this compact prompt principle if needed:

> Preserve each Design Confidence subcategory's meaning. When a change improves one design dimension but worsens another, reflect both effects in their relevant subcategories. Cross-functional trade-offs may be justified in the overall Design Confidence judgment, but a subcategory should be positive only when that subcategory itself improved.

The deterministic layer remains simple and conservative:

- Keep `rating + score_materiality -> raw_points`.
- Add calibrated `points` and `calibration_notes` when final points differ from raw points.
- Apply calibration at the Design Confidence subcategory level, not to Completion Outlook score or Completion Outlook pillars.
- Use same-direction double-counting control as the primary rule, triggered only by strong matching Completion Outlook pillar movement at `>= +3.0` or `<= -3.0` points. If the matching pillar moved strongly positive and is now still negative, cap strongly positive mapped Design Confidence at `+2.5`; if it moved strongly positive and is now neutral or positive, cap at `+1.5`. If the matching pillar moved strongly negative and is now still positive, soften strongly negative mapped Design Confidence to `-2.5`; if it moved strongly negative and is now neutral or negative, soften to `-1.5`.
- Preserve opposite-direction counterweight behavior: when Design Confidence challenges Completion Outlook movement, keep stronger supported points. Examples include Completion Outlook improving while evidence quality weakens, or Completion Outlook worsening because rigor, patient relevance, governance, or evidence interpretability improved.
- Allow Design Confidence to speak when Completion Outlook movement is flat or small, especially for Trial description, scenario-readiness, operational-assumption, or cross-pillar quality changes that are not directly reflected in Completion Outlook.
- Do not use total Completion Outlook score movement or previous-score thresholds as Design Confidence calibration triggers. Total score movement is an aggregate and can reflect other pillars; the total is where subcategory trade-offs reconcile, not the trigger for subcategory-level softening.
- Preserve subcategory meaning. Positive points should stay in the subcategory that improved; negative or neutral counter-impact should remain visible in the relevant other subcategory. Compensation happens through the total Design Confidence sum, not by making unrelated subcategories positive.
- Operational assumptions and all other supported packet evidence follow the same Design Confidence scoring and calibration rules. Planning assumptions may improve or worsen Operational Burden Balance because they affect whether the scenario feels operationally proportionate and executable. They may also create counter-effects in other Design Confidence subcategories, such as endpoint maturity or evidence sufficiency, when supported by the rationale. The special rule for planning assumptions is only that they must not explain Completion Outlook movement; within Design Confidence they are handled like other supported current-scenario evidence. Preserve subcategory meaning through the LLM rationale rather than through field-family-specific numeric caps.
- Shortcut-driven ease should not create strong positive Operational Burden Balance or Design Confidence. If easier completion comes from weaker randomization, masking, comparator, endpoint rigor, arms, governance, or development ambition, cap positive feasibility credit and preserve evidence/phase critique.
- Negative critiques should generally remain visible when supported. Soften negative points only when they duplicate strong same-direction negative Completion Outlook or pillar movement; do not soften them merely because another subcategory improved.
- The deterministic layer should mostly cap or soften excessive same-pillar, same-direction amplification. It should not invent new positive or negative points and should not perform deep clinical interpretation beyond matching-pillar movement and provider evidence fields.

Participant-facing Design Confidence treemap display shows signed subcategory points and the short rationale only. `rating` and `score_materiality` remain internal for scoring, validation, and audit; those labels are not shown as participant-facing treemap text.

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
    "summary_ui": "Brief study summary used to interpret scenario intent and possible structured/text tensions.",
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

`structured_features` dropdown/numeric fields are the primary source of truth. Trial description fields in `text_context` are secondary and should be used for coherence checking, contradiction detection, and narrative context rather than as the main source of scoring. Narrative packets should send canonical submitted structured values first and display labels separately, so future providers can reason from readable labels without losing scoring value provenance.

Missing or brief Trial description fields should not be heavily penalized unless they directly contradict structured trial features or make an otherwise important design claim impossible to interpret.

For structured Trial Features, narrative packets should use taxonomy option keys as the canonical value where an option key exists, and should include human-readable labels separately in `structured_feature_display_values`. For example, `endpoint_structure_ml = MULTI_COMPOSITE` should be paired with display label `Multi/Composite`. Numeric fields keep numeric values. The packet should include `field_dictionary_version`, `structured_feature_meanings`, and `text_context_field_meanings` so prompts and providers know which taxonomy meanings and text fields apply without requiring an external lookup or repeating the full field dictionary in every packet. Narrative field meanings must be generated from the production taxonomy source path in `src/prep/pipeline.py`, not patched only into `models/taxonomy_01.json`, so rerunning `notebooks/production_01.ipynb` preserves them.

Reference-pack routing should be active but compact. The packet should include only selected pack IDs, tags, roles, and `Prompt-Safe Summary` text from `frontend/data/docs/narrative_reference_packs`, not full source documents. Default V1 inclusion is `core_clinical_development_v1`, `strategic_context_2026_v1`, and `ich_e8_quality_by_design_v1`. Operational/governance scenarios may add `ich_e6_r3_gcp_v1`; endpoint/statistical scenarios may add `ich_e9_r1_estimands_v1` and `ich_e9_statistical_principles_v1`. Reference packs are secondary context: the scenario packet remains authoritative, and the provider should record used pack IDs in `trace.reference_pack_ids_used`.

Narrative packets should keep scenario edit facts separate from model explanation facts:

- `iteration_context.field_changes` records what the participant changed, with baseline, previous, and current values/labels. This covers structured Trial Features, changed text-context fields, and operational assumptions when available.
- `model_interpretation.xgboost_impact_changes` records what moved in the model explanation, with baseline, previous, and current impact values plus deltas. `changed_since_previous` marks local movement since the last prediction; `changed_from_baseline` marks accumulated drift from the original trial. These entries are XGBoost/SHAP explanation facts at pillar or subcategory level, not proof of clinical causality and not necessarily limited to fields the participant directly edited.

The LLM should use `field_changes` to explain what changed in the scenario and `xgboost_impact_changes` to weight which model-explanation movements were material. It should not infer that every model impact movement was directly caused by a single changed field.

The active simulator does not run a pre-prediction structured/text consistency check. Editable text fields remain narrative context and are submitted with the scenario, but they do not create a correction gate before scoring.

## 12. Output JSON Contract

The LLM should return structured JSON. The application should validate the response, reject or downgrade malformed scoring fields, and calculate Design Confidence and Total Scenario Score deterministically.

The LLM should not return the final score as an authority. It should return review ratings, evidence fields, narrative, and continuity fields.

Target contract:

```json
{
  "review_metadata": {
    "review_mode": "hidden_baseline | first_visible_iteration | later_visible_iteration",
    "participant_visible": true
  },
  "completion_outlook_analysis": {
    "score_movement_summary": "short explanation of the observed Completion Score movement",
    "main_model_signals": [],
    "interpretive_hypotheses": [],
    "model_limits": []
  },
  "design_confidence_subcategories": {
    "phase_intent_alignment": {
      "evidence_fields": [],
      "rationale": "...",
      "rating": "strong | supportive | balanced | weak | conflicting"
    },
    "endpoint_evidence_strength": {
      "evidence_fields": [],
      "rationale": "...",
      "rating": "strong | supportive | balanced | weak | conflicting"
    },
    "target_population_alignment": {
      "evidence_fields": [],
      "rationale": "...",
      "rating": "strong | supportive | balanced | weak | conflicting"
    },
    "operational_burden_balance": {
      "evidence_fields": [],
      "rationale": "...",
      "rating": "strong | supportive | balanced | weak | conflicting"
    }
  },
  "pillar_reviews": {
    "therapeutic_context": {
      "completion_interpretation": "...",
      "design_adjustment_interpretation": "...",
      "collateral_impacts": []
    },
    "scientific_challenge": {
      "completion_interpretation": "...",
      "design_adjustment_interpretation": "...",
      "collateral_impacts": []
    },
    "patient_profile": {
      "completion_interpretation": "...",
      "design_adjustment_interpretation": "...",
      "collateral_impacts": []
    },
    "execution_framework": {
      "completion_interpretation": "...",
      "design_adjustment_interpretation": "...",
      "collateral_impacts": []
    }
  },
  "tradeoff_review": {
    "central_tension": "...",
    "what_completion_gained": "...",
    "what_design_confidence_gained": "...",
    "what_may_have_been_sacrificed": "...",
    "main_uncertainty": "..."
  },
  "participant_review": {
    "overall_completion_comment": "...",
    "overall_design_comment": "...",
    "most_impactful_pillar_1": "...",
    "most_impactful_pillar_2": "...",
    "interaction_summary": "...",
    "medical_development_question": "...",
    "clinops_execution_question": "..."
  },
  "facilitator_view_optional": {
    "shortcut_risk": "low | moderate | high",
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
    "main_completion_drivers_considered": [],
    "main_design_subcategories_considered": [],
    "operational_statuses_considered": [],
    "reference_pack_ids_used": [],
    "compared_against": "previous_prediction",
    "should_repeat_prior_warning": false
  }
}
```

`facilitator_view_optional` may be omitted in the first implementation. The minimum provider contract is the Completion Outlook review, Design Confidence subcategories, participant review, continuity fields, and trace fields needed for validation and replay.

### Output Style And Length Requirements

Phase 4 prompt refinement should keep the Scenario Review detailed enough for serious-game discussion but bounded enough for live play and trace review.

Provider output style:

- Use concise clinical-development prose, not marketing language and not technical model jargon.
- Use conditional language such as `may`, `could`, `appears`, and `would need support`; avoid absolute clinical claims.
- Do not recommend exact next edits. End with questions that support discussion.
- Do not mention SHAP, XGBoost, feature impact, model movement, or pillar delta in participant-facing fields.
- Do not calculate, mention, or estimate Design Confidence points, Total Scenario Score, or subcategory point values.

Expert analysis requirements:

- The provider should write as a senior clinical-development and medical-strategy reviewer evaluating a scenario for serious-game discussion, not as a trial optimizer.
- The output should make a clear expert judgment about what the scenario appears to strengthen, weaken, or leave uncertain, while preserving conditional language.
- Participant-facing prose should usually follow `because / however / therefore` logic: identify the packet-supported signal, name the trade-off or limitation, then state the implication for discussion.
- The analysis should use relevant expert lenses when supported by packet evidence: evidence interpretability, development intent fit, target-population relevance, operational proportionality, shortcut risk, governance and oversight adequacy, and cross-pillar tension between Completion Outlook and Design Confidence.
- The provider must not present the Completion Score as clinical truth, infer regulatory acceptability, efficacy, safety, or feasibility beyond packet evidence, imply that a higher Completion Score means a better trial design, or turn the review into a prescription for the next edit.

Compact examples:

```text
Good Completion Outlook comment:
The Completion Outlook appears to improve because the edited scenario looks easier to complete on the Completion Outlook score inputs. However, that improvement should be read as operational or structural favorability, not as proof that the revised design would answer the development question better.

Good Design Confidence comment:
The Design Confidence signal is more cautious because the scenario may have reduced evidentiary rigor relative to the stated development intent. Therefore, the team should be ready to defend whether the completion gain is worth the loss of interpretability.

Weak comment to avoid:
The score went up and the design is better; change the endpoint and population this way next.

Completion improves but evidence weakens:
The scenario may look more completion-favorable because it simplifies evidence generation or execution. However, if endpoint rigor, comparator credibility, masking, or duration support weaken, the review should say that the completion gain may come with lower decision interpretability.

Completion declines but design improves:
The scenario may look harder to complete because it increases burden, duration, endpoint ambition, or population specificity. However, if those changes better match the development question, the review should explain why lower completion outlook may coexist with stronger design confidence.

Operational burden increases without evidence gain:
The scenario may add enrollment, sites, duration, arms, or oversight burden. However, if the packet does not show a matching evidence or population-fit gain, the review should flag proportionality rather than treating operational ambition as inherently positive.
```

### Golden / One-Shot Example Policy

The current implementation has contract fixtures in `src/narratives/contract_fixtures.py` and compact style examples in this prompt architecture, but it does not yet need full one-shot examples embedded in the live provider prompt.

Near-term recommendation:

- Create 3-5 golden prompt examples outside the live prompt first, for calibration and regression review.
- Keep them in documentation or fixture-style artifacts until the target schema, prompt modes, and participant output format are stable.
- Add full one-shot examples to the live provider prompt only if schema/rules/reference packs are not enough to produce consistent output.

Each golden example should include:

- Scenario setup and trial context.
- Changed `structured_features` and changed Trial description field evidence.
- Whether the mode is `hidden_baseline`, `first_visible_iteration`, or `later_visible_iteration`.
- Expected Completion Outlook framing, including early-termination risk-pattern language and model-boundary wording.
- Expected Design Confidence challenge, including whether it should moderate a clear Completion Outlook increase or decrease.
- Relevant Design Confidence subcategories and evidence fields.
- Any text/structured-field consistency note expected.
- Optional therapeutic-area context and whether a TA `.md` pack is present or missing.
- Optional regulatory or finance/cost lens only when materially relevant.
- Forbidden wording, especially causal field claims, hidden-baseline Design Confidence comparisons, unsupported disease/regulatory/cost claims, and planned enrollment/sites/total-duration claims as Completion Outlook drivers.
- Ideal short participant output: Completion Outlook Analysis, Design Confidence Analysis, and Key Questions.

Recommended initial golden examples:

1. Completion Outlook improves but Design Confidence weakens because evidence value or interpretability is reduced.
2. Completion Outlook worsens but Design Confidence strengthens because rigor, patient relevance, governance, or justified ambition improves.
3. Text/structured contradiction where selected categorical/numeric fields prevail and a consistency note is shown.
4. Operational burden increases without matching evidence gain.
5. Therapeutic-area/pathway change requiring cautious interpretation without overclaiming mechanism, efficacy, safety, or regulatory significance.

Golden examples should be used as prompt-regression references and teaching artifacts first. If later embedded in the live prompt, they must be compact enough not to crowd the packet, must match the current response schema, and must be updated whenever the schema or participant UI contract changes.

Required field lengths:

- Each Design Confidence subcategory `rationale`: 1 sentence, usually 18-35 words, maximum 45 words.
- `completion_outlook_analysis.score_movement_summary`: 1 sentence, maximum 35 words.
- Each item in `main_model_signals`, `interpretive_hypotheses`, and `model_limits`: maximum 25 words.
- Each `pillar_reviews.*.completion_interpretation`: 1 sentence, maximum 30 words.
- Each `pillar_reviews.*.design_adjustment_interpretation`: 1 sentence, maximum 30 words.
- Each `pillar_reviews.*.collateral_impacts` item: maximum 20 words.
- Each `tradeoff_review` string field: 1 sentence, maximum 35 words.
- `participant_review.overall_completion_comment`: 1 short paragraph of 2-3 sentences, maximum 85 words.
- `participant_review.overall_design_comment`: 1 short paragraph of 2-3 sentences, maximum 85 words.
- `participant_review.most_impactful_pillar_1` and `participant_review.most_impactful_pillar_2`: each 1 short paragraph of 2 sentences, maximum 70 words, naming the pillar and explaining why it matters.
- `participant_review.interaction_summary`: 1-2 sentences, maximum 55 words, explaining how Completion Outlook and Design Confidence interact.
- Each participant debate question: one question, maximum 25 words.
- `continuity.storyline_update`: 1 sentence, maximum 35 words.
- Trace arrays should contain short field names or compact labels, not full narrative sentences.

Narrative detail target:

- The participant panel should be readable in roughly 75-120 seconds.
- The full provider JSON should contain enough rationale for audit, but the main participant review should stay concise.
- Detailed evidence lives in `evidence_fields`, trace, and validation output; participant-facing narrative should explain the trade-off rather than list every field.
- The participant answer should be organized in this order: overall Completion Outlook comment, overall Design Confidence comment, two most impactful pillar/interaction comments, then three debate questions.
- The two pillar comments should focus on the most material dimensions, not all four pillars. They should surface interactions such as completion improved but evidence weakened, completion declined but design became more defensible, or operational burden increased without clear evidence gain.
- `tradeoff_review.central_tension` should summarize the single most important Completion Outlook versus Design Confidence trade-off in one sentence. It is mainly for audit/storage and can later feed a facilitator or compact participant heading.
- The debate questions should be open-ended, not answerable with yes/no, and should not ask whether a specific field should be changed. They should elevate the discussion by asking what evidence standard, strategic rationale, population trade-off, governance burden, field-level challenge, or operational proportionality would make the scenario defensible.
- The medical/development question should focus on evidence value, development decision, endpoint interpretability, or patient relevance. The clinops/execution question should focus on feasibility, access, oversight, data reliability, participant/site burden, or risk-proportionate conduct.
- When `strategic_context_2026_v1` is available, questions may raise current strategic themes such as access, representativeness, decentralised or digital data collection, estimand clarity, data reliability, and governance proportionality, but only when supported by packet evidence.
- The review should be substantial enough to support discussion but must not reveal an optimization recipe or tell participants exactly which field to change next.

### Evidence-First Review Sequence

Phase 4 prompt/schema work must make the provider reason in this order for every Design Confidence subcategory:

1. Select packet-supported `evidence_fields`.
2. Write the subcategory `rationale` from those evidence fields.
3. Assign the subcategory `rating` that follows from the evidence and rationale.
4. Assign qualitative `score_materiality` from supported-evidence strength and context guardrails.

The provider should not choose a rating or score materiality first and then search for a justification. This evidence-first sequence is required for output quality and auditability. It makes the rating and materiality inspectable: a facilitator or developer can open the stored trace, see which packet fields were cited, read the rationale, and understand why the rating was valid or why the application gave it zero score effect.

The JSON object may be parsed without relying on key order, but prompt examples and response schemas should present fields in the same conceptual order where possible:

```json
{
  "evidence_fields": ["endpoint_rigor_ml", "comparator_benchmark_ml"],
  "rationale": "The endpoint and comparator choices weaken decision interpretability for the stated intent.",
  "rating": "weak",
  "score_materiality": "moderate"
}
```

Audit rules:

- `evidence_fields` must cite packet fields or allowed packet evidence references.
- `rationale` must explain how those fields support the rating in clinical-development terms.
- `rating` must be one of the allowed labels and must not mention point values.
- `score_materiality` must be one of `minimal`, `low`, `moderate`, `high`, or `very_high`; it controls app-owned point magnitude but is still qualitative provider evidence, not provider-owned scoring.
- Unsupported evidence references are preserved in trace/debug output but have zero scoring effect.
- A malformed, missing, or incomplete subcategory should suppress Design Confidence and Total Scenario Score for that review rather than silently counting as neutral.
- Provider traces should store the original output JSON, normalized validated review, validation errors, supported/unsupported evidence fields, app-calculated subcategory points, Design Confidence, Total Scenario Score, prompt/rubric versions, provider/model namespace, and input hash so score rationale can be audited later.

Malformed or incomplete provider responses should be prevented and handled as follows:

- Phase 4 prompt/schema work should make the provider schema strict enough that all required top-level sections and all four Design Confidence subcategories are always requested.
- The prompt should explicitly require all four subcategories on every review: `phase_intent_alignment`, `endpoint_evidence_strength`, `target_population_alignment`, and `operational_burden_balance`.
- The mock, prompt, and provider checkers should assert that all four subcategories are present, valid, and evidence-first.
- The provider wrapper may make one bounded repair/retry attempt for malformed JSON, missing required fields, or incomplete required subcategories. That retry should use the same packet and a clearly recorded retry reason; it must not ask a different provider to reinterpret a clinically valid but unfavorable review.
- If the retry still fails validation, the UI should show Completion Score only and mark Scenario Review unavailable for the current scenario. Participants should use the normal Review Scenario action again after addressing the issue or retrying later.
- The application should not show a partial Design Confidence score and should not treat a missing subcategory as neutral.
- A future response-repair step is allowed only if it is deterministic, auditable, uses the same provider output and packet evidence, and cannot invent missing clinical reasoning.

The provider prompt should also include qualitative `rating_guidance_by_subcategory` for the allowed rating labels. This guidance explains labels such as `supportive`, `balanced`, `weak`, and `conflicting` in clinical-development terms so the LLM can choose the right category. It is not model-owned scoring, and it should not expose or ask the provider to calculate point values.

Participant-facing narrative should translate `model_interpretation` evidence into clinical trial / pharma development language. The provider may use `field_changes`, `xgboost_impact_changes`, `score_delta`, and pillar/subcategory movement internally, but visible explanations should avoid technical model terms such as SHAP, feature impact, XGBoost movement, or pillar delta unless the facilitator view explicitly asks for model diagnostics. Preferred language should discuss endpoint maturity, evidence strength, comparator credibility, blinding/control implications, recruitment burden, trial duration, patient population fit, operational complexity, execution feasibility, development strategy, design shortcut risk, and regulatory persuasiveness.

The LLM does not return `Design Confidence`, `Total Scenario Score`, or final Design Confidence subcategory point contributions. The application derives them from validated `design_confidence_subcategories`, evidence fields, and the documented deterministic mapping. If a provider returns app-owned score fields, validation should mark them as ignored and the application should still calculate its own values. This keeps plotted Design Confidence contributions reproducible.

The application calculates:

```text
phase_intent_alignment_points = deterministic_map(validated_phase_intent_alignment_rating, score_materiality, context_guardrails)
endpoint_evidence_strength_points = deterministic_map(validated_endpoint_evidence_strength_rating, score_materiality, context_guardrails)
target_population_alignment_points = deterministic_map(validated_target_population_alignment_rating, score_materiality, context_guardrails)
operational_burden_balance_points = deterministic_map(validated_operational_burden_balance_rating, score_materiality, context_guardrails)

design_confidence =
    phase_intent_alignment_points
  + endpoint_evidence_strength_points
  + target_population_alignment_points
  + operational_burden_balance_points

total_scenario_score = clamp(
    completion_score + design_confidence,
    0,
    100
)
```

Implemented mapping envelope:

```text
Each Design Confidence subcategory = -5.0 to +5.0
Allowed increments = 0.5
Default = 0.0
Typical movement = -2.5 to +2.5
Strong movement = requires multiple explicit, supported signals
```

The provider supplies qualitative `rating` and `score_materiality`; it does not supply numeric points. The middle rating, `supportive`, allows the review to recognize directionally favorable design quality without forcing every positive observation into the top rating. `balanced` should map to `0.0` regardless of materiality.

The final score should preserve half-point values when they occur and display one decimal only when needed. A design subcategory rating should affect Design Confidence only when the LLM provides supporting `evidence_fields`; otherwise the point effect should be zero and the issue can remain narrative-only. Evidence fields must also reference evidence available in the review packet. Unsupported evidence references are preserved for auditability but do not move Design Confidence.

Allowed evidence references may cite:

- direct structured/text fields, such as `endpoint_rigor_ml`, `primary_outcomes_ui`, or `text_context.primary_outcomes_ui`
- operational fields and nested metadata, such as `operational_assumptions.planned_enrollment.support_level`
- clarification context, such as `clarification_context.user_clarifications`
- Step-2 delta evidence, such as `field_changes.endpoint_rigor_ml`
- model explanation sections or names, such as `xgboost_impact_changes`, `Execution Framework`, or `xgboost_impact_changes.Execution Framework`

Guardrails:

- If all validated Design Confidence subcategories are `balanced`, neutral, or otherwise non-concerning, `Design Confidence = 0`.
- A positive Design Confidence adjustment requires evidence that the participant strengthened design confidence, not merely that the design avoided obvious concerns.
- Unsupported `evidence_fields` have zero scoring effect even when the rating is non-neutral.
- Benchmark-typical operational assumptions are neutral by default; they do not create a positive Design Confidence adjustment unless supported by broader design improvements.
- Low-confidence operational benchmark metadata should be narrative-first. It should affect points only when multiple conflict signals agree.

## 13. Plot Integration Guidance

Plot integration should preserve source clarity while allowing an adjusted-score view.

Use a toggle:

```text
Completion Score View
Total Scenario Score View
```

### Completion Score View

This is the existing model-first view:

- Gauge: `Completion Score`.
- Bar chart: four XGBoost / SHAP-derived completion pillars.
- Treemap: existing XGBoost / SHAP-derived completion hierarchy.
- Labels should remain tied to Completion Score drivers.

### Total Scenario Score View

This is the adjusted serious-game view:

- Gauge: `Total Scenario Score`.
- Component cards:
  - `Completion Score`.
  - `Design Confidence`.
  - `Total Scenario Score`.
- Bar chart should keep the four familiar participant-facing pillars:
  - `Therapeutic Context`.
  - `Scientific Challenge`.
  - `Patient Profile`.
  - `Execution Framework`.
- Each pillar can include its existing Completion Outlook subcategories plus one Design Confidence subcategory:
  - `Therapeutic Context -> Phase & Intent Alignment`.
  - `Scientific Challenge -> Endpoint & Evidence Strength`.
  - `Patient Profile -> Target Population Alignment`.
  - `Execution Framework -> Operational Burden Balance`.
- Use clear provenance labels or visual encoding so users can see which subcategory is model-derived Completion Outlook and which subcategory is app-scored Design Confidence.

Recommended adjusted treemap structure:

```text
Total Scenario Score
├── Therapeutic Context
│   ├── Therapeutic Area Profile
│   ├── Development Phase and Goal
│   └── Phase & Intent Alignment
├── Scientific Challenge
│   ├── Biological Profile
│   ├── Protocol Architecture
│   └── Endpoint & Evidence Strength
├── Patient Profile
│   ├── Clinical Severity
│   ├── Population Scope
│   └── Target Population Alignment
└── Execution Framework
    ├── Methodological Setup
    ├── Trial Complexity Footprint
    └── Operational Burden Balance
```

The treemap should make source boundaries explicit even when the participant-facing chart is integrated:

```text
Completion Outlook subcategories = XGBoost / SHAP-derived
Design Confidence subcategories = structured Scenario Review, app-scored
```

Do not create fake SHAP attribution. Design Confidence values are not SHAP values and should not be described as model drivers. Use terms such as `Design Confidence adjustment` or `Scenario Review adjustment` rather than `SHAP drivers`.

Operational assumptions should not be redistributed into the XGBoost `Execution Framework` Completion Outlook value. In adjusted view, Planned Enrollment, Planned Sites, and Planned Duration can contribute only through `Operational Burden Balance`.

The minimal participant panel does not need to expose all validation/debug fields. It should show Completion Score, Design Confidence, Total Scenario Score when enabled, the four familiar pillars, signed subcategory contribution direction, concise participant-review narrative, and three expert questions. It should not expose raw LLM rating labels such as `supportive`, `weak`, or `conflicting` in the participant panel; those categories are consumed by the application to calculate and audit the score. Future facilitator or debug views should consider exposing packet-supported versus unsupported `evidence_fields` and raw subcategory ratings from `design_confidence_subcategories` so facilitators can audit whether a review rating was grounded in packet evidence.

Treemap signed-value rule:

- Tile labels show signed point contribution.
- Color indicates positive versus negative contribution.
- Tile size should use absolute magnitude or a fixed group sizing rule, because treemap area cannot directly represent negative values.
- The root label should show the Total Scenario Score when combined view is enabled, but the chart should not imply that negative tile areas add arithmetically as positive area.
- Completion Outlook and Design Confidence provenance should remain inspectable so users do not confuse app-scored design contributions with SHAP-derived model impacts.

### Score Visibility, Pending State, And Delta Policy

Hidden baseline:

- Completion Outlook may be participant-visible because it is the original visible model score.
- Design Confidence and Total Scenario Score must remain participant-hidden.
- Do not show previous-value cards or deltas for hidden baseline Design Confidence or Total Scenario Score.

Pending review after a feature change:

- Completion Outlook, Design Confidence, and Total Scenario Score should keep showing the latest successful reviewed values until the participant clicks `Review Scenario`.
- Each affected score view should clearly mark that a review is pending. Completion Outlook already shows `Score update pending`; Design Confidence and Total Scenario Score should receive equivalent pending-review signaling.
- Design Confidence and Total Scenario Score should not disappear merely because a feature value changed. They should disappear only when no participant-visible review exists yet, such as immediately after hidden baseline generation.
- If the participant changes a value and then restores it to the latest reviewed value, the pending state should clear and the previous reviewed score views should restore.

Previous-value and delta policy:

- Baseline: no Design Confidence or Total Scenario Score previous-value cards.
- First visible iteration:
  - Completion Outlook compares current Completion Outlook against baseline Completion Outlook.
  - Design Confidence shows no previous-value delta because participants did not see a baseline Design Confidence score.
  - Total Scenario Score compares current Total Scenario Score against baseline Completion Outlook, with points and percent variance calculated from those two values.
- Second and later visible iterations:
  - Completion Outlook compares current Completion Outlook against previous visible Completion Outlook.
  - Design Confidence compares current Design Confidence against previous visible Design Confidence.
  - Total Scenario Score compares current Total Scenario Score against previous visible Total Scenario Score.

Bar chart delta policy:

- Completion Outlook bar chart shows pillar `+/- pts` variance against the previous Completion Outlook snapshot, including first iteration versus baseline.
- Design Confidence bar chart should not show previous-point variance on the first visible iteration.
- First visible Total Scenario Score bar chart compares current combined pillar values against baseline Completion Outlook pillar values.
- From the second visible iteration onward, Design Confidence bar chart variance compares current Design Confidence pillar contributions against previous visible Design Confidence pillar contributions.
- From the second visible iteration onward, Total Scenario Score bar chart variance compares current combined pillar values against previous visible Total Scenario Score pillar values.

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
- Generate fresh medical/development and ClinOps questions at each visible iteration. Assume participants discussed the previous questions, so new questions should respond to the latest value changes, current dilemma, or a high-value challenge raised by the trial context.

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
- Scenario Review ratings per iteration.
- Design Confidence per iteration.
- Total Scenario Score per iteration.
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
- Design Confidence.
- Main gain.
- Main trade-off.
- Resolved concerns.
- Persistent concerns.
- New concerns.
- Storyline update.

For the next prediction, pass baseline review, previous review, compact storyline memory, current delta packet, and current snapshot. After several iterations, the system should pass a compact case memory summary rather than the full raw history every time. This avoids long context, repeated warnings, and drift in the narrative. Raw history can still be stored for audit, export, or facilitator debrief.

### Review Regeneration And No-Op Policy

The application should decide whether a new Scenario Review is needed before calling the LLM. This prevents the narrative from changing when the user has not materially changed the scenario.

Do not call the LLM and do not create a new storyline step when:

- No Completion Outlook score-input Trial Features changed.
- No active operational assumptions changed.
- No editable text fields changed materially.
- Completion Score and operational metadata are unchanged.

For no-op predictions, reuse the latest validated review and leave Design Confidence, Total Scenario Score, and storyline memory unchanged.

For minor text-only edits, use a materiality gate before triggering a full review. The first implemented gate normalizes case, whitespace, and punctuation before comparing text snapshots; richer semantic materiality can be added later:

```text
Normalize text -> compare to previous text -> classify as no-op, minor wording, or material meaning change.
```

Examples:

- Typo, punctuation, casing, whitespace, or a single wording cleanup = no new review.
- A short clarification that does not alter endpoint, population, intervention, or operational meaning = no full review; optionally update displayed text only.
- A text edit that changes endpoint intent, population scope, intervention description, rationale, or creates a structured-field contradiction = material text change and may trigger a new Scenario Review.

If a text-only material change triggers review, the narrative should state that the design variables did not change and the review changed only because the textual rationale/context changed. The application should avoid presenting this as a new model-score movement.

Do not run a clarification gate before prediction for `structured_features` / `text_context` mismatches. If Trial description field changes are material, submit the scenario through the normal `Predict Trial Completion` flow and let Scenario Review interpret the submitted structured fields, operational assumptions, and `text_context` after scoring.

### Live Regression Targets

Keep a small named trial set for manual live Scenario Review calibration after schema or prompt changes:

- `NCT03386721` - Simlukafusp alfa (ROCHE), Oncology, 2018: review whether Execution Framework / Operational Burden Balance narratives are specific, expert, auditable, and strategically useful without prescribing the next edit.
- `NCT03896581` - `[BE COMPLETE]` Bimekizumab (UCB), Musculoskeletal, 2019: change Pathway Profile from `Interleukin Cytokine` to `Kinase Inhibitor` and review whether Scenario Review explains the clinical-development meaning of a pathway-class change without overclaiming mechanism, efficacy, or regulatory implications.

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
  - `NARRATIVE_LLM_MAX_OUTPUT_TOKENS=12000`
  - `NARRATIVE_LLM_TIMEOUT_SECONDS`
  - `NARRATIVE_LLM_MAX_RETRIES`
- Current setup status: local `.env` can hold these values, and `src/narratives/provider_config.py` reads and validates them without making any LLM API call. `scripts/check_narrative_openai_smoke.py` and `scripts/check_narrative_gemini_smoke.py` can run opt-in API smoke tests when `RUN_NARRATIVE_OPENAI_SMOKE=1` or `RUN_NARRATIVE_GEMINI_SMOKE=1` is set; they skip by default to avoid accidental network calls or API spend. `src/narratives/provider.py` contains real OpenAI and Gemini invocation helpers behind the same normalized provider result shape. `frontend/views/trial_simulator.py` uses the deterministic mock provider by default and routes both hidden baseline and visible Scenario Review calls through the live provider chain only when `NARRATIVE_LIVE_REVIEW_ENABLED=1`. Live wrapper checks have validated full fixture reviews using the normalized provider boundary.
- Provider config, prompt/schema fixtures, opt-in OpenAI/Gemini smoke testing, and opt-in simulator UI routing now target the active Scenario Review contract with `completion_outlook_analysis`, `design_confidence_subcategories`, required qualitative `score_materiality`, `review_metadata`, `scenario_consistency_note`, and text-change evidence in the packet context. If live routing is enabled and all configured providers fail or validation does not produce a complete Scenario Review, the participant panel shows Completion Score only and marks Scenario Review unavailable for the current scenario; participants should use the normal Review Scenario action again after addressing the issue or retrying later. It does not reuse stale Design Confidence for a new packet.
- Active provider prompts use packet-section names consistently. `Completion Outlook score` is `model_interpretation.completion_score`; `Completion Outlook score inputs` are selected categorical/numeric fields in `structured_features` that feed the score, identified through `model_interpretation.direct_xgboost_shap_fields` and `model_interpretation` score evidence, with readable labels in `structured_feature_display_values` and meanings in `structured_feature_meanings`; `Trial description fields` are `text_context` fields with meanings in `text_context_field_meanings`, UI labels `Title` (top study title), `Summary`, `Conditions`, `Interventions`, and `Primary Outcomes`, and JSON keys `title`, `summary_ui`, `conditions_ui`, `interventions_ui`, and `primary_outcomes_ui`; `Planning assumptions` are `operational_assumptions.planned_enrollment`, `operational_assumptions.planned_sites`, and `operational_assumptions.planned_duration_months`; `Review controls` are `review_controls`; `Completion Outlook narrative` is `completion_outlook_analysis`; `Design Confidence narrative` is `design_confidence_analysis`; and `Design Confidence subcategory ratings` are `design_confidence_subcategories`.
- Participant-facing Completion Outlook wording should avoid internal model vocabulary. Use plain phrases such as `Completion Outlook score inputs`, `score inputs`, `score pattern`, `score-driving fields`, or `early-termination risk pattern` when explaining the scoring boundary; avoid phrases such as `model-facing`, `model signal`, `model-score inputs`, `model suggests`, `model indicates`, `model registers`, `model-derived`, `model interpretation`, `in the model`, `model's...`, or `the model reflects`. The provider prompt asks the model to replace any remaining internal model-language phrase before finalizing participant-facing text.
- Planning-assumption fields are outside Completion Outlook: planned enrollment, planned site count, and planned total duration. If the latest change is limited to these fields, the Completion Outlook score is unchanged because they do not feed the Completion Outlook score. If the latest change is limited to Trial description fields, or combines Trial description fields with planning assumptions, and no structured Completion Outlook score input changed, the Completion Outlook narrative should use the provided stable-mode sentence when `review_controls.required_completion_outlook_sentence` is present: `The Completion Outlook score remains stable because the latest changes are not directly used to calculate the Completion Outlook score. Nevertheless, the updated scenario details are considered in Design Confidence.` It should not name or summarize planning-assumption details such as enrollment, site count, total duration, planned duration, primary duration, resource allocation, or operational footprint. Those planning assumptions remain Design Confidence context for proportionality and executability. If other Completion Outlook score inputs also changed, the Completion Outlook narrative should be explained by those score-input changes only. Other operational trial features may still be discussed in Completion Outlook when they are actual Completion Outlook score inputs and packet evidence supports them.
- Mixed structured-plus-planning changes use a narrow review-control mode, `structured_score_inputs_only`, rather than deterministic replacement. In this mode the provider should still write the Completion Outlook narrative, but only from changed structured Completion Outlook score inputs and aligned Trial description field context; changed planning-assumption fields are passed in `completion_outlook_forbidden_latest_fields` and remain Design Confidence context. The same exclusion applies to planning-assumption proxy phrases such as `operational footprint`, `operational scale`, `site expansion`, `larger enrollment`, `scaled execution`, or `site performance`.
- Resource, staffing, and budget implications are qualitative Operational Burden Balance considerations. The system should not estimate monetary cost, affordability, or financial feasibility without explicit financial inputs. Added resource intensity should influence Design Confidence through proportionality: whether the added burden is justified by evidence, patient-relevance, governance, or interpretability value gained.
- Operational simplification may receive Operational Burden Balance credit when it genuinely improves executability. When simplification is achieved mainly by weakening comparator, masking, allocation, endpoint rigor, or evidence ambition, strong positive credit (`+3` to `+5`) requires independent operational value or a context where lower evidence ambition is appropriate, such as a coherent safety-extension/proportionality rationale. Removing randomization, masking, comparator structure, arms, or endpoint rigor is not independent operational value by itself. In shortcut scenarios, Operational Burden Balance should usually be bounded unless a separate access, safety-extension, oversight, patient-burden, or proportionality gain is present. Otherwise shortcut-driven feasibility should be framed as bounded and usually low or moderate materiality, so it does not overpower Endpoint Evidence Strength or Phase & Intent concerns.
- When multiple core evidence-quality controls are removed together in a shortcut scenario, unchanged target-population relevance should not by itself keep total Design Confidence positive. Endpoint Evidence Strength and Phase & Intent Alignment should carry the main penalty unless the packet provides a clear safety-extension, exploratory-signal, access, or proportionality rationale for lower evidence ambition. This is applied through `review_controls.shortcut_design_confidence_rule`, not as a deterministic score override.
- For hard product-boundary cases, the app may pass narrow `review_controls` to the provider. These controls should define the Completion Outlook mode, latest-change focus, forbidden latest fields for Completion Outlook, and required question focus. They should not reduce Design Confidence into a template; Design Confidence should continue to use evidence-first expert reasoning.
- The deterministic Completion Outlook boundary is shared in `src/narratives/review_controls.py` and applies in both the eval harness and live Scenario Review storage path. For operational-only and stable non-score-input modes, only `completion_outlook_analysis` is normalized; Design Confidence narrative, subcategory ratings, rationales, questions, and scores remain provider-generated.
- When `review_controls` are present, participant-facing narratives should explain the latest change without re-labeling older cumulative issues as newly changed. Older issues may remain relevant to the current full scenario, but they should not be described as if they were introduced by the latest edit.
- For later visible iterations, the participant questions should work as a set: the medical/development question should focus on the medical or evidence implication of the newest material scenario change; the clinical-operations question should raise an operational-development debate using the trial or latest change as a concrete example; and the strategic/field question should step back to a broader Therapeutic Area or field-level challenge exposed by the scenario. If an older dilemma remains relevant, it should be reframed through the newest material change rather than repeated in the same question frame or opening stem. Questions should be framed as general discussion prompts without naming or addressing responsible parties or participants, and should not use `team`, `sponsor`, `sponsors`, `investigator`, `investigators`, `stakeholder`, `stakeholders`, `you`, or `your`; the prompt asks the model to rewrite any question containing those words into impersonal field-level wording before finalizing. The strategic/field question should vary its lens across evidence standard, access, governance, data reliability, representativeness, feasibility, and interpretability rather than repeatedly using the same opening frame. When the latest change is limited to planning assumptions, the medical/development question should explicitly mention the latest planning context, such as enrollment, site count, duration, planning burden, operational scale, or proportionality, while connecting current evidence ambition to whether the added burden is justified. When the latest change creates a `structured_features` / `text_context` conflict, at least one question should focus on resolving or reconciling the scenario before relying on it; questions should not ask participants how to operationalize stale contradictory Trial description detail.
- Trial description fields do not directly feed the Completion Outlook score. They may support the Completion Outlook narrative only when they align with, clarify, or add non-conflicting detail to selected Completion Outlook score inputs. This conflict rule applies across all Trial description fields in `text_context` and all relevant `structured_features`, not only intervention descriptions. Completion Outlook score inputs define the score-interpreted scenario when they directly conflict with Trial description fields. Only the conflicting Trial description field detail should be treated as stale scenario text superseded by the structured_features value; it should not be used as Completion Outlook evidence or as evidence that the selected structured design has the contradicted modality, delivery burden, endpoint, or population feature. Non-conflicting Trial description field details and latest `text_context` changes remain valid context when they clarify population, endpoints, intervention rationale, or trial context. In the participant warning, "text is used as supporting context" means aligned or non-conflicting Trial description field content; the directly conflicting detail remains stale scenario text superseded by the corresponding `structured_features` value.
- `structured_features` / `text_context` conflict is a scenario-readiness warning. It may affect Design Confidence because the scenario is not internally aligned enough to rely on, but it should usually affect the most relevant Design Confidence subcategory and should not drive multiple strong negative subcategory ratings unless non-conflicting structured fields independently support those penalties.
- When only the three planning-assumption fields changed and the Completion Outlook score delta is `0.0`, the app may deterministically set the participant-facing Completion Outlook boundary sentence before storing/reporting the trace. This fixed sentence is exclusive to the planning-assumption-only boundary mode and must not be reused for `structured_features` / `text_context` consistency cases or intervention-modality changes. This is a product boundary, not a clinical judgment, and should not alter Design Confidence ratings, score materiality, subcategory rationales, or Total Scenario Score calculation.
- Do not add post-narrative deterministic cleanup for participant wording or question rewriting at this stage. Internal-language leaks and repeated/similar questions should be handled by prompt wording and eval findings only, except for the existing fixed planning-assumption Completion Outlook boundary sentence and provider-neutral unavailable-review error formatting.
- Latest three-trial live Gemini run `first_wave_operational_shortcut_cap_3trials_1` returned 12/12 reviewed visible iterations, 0 failed checks, and 3 warning checks. The operational shortcut cap behaved as intended, with shortcut-driven simplification receiving only limited Operational Burden Balance credit. Remaining warnings were question opening-frame repetition and one scenario-readiness dominance review item, so no urgent prompt change is required before the next broader wave.
- Broader five-trial live run `first_wave_broader_trials_5_1` returned 20/20 reviewed visible iterations, 3 failed checks, and 10 warning checks. Follow-up adjustments are eval/prompt-boundary only: avoid `model signals` by using score-pattern wording, skip the positive Target Population Alignment expectation when the synthetic population edit conflicts with a prevention/vaccine-style trial objective, and require operational-only medical questions to reference planning burden, scale, or proportionality.
- After `first_wave_broader_trials_5_2`, the main residual quality issue is question generation rather than core narrative/scoring logic. The active contract now adds `key_questions.strategic_field_question` as a third participant question. It should raise a broader Therapeutic Area or field-level development-design challenge using the current scenario as the example, while the existing medical/development and clinical-operations questions remain focused on the newest scenario change and operational-development implications.
- Latest five-trial live run `first_wave_three_question_contract_5_1` returned 19/20 reviewed visible iterations, 4 failed checks, and 20 warning checks. One fail was a transient Gemini `ServerError`, not a prompt issue. Follow-up changes remain light: remove participant-facing `in the model` leakage by strengthening score-pattern replacement language, make the patient-relevance expectation skip broader prevention/vaccine contexts when refractory/metastatic edits conflict with the base objective, and vary the strategic/field question lens to reduce repeated `What evidence standard...` / `How should the field balance...` openings.
- Any OpenAI model used in later validation should be pinned to an explicit snapshot rather than a floating alias. OpenAI Pro/high-reasoning profiles can be considered for slower, high-quality hidden baseline generation or offline review, but they are not the default live interactive path after the June 2026 Gemini Flash-Lite decision.
- Any Gemini model used in production or fallback should be configured with an explicit model ID rather than hard-coded in product logic. The current live interactive candidate is `gemini-3.1-flash-lite`; a Pro-class Gemini model can be evaluated later for slower offline or fallback review quality.
- Future provider-chain mode should try the configured primary provider first, then the configured fallback only for provider/network/rate-limit/unavailable failure. Do not fallback when the primary provider returns valid but unfavorable clinical reasoning, or when the provider returns malformed/invalid review JSON; that would create provider-shopping behavior and hide prompt/contract problems that should be fixed.
- Cache and trace keys must include provider, model name, and live generation-control namespace so OpenAI and Gemini outputs, or outputs produced with different reproducibility settings, are not treated as interchangeable.
- For live provider-chain calls, if the same input packet and same live generation-control namespace already have a validated cached review, reuse it before calling any provider. This keeps provider fallback transparent to participants: the app does not regenerate a review just because the provider that answered last time differs from the provider that would answer now.
- Keep the implementation deliberately small: one provider config reader, one prompt/schema builder, and one normalized provider result shape shared by mock, OpenAI, and Gemini. Avoid separate scoring, packet-building, cache, or UI code paths per provider.
- In future provider-chain mode, provider fallback should be bounded and auditable. Try at most the configured primary and one configured fallback for a given packet. Store which provider failed, why it failed, and which provider generated the accepted review. Do not silently retry multiple times or cascade across many models.
- Provider identity should remain transparent to participants. The participant panel should not say whether OpenAI, Gemini, or another live provider produced a review. Provider and model names belong in trace/debug/facilitator metadata only.
- If all configured live providers fail, return an unavailable Scenario Review state and show Completion Score only. Do not reuse stale Design Confidence for the new packet.
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
- The main artificial slowdown drivers are primary-provider timeout, retry count, reasoning effort, output budget, and hidden-baseline generation timing. With timeout 60 seconds, one retry, primary plus fallback, the worst-case provider wait can approach 240 seconds before normal generation overhead.
- Cache behavior should be interpreted carefully: validated successful reviews can be replayed for the same packet and same generation-control namespace; malformed, incomplete, provider-error, or validation-failed reviews should not be treated as reusable Scenario Reviews.
- Local Completion Score prediction API latency was not measured in these provider benchmarks because the local `/health` endpoint was unavailable during the shell check. The observed multi-minute UI delay was dominated by live LLM generation and provider fallback, not by known XGBoost scoring work.

Implementation-time cost-control decision after the first billing check:

- During active coding and UI iteration, keep live review disabled by default with `NARRATIVE_LIVE_REVIEW_ENABLED=0`. This makes the deterministic mock path the default and prevents accidental API spend.
- For many low-cost live tests per day, use a single cheap primary provider rather than a fallback chain. The current validated local development profile is Gemini-only with no effective fallback: `NARRATIVE_LLM_PROVIDER=gemini`, `NARRATIVE_LLM_FALLBACK_PROVIDER=gemini`, `GEMINI_NARRATIVE_MODEL=gemini-3.1-flash-lite`, `NARRATIVE_LLM_MAX_OUTPUT_TOKENS=12000`, `NARRATIVE_LLM_TIMEOUT_SECONDS=60`, and `NARRATIVE_LLM_MAX_RETRIES=0`. The config loader collapses same-provider fallback to `None`.
- Keep the OpenAI model configured as a cheaper reserve option, such as `gpt-5.4-mini`, rather than `gpt-5.5` during implementation. `gpt-5.5` should be reserved for rare high-quality/offline validation, not repeated local development testing.
- This implementation-time profile is intentionally different from the later production resilience design. Production can re-enable a primary/fallback chain after latency, reliability, budget, and provider-output quality are calibrated.

Open live-play calibration items before rollout:

- Decide whether hidden baseline generation remains on Simulation Mode toggle or is deferred until first `Predict Trial Completion`. Toggle-time generation makes the first prediction faster after the wait; first-predict generation keeps toggle responsive but can make the first prediction require both baseline and visible-review calls.
- Continue monitoring Gemini's full visible-iteration JSON reliability during representative live play. The current hardening uses SDK response schema, omitted/default temperature, primary `thinking_level=high`, a 12000-token primary output ceiling, and one explicit malformed/MAX_TOKENS retry with lower thinking and a 16000-token ceiling, while preserving app-owned validation and scoring.
- Measure the local `/predict` API separately once the API is running, so model scoring time is separated from provider time in latency budgets.
- Repeat timing tests with representative real trial scenarios, not only contract fixtures, before setting production timeout/retry defaults.
- Make a first qualitative assessment of live participant-review text across several representative real scenarios before deciding whether the prompt should become shorter, more structured, or model-specific.
- Update the participant UI so Design Confidence can be integrated with Completion Score into the Total Scenario Score, while still preserving the distinction between XGBoost Completion Outlook drivers and app-owned Scenario Review contributions.
- During live testing, expose compact timing diagnostics for successful and failed Scenario Reviews. The diagnostics should separate hidden-baseline lookup/generation time, visible-review provider/store time, total visible workflow time, provider latency, attempts, cache hits, configured timeout, applied provider timeout, response length, and validation status. These diagnostics belong in an expander for calibration/debugging, not in the main participant narrative.

Current mid-way development sanity check:

This review has been superseded by the four-pillar Design Confidence plan. It remains useful as historical context for why prompt size, provider reliability, and participant display need explicit calibration.

1. Prompt, token, output, and cost audit. First inspect exactly what the application sends to the LLM for `hidden_baseline`, `first_visible_iteration`, and `later_visible_iteration` prompts: prompt instructions, response contract, packet JSON, field-change evidence, XGBoost movement evidence, `text_context` Trial description fields, baseline/previous review context, operational assumptions, and clarification context. For each representative packet, record prompt character count, approximate input tokens, configured output budget, actual response length, parser/validation result, cache behavior, and estimated cost for the selected model. Then compare observed wall-clock time against this input/output volume and provider limitations. The audit should make the LLM input readable as if a facilitator had written the prompt manually, while keeping secrets and raw provider outputs out of participant UI.
2. Design Confidence and participant display review. Reassess whether the four design subcategories, rating-to-point mapping, and participant wording are coherent, legitimate, and easy to explain. The review should test whether participants can understand how Completion Score features, Scenario Review evidence, and Design Confidence differ.
3. Implementation plan for prompt/UI changes. Only after the first two reviews decide what to change in code: which packet fields should be removed, summarized, or added; whether hidden baseline generation should be deferred or cached durably; which provider/model profile should be used for interactive play; and how the Total Scenario Score, Completion Outlook drivers, and Design Confidence contributions should be rendered in the simulator.

Gemini JSON reliability finding from the NCT02741128 live audit:

- The real visible-iteration prompt was moderate in size, about 13.9k characters and 3.5k estimated input tokens. Prompt volume alone did not explain 45-second waits.
- The malformed JSON failures were caused by Gemini spending too much of the generation budget on hidden thinking and leaving too little visible budget to complete the JSON object. With `NARRATIVE_LLM_MAX_OUTPUT_TOKENS=2500`, one failed call used about 1.6k thinking tokens and cut the JSON mid-string.
- The Gemini provider path now uses the SDK response schema and the Gemini 3 `thinking_level` control. The current production-style setting is `gemini-3.1-flash-lite`, omitted/default temperature, primary `thinking_level=high`, and a 12000-token primary output ceiling. The output ceiling is a completion-safety margin, not a quality knob; in earlier medium-thinking benchmarks, ceilings from 4000 through 12000 produced the same visible review quality and actual token use for the tested scenarios. The 12000-token default is retained to prepare for longer future reviews.
- The explicit metadata-visible retry remains bounded to one attempt. It is reserved for malformed JSON or provider `MAX_TOKENS`; the retry lowers thinking to `low` and uses a 16000-token ceiling because excessive thinking can consume output budget and harm JSON completion.
- Later temperature/thinking evals on five-trial Scenario Review waves showed omitted/default temperature gave better visible quality than explicit `0` or `0.3`, while explicit high thinking reduced failed/warning checks versus default medium thinking. Higher thinking did not make outputs deterministic: duplicate runs still drifted in Design Confidence scoring and wording. Therefore high thinking is a quality setting, not a reproducibility guarantee; reproducibility-sensitive evals should keep using duplicate traces and drift inspection.
- Final-settings quality/reproducibility assessment should use `scripts/run_final_narrative_quality_plan.py`. The default wave runs 10 live Gemini trials with omitted/default temperature and explicit high thinking to detect quality, adherence, and scoring patterns, then runs the first 3 trials twice under the same settings to inspect reproducibility drift. The helper also writes a generation-control comparison report so the review can separate systematic prompt issues from expected live-provider variability.
- Final narrative validation should use `scripts/run_final_narrative_validation_plan.py`. This is separate from generation-control testing: it keeps omitted/default temperature and Gemini high thinking, then runs a boundary-behavior wave, a 12-trial credible-storyline candidate wave, and a duplicate reproducibility wave. The boundary wave is non-cumulative: each boundary iteration resets to the same baseline so it can isolate latest-change behavior. It covers the key input combinations: structured score-input only, Trial description only, planning assumptions only, structured + Trial description, Trial description + planning assumptions, structured + planning assumptions, all three input types together, `structured_features` / `text_context` contradiction, aligned non-conflict structured/text version, and shortcut simplification. The storyline wave remains cumulative because it is deliberately built as candidate material for later one-shot example selection; it targets Oncology plus UCB-relevant therapeutic areas and prefers UCB-sponsored candidates when available without making sponsor identity a hard filter. It should be reviewed for credibility, cross-functional tension, narrative coherence across 2-4 iterations, Design Confidence scoring quality, question quality, and whether the example is `presentation_ready`, `good_after_light_edit`, `useful_for_stress_test_only`, or `discard`. Do not embed one-shot examples into the live provider prompt until candidate examples are selected and an A/B check shows they improve quality without making output formulaic.
- Final validation wave `final_validation_boundary_10_1`, `final_validation_storyline_candidates_12_1`, `final_validation_repro_3_a`, `final_validation_repro_3_b`, and `final_validation_repro_3_comparison` completed on 2026-06-15. Boundary behavior was stable at 110/110 reviewed visible iterations. Storyline candidates completed 48/48 reviewed visible iterations and are the source of truth for later one-shot selection. Duplicate storyline reproducibility was 12/12 exact iteration matches and 12/12 score matches. The next step is human qualitative inspection of full storylines, changed fields, narratives, subcategory rationales, score movements, and questions before any example is selected or embedded.
- Real-provider diagnostics now record token usage metadata when available, including Gemini prompt, candidate, thought, cached-content, and total token counts; they also record finish metadata such as finish reason and safety-rating count when exposed by the SDK.

Current structured/text consistency-check status:

- Removed from the active simulator.
- `Predict Trial Completion` proceeds directly with edited structured fields, text fields, and operational assumptions.
- The deterministic alignment module and checker are not part of the active workflow.
- The active Scenario Review path should not return `clarification_needed` before scoring.

Implemented structured-field red flags:

These are UI coherence checks for impossible or internally incompatible structured Trial Feature combinations. They are not Design Confidence scoring rules and should stay separate from the later Design Confidence scoring-calibration work. Implementation should mirror the existing placebo consistency behavior: run immediately when fields change, highlight involved controls with red background, do not add amber states, do not add a compact warning card for now, and do not auto-correct except for the existing placebo sync behavior. Red-highlighted fields should remain highlighted until the incompatible combination is resolved, but they must not disable `Review Scenario` or block Scenario Review generation.

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

Do not include the following as red flags in this first pass because they can be unusual but not impossible: `Phase 3` with `Single Group`, `Confirmatory / Registration` with `Single Group`, `Hard Clinical (Survival/Death)` with short endpoint duration, `Advanced / Metastatic` with `Adjuvant / Neoadjuvant`, `Prevention` with patients only, `Healthy Volunteers = Yes` with non-healthy patient fields, or `Data Monitoring Committee = No` in high-risk or pivotal trials. These may remain Design Confidence considerations or future amber warnings, but not hard red structured-field incompatibility flags.

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

- Current prototype trace should remain simple and session-state compatible. It should store `input_packet`, provider/mock `output_json`, `validated_review`, validation status/errors, app-owned Design Confidence, Total Scenario Score, Design Confidence subcategory contributions, clarification issues, user clarifications, changed fields, score movement, provider/model identity, and compact storyline memory.
- Current real-provider traces store prompt template version, response schema version, configured/applied generation controls, attempts, latency, parse status, response text length, token usage when available, finish metadata when available, malformed-JSON retry metadata, and fallback-after metadata in provider metadata. The UI may expose these fields in a compact technical diagnostics expander when live Scenario Review is unavailable, without showing API keys, raw prompts, or raw provider output. Add prompt template hashes only if prompt version strings are not enough for audit. Add a compact evidence-audit summary only if unsupported evidence fields become hard to inspect from `design_confidence_subcategories`.
- Defer until durable provider tracing: raw provider response, parsed JSON response, provider response ID, system fingerprint, and provider-specific safety/refusal metadata beyond compact finish/safety counts. These fields are not meaningful for the deterministic mock reviewer and are not required for the current mock-default simulator path.
- Defer until durable storage: database/file persistence, shared trial-level baseline review records, cross-team replay, facilitator export, retention policy, privacy controls, and schema migration strategy.
- Do not expand the prompt packet just because the trace stores more audit data. Store enough for audit; send only curated current-context fields to the LLM.

Gemini prompt-size guidance:

- Keep each live Gemini narrative input prompt under roughly `10k` to `20k` input tokens where practical.
- Treat this as an operating target, not a hard validator limit. A larger prompt may be acceptable for an exceptional baseline or offline review, but it should trigger prompt-size diagnostics and a review of what can be summarized.
- Do not send full raw iteration history, full source documents, raw reference PDFs, raw database tables, or verbose prior narratives. Send compact baseline memory, compact previous-review memory, selected reference-pack summaries, selected local context statistics, current field changes, and material XGBoost movement evidence.
- If representative visible-iteration prompts drift above this range, reduce prompt volume before increasing provider timeouts or output ceilings.

The goal is to make repeated runs as consistent as possible while acknowledging that exact determinism is not guaranteed for LLM outputs.

Provider abstraction should be thin. The application should own payload construction, validation, Design Confidence calculation, persistence, cache lookup, and UI rendering. Provider-specific code should own only model invocation and response normalization. The V1 provider boundary includes the deterministic mock provider, explicit unsupported-provider failure path, and real OpenAI/Gemini invocation behind the same normalized result shape.

Real-provider prompts use a funnel instruction, currently implemented in `src/narratives/prompt_builder.py` and validated by `scripts/check_narrative_prompt_builder.py`:

- Use prompt mode `hidden_baseline` for the original trial before participant changes. This mode creates hidden baseline context, qualitative baseline score interpretation, baseline strengths/concerns, consistency flags, and compact memory. It must not write as if the participant changed the scenario and must not expose participant-facing baseline Design Confidence, Total Scenario Score, or hidden numeric quality score.
- Use prompt mode `first_visible_iteration` for the first participant-modified scenario. This mode can compare Completion Outlook to the visible original Completion Score, but must not claim Design Confidence improved or worsened versus a hidden baseline score.
- Use prompt mode `later_visible_iteration` for later participant-modified scenarios. This mode can use previous visible review context for continuity, but Design Confidence wording should stay grounded in supported field changes and evidence rather than unsupported score-to-score storytelling.
- In visible modes, use `iteration_context.field_changes` to identify what the participant changed.
- Use `model_interpretation.xgboost_impact_changes` to understand model movement and materiality. In `hidden_baseline` mode, do not invent participant edits when `field_changes` is empty.
- Treat XGBoost/SHAP movement as model explanation evidence, not proof of clinical causality.
- Translate score evidence into clinical trial / pharma development language for participant-facing text. Explain why the revised scenario may look more or less completion-like, robust, feasible, governed, strategically aligned, risk-reduced, simplified, or less evidence-generating in terms of supported evidence such as endpoint timing, comparator choice, population scope, oversight, operational burden, scientific challenge, or development strategy rather than exposing raw model vocabulary. Total duration, planned enrollment, planned site count, and operational benchmark assumptions must not be cited as Completion Outlook drivers; maximum primary endpoint duration may be used only when present as Completion Outlook score evidence. Do not equate a higher Completion Score with simplification by default, but do flag simplification or value loss when the evidence points that way.
- Participant-facing narratives should state unresolved concerns rather than prescribe exact redesign paths. It is acceptable to say a scenario has unresolved bias-control, interpretability, proportionality, or scenario-readiness concerns; it should not tell participants to switch to a specific comparator, randomization, blinding, endpoint, modality, or population.
- Produce design subcategory ratings, rationale, evidence fields, participant-facing narrative, continuity, and trace fields.
- Do not calculate or return `Design Confidence`, `Total Scenario Score`, or Design Confidence point values. The application calculates those from the validated subcategory ratings.

Use a deterministic input hash based on prompt version, rubric version, baseline snapshot, current snapshot, storyline memory, and `text_context` Trial description fields. If the same input hash is reviewed again with the same provider/model cache namespace, reuse the stored validated review instead of calling the provider again. Generate the baseline review once per selected study and store it for the session. Hashable review context should avoid session-specific trace IDs; use stable input hashes and iteration IDs instead.

Validation and failure behavior:

- If the LLM provider call fails, show Completion Score only and mark Design Confidence as unavailable for the current snapshot.
- Do not reuse stale Design Confidence for a new snapshot.
- If JSON is malformed or fails schema validation, discard scoring fields and either show no narrative or show only validated narrative fields.
- If a design subcategory rating is valid but lacks required `evidence_fields`, set its point contribution to zero and keep the issue narrative-only.
- If partial JSON validates, the application may render validated narrative sections, but Total Scenario Score should be calculated only from validated scoring fields.
- Store validation status and failure reason with the review trace.

## 17. Fields And Source-Of-Truth Principle

The structured feature registry remains the primary design source of truth for the narrative layer.

The LLM narrative layer should treat `structured_features` dropdown and numeric fields as the primary source of truth. Trial description fields in `text_context` are secondary. They should help detect contradiction, missing rationale, or narrative inconsistency. Missing or brief Trial description fields should not be heavily penalized unless they directly contradict structured trial features.

If `structured_features` and Trial description fields conflict, the LLM should flag the inconsistency rather than silently penalize Design Confidence. For example, if `adult_ml` says the scenario is adult-only but `text_context.summary_ui` says the intended treatment population includes pediatric or adolescent participants, the LLM may flag a target-population concern.

Trial description fields in `text_context` are untrusted context. The provider prompt must instruct the model to ignore any instructions, scoring requests, or role changes embedded inside `text_context.title`, `text_context.summary_ui`, `text_context.interventions_ui`, `text_context.primary_outcomes_ui`, `text_context.conditions_ui`, or clarifications. Trial description fields can provide rationale, context, or contradiction evidence, but they must not override `structured_features` unless a future UI explicitly marks them as participant rationale.

Trial Description / Structured-Feature Conflict Handling:

- This rule applies across all Trial description fields in `text_context` and all relevant `structured_features`, not only intervention descriptions.
- For obvious material mismatches, pause the prediction workflow before new scoring and ask the participant to correct the scenario or add an explanation.
- For softer tensions, continue Scenario Review and flag the inconsistency in narrative context.
- Route it to the affected Design Confidence subcategory.
- Require `evidence_fields` before it can affect Design Confidence.
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

The field set should support the Scenario Review and Design Confidence rubric by evaluating:

- Phase and intent alignment.
- Target population alignment.
- Endpoint and evidence strength.
- Operational burden balance.
- Cross-pillar collateral impacts.
- Change integrity.

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
- Scenario Review ratings.
- Design Confidence subcategory contributions.
- Design Confidence.
- Total Scenario Score.
- Input hash.
- Prompt version.
- Rubric version.
- Validation status.
- Failure reason, if any.
- Compact storyline memory.

Storage should keep Scenario Review ratings, Design Confidence subcategory contributions, Design Confidence, and Total Scenario Score as explicit fields rather than storing only a derived narrative explanation.

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
- No redistribution of Design Confidence into XGBoost / SHAP Completion Score pillars.
- No feature-level LLM pseudo-SHAP in v1.
- No LLM-generated final score.
- No nearest-neighbor / similarity cohorts in v1.
- No full feature-norm benchmark table in v1.

## 20. V1 Roadmap Summary

V1 serious-game narrative layer:

- Keep XGBoost unchanged.
- Use Planned Enrollment, Planned Sites, and Planned Duration as deterministic operational assumptions.
- Classify operational assumptions against similar-trial benchmarks.
- Use operational assumptions as structured inputs into Scenario Review.
- Make Scenario Review bidirectional.
- Map Design Confidence into four participant-facing subcategories aligned to the existing Completion Outlook pillars: Phase & Intent Alignment, Endpoint & Evidence Strength, Target Population Alignment, and Operational Burden Balance.
- Calculate reconciled Design Confidence in application logic.
- Calculate Total Scenario Score additively when the combined view is enabled.
- Keep Completion Score View XGBoost-first.
- In Total Scenario Score View, show the four familiar pillars with clear provenance for Completion Outlook subcategories versus Design Confidence subcategories.
- Show a narrative panel explaining design trade-offs.
- Store compact storyline memory so later predictions build on earlier changes.

Implementation staging:

1. Contract fixtures: replace or extend the current fixtures with scenarios that test the four Design Confidence subcategories. Include at least baseline, Completion Outlook score-input edit, operational-only edit, material Trial description edit, `structured_features` / `text_context` review, no-op/minor Trial description edit, score improves but evidence value weakens, score improves and Design Confidence remains neutral, score declines but Design Confidence improves, endpoint description contradiction, biomarker/population mismatch, phase/intent ambition versus weak endpoint or comparator support, modality/risk-governance mismatch, and no-adjustment despite large Completion Outlook movement. Current artifact to migrate: `src/narratives/contract_fixtures.py`, validated by `scripts/check_narrative_contract_fixtures.py`.
2. Deterministic review packet builder: assemble baseline/current/previous snapshots, changed fields, explicit field-change deltas, operational metadata, score deltas, XGBoost impact movements, `text_context` Trial description fields, user clarification context, field dictionary version, canonical taxonomy option-key values, display labels, and compact storyline memory without calling an LLM. Current implementation artifact: `src/narratives/packet_builder.py`, validated by `scripts/check_narrative_packet_builder.py`.
3. Validation and scoring engine: validate review JSON, enforce packet-supported `evidence_fields`, derive Design Confidence subcategory points from `rating + score_materiality + context guardrails`, enforce default-zero and supported-evidence gates, preserve 0.5 increments across the `-5.0..+5.0` subcategory scale, avoid fake balancing, require subcategory totals to reconcile to Design Confidence, and calculate Total Scenario Score when enabled. Current implementation artifact: `src/narratives/scoring.py`, validated by `scripts/check_narrative_scoring.py`.
4. Mock reviewer: use deterministic fake JSON responses based on the fixtures to test validation, scoring math, no-op behavior, text-materiality behavior, and failure handling. Current implementation artifact: `src/narratives/mock_reviewer.py`, validated by `scripts/check_narrative_mock_reviewer.py`.
5. Storage and replay: persist validated review traces in session state first, including input hash, validation status, Design Confidence, Total Scenario Score, design subcategory contributions, and compact storyline memory. Reuse cached reviews for identical input hashes. Current artifact to migrate: `src/narratives/review_store.py`, validated by `scripts/check_narrative_review_store.py`. It supports direct mock replay now and an optional provider-chain path for future live-provider activation without reusing mock cache entries as real-provider reviews. This is prototype storage only and does not yet satisfy the durable cross-team baseline requirement.
6. Pre-prediction consistency check: removed from the active simulator. Do not reintroduce `Check Scenario`, deterministic structured/text gates, or a lightweight LLM consistency pass without a new explicit product decision.
7. Minimal UI panel: render Completion Score, Design Confidence, Total Scenario Score when enabled, Scenario Review, and compact four-pillar Design Confidence rows. Design Confidence treemap leaves include concise rationale details prepared by the Streamlit-free `frontend/utils/scenario_review_plot_data.py` helper, so local checkers can validate plot data without importing the full Streamlit view. Do not overexpose supported/unsupported evidence fields in the participant panel by default; reserve them for future facilitator/debug views. Current implementation artifacts: `frontend/views/trial_simulator.py` and `frontend/utils/scenario_review_plot_data.py`, using the provider-free packet builder, mock reviewer, and session-state review store.
8. Hidden baseline continuity: generate/store the hidden baseline review and verify that later iteration reviews use baseline review, previous review, compact non-numeric baseline quality summary, and compact storyline memory consistently. Current implementation is session-level only; it does not yet provide cross-team durable baseline reuse. Current implementation artifacts: `src/narratives/packet_builder.py`, `frontend/views/trial_simulator.py`, and `scripts/check_narrative_packet_builder.py`.
9. Thin LLM provider wrapper: add provider config first, then opt-in config-path API smoke tests, then provider-chain invocation through the same normalized result shape for mock, OpenAI, and Gemini, then an explicit simulator UI activation flag. Provider config reads API keys from environment variables or secret managers; it never stores keys in code. Current implementation artifacts: `src/narratives/provider.py`, `src/narratives/provider_config.py`, `src/narratives/prompt_builder.py`, `src/narratives/review_store.py`, and the opt-in routing in `frontend/views/trial_simulator.py`, validated by `scripts/check_narrative_provider.py`, `scripts/check_narrative_provider_config.py`, `scripts/check_narrative_prompt_builder.py`, `scripts/check_narrative_review_store.py`, and the skipped-by-default `scripts/check_narrative_openai_smoke.py` / `scripts/check_narrative_gemini_smoke.py`; the simulator still uses the deterministic mock provider by default unless `NARRATIVE_LIVE_REVIEW_ENABLED=1`. Provider code invokes the model and normalizes JSON only; the application owns scoring. Live-provider UI routing is available, but durable provider tracing and live-play calibration remain future work.
10. First adjusted-score visual: replace the current Final Candidate Score / seven-domain grouped chart with the Total Scenario Score view when implementation reaches UI migration. The active target keeps the four familiar pillars and adds one Design Confidence subcategory under each pillar; Completion Outlook and Design Confidence provenance must remain visible in trace/facilitator output.
11. Durable baseline store: add a database-backed baseline review repository keyed by trial/version and input hash. It should use create-if-missing semantics so the first team creates the hidden baseline and later teams reuse it.
12. Two-branch adjusted treemap: add only after the simpler adjusted view is stable and understandable; defer to V1.1 if it slows the first implementation.
13. Calibration/playtesting: review examples and tune rating-to-point mapping if Design Confidence is too strong or too weak.
14. Golden / one-shot example calibration: create 3-5 external golden examples for prompt-quality review before deciding whether any compact one-shot examples should be embedded in the live provider prompt.
15. Prompt enhancement migration checkpoint: before implementing the next prompt/schema migration, align this architecture document and `implementation_plan.md` with accepted durable decisions from `prompt_enhancement_plan.md`, then proceed through staged fixtures, packet builder, prompt/schema, mock/provider normalization, scoring/storage, prompt export review, UI integration, regression/live review, and only then provider settings tuning.

## 21. Open Questions

- Whether later versions add, remove, or reorder fields beyond the v1 field-selection policy above.
- Exact storage mechanism.
- Exact participant versus facilitator UI placement.
- Exact number of previous iterations to keep raw before summarization.
- Exact Design Confidence calibration examples after v1 playtesting.
- Whether facilitator view is hidden behind an expander or separate mode.
- Whether final governance recommendation is generated by participants, LLM, or both.
