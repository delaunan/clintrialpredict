# Prompt Enhancement Plan

## Scope

Architecture scope: `architecture_narratives`

Purpose: define the next prompt-engineering roadmap for the Scenario Review / Design Confidence layer before changing prompt behavior, provider settings, scoring, or UI rendering.

This plan complements:

- `docs/architecture_narratives.md` for durable narrative architecture decisions.
- `docs/narrative_prompt_engineering_brief.md` for current prompt anatomy and LLM flow understanding.
- `implementation_plan.md` for implementation phase tracking.

## Product Direction

The current narrative system should be reshaped around three participant-facing parts:

1. `Completion Outlook Analysis`
   - What the XGBoost risk-pattern model says changed.
   - Why the revised profile appears more or less similar to historically completed versus early-terminated trials.
   - What model-facing fields, SHAP movements, pillar/subcategory movements, and changed structured fields support that interpretation.

2. `Design Confidence Analysis`
   - Whether the resulting design is strategically, clinically, scientifically, and operationally defensible.
   - Whether the Completion Outlook movement appears supported by stronger design logic or challenged by weaker evidence value, reduced relevance, shortcut risk, or disproportionate burden.
   - The rationale behind Design Confidence subcategory ratings.

3. `Two Key Questions`
   - One medical/development question.
   - One clinical-operations/execution question.
   - Questions should challenge the team without prescribing the next edit.

The participant narrative should become easier to read and easier to place visually under the score areas in `frontend/views/trial_simulator.py`.

## Target Participant Output Structure

Canonical provider-facing structure:

```json
{
  "review_metadata": {
    "review_mode": "first_visible_iteration",
    "participant_visible": true
  },
  "completion_outlook_analysis": {
    "risk_pattern_summary": "...",
    "driver_summary": "...",
    "main_model_signals": ["..."],
    "interpretive_hypotheses": [
      {
        "signal": "...",
        "possible_pattern": "...",
        "context_modifiers": ["..."],
        "boundary": "..."
      }
    ],
    "movement_explanation": "...",
    "model_boundary_note": "..."
  },
  "design_confidence_subcategories": {
    "phase_intent_alignment": {
      "rating": "...",
      "evidence_fields": ["..."],
      "rationale": "...",
      "short_rationale": "...",
      "optional_lenses_used": [],
      "regulatory_or_finance_note": ""
    },
    "endpoint_evidence_strength": {
      "rating": "...",
      "evidence_fields": ["..."],
      "rationale": "...",
      "short_rationale": "...",
      "optional_lenses_used": [],
      "regulatory_or_finance_note": ""
    },
    "target_population_alignment": {
      "rating": "...",
      "evidence_fields": ["..."],
      "rationale": "...",
      "short_rationale": "...",
      "optional_lenses_used": [],
      "regulatory_or_finance_note": ""
    },
    "operational_burden_balance": {
      "rating": "...",
      "evidence_fields": ["..."],
      "rationale": "...",
      "short_rationale": "...",
      "optional_lenses_used": [],
      "regulatory_or_finance_note": ""
    }
  },
  "design_confidence_analysis": {
    "summary": "...",
    "confidence_rationale": "...",
    "supporting_evidence": ["..."],
    "limiting_evidence": ["..."]
  },
  "key_questions": {
    "medical_development_question": "...",
    "clinical_operations_question": "..."
  }
}
```

The structured Design Confidence subcategories should remain available for validation, scoring, trace, and visual decomposition. The participant narrative should be organized into the three larger blocks above rather than exposing every internal category as equal narrative sections.

Implementation decision:

- Use one shared response schema with mode-specific constraints.
- Add top-level `review_metadata.review_mode` and `review_metadata.participant_visible` so the same schema can support hidden baseline, first visible iteration, and later visible iteration behavior.
- Mode-specific rules should control required and forbidden wording rather than creating three unrelated schemas.

Suggested participant text length:

- Completion Outlook Analysis: 90-140 words.
- Design Confidence Analysis: 120-180 words.
- Two Key Questions: 20-30 words each.
- Total participant narrative target: roughly 300-380 words.

## Completion Outlook Rules

Completion Outlook must be framed as a model-grounded, movement-aware risk-pattern interpretation.

Required framing:

- The Completion Score reflects how similar the revised trial profile appears to historical patterns associated with completed versus early-terminated trials.
- It is an early-termination risk-pattern assessment, not a prediction of clinical success.
- It is not a claim that the trial will complete.
- It is not a claim that the design is clinically better.
- It should describe model-supported patterns, correlations, and plausible historical-profile interpretations, not causal certainty.
- It may try to make sense of why the revised profile appears riskier or less risky, but only as an evidence-bounded interpretation grounded in score movement, SHAP/pillar/subcategory movement, changed fields, and surrounding context.
- Participant-facing wording should prefer "lower/higher risk of early termination" or "more/less similar to completed-trial patterns" over "higher/lower chance of completion."

Allowed evidence for Completion Outlook:

- Model-facing structured fields.
- Changed structured fields that feed the XGBoost pipeline.
- Completion Score and score movement.
- Pillar impacts and pillar deltas.
- SHAP/subcategory/feature impact changes.
- Top positive, negative, and movement drivers when available.

Forbidden Completion Outlook evidence:

- Planned enrollment.
- Planned site count.
- Planned total duration / operational planned duration.
- Operational benchmark metadata.
- Any non-XGBoost field used only for Scenario Review / Design Confidence.

Operational assumptions may be discussed under Design Confidence, especially `Operational Burden Balance`, but must not be cited as Completion Outlook drivers.

Duration distinction:

- `primary_duration_months_ml` / maximum primary endpoint duration is a model-facing XGBoost field and may be used in Completion Outlook when present in model evidence.
- Planned total duration / operational duration assumption is outside XGBoost and must not be cited as a Completion Outlook driver.

## Completion Outlook Interpretation Depth

Completion Outlook should not be a shallow restatement of score movement. It should help participants understand why the model may see a profile as more or less completion-like. However, this interpretation must remain bounded.

Required interpretation pattern:

1. Name the model signal.
   - Which score, pillar, subcategory, SHAP, or model-facing field movement is relevant?

2. Offer a possible historical-pattern meaning.
   - What might this signal represent in prior trial profiles?

3. Name context modifiers.
   - Which surrounding fields change how the signal should be interpreted?

4. State the boundary.
   - What should the participant not conclude from the model signal?

Candidate response shape:

```json
"completion_outlook_analysis": {
  "review_mode": "first_visible_iteration",
  "participant_visible": true,
  "risk_pattern_summary": "...",
  "driver_summary": "...",
  "main_model_signals": ["..."],
  "interpretive_hypotheses": [
    {
      "signal": "...",
      "possible_pattern": "...",
      "context_modifiers": ["phase_ml", "therapeutic_modality_ml", "patient_severity_ml"],
      "boundary": "Do not interpret this field as directly causing completion risk."
    }
  ],
  "movement_explanation": "...",
  "model_boundary_note": "..."
}
```

This applies to every field, not only obviously ambiguous fields. Any model-facing field may behave as a marker of historical trial profile, complexity, governance, evidence ambition, patient risk, operational footprint, or therapeutic-area norms. The LLM should not simplify a field into a one-direction causal story.

Examples of unsafe simplification:

- "A DMC improves completion because oversight is good."
- "A DMC worsens completion because the trial is risky."
- "Placebo improves evidence quality."
- "Longer duration reduces feasibility."
- "Broader population improves generalizability."

Safer style:

- "The signal should be read as a historical-pattern marker rather than a direct causal effect."
- "In this profile, the field is better interpreted alongside phase, modality, endpoint rigor, population severity, and therapeutic context."
- "The model movement may reflect a more complex or risk-sensitive historical trial profile, but the packet does not prove that this specific field causes completion risk."

Interaction rule:

- The LLM may describe field interactions as hypotheses when they are grounded in packet context and model movements.
- It must not claim the XGBoost model learned a specific named interaction unless explicit interaction tooling is added later.
- Preferred wording: "should be read alongside", "may indicate a combined profile", "could be acting as a marker", "the packet does not prove the mechanism."

Redline:

```text
Completion Outlook explanations must be framed as historical-pattern hypotheses grounded in model evidence and packet fields. They must not present field effects as clinical causality, operational truth, or design recommendations.
```

## Design Confidence Rules

Design Confidence should act as a counterweight and interpretation layer, not as a multiplier of Completion Outlook.

Design Confidence remains an application-calculated score adjustment for quality of trial design. Its narrative role is to challenge the scenario logic and moderate clear Completion Outlook increases or decreases when design-quality evidence supports doing so. It is not a second completion predictor.

Core rules:

- High Completion Outlook does not imply high Design Confidence.
- Design Confidence should ask what idea the team needs to defend given the Completion Outlook movement.
- Positive Completion Outlook movement should be challenged if it appears to come from simplification, weaker endpoint rigor, easier comparator logic, narrower population, reduced follow-up, reduced governance, or other loss of evidence value.
- Negative Completion Outlook movement can receive positive Design Confidence if the added difficulty reflects stronger evidence, better patient relevance, strategically justified ambition, appropriate oversight, or proportionate governance.
- Negative Completion Outlook movement can still receive negative Design Confidence if the added difficulty is poorly justified, incoherent, or disproportionate.
- If Completion Outlook barely moves, Design Confidence should still evaluate coherence and defensibility, but it should avoid inventing a dramatic counter-story.
- Benchmark-typical operational assumptions are neutral by default. They become score-relevant only when combined with design evidence that supports or conflicts with the scenario.

The LLM should provide the rationale that leads to `strong`, `supportive`, `balanced`, `weak`, or `conflicting`, but the app must continue to calculate points deterministically.

Calibration safeguards:

- Positive Design Confidence requires explicit evidence that the scenario strengthened design defensibility, evidence value, patient relevance, strategic fit, governance, or proportionality.
- A model-favorable simplification is not enough for positive Design Confidence by itself.
- If Completion Outlook improves because the scenario appears easier to execute, simpler, shorter, narrower, less rigorous, or less governed, Design Confidence should be neutral or negative unless separate evidence shows the design became more defensible.
- If Completion Outlook worsens because the scenario adds rigor, patient relevance, endpoint interpretability, governance, or justified ambition, Design Confidence may be positive.
- If Completion Outlook worsens because the scenario adds incoherent or disproportionate burden, Design Confidence may also be negative.
- The prompt and scoring checks should prevent double-rewarding the same model-favorable simplification as both lower early-termination risk and stronger design confidence.

## Cross-Pillar Evidence Rule

The participant-facing pillar location is not an evidence boundary.

Some fields visually grouped under one Completion Outlook pillar can legitimately affect the Design Confidence subcategory displayed under another pillar.

Examples:

- Endpoint rigor may sit under Scientific Challenge but also affect Phase & Intent Alignment.
- Therapeutic modality may sit under Therapeutic Context but also affect Operational Burden Balance.
- Pediatric or older-adult inclusion may sit under Patient Profile but also affect governance, oversight, and operational burden.
- Planned total duration is outside XGBoost but may affect Endpoint & Evidence Strength and Operational Burden Balance.
- Comparator strategy may affect Endpoint & Evidence Strength, Phase & Intent Alignment, and Target Population Alignment depending on the scenario.

Prompt and trace language should distinguish:

```text
Display pillar location != evidence source boundary
```

The packet, response contract, validation trace, and UI rationale should preserve evidence fields so cross-pillar reasoning remains auditable.

## Design Confidence Subcategory Prompting

Each Design Confidence subcategory should be prompted with a clear core question, positive evidence patterns, concern evidence patterns, neutral/default behavior, optional regulatory or finance/cost lenses, and forbidden overclaims.

The four subcategories are:

1. `Phase & Intent Alignment`
   - Core question: does the scenario's phase, purpose, therapeutic context, modality, and evidence ambition fit the decision the trial appears to support?
   - Positive evidence may include phase matching development ambition, purpose aligning with endpoint/comparator, credible strategic ambition for target/modality maturity, and text/structured consistency.
   - Concern evidence may include confirmatory ambition with weak exploratory evidence, registration-style intent with limited comparator support, early-phase designs overclaiming decision value, or novel modality without enough endpoint/governance clarity.

2. `Endpoint & Evidence Strength`
   - Core question: does the design generate interpretable, decision-useful evidence, or does it appear easier while weakening evidentiary strength?
   - Positive evidence may include clinically meaningful endpoints, credible comparator/control, bias-control features where relevant, endpoint timing that fits outcome maturity, and biomarker/adaptive features that clarify interpretation.
   - Concern evidence may include softer endpoints for high-stakes intent, weakened comparator support, endpoint timing that may undercut interpretability, unclear composite/multiple endpoint logic, or bias-prone design where bias matters.

3. `Target Population Alignment`
   - Core question: does the population match the intended patient and indication question, or does it appear to improve the risk profile by reducing relevance?
   - Positive evidence may include severity and line-of-therapy fit, justified age/sex scope, coherent rare-disease or biomarker restriction, broader relevance without destroying interpretability, and aligned condition text/structured population fields.
   - Concern evidence may include narrowing mainly for ease, unsupported exclusion of older adults/children/severe patients, healthy-volunteer mismatch, biomarker strategy that weakens generalizability without rationale, or disease text conflicting with structured fields.

4. `Operational Burden Balance`
   - Core question: are enrollment, sites, planned total duration, arms, oversight, and delivery demands proportionate to evidence ambition, patient risk, and modality?
   - Positive evidence may include operational burden justified by endpoint rigor or patient relevance, ambitious but coherent enrollment/sites/duration assumptions, oversight matching modality/severity/safety uncertainty, and delivery complexity proportionate to therapeutic context.
   - Concern evidence may include more operational burden without stronger evidence value, optimistic assumptions for patient context, reduced oversight despite higher-risk modality/population, or complexity without clear scientific or strategic gain.
   - Cost/finance comments are most likely to be relevant here, but they must remain qualitative unless actual cost data exists in the packet or reference context.

Optional regulatory and finance/cost lenses:

- These are optional lenses, not mandatory sections.
- Use regulatory-style comments only when phase, endpoint, comparator, population, governance, safety monitoring, data reliability, or strategic ambition makes the issue material.
- Use finance/cost comments only when operational footprint, modality, enrollment, sites, duration, complexity, or evidence value makes resource proportionality material.
- Do not infer formal regulatory acceptability, payer acceptance, market size, budget impact, financial return, or exact trial cost.
- Frame these comments as strategic considerations or questions for discussion.

Conditional-language rule:

- Participant-facing interpretation must be conditional, bounded, and evidence-linked.
- Prefer wording such as `may`, `might`, `could`, `appears to`, `may suggest`, `could raise`, `would need support from`, `one interpretation is`, and `the team may need to defend`.
- Avoid categorical wording such as `will`, `proves`, `demonstrates`, `regulators would`, `payers will`, `this is financially viable/unviable`, `this endpoint is acceptable/unacceptable`, or `this design is better/worse` as fact.

Preferred examples:

- "This could raise a regulatory-style question about whether the endpoint and comparator are strong enough for the intended decision."
- "The larger operational footprint may be harder to justify unless it produces clearer evidence or broader patient relevance."
- "The profile appears less similar to historically completed trials, possibly because the changed fields resemble patterns associated with higher early-termination risk."

Avoid:

- "Regulators would reject this endpoint."
- "This trial costs too much."
- "These changes increase termination risk."

Candidate subcategory response shape:

```json
{
  "rating": "weak",
  "evidence_fields": ["endpoint_rigor_ml", "comparator_benchmark_ml"],
  "rationale": "...",
  "short_rationale": "Endpoint support weakened for the intended decision.",
  "optional_lenses_used": ["regulatory"],
  "regulatory_or_finance_note": "This could raise a regulatory-style question about whether the endpoint and comparator support the intended decision."
}
```

`optional_lenses_used` may be empty. Most reviews should not force regulatory or finance/cost comments unless the scenario makes them material.

## Optional Therapeutic-Area Reference Packs

Prepare the prompt system for optional therapeutic-area context packs.

Proposed folder:

```text
frontend/data/docs/narrative_reference_packs/
```

Proposed naming rule:

- Use the XGBoost canonical `therapeutic_area_ml` value from the pipeline/taxonomy as the source name, such as `ONCOLOGY` or `NEUROLOGY`.
- Store as `.md`.
- Add safe filename handling if canonical values contain spaces, punctuation, or path-unsafe characters.
- TA packs may live in the same reference-pack folder as the general packs. A separate `therapeutic_areas/` folder is not required.

Example:

```text
frontend/data/docs/narrative_reference_packs/ONCOLOGY.md
frontend/data/docs/narrative_reference_packs/MUSCULOSKELETAL.md
frontend/data/docs/narrative_reference_packs/INFECTIONS.md
```

Lookup policy:

- The packet builder should look up a TA pack using the current canonical `therapeutic_area_ml` value.
- Raw canonical values should be converted to a safe expected filename before filesystem access.
- Safe filename rule: trim whitespace, preserve canonical case, replace path separators and unsafe filename characters with `_`, append `.md`, and never allow directory traversal.
- The original canonical value and expected filename should be recorded in the packet trace.
- Missing files should not fail packet construction or review generation.

Packet behavior when a TA pack exists:

```json
"therapeutic_area_context": {
  "canonical_value": "ONCOLOGY",
  "expected_filename": "ONCOLOGY.md",
  "pack_found": true,
  "pack_id": "therapeutic_area.ONCOLOGY",
  "prompt_safe_summary": "..."
}
```

Packet behavior when missing:

```json
"therapeutic_area_context": {
  "canonical_value": "ONCOLOGY",
  "expected_filename": "ONCOLOGY.md",
  "pack_found": false,
  "instruction": "Use general clinical-development reasoning only; do not invent specific current standards, prevalence, efficacy, safety, or regulatory conclusions."
}
```

Prompt rules:

- Use therapeutic-area context when provided.
- If no therapeutic-area pack is provided, use general clinical-development and therapeutic-area knowledge plus packet evidence, but keep wording cautious.
- Broad therapeutic-area reasoning is allowed without a TA pack.
- Do not invent specific current treatment standards, prevalence, efficacy, safety, guideline, or regulatory facts.
- Do not use unsupported therapeutic-area knowledge to create point-moving Design Confidence evidence.

This prepares the architecture for future knowledge expansion without requiring TA packs immediately.

## Therapeutic Context And Conditions

The prompt should use more than static documentation. It should reason from:

- Therapeutic area.
- Conditions text.
- GBD/condition category.
- Rare-disease flag.
- Phase.
- Strategic ambition.
- Target precedent.
- Pathway profile.
- Modality.
- Innovation tier.
- Trial title and summary.

However, the LLM should remain bounded:

- General clinical-development knowledge is allowed.
- General therapeutic-area knowledge is allowed.
- Conditions and therapeutic area may be used to frame plausible considerations.
- Specific disease facts should be used only when present in packet evidence or supplied reference packs.
- The model must not infer unprovided efficacy, safety, standard-of-care, regulatory, or epidemiology claims.

Future optional condition packs may be useful, but therapeutic-area packs should come first because they are broader and easier to maintain.

## Free-Text And Structured-Field Consistency

Structured categorical and numeric scenario fields should prevail when they conflict with free-text fields. Free text should remain available as context, rationale, or contradiction evidence, but it should not silently override the selected scenario fields.

Participant-facing consistency note:

```text
Some scenario details are not fully aligned across free-text fields and selected fields. In this case the value in the selected fields drive the analysis, while the text is used as supporting context (Intervention text, Therapeutic Modality).
```

Use this note only when a clear mismatch remains. The fields in parentheses should use participant-readable labels and should identify the relevant free-text field and selected categorical/numeric field.

Planned response field:

```json
"scenario_consistency_note": {
  "show": true,
  "message": "Some scenario details are not fully aligned across free-text fields and selected fields. In this case the value in the selected fields drive the analysis, while the text is used as supporting context.",
  "fields": ["Intervention text", "Therapeutic Modality"]
}
```

Rules:

- Do not block review generation because of a text/feature mismatch.
- Do not let free text override structured categorical or numeric scenario choices.
- Completion Outlook should follow model-facing structured fields and model evidence.
- Design Confidence may use the mismatch as evidence of reduced coherence when supported.
- Key questions may challenge the team to clarify or defend the mismatch when it is material.

## Text-Change Evidence

The LLM should clearly see what changed in free-text fields, especially when the edit adds new information rather than simply aligning text with structured categorical or numeric choices.

Planned packet field:

```json
"text_change_evidence": {
  "summary_ui": {
    "changed": true,
    "previous_excerpt": "...",
    "current_excerpt": "...",
    "changed_terms_added": ["registrational", "overall survival"],
    "changed_terms_removed": ["exploratory", "response rate"],
    "change_type": "new_information"
  }
}
```

Suggested `change_type` values:

- `alignment_only`: text was edited mainly to match selected structured fields.
- `new_information`: text adds new clinical, design, operational, or strategic meaning.
- `contradiction`: text conflicts with selected structured fields.
- `minor_cleanup`: wording change has no material scenario meaning.

Implementation guidance:

- Start simple and deterministic.
- Include previous/current excerpts and simple added/removed terms.
- In v1, deterministic code may set only `changed`, excerpts, and simple added/removed terms when reliable.
- `change_type` can be LLM-assessed or assigned by light deterministic rules; it should not require a complex semantic classifier in the first implementation pass.
- Let the LLM interpret meaning conditionally, but require it to cite packet evidence.
- Preserve text-change evidence in traces so later prompt-regression reviews can inspect why a text edit mattered.

## Prompt Modes And Narrative Lifecycle

Use separate prompt modes for the three different narrative lifecycle stages, while keeping one shared response schema with mode-specific constraints.

### Mode 1: Hidden Baseline Context

Prompt mode:

```text
hidden_baseline
```

Participant visibility:

- Not visible by default.

Primary job:

- Interpret the original trial and current opening state.
- Learn from the original Completion Score, study text, therapeutic area, condition context, structured fields, and operational opening values.
- Create qualitative baseline context, baseline strengths, baseline concerns, text/structured consistency flags, and compact memory for later visible reviews.

Allowed:

- Interpret why the opening Completion Outlook appears as it does.
- Internally assess baseline design confidence qualitatively.
- Inspect operational fields as baseline context for later Design Confidence.

Forbidden:

- Expose participant-facing baseline Design Confidence.
- Expose baseline Total Scenario Score.
- Create participant-visible design-score comparison language.
- Suggest that participants have already seen a Design Confidence baseline.

Trace policy:

```json
{
  "participant_visible": false,
  "numeric_design_context_policy": "hidden_qualitative_only",
  "design_confidence": null,
  "total_scenario_score": null
}
```

Implementation decision:

- Hidden baseline should use the same structured Design Confidence subcategory review where practical.
- Hidden baseline may produce qualitative subcategory ratings, rationales, and evidence fields.
- Hidden baseline subcategory ratings and evidence fields should still be validated for trace quality.
- Hidden baseline validation may produce a validated qualitative review, but it should not calculate or store participant-visible `design_confidence` or `total_scenario_score`.
- Hidden baseline must suppress participant-visible numeric Design Confidence and Total Scenario Score.
- The stored hidden baseline context may inform later qualitative continuity, but first visible iteration must not compare Design Confidence against hidden baseline numeric or pseudo-numeric values.

### Mode 2: First Visible Iteration

Prompt mode:

```text
first_visible_iteration
```

Participant visibility:

- Visible after the first participant scenario edit.

Primary job:

- Compare Completion Outlook against the visible original Completion Score.
- Explain model-grounded score movement and risk-pattern interpretation.
- Evaluate current Design Confidence without comparing against a hidden baseline Design Confidence score.
- Challenge the participant based on the design trade-off created by the current changes.

Allowed:

- "Completion Outlook moved from X to Y..."
- "The model now sees the revised profile as more/less similar to historically completed trials..."
- "Current Design Confidence is cautious/supportive because..."

Forbidden:

- "Design Confidence improved versus baseline."
- "Design Confidence declined versus baseline."
- "The team resolved the baseline Design Confidence concern."
- Any language implying the participant had already seen a baseline design score.

Governing distinction:

```text
Completion Outlook is comparative.
Design Confidence is current-scenario evaluative.
```

### Mode 3: Later Visible Iteration

Prompt mode:

```text
later_visible_iteration
```

Participant visibility:

- Visible after the second and later participant scenario edits.

Primary job:

- Explain the current changes, current Completion Outlook movement, current Design Confidence, unresolved concerns, newly introduced concerns, and continuity from prior visible reviews.
- Keep Design Confidence change-aware, but not primarily score-variance driven.

Allowed:

- "This iteration introduces a new concern..."
- "The previous concern remains unresolved..."
- "The current edit strengthens endpoint interpretability..."
- "The Design Confidence signal is more cautious because..."

Use caution with:

- "Design Confidence improved/worsened versus previous."

This may be used only when current field changes and validated evidence clearly support the statement.

Preferred distinction:

```text
Completion Outlook narrative is model-grounded, movement-aware risk-pattern interpretation.
Design Confidence narrative is evidence-and-change evaluative.
```

Rationale:

- Participants can see original and current Completion Scores, so Completion Outlook movement is visible and legitimately comparative.
- Participants do not see a baseline Design Confidence score, so first visible Design Confidence should not be framed as a hidden-score movement.
- Later Design Confidence should focus on current changes, evidence logic, unresolved/resolved concerns, and participant challenge rather than score-to-score storytelling.

## Treemap And Visual Rationale

The treemap / contribution visual should eventually include a condensed rationale for each Design Confidence subcategory.

Candidate trace shape:

```json
"design_confidence_assessment": {
  "subcategories": {
    "operational_burden_balance": {
      "points": -2,
      "rating": "weak",
      "short_rationale": "Operational scale increased without matching evidence gain.",
      "evidence_fields": [
        "operational_assumptions.planned_enrollment",
        "number_of_arms_ml",
        "endpoint_rigor_ml"
      ]
    }
  }
}
```

UI use:

- Tile label.
- Signed points.
- One-line rationale.
- Evidence fields in tooltip, expander, or facilitator/debug view.

The goal is to make the rationale visible without overcrowding the chart.

Source-of-truth decision:

- Treemap short rationale applies to Design Confidence subcategories only.
- It should explain in a few words why the subcategory is `strong`, `supportive`, `balanced`, `weak`, or `conflicting`.
- Prefer an LLM-provided `short_rationale` field validated against the same evidence fields as the full subcategory rationale.
- If no valid `short_rationale` is present, the UI may derive a clipped display phrase from the validated subcategory rationale.
- The app should not invent a separate rationale that is disconnected from the validated review trace.

## UI Placement Idea

Participant-facing layout direction:

- Under Completion Outlook score/plot:
  - Completion Outlook Analysis narrative box.

- Under Design Confidence score/plot:
  - Design Confidence Analysis narrative box.

- Under combined Scenario Review area or below both score sections:
  - Two Key Questions.

The narrative should remain readable and not become a diagnostics panel. Raw evidence fields, validation details, provider traces, and benchmark details belong in expanders or facilitator/debug views.

## Reliability Boundaries

Completion Outlook can reliably explain:

- What model-facing fields changed.
- Whether the Completion Score moved up/down.
- Which SHAP/pillar/subcategory movements were material.
- Whether the revised profile became more or less similar to completed-trial patterns in the training data.

Completion Outlook should not claim:

- The trial will complete.
- The intervention is efficacious or safe.
- The design is clinically better.
- A field caused completion.
- An operational assumption drove Completion Score when it is not an XGBoost input.
- Planned total duration drove Completion Score.

Design Confidence can evaluate:

- Coherence of phase, intent, endpoint, comparator, population, modality, governance, and operational burden.
- Whether the design appears more defensible or less defensible than the Completion Outlook alone suggests.
- Whether a high Completion Outlook may reflect score-seeking simplification.
- Whether a lower Completion Outlook may reflect rigorous or patient-relevant design ambition.

## Live Prompt Review Log

This section centralizes live prompt-quality observations from manual scenario testing. Use it to refine the prompt, update fixtures, and later create one-shot examples if rule-based prompting is not enough.

### Live Test Iteration 1: Randomized Active-Controlled Evidence Upgrade

Baseline scenario:

- Trial: `NCT03287245`, idasanutlin, hematology, polycythemia vera.
- Baseline structure: Phase 2, single-arm, open-label, patients with hydroxyurea-resistant/intolerant polycythemia vera.
- Baseline Completion Outlook: `46.9`.

Edited scenario:

- `number_of_arms_ml`: changed from `1` to `2`.
- `allocation_method_ml`: changed from not specified / not applicable to randomized.
- Comparator / placebo / endpoint fields were changed toward active-controlled, placebo-controlled, hard clinical endpoint evidence.
- `primary_duration_months_ml` / maximum primary endpoint duration was set to `12.0`.
- Study summary text added a simulated randomized-control-arm sentence.

Observed scores:

- Completion Outlook stayed flat at `46.9`.
- Design Confidence increased to `+6.0`.
- Total Scenario Score increased to `52.9`.

What worked:

- The review correctly identified the central tension: stronger evidence interpretability versus added design and execution burden.
- Design Confidence moved positively for randomized, active-controlled, hard clinical endpoint evidence.
- Completion Outlook did not increase merely because the design became more rigorous.
- The participant questions were open-ended and relevant to evidence standard and operational proportionality.

Issues to correct in the prompt:

- Completion Outlook over-explained with unsupported assumptions. It mentioned a need for a "larger, more diverse patient population" even though no diversity-related or population-breadth field was changed or clearly model-supported.
- Completion Outlook used broad clinical-development generalization too assertively: "Historically, randomized trials with active comparators and hard clinical endpoints face greater recruitment and retention challenges..." This is directionally plausible but too factual/causal unless grounded in changed model evidence.
- Completion Outlook should stay closer to score movement, non-movement, model-facing changed fields, and resemblance to historical completion / early-termination risk patterns.
- Design Confidence wording was directionally correct but too strong. Phrases such as "significantly improves" and "the design is now more robust" should be softened.
- Design Confidence should more explicitly keep the counterweight: stronger evidence generation may also create cost, governance, operational burden, feasibility, or proportionality concerns when supported by the packet.
- Design Confidence treemap subcategory tiles did not show the short rationale needed to evidence why each rating was strong, supportive, balanced, weak, or conflicting.

Preferred prompt wording direction:

- Prefer "may improve interpretability" instead of "significantly improves interpretability."
- Prefer "appears more robust from an evidence-generation perspective" instead of "the design is now more robust."
- Prefer "may resemble patterns associated with greater execution burden where comparator structure and endpoint ambition increase" instead of broad factual claims about historical recruitment and retention.
- When Completion Outlook is flat, explicitly state that the edited fields did not materially shift the scenario's resemblance to historical completion or early-termination patterns.
- If operational or feasibility language is used in Completion Outlook, it must be framed as a cautious risk-pattern interpretation and tied to model-supported fields, not as clinical truth.

Candidate improved Completion Outlook wording:

```text
The Completion Outlook remains essentially unchanged, suggesting that the added comparator structure and endpoint ambition do not materially shift the model's resemblance to historical completion or early-termination patterns in this scenario. The randomized, active-controlled structure and longer endpoint horizon may introduce execution burden, but this should be read as a risk-pattern interpretation rather than proof that the trial would be harder to complete. The main discussion point is therefore the trade-off between stronger evidentiary interpretability and a design that may require more operational discipline to execute.
```

Candidate one-shot lesson:

- Use this scenario later as a first-visible-iteration example where Completion Outlook is flat but Design Confidence improves.
- The one-shot should demonstrate that the model score did not materially move, while Design Confidence may increase because evidence interpretability improved.
- The one-shot should also demonstrate the redline: do not invent population diversity, recruitment, retention, cost, or feasibility facts unless supported by changed evidence fields or kept in cautious Design Confidence language.
- The one-shot should include Design Confidence subcategory `short_rationale` values suitable for treemap labels or tooltips.

### Live Test Iteration 2: Revert To Simpler Single-Arm Evidence Strategy

Starting point:

- Previous iteration had randomized / active-controlled / hard clinical endpoint evidence.
- Previous scores: Completion Outlook `46.9`, Design Confidence `+6.0`, Total Scenario Score `52.9`.

Edited scenario:

- `number_of_arms_ml`: changed from `2` back to `1`.
- `allocation_method_ml`: changed from randomized back to not specified / not applicable.
- Comparator / placebo / control fields were moved back toward weaker or single-arm settings.
- Endpoint evidence was moved toward surrogate or less rigorous evidence.
- `primary_duration_months_ml` / maximum primary endpoint duration remained `12.0`.
- Study summary text described a single-arm design interpreted against clinical context rather than a randomized control arm.

Observed scores:

- Completion Outlook increased only slightly from `46.9` to `47.2`.
- Design Confidence decreased from `+6.0` to `-3.5`.
- Total Scenario Score decreased from `52.9` to `43.7`.

What worked:

- The small Completion Outlook movement was appropriate: the simpler design did not create an exaggerated score gain.
- Design Confidence dropped strongly, which matched the expected interpretation: operational simplicity came with weaker comparative interpretability.
- Total Scenario Score moved down because the Design Confidence penalty outweighed the small Completion Outlook increase.
- The central tension was clear: a simpler single-arm structure may not match pivotal / registration intent.
- The clinical operations question about assessment bias was useful and discussion-oriented.

Issues to correct in the prompt:

- Completion Outlook again used overly broad feasibility language: "historically associated with lower operational complexity and faster recruitment." "Faster recruitment" should not be stated unless directly supported by model-facing changed fields or packet evidence.
- Completion Outlook described the result as an improvement even though `+0.3 pts` is marginal. It should call this essentially stable or only slightly higher.
- Design Confidence was directionally correct but too categorical in phrases such as "less robust" and "significant interpretability gap."
- Design Confidence treemap subcategory tiles still lacked the short rationale needed to explain the rating in a few words.
- Some proposed model-grounding language can become too complex for participants. Phrases such as "on model-supported design features," "more execution-favorable," and "does not materially alter the scenario's resemblance" are directionally right but should be simplified for live use.

Preferred prompt wording direction:

- Prefer "The Completion Outlook is almost unchanged, with only a small increase" over "The Completion Outlook appears to improve" for tiny score movements.
- Prefer "the simpler structure may look easier to run in the score pattern" over "more execution-favorable on model-supported design features."
- Prefer "the score pattern is still close to the previous scenario" over "does not materially alter the scenario's resemblance."
- Prefer "may be less convincing for a registration-focused decision" over "less robust for a pivotal registration intent" when writing for participants.
- Prefer "important interpretability concern" over "significant interpretability gap."
- Keep clinical-development wording serious, but avoid unnecessary technical phrasing when a simpler sentence preserves the boundary.

Candidate improved Completion Outlook wording:

```text
The Completion Outlook is almost unchanged, with only a small increase. Moving back to a single-arm structure may look slightly easier in the score pattern, but this should not be read as proof that recruitment, retention, or completion would improve. The main discussion point is that a simpler design may reduce some execution burden while weakening how clearly the trial can support the intended development decision.
```

Candidate one-shot lesson:

- Use this scenario later as a first-visible or later-visible iteration example where Completion Outlook is nearly flat but Design Confidence drops.
- The one-shot should demonstrate that a small Completion Outlook gain must not be over-narrated as a meaningful improvement.
- The one-shot should show Design Confidence challenging score-seeking simplification.
- The one-shot should use simpler participant language while preserving risk-pattern and anti-causality boundaries.

### Live Test Iteration 3: Narrower Refractory / Advanced Population

Starting point:

- Previous scenario was single-arm, open-label, weaker comparator/control, surrogate endpoint evidence.
- Previous scores: Completion Outlook `47.2`, Design Confidence `-3.5`, Total Scenario Score `43.7`.

Edited scenario:

- `rare_condition_status_ml`: changed from unlikely to yes / likely.
- `patient_severity_ml`: changed from chronic progressive to advanced / metastatic.
- `line_of_therapy_ml`: changed from later-line to refractory / relapsed.
- Single-arm, open-label, surrogate-based structure was kept.
- `primary_duration_months_ml` / maximum primary endpoint duration remained `12.0`.
- Study summary text added that the target population would be narrowed toward more advanced or treatment-resistant disease, potentially increasing clinical relevance while making evidence more dependent on patient selection.

Observed scores:

- Completion Outlook decreased from `47.2` to `45.5`.
- Design Confidence increased from `-3.5` to `-1.5`.
- Total Scenario Score increased slightly from `43.7` to `44.0`.

What worked:

- The score pattern was useful: narrower / more severe population lowered Completion Outlook, while Design Confidence improved because target-population alignment became more defensible.
- Design Confidence correctly recognized better alignment with refractory / relapsed patients.
- The central tension was directionally right: patient relevance and operational simplicity versus weak comparative rigor for registration intent.

Issues to correct in the prompt:

- Completion Outlook again introduced unsupported operational assumptions: "may facilitate recruitment and site execution." This should not be stated unless recruitment/site execution fields changed and are in the supported evidence context.
- Completion Outlook again used broad historical generalization too strongly: "Historically, such designs face challenges..." This should be made conditional and grounded in the current evidence fields.
- Completion Outlook mixed design-quality critique into the Completion Outlook paragraph. Lack of control group, surrogate endpoints, masking, and interpretability belong mainly in Design Confidence unless directly tied to model-facing Completion Outlook evidence.
- Design Confidence duplicated itself: it repeated the "significant interpretability gap" and "team must defend..." idea twice.
- Design Confidence remained too categorical. "Well-aligned," "significant interpretability gap," and "required for pivotal registration" should be softened when not directly established by the packet.
- The participant language still occasionally overcomplicates or overstates. It should remain serious but shorter, less repetitive, and more conditional.
- Design Confidence treemap subcategory tiles still lacked visible short rationale, so participants can see ratings but not the rationale behind the rating.
- The two participant questions were too similar to prior iterations. The prompt should assume participants already discussed the previous questions and should generate new, iteration-specific questions after each reviewed change.
- Questions should route naturally from the latest value changes and the dilemma they create. They can be general and high-value, but they should connect to the current trial context or to a ClinOps / development challenge raised by the latest scenario.

Preferred prompt wording direction:

- Prefer "The Completion Outlook moves down, which is consistent with a narrower and more severe population profile appearing riskier in the score pattern" over operational claims about recruitment or site execution.
- Prefer "The single-arm, surrogate-endpoint structure may still make the evidence less convincing for a registration-focused decision" over "lacks the comparative rigor required for pivotal registration."
- Prefer "appears better aligned with the intended refractory / relapsed population" over "is well-aligned."
- Avoid repeating the same concern twice in Design Confidence; compress to one trade-off sentence.
- Keep Completion Outlook focused on score movement and changed model-facing fields; keep interpretability and evidence adequacy mainly in Design Confidence.
- Generate fresh key questions each iteration. Avoid reusing the same evidence-standard or operations-balance question unless the latest change truly makes the same question newly relevant.
- The medical/development question should challenge the current design dilemma created by the latest change. The ClinOps question should raise a high-value execution, governance, data reliability, bias, burden, or proportionality issue linked to the current scenario.

Candidate improved Completion Outlook wording:

```text
The Completion Outlook decreases from the prior scenario, which suggests that the narrower refractory / advanced population profile looks riskier in the score pattern. This should not be read as proof that recruitment or completion would be worse. The main point is that the scenario may now be more clinically focused, but the completion-risk signal has moved in the less favorable direction.
```

Candidate improved Design Confidence wording:

```text
Design Confidence improves modestly because the population appears more closely aligned with a refractory / relapsed PV decision context. However, the single-arm, surrogate-endpoint structure may still be less convincing for a registration-focused decision, so the stronger patient relevance does not fully offset the evidence-interpretability concern.
```

Candidate one-shot lesson:

- Use this scenario later as an example where Completion Outlook worsens while Design Confidence improves modestly.
- The one-shot should demonstrate separation between patient-relevance value and completion-risk pattern.
- The one-shot should avoid unsupported claims about recruitment, site execution, and speed.
- The one-shot should show concise, non-duplicative Design Confidence reasoning.
- The one-shot should show that the two participant questions evolve across iterations and do not repeat the same wording or same discussion target.

### Live Test Iteration 4: Free-Text / Structured-Field Contradiction

Starting point:

- Previous scenario was single-arm, open-label, refractory / relapsed, advanced / metastatic, rare-condition status yes, surrogate-based evidence.
- Previous scores: Completion Outlook `45.5`, Design Confidence `-1.5`, Total Scenario Score `44.0`.

Edited scenario:

- `pathway_profile_ml`: changed from enzyme modulator to GPCR target.
- Structured treatment fields remained aligned with a small-molecule oral intervention:
  - `therapeutic_modality_ml`: small molecule.
  - `delivery_profile_ml`: simple oral pill/tablet.
- Intervention free text added a deliberately contradictory description:
  - cell-based immunotherapy.
  - individualized manufacturing.
  - infusion-site coordination.

Observed scores:

- Completion Outlook decreased slightly from `45.5` to `45.2`.
- Design Confidence decreased from `-1.5` to `-7.5`.
- Total Scenario Score decreased from `44.0` to `37.7`.

What worked:

- Design Confidence appropriately became much more negative when the scenario contained incoherent intervention information.
- The clinical operations question picked up the manufacturing-coordination issue, which is relevant if the free text were true.
- Total Scenario Score appropriately reflected the negative Design Confidence pressure.

Issues to correct in the prompt:

- The review did not appear to show the required `scenario_consistency_note` despite a clear contradiction between structured fields and free text.
- The review treated the free-text cell-therapy/manufacturing description as if it were true scenario evidence, even though selected structured fields still said small molecule and simple oral delivery.
- The prompt must more strongly state that selected structured/categorical fields prevail when they conflict with free text.
- Contradictory free text may be used as a coherence concern in Design Confidence, but it must not replace the structured fields as the analysis source of truth.
- Completion Outlook must not use contradictory free-text manufacturing, site coordination, infusion burden, or retention burden as model-facing drivers when the selected model fields remain small molecule / simple oral.
- The narrative introduced an apparent "female-only population" restriction. This was not part of the intended Iteration 4 change and should not be invented unless supported by changed structured fields or text evidence.
- The two participant questions again repeated earlier evidence-standard and assessment-bias patterns too closely.
- Design Confidence again used categorical language such as "typically required for registration-enabling studies" and "can provide the necessary evidence." This should be softened.

Preferred prompt wording direction:

- If text and structured fields conflict, start with a short consistency note such as:

```text
Some scenario details are not fully aligned across selected fields and free-text fields. The selected fields drive the analysis, while the text is used as supporting context (Therapeutic Modality, Intervention text).
```

- In Completion Outlook, prefer:

```text
The Completion Outlook changes only slightly. Because the selected intervention fields still describe a small-molecule oral treatment, the contradictory cell-therapy wording should not be treated as a model driver. The main risk-pattern signal is therefore limited to the changed structured fields, while the text mismatch mainly raises a design-coherence question.
```

- In Design Confidence, prefer:

```text
Design Confidence falls because the intervention description is internally inconsistent: selected fields describe an oral small molecule, while the intervention text describes cell-based manufacturing and infusion logistics. The selected fields should drive the score interpretation, but the mismatch makes the scenario harder to defend unless clarified.
```

Candidate one-shot lesson:

- Use this scenario as the primary one-shot example for structured/free-text contradiction handling.
- The one-shot should demonstrate a visible `scenario_consistency_note`.
- The one-shot should show that structured fields prevail in Completion Outlook.
- The one-shot should allow Design Confidence to penalize coherence, but only as a text/field mismatch concern, not as if the cell-therapy text had replaced selected values.
- The one-shot should generate new participant questions that focus on clarification and governance of inconsistent scenario evidence rather than repeating the earlier evidence-standard question.

## UI Review Notes From Live Testing

These notes should be implemented after the narrative-quality review is complete.

### Pending Review Visibility

When the participant changes any feature after the latest reviewed scenario:

- Completion Outlook should keep showing the latest successful Completion Score, but display `Score update pending`.
- Design Confidence should keep showing the previous reviewed Design Confidence, but display a pending-review notice until `Review Scenario` is clicked.
- Total Scenario Score should keep showing the previous reviewed Total Scenario Score, but display a pending-review notice until `Review Scenario` is clicked.
- Design Confidence and Total Scenario Score should not disappear merely because a feature value changed. Hiding them is valid only for hidden baseline generation or when no participant-visible review exists yet.
- If the user realigns the changed value back to the reviewed value, the pending state should clear and the previous reviewed views should restore.

### Previous-Value And Delta Display Policy

Baseline:

- Completion Outlook may show the baseline score and drivers.
- Design Confidence should not show a previous value, delta card, or participant-visible score.
- Total Scenario Score should not show a previous value, delta card, or participant-visible score.

First visible iteration:

- Completion Outlook delta card compares current Completion Outlook against baseline Completion Outlook.
- Design Confidence has no previous-value delta card, because no participant-visible Design Confidence baseline existed.
- Total Scenario Score delta card uses baseline Completion Outlook as the previous value, then compares current Total Scenario Score against that baseline Completion Outlook. The point and percent variance should be based on those two values.

Second and later visible iterations:

- Completion Outlook delta card compares current Completion Outlook against previous visible Completion Outlook.
- Design Confidence delta card compares current Design Confidence against previous visible Design Confidence.
- Total Scenario Score delta card compares current Total Scenario Score against previous visible Total Scenario Score.

Bar chart delta policy:

- Completion Outlook bar chart keeps per-pillar `+/- pts` variance from the previous Completion Outlook snapshot, including first iteration versus baseline.
- Design Confidence bar chart should not show previous-point variance on the first visible iteration.
- First visible Total Scenario Score bar chart compares the current combined pillar values against baseline Completion Outlook pillar values.
- From the second visible iteration onward, Total Scenario Score bar chart variance compares current combined pillar values against previous visible Total Scenario Score pillar values.
- From the second visible iteration onward, Design Confidence bar chart variance compares current Design Confidence pillar contributions against previous visible Design Confidence pillar contributions.

## Regression Acceptance Criteria

Prompt and schema changes should be tested against concrete pass/fail criteria, not only subjective quality.

Completion Outlook acceptance checks:

- Pass if Completion Outlook does not cite planned enrollment or planned site count as score drivers. Verification: automated evidence-field check plus manual wording review.
- Pass if Completion Outlook does not cite planned total duration / operational duration assumption as a score driver. Verification: automated evidence-field check plus manual wording review.
- Pass if Completion Outlook may cite `primary_duration_months_ml` / maximum primary endpoint duration when it is present as model-facing XGBoost evidence. Verification: automated packet/field check.
- Pass if Completion Outlook describes lower/higher risk of early termination or resemblance to completed/terminated historical patterns, rather than promising chance of completion. Verification: manual or LLM-review.
- Pass if Completion Outlook does not claim that any field caused completion or caused early termination risk. Verification: manual or LLM-review.
- Pass if interpretive hypotheses include model signal, possible pattern, context modifiers, and boundary when used. Verification: automated schema check plus manual quality review.

Design Confidence acceptance checks:

- Pass if first visible iteration does not say "Design Confidence improved versus baseline" or equivalent hidden-baseline comparison language. Verification: manual or LLM-review.
- Pass if every non-neutral Design Confidence rating cites supported evidence fields. Verification: automated validation.
- Pass if positive Design Confidence is backed by explicit design-quality evidence, not merely model-favorable simplification. Verification: manual or LLM-review.
- Pass if Design Confidence can challenge a high Completion Outlook when evidence value, patient relevance, governance, or proportionality is weakened. Verification: golden examples plus manual review.
- Pass if Design Confidence can be positive despite lower Completion Outlook when added difficulty reflects justified rigor, relevance, or governance. Verification: golden examples plus manual review.
- Pass if the Design Confidence narrative avoids duplicating the same concern in the same participant-facing block. Verification: manual or LLM-review.

Key-question acceptance checks:

- Pass if the two questions differ materially from the previous visible iteration unless the latest change genuinely reopens the same dilemma. Verification: golden examples plus manual review.
- Pass if each question is linked to the latest value changes, current narrative tension, or a high-value clinical-development / ClinOps dilemma relevant to the trial. Verification: manual or LLM-review.
- Pass if questions remain open-ended and not answerable with yes/no. Verification: automated pattern check plus manual review.

Therapeutic-area context acceptance checks:

- Pass if a matching TA `.md` file exists and appears in the packet/trace. Verification: automated packet check.
- Pass if a missing TA `.md` file does not break packet construction or review generation. Verification: automated packet/review check.
- Pass if missing TA context leads to cautious general therapeutic-area wording. Verification: manual or LLM-review.
- Pass if unsupported specific disease facts, treatment standards, prevalence, efficacy, safety, guideline, or regulatory claims are not introduced. Verification: manual or LLM-review.

Format and UI-readiness acceptance checks:

- Pass if output fits the target participant length range or a documented revised range. Verification: automated word/character count plus manual readability review.
- Pass if Design Confidence treemap short rationales are present or safely derivable from validated subcategory rationale. Verification: automated schema check.
- Pass if evidence fields used for rationale remain available for tooltip, expander, or facilitator/debug view. Verification: automated trace check.
- Pass if `scenario_consistency_note` appears only when a clear text/structured-field mismatch remains. Verification: golden examples plus manual review.
- Pass if the consistency note uses participant-readable field labels in parentheses. Verification: automated schema/string check plus manual review.
- Pass if text-change evidence is available when material free-text fields changed. Verification: automated packet check.

Lifecycle acceptance checks:

- Pass if hidden baseline stores qualitative context while suppressing participant-visible Design Confidence and Total Scenario Score. Verification: automated trace check.
- Pass if first visible iteration can compare Completion Outlook to visible original Completion Score. Verification: manual or LLM-review.
- Pass if first visible iteration evaluates Design Confidence as current-scenario design defensibility, not hidden-score variance. Verification: manual or LLM-review.
- Pass if later visible iterations use prior visible review for continuity without overusing Design Confidence score-to-score storytelling. Verification: manual or LLM-review.
- Pass if pending feature changes preserve previous visible Design Confidence and Total Scenario Score while clearly marking those views as pending review. Verification: focused UI smoke.
- Pass if first visible Total Scenario Score delta uses baseline Completion Outlook as the previous value, then later iterations use previous visible Total Scenario Score. Verification: automated snapshot/delta test plus UI smoke.

## Implementation Sequence

1. Update narrative architecture docs.
   - Add the three-part participant output model.
   - Add the canonical provider-facing schema.
   - Add Completion Outlook boundary rules.
   - Add Completion Outlook interpretation-depth rules and anti-causality redlines.
   - Add operational-assumption exclusion from Completion Outlook.
   - Add the endpoint-duration versus planned-total-duration distinction.
   - Add Design Confidence counterweight rules.
   - Add Design Confidence calibration safeguards.
   - Add hidden baseline / first visible / later visible prompt-mode lifecycle.
   - Add optional TA pack policy.
   - Add regression acceptance criteria.

2. Update the implementation plan.
   - Add prompt redesign as the next prompt-engineering phase.
   - Add optional TA knowledge substrate work as a separate phase.

3. Update packet builder.
   - Add optional therapeutic-area context lookup.
   - Include `therapeutic_area_context` with `pack_found`.
   - Record canonical therapeutic-area value and safe expected filename.
   - Add `text_change_evidence` for material free-text edits.
   - Preserve missing-pack trace without failing.

4. Update prompt builder and response schema.
   - Reshape participant-facing output into three parts.
   - Add prompt modes: `hidden_baseline`, `first_visible_iteration`, and `later_visible_iteration`.
   - Add `scenario_consistency_note`.
   - Keep structured subcategory ratings for scoring.
   - Strengthen Completion Outlook and operational-field boundaries.
   - Require interpretive hypotheses to include model signal, possible pattern, context modifiers, and boundary when used.
   - Add cross-pillar evidence rule.
   - Require lower/higher early-termination risk wording instead of chance-of-completion claims.

5. Update validation/scoring only if needed.
   - Preserve app-owned scoring.
   - Ensure new narrative fields are required/validated.
   - Preserve evidence-field validation.

6. Update review storage.
   - Store the three participant-facing blocks.
   - Store condensed treemap rationales if added.
   - Preserve prior trace compatibility where needed.

7. Update UI.
   - Add narrative box under Completion Outlook score/plot.
   - Add narrative box under Design Confidence score/plot.
   - Add two key questions below the score narratives.
   - Keep diagnostics behind expanders.

8. Run regression.
   - Use the existing fixture suite.
   - Regenerate prompt/context exports only if a human-readable comparison artifact is useful.
   - Apply the acceptance criteria above.
   - Run named live cases:
     - `NCT03386721`
     - `NCT03896581`
   - Compare before/after narrative quality.

9. Tune provider settings only after content architecture is stable.
   - Compare prompt quality, latency, token use, malformed JSON rate, validation failures, and reproducibility.

## Open Questions

- Should participant narratives always show fixed section labels, or read as a continuous narrative divided visually by score area?
- Should `central_tension` be participant-visible, facilitator-only, or trace-only?
- Should therapeutic-area packs be selected only by `therapeutic_area_ml`, or also by condition/GBD category when available?
- How much disease-specific reasoning should be allowed without an explicit TA or condition pack?
- Should treemap rationales be generated by the LLM, derived from validated subcategory rationale, or both?
- Should the default reading-time target be 45-75 seconds or 75-120 seconds?

## Immediate Next Step

Before editing prompt code, review this plan together with:

- `docs/narrative_prompt_engineering_brief.md`
- `docs/architecture_narratives.md`
- `implementation_plan.md`

Then perform Step 0 baseline capture, promote any remaining durable decisions into `docs/architecture_narratives.md`, and start the staged packet/prompt/schema migration. The deleted educational Word exports should not be treated as implementation dependencies; regenerate them only if they are useful for human review.

## Current-To-Target Comparison Before Implementation

Before implementing this plan in code, compare the current prompt system against the target design so useful current behavior is preserved.

Current prompt strengths to preserve:

- Explicit senior clinical-development / medical-strategy reviewer role.
- Strict JSON-only provider response.
- App-owned Design Confidence / Total Scenario Score calculations.
- Evidence-first Design Confidence sequence: evidence fields, rationale, then rating.
- Provider is forbidden from returning app-owned score fields.
- Reference packs are secondary to scenario packet evidence.
- User-editable trial text is treated as context, not instruction.
- Hidden baseline context exists and is not participant-visible by default.
- Provider traces keep model/settings/validation metadata without secrets.
- Fallback/retry behavior is bounded.
- Output style requires conditional language and avoids direct optimization instructions.

Target enhancements to add:

- Three participant-facing blocks: Completion Outlook Analysis, Design Confidence Analysis, and Two Key Questions.
- Top-level `review_metadata` with `review_mode` and `participant_visible`.
- Three prompt modes: `hidden_baseline`, `first_visible_iteration`, and `later_visible_iteration`.
- Completion Outlook framed as early-termination risk-pattern interpretation.
- Completion Outlook interpretive hypotheses with model signal, possible pattern, context modifiers, and boundary.
- Planned enrollment, planned site count, planned total duration, and operational benchmark metadata excluded from Completion Outlook.
- `primary_duration_months_ml` / maximum primary endpoint duration explicitly allowed as XGBoost Completion Outlook evidence.
- Design Confidence as a quality-of-design adjustment and challenge layer, not a second completion predictor.
- Design Confidence subcategory objects with rating, evidence fields, rationale, short rationale, optional lenses, and optional regulatory/finance note.
- Optional TA `.md` lookup by XGBoost canonical therapeutic-area value.
- `scenario_consistency_note` for clear free-text / selected-field mismatch.
- `text_change_evidence` for material free-text edits.
- Treemap short rationales for Design Confidence subcategories.
- Regression acceptance criteria with automated/manual/LLM-review classification.

Things not to lose during implementation:

- Do not weaken XGBoost / SHAP / calibration separation.
- Do not let the LLM calculate Design Confidence points or Total Scenario Score.
- Do not expose hidden baseline Design Confidence as a participant-visible baseline score.
- Do not turn the review into a recommendation engine.
- Do not make operational assumptions part of Completion Outlook.
- Do not remove supported-evidence validation for non-neutral Design Confidence.
- Do not add TA packs in a way that can fail review generation when missing.
- Do not use broad therapeutic knowledge to invent unsupported specific disease, regulatory, efficacy, safety, prevalence, or cost claims.

## Implementation Readiness Checklist

The plan is ready to move into code when these decisions are explicitly accepted:

- Canonical response schema is final enough for fixture migration.
- `review_metadata` replaces repeated per-block mode metadata.
- Hidden baseline policy is final: qualitative subcategory validation is allowed, numeric participant score suppressed.
- TA file naming uses XGBoost canonical therapeutic-area values and safe filename conversion.
- `text_change_evidence.change_type` v1 remains light-touch and does not require a complex semantic classifier.
- Treemap rationales apply to Design Confidence subcategories only.
- Acceptance criteria are sufficient for first regression pass.

## Suggested Implementation Plan

Implementation should be incremental. Do not attempt packet, prompt, scoring, storage, and UI changes in one pass. Each step should preserve the ability to run the existing narrative checker suite or an updated equivalent.

### Step 0: Baseline Capture And Migration Guardrails

Before editing narrative behavior, capture the current behavior so the migration can be compared against it.

Actions:

- Export the current prompt packet with `scripts/export_narrative_prompt_brief.py`.
- Save a copy of the current exported prompt/context under ignored `data/understanding_narratives/` for local comparison.
- Run the current narrative checker suite to establish a clean baseline.
- Identify any current aliases or old schema fields that the UI still depends on.

Verification:

- `python scripts/export_narrative_prompt_brief.py --fixture operational_only_ambitious_enrollment_v2 --out /tmp/narrative_prompt_export`
- `python scripts/check_narrative_contract_fixtures.py`
- `python scripts/check_narrative_packet_builder.py`
- `python scripts/check_narrative_scoring.py`
- `python scripts/check_narrative_mock_reviewer.py`
- `python scripts/check_narrative_review_store.py`
- `python scripts/check_narrative_provider_config.py`
- `python scripts/check_narrative_prompt_builder.py`
- `python scripts/check_narrative_provider.py`
- `python scripts/check_narrative_live_snapshot_flow.py`

Guardrails:

- No XGBoost, SHAP, therapeutic-area calibration, preprocessing, taxonomy encoding, or `/predict` scoring behavior should change in this migration.
- If any scoring/prep/model artifact path is touched unexpectedly, stop and run the parity gates before proceeding.

### Step 1: Documentation Alignment

Update durable architecture documents before code:

- Move accepted parts of this plan into `docs/architecture_narratives.md`.
- Update `implementation_plan.md` with the concrete implementation phase and verification gates.
- Keep `prompt_enhancement_plan.md` as the detailed working plan and comparison artifact.
- Add an explicit migration note that the implementation changes provider contract shape, not XGBoost scoring.

Verification:

- `git diff --check` for changed docs.

### Step 2: Golden Examples And Contract Fixtures

Purpose:

- Golden examples define qualitative expectations.
- Contract fixtures define executable schema/scoring expectations.

Actions:

- Create or update 3-5 external golden examples before deciding whether to embed any examples in the live prompt.
- Update `src/narratives/contract_fixtures.py` after the golden examples clarify expected behavior.

Update `src/narratives/contract_fixtures.py` to reflect:

- `review_metadata`.
- Three-part participant narrative.
- Structured `design_confidence_subcategories`.
- `scenario_consistency_note` cases.
- `text_change_evidence` cases.
- Hidden baseline qualitative-only policy.
- Acceptance examples for early-termination risk wording and no hidden-baseline Design Confidence comparison.
- A positive Completion Outlook / weaker Design Confidence case.
- A negative Completion Outlook / stronger Design Confidence case.
- A TA pack found case and missing TA pack case if fixture data can cover both cheaply.

Verification:

- `python scripts/check_narrative_contract_fixtures.py`
- `python -m py_compile src/narratives/contract_fixtures.py scripts/check_narrative_contract_fixtures.py`

### Step 3: Packet Builder And Evidence Assembly

Update `src/narratives/packet_builder.py` to add:

- optional `therapeutic_area_context`
- safe TA filename lookup
- expected filename and pack-found trace
- `text_change_evidence`
- enough metadata for prompt mode inference
- explicit duration provenance so `primary_duration_months_ml` and planned total duration cannot be confused
- text/structured field labels needed by `scenario_consistency_note`
- reference-pack IDs available for trace validation

Verification:

- `python scripts/check_narrative_packet_builder.py`
- `python scripts/check_narrative_live_snapshot_flow.py`
- targeted export with `scripts/export_narrative_prompt_brief.py`
- inspect `/tmp/narrative_prompt_export/02_packet.json`

### Step 4: Prompt Contract And Schema

Update `src/narratives/prompt_builder.py` to:

- emit the new response contract
- support `hidden_baseline`, `first_visible_iteration`, and `later_visible_iteration`
- add Completion Outlook early-termination risk framing
- add Design Confidence challenge-layer rules
- add consistency-note and text-change instructions
- add optional TA context instructions
- preserve JSON-only, evidence-first, conditional-language, and forbidden-score rules
- preserve prompt-injection protection for user-editable text
- update Gemini response schema to match the new contract
- update prompt export docs so the educational Word exports reflect the new shape

Verification:

- `python scripts/check_narrative_prompt_builder.py`
- `python scripts/check_narrative_reference_packs.py`
- inspect regenerated `data/understanding_narratives` exports

### Step 5: Mock Reviewer And Provider Normalization

Update provider-facing test doubles before changing UI:

- Update `src/narratives/mock_reviewer.py` to emit the new schema.
- Update `src/narratives/provider.py` only as needed for new schema validation, retry, and normalized metadata.
- Preserve provider failure behavior: no stale Design Confidence for a new packet.
- Preserve no fallback-provider shopping for malformed clinical JSON unless policy changes later.

Verification:

- `python scripts/check_narrative_mock_reviewer.py`
- `python scripts/check_narrative_provider.py`
- `python -m py_compile src/narratives/mock_reviewer.py src/narratives/provider.py`

### Step 6: Validation, Scoring, And Storage

Update only what is necessary in:

- `src/narratives/scoring.py`
- `src/narratives/review_store.py`

Preserve:

- app-owned scoring
- supported-evidence gates
- no partial Design Confidence when required fields are malformed
- hidden baseline qualitative-only numeric suppression
- storage of `review_metadata`, three narrative blocks, `scenario_consistency_note`, `text_change_evidence`, TA context, and short rationales
- compatibility aliases only where needed to keep the current simulator panel working during migration

Verification:

- `python scripts/check_narrative_scoring.py`
- `python scripts/check_narrative_review_store.py`
- `python scripts/check_narrative_mock_reviewer.py`
- `python scripts/check_narrative_provider.py`

### Step 7: Prompt Export And Human Review Checkpoint

Before touching the simulator UI, regenerate the educational exports and inspect them.

Actions:

- Run the prompt export helper for at least:
  - `operational_only_ambitious_enrollment_v2`
  - a text contradiction fixture
  - a score-up/design-down fixture
  - a score-down/design-up fixture
- Regenerate Word exports under ignored `data/understanding_narratives/` if the export format is still useful.
- Confirm the prompt remains understandable and does not become overstuffed.

Verification:

- exported prompt includes the new schema and mode rules
- packet includes TA context and text-change evidence when relevant
- response contract still forbids provider-owned score fields
- Completion Outlook instructions exclude planned enrollment/sites/total duration

### Step 8: UI Integration

Update `frontend/views/trial_simulator.py` only after the provider/mock path is stable:

- Completion Outlook narrative box under Completion Outlook score/plot.
- Design Confidence narrative box under Design Confidence score/plot.
- Two Key Questions area.
- Optional consistency note.
- Design Confidence treemap short rationale display or tooltip/expander.
- Diagnostics remain behind expanders.
- Keep score-view radio behavior view-only; it must not create pending state or trigger provider calls.
- Keep hidden baseline participant-hidden by default.

Verification:

- `python -m py_compile frontend/views/trial_simulator.py`
- focused Streamlit smoke
- browser check at the current target breakpoints

### Step 9: Regression And Live Prompt Review

Run automated checks first:

- all `scripts/check_narrative_*.py`
- `python -m py_compile` for changed modules
- `git diff --check`

Then run manual/golden-example review:

- no operational assumptions in Completion Outlook
- no planned total duration as Completion Outlook driver
- no hidden-baseline Design Confidence comparison in first visible iteration
- no causal field claims
- supported evidence for non-neutral Design Confidence
- TA pack found/missing behavior
- conditional regulatory/finance wording
- text/structured mismatch note behavior
- short rationale clarity in Design Confidence treemap/subcategory display
- first visible iteration does not compare Design Confidence to hidden baseline
- later visible iteration uses continuity without inventing unsupported reversals

Then run named live cases:

- `NCT03386721`
- `NCT03896581`

Add one browser/UI pass after live prompt review:

- Simulation Mode opening stays usable.
- Review Scenario button behavior remains stable.
- Score views do not overlap narrative boxes at target breakpoints.
- Diagnostics remain accessible but not participant-primary.

### Step 10: Provider Settings Review

Tune provider settings only after content architecture is stable:

- model choice
- thinking/reasoning level
- output token ceiling
- timeout
- retry policy
- reproducibility versus quality
- cost and latency

Do not use provider settings to compensate for unclear prompt or packet design. If outputs are weak because evidence is missing or instructions conflict, fix packet/prompt content first.
