# Prompt Enhancement Plan

## Scope

Architecture scope: `architecture_narratives`

Purpose: define the next prompt-engineering roadmap for the Scenario Review / Design Confidence layer before changing prompt behavior, provider settings, scoring, or UI rendering.

This plan complements:

- `docs/architecture_narratives.md` for durable narrative architecture decisions.
- `docs/narrative_prompt_engineering_brief.md` for current prompt anatomy and LLM flow understanding.
- `implementation_plan.md` for implementation phase tracking.

## Product Direction

The current narrative system should be reshaped around four concise participant-facing parts:

1. `Completion Outlook Analysis`
   - What the XGBoost risk-pattern model says changed.
   - Why the revised profile appears more or less similar to historically completed versus early-terminated trials.
   - What Completion Outlook score inputs, model_interpretation evidence, pillar/subcategory movements, and changed structured_features support that interpretation.

2. `Design Confidence Analysis`
   - Whether the resulting design is strategically, clinically, scientifically, and operationally defensible.
   - Whether the Completion Outlook movement appears supported by stronger design logic or challenged by weaker evidence value, reduced relevance, shortcut risk, or disproportionate burden.
   - The rationale behind Design Confidence subcategory ratings.

3. `Main Tension`
   - One sentence stating the central cross-functional tension raised by the current scenario.

4. `Key Questions`
   - One medical/clinical-development question grounded in the trial and latest material scenario change.
   - One strategic development question about the broader development path or field-level implication.
   - Questions should support general debate without prescribing the next edit or addressing participants directly.

The participant narrative should become easier to read and easier to place visually under the score areas in `frontend/views/trial_simulator.py`.

## Target Participant Output Structure

Canonical provider-facing structure:

```json
{
  "review_metadata": {
    "review_mode": "first_visible_iteration",
    "visible": true
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
      "evidence_fields": ["..."],
      "rationale": "...",
      "rating": "...",
      "score_materiality": "...",
      "short_rationale": "...",
      "optional_lenses_used": [],
      "regulatory_or_finance_note": ""
    },
    "endpoint_evidence_strength": {
      "evidence_fields": ["..."],
      "rationale": "...",
      "rating": "...",
      "score_materiality": "...",
      "short_rationale": "...",
      "optional_lenses_used": [],
      "regulatory_or_finance_note": ""
    },
    "target_population_alignment": {
      "evidence_fields": ["..."],
      "rationale": "...",
      "rating": "...",
      "score_materiality": "...",
      "short_rationale": "...",
      "optional_lenses_used": [],
      "regulatory_or_finance_note": ""
    },
    "operational_burden_balance": {
      "evidence_fields": ["..."],
      "rationale": "...",
      "rating": "...",
      "score_materiality": "...",
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
  "main_tension": "...",
  "key_questions": {
    "medical_clinical_development_question": "...",
    "strategic_development_question": "..."
  }
}
```

The structured Design Confidence subcategories should remain available for validation, scoring, trace, and visual decomposition. The participant narrative should expose only the concise participant-facing sections above rather than every internal category as equal narrative sections.

Implementation decision:

- Use one shared response schema with mode-specific constraints.
- Add top-level `review_metadata.review_mode` and `review_metadata.visible` so the same schema can support hidden baseline, first visible iteration, and later visible iteration behavior. Keep `participant_visible` only as a temporary legacy fallback during migration.
- Mode-specific rules should control required and forbidden wording rather than creating three unrelated schemas.

Suggested participant text length:

- Completion Outlook Analysis: 90-140 words.
- Design Confidence Analysis: 120-180 words.
- Main Tension: one sentence.
- Key Questions: 20-30 words each.
- Total participant narrative target: roughly 300-380 words.

## Completion Outlook Rules

Completion Outlook must be framed as a model-grounded, movement-aware risk-pattern interpretation.

Required framing:

- The Completion Score reflects how similar the revised trial profile appears to historical patterns associated with completed versus early-terminated trials.
- It is an early-termination risk-pattern assessment, not a prediction of clinical success.
- It is not a claim that the trial will complete.
- It is not a claim that the design is clinically better.
- It should describe Completion Outlook score-input patterns, correlations, and plausible historical-profile interpretations, not causal certainty.
- It may try to make sense of why the revised profile appears riskier or less risky, but only as an evidence-bounded interpretation grounded in score movement, `model_interpretation` evidence, changed `structured_features`, and surrounding context.
- Participant-facing wording should prefer "lower/higher risk of early termination" or "more/less similar to completed-trial patterns" over "higher/lower chance of completion."

Allowed evidence for Completion Outlook:

- Completion Outlook score inputs in `structured_features`.
- Changed `structured_features` that feed the Completion Outlook score.
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

- `primary_duration_months_ml` / maximum primary endpoint duration is a Completion Outlook score input and may be used in Completion Outlook when present in `model_interpretation` evidence.
- Planned total duration / operational duration assumption is outside XGBoost and must not be cited as a Completion Outlook driver.

## Completion Outlook Interpretation Depth

Completion Outlook should not be a shallow restatement of score movement. It should help participants understand why the model may see a profile as more or less completion-like. However, this interpretation must remain bounded.

Required interpretation pattern:

1. Name the model signal.
   - Which score, pillar, subcategory, impact movement, or Completion Outlook score-input movement is relevant?

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
  "visible": true,
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

This applies to every Completion Outlook score input, not only obviously ambiguous fields. Any score input may behave as a marker of historical trial profile, complexity, governance, evidence ambition, patient risk, operational profile, or therapeutic-area norms. The LLM should not simplify a field into a one-direction causal story.

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
Completion Outlook explanations must be framed as historical-pattern hypotheses grounded in `model_interpretation` evidence and packet fields. They must not present field effects as clinical causality, operational truth, or design recommendations.
```

Participant-facing narratives should state unresolved concerns, not prescribe exact redesign paths. For example, prefer saying that a non-randomized, open-label structure remains an interpretability concern over saying the design would need to transition to randomized or blinded conduct.

## Design Confidence Rules

Design Confidence should act as a counterweight and interpretation layer, not as a multiplier of Completion Outlook.

Design Confidence remains an application-calculated score adjustment for quality of trial design. Its narrative role is to challenge the scenario logic and moderate clear Completion Outlook increases or decreases when design-quality evidence supports doing so. It is not a second completion predictor.

Scenario-state rule:

- Scenario edits are cumulative: each review evaluates the current full scenario after all retained participant edits.
- Design Confidence is not a running accumulated penalty or bonus. It is recalculated from the current packet evidence on every reviewed scenario.
- Prior visible reviews provide continuity, deltas, and storyline memory only. They should help identify unresolved concerns, newly introduced concerns, and concerns resolved by current fields.
- If a prior weakness is fixed by current structured fields or text clarification, Design Confidence should stop penalizing that old weakness, while the narrative may briefly acknowledge that the concern was resolved.
- Design Confidence delta should compare the current recalculated Design Confidence against the prior visible recalculated Design Confidence; it should not add the prior Design Confidence points to the new score.

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

### Implemented Wider Subcategory Scale

Implemented scoring-contract target: each Design Confidence subcategory now spans `-5.0..+5.0` in `0.5` increments.

This is implemented as an app-owned scoring migration, not by asking the LLM to return numeric points. The LLM builds the critical narrative lens from the current packet: Completion Outlook score and delta, `model_interpretation` score evidence, `iteration_context.field_changes`, `operational_assumptions` planning assumptions, `text_context` Trial description fields, consistency notes, therapeutic-area reference context, hidden-baseline qualitative memory, and prior visible review context. For each subcategory it selects evidence fields, explains the current design judgment, assigns a qualitative rating, and assigns a qualitative `score_materiality` level. The app then maps `rating + score_materiality + context guardrails` into points.

Provider-owned qualitative fields:

```json
{
  "rating": "supportive",
  "score_materiality": "minimal | low | moderate | high | very_high",
  "evidence_fields": ["..."],
  "rationale": "...",
  "short_rationale": "..."
}
```

Provider-owned numeric point fields remain forbidden.

Implemented base mapping:

| Rating | Minimal | Low | Moderate | High | Very High |
| --- | ---: | ---: | ---: | ---: | ---: |
| `strong` | `+3.0` | `+3.5` | `+4.0` | `+4.5` | `+5.0` |
| `supportive` | `+0.5` | `+1.0` | `+1.5` | `+2.0` | `+2.5` |
| `balanced` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| `weak` | `-0.5` | `-1.0` | `-1.5` | `-2.0` | `-2.5` |
| `conflicting` | `-3.0` | `-3.5` | `-4.0` | `-4.5` | `-5.0` |

Anti-inflation and context guardrails:

- Default `score_materiality` is `minimal` unless the rationale identifies a concrete reason for larger score movement.
- `high` positive `score_materiality` is rare. It requires supported evidence that the scenario materially improves decision usefulness, evidence interpretability, patient relevance, governance, or proportionality, not merely that the Completion Outlook is favorable.
- If the corresponding Completion Outlook pillar is already strongly positive, positive Design Confidence for that same pillar should normally be capped at `+0.5` or `+1.0` unless the scenario explicitly resolves a prior current-scenario weakness or adds new design-quality evidence not already captured by Completion Outlook.
- If Completion Outlook improves because the scenario is simpler, narrower, shorter, less governed, less blinded, less controlled, or operationally easier, positive Design Confidence should be blocked unless separate evidence shows the design became more defensible.
- If Completion Outlook declines because the scenario adds rigor, patient relevance, endpoint interpretability, governance, or justified ambition, positive Design Confidence may be larger, but only in the subcategory directly supported by that evidence.
- Operational assumptions can support `operational_burden_balance`, but operational burden alone should not produce a positive score. If enrollment, site count, planned duration, arms, oversight, or delivery burden increase without matching evidence or patient-relevance gain, `operational_burden_balance` should be neutral or negative.
- Operational simplification caused mainly by weaker comparator, masking, allocation, endpoint rigor, or evidence ambition should not receive strong positive `operational_burden_balance` unless the packet shows an independent gain in access, feasibility, oversight, data reliability, patient burden, or proportionality.
- Use packet-section names consistently in prompt wording. `Completion Outlook score` is `model_interpretation.completion_score`; `Completion Outlook score inputs` are selected categorical/numeric fields in `structured_features` that feed the score, identified through `model_interpretation.direct_xgboost_shap_fields` and `model_interpretation` score evidence, with readable labels in `structured_feature_display_values` and meanings in `structured_feature_meanings`; `Trial description fields` are `text_context` fields with meanings in `text_context_field_meanings`, UI labels `Title` (top study title), `Summary`, `Conditions`, `Interventions`, and `Primary Outcomes`, and JSON keys `title`, `summary_ui`, `conditions_ui`, `interventions_ui`, and `primary_outcomes_ui`; `Planning assumptions` are `operational_assumptions.planned_enrollment`, `operational_assumptions.planned_sites`, and `operational_assumptions.planned_duration_months`; `Completion Outlook narrative` is `completion_outlook_analysis`; `Design Confidence narrative` is `design_confidence_analysis`; and `Design Confidence subcategory ratings` are `design_confidence_subcategories`.
- Trial description fields do not directly feed the Completion Outlook score. They may support the Completion Outlook narrative only when they align with, clarify, or add non-conflicting detail to selected Completion Outlook score inputs. This conflict rule applies across all Trial description fields in `text_context` and all relevant `structured_features`, not only intervention descriptions. Completion Outlook score inputs define the score-interpreted scenario when they directly conflict with Trial description fields. Only the conflicting Trial description field detail should be treated as stale scenario text superseded by the structured_features value; it should not be used as Completion Outlook evidence or as evidence that the selected structured design has the contradicted modality, delivery burden, endpoint, or population feature. Structured_features/text_context conflict is a scenario-readiness warning and should not drive multiple strong negative subcategory ratings unless non-conflicting structured_features values independently support those penalties. In the participant warning, "text is used as supporting context" means aligned or non-conflicting Trial description field content.
- Hidden baseline memory and prior visible reviews should shape continuity and identify unresolved or resolved concerns. They must not act as accumulated numeric penalties or bonuses.
- Unsupported evidence fields force point effect to `0.0` even when the narrative is plausible.

The wider scale should make strong negative critique and strong positive design-quality improvement easier to express, but it should not make ordinary supportive categories drift upward. Most visible subcategory movements should still sit between `-2.5` and `+2.5`; `+4.0/+5.0` and `-4.0/-5.0` should be reserved for rare, well-supported design signals.

Implemented scoring-calibration refinement:

Design Confidence is the qualitative critical lens on Completion Outlook, not a second completion predictor and not a bonus multiplier. Completion Outlook mainly expresses predicted completion versus early-termination risk. That score can move because a scenario is simpler, riskier, higher quality, lower quality, more burdensome, or more conventional. Design Confidence should therefore add most value when it qualifies, challenges, or contextualizes Completion Outlook, not when it repeats the same signal.

This refinement supersedes the earlier stricter already-positive-pillar cap language. The governing model is matching-pillar same-direction double-counting control plus opposite-direction counterweight preservation.

Use only one compact prompt addition if needed:

> Preserve each Design Confidence subcategory's meaning. When a change improves one design dimension but worsens another, reflect both effects in their relevant subcategories. Cross-functional trade-offs may be justified in the overall Design Confidence judgment, but a subcategory should be positive only when that subcategory itself improved.

Implemented behavior:

- Keep the LLM contract unchanged: `rating`, `score_materiality`, evidence fields, rationale, and `short_rationale`; no provider-owned numeric Design Confidence points.
- Preserve `raw_points` from the current deterministic `rating + score_materiality` map.
- Add calibrated `points` and `calibration_notes` when the app caps or softens the raw score.
- Apply caps at the Design Confidence subcategory level only.
- Primary rule: avoid same-direction double-counting, triggered only by strong matching Completion Outlook pillar movement at `>= +3.0` or `<= -3.0` points. If the matching pillar moved strongly positive and is now still negative, cap strongly positive mapped Design Confidence at `+2.5`; if it moved strongly positive and is now neutral or positive, cap at `+1.5`. If the matching pillar moved strongly negative and is now still positive, soften strongly negative mapped Design Confidence to `-2.5`; if it moved strongly negative and is now neutral or negative, soften to `-1.5`.
- Preserve opposite-direction counterweight. If Completion Outlook improves while design confidence weakens, or Completion Outlook worsens while design quality improves, preserve stronger supported Design Confidence points.
- Let Design Confidence move when Completion Outlook is flat or barely changed, especially for Trial description, operational assumptions, scenario-readiness, or cross-pillar quality effects.
- Do not use total Completion Outlook score movement or previous-score thresholds as Design Confidence calibration triggers. Total score movement is an aggregate and can reflect other pillars; the total is where subcategory trade-offs reconcile, not the trigger for subcategory-level softening.
- Preserve subcategory meaning: positive impact belongs only in the subcategory that itself improved; the counter-impact belongs in the relevant other subcategory; total Design Confidence is where compensation happens.
- Operational assumptions and all other supported packet evidence follow the same Design Confidence scoring and calibration rules. Planning assumptions may improve or worsen Operational Burden Balance because they affect whether the scenario feels operationally proportionate and executable. They may also create counter-effects in other Design Confidence subcategories, such as endpoint maturity or evidence sufficiency, when supported by the rationale. The special rule for planning assumptions is only that they must not explain Completion Outlook movement; within Design Confidence they are handled like other supported current-scenario evidence. Preserve subcategory meaning through the LLM rationale plus global double-counting/saturation controls, not through field-family-specific numeric caps.
- Shortcut-driven ease from weaker evidence controls should receive at most bounded feasibility credit and should preserve evidence/phase critique.
- Negative critique should generally remain visible when supported; soften it only when it duplicates a strong same-direction negative Completion Outlook or pillar movement.
- The deterministic layer should mostly cap or soften excessive same-pillar, same-direction amplification. It should not create new positives or negatives and should not attempt deep clinical interpretation beyond matching-pillar movement and evidence-field alignment.

Participant-facing treemap display now shows signed subcategory points plus the `short_rationale`; `rating` and `score_materiality` labels are hidden from participant-facing treemap text while remaining internal for scoring, validation, and audit.

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
  "evidence_fields": ["endpoint_rigor_ml", "comparator_benchmark_ml"],
  "rationale": "...",
  "rating": "weak",
  "score_materiality": "moderate",
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

## Trial Description Fields And Structured-Feature Consistency

`structured_features` values should prevail when they directly conflict with Trial description fields in `text_context`. Trial description fields remain available as aligned context, rationale, or contradiction evidence, but they should not silently override the selected scenario fields.

Participant-facing consistency note:

```text
Some scenario details are not fully aligned across Trial description fields and structured fields. In this case the value in the structured fields drives the analysis, while the Trial description fields are used as supporting context. (Intervention text, Therapeutic Modality)
```

Use this note only when a clear mismatch remains. The fields in parentheses should use participant-readable labels and should identify the relevant Trial description field and selected categorical/numeric field.

Planned response field:

```json
"scenario_consistency_note": {
  "show": true,
  "message": "Some scenario details are not fully aligned across Trial description fields and structured fields. In this case the value in the structured fields drives the analysis, while the Trial description fields are used as supporting context. (Intervention text, Therapeutic Modality)",
  "fields": ["Intervention text", "Therapeutic Modality"]
}
```

Rules:

- Do not block review generation because of a `text_context` / `structured_features` mismatch.
- Do not let Trial description fields override structured categorical or numeric scenario choices.
- Completion Outlook should follow Completion Outlook score inputs and `model_interpretation` evidence.
- Design Confidence may use the mismatch as evidence of reduced coherence when supported.
- Key questions may challenge the team to clarify or defend the mismatch when it is material.

## Text-Change Evidence

The LLM should clearly see what changed in Trial description fields, especially when the edit adds new information rather than simply aligning `text_context` with structured categorical or numeric choices.

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

- `alignment_only`: a Trial description field was edited mainly to match selected structured fields.
- `new_information`: text adds new clinical, design, operational, or strategic meaning.
- `contradiction`: a Trial description field conflicts with selected structured fields.
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
  "visible": false,
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
  - Key Questions.

The narrative should remain readable and not become a diagnostics panel. Raw evidence fields, validation details, provider traces, and benchmark details belong in expanders or facilitator/debug views.

## Reliability Boundaries

Completion Outlook can reliably explain:

- What Completion Outlook score inputs changed.
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
- `text_context.summary_ui` added a simulated randomized-control-arm sentence.

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

- Completion Outlook over-explained with unsupported assumptions. It mentioned a need for a "larger, more diverse patient population" even though no diversity-related or population-breadth field was changed or clearly supported by Completion Outlook score inputs.
- Completion Outlook used broad clinical-development generalization too assertively: "Historically, randomized trials with active comparators and hard clinical endpoints face greater recruitment and retention challenges..." This is directionally plausible but too factual/causal unless grounded in changed `model_interpretation` evidence.
- Completion Outlook should stay closer to score movement, non-movement, changed Completion Outlook score inputs, and resemblance to historical completion / early-termination risk patterns.
- Design Confidence wording was directionally correct but too strong. Phrases such as "significantly improves" and "the design is now more robust" should be softened.
- Design Confidence should more explicitly keep the counterweight: stronger evidence generation may also create cost, governance, operational burden, feasibility, or proportionality concerns when supported by the packet.
- Design Confidence treemap subcategory tiles did not show the short rationale needed to evidence why each rating was strong, supportive, balanced, weak, or conflicting.

Preferred prompt wording direction:

- Prefer "may improve interpretability" instead of "significantly improves interpretability."
- Prefer "appears more robust from an evidence-generation perspective" instead of "the design is now more robust."
- Prefer "may resemble patterns associated with greater execution burden where comparator structure and endpoint ambition increase" instead of broad factual claims about historical recruitment and retention.
- When Completion Outlook is flat, explicitly state that the edited fields did not materially shift the scenario's resemblance to historical completion or early-termination patterns.
- If operational or feasibility language is used in Completion Outlook, it must be framed as a cautious risk-pattern interpretation and tied to Completion Outlook score inputs, not as clinical truth.

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
- `text_context.summary_ui` described a single-arm design interpreted against clinical context rather than a randomized control arm.

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

- Completion Outlook again used overly broad feasibility language: "historically associated with lower operational complexity and faster recruitment." "Faster recruitment" should not be stated unless directly supported by changed Completion Outlook score inputs or packet evidence.
- Completion Outlook described the result as an improvement even though `+0.3 pts` is marginal. It should call this essentially stable or only slightly higher.
- Design Confidence was directionally correct but too categorical in phrases such as "less robust" and "significant interpretability gap."
- Design Confidence treemap subcategory tiles still lacked the short rationale needed to explain the rating in a few words.
- Some proposed score-grounding language can become too complex for participants. Phrases such as "on score-supported design features," "more execution-favorable," and "does not materially alter the scenario's resemblance" are directionally right but should be simplified for live use.

Preferred prompt wording direction:

- Prefer "The Completion Outlook is almost unchanged, with only a small increase" over "The Completion Outlook appears to improve" for tiny score movements.
- Prefer "the simpler structure may look easier to run in the score pattern" over "more execution-favorable on score-supported design features."
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
- `text_context.summary_ui` added that the target population would be narrowed toward more advanced or treatment-resistant disease, potentially increasing clinical relevance while making evidence more dependent on patient selection.

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
- Completion Outlook mixed design-quality critique into the Completion Outlook paragraph. Lack of control group, surrogate endpoints, masking, and interpretability belong mainly in Design Confidence unless directly tied to Completion Outlook score-input evidence.
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
- Keep Completion Outlook focused on score movement and changed Completion Outlook score inputs; keep interpretability and evidence adequacy mainly in Design Confidence.
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

### Live Test Iteration 4: Trial Description / Structured-Feature Intervention Contradiction Example

Starting point:

- Previous scenario was single-arm, open-label, refractory / relapsed, advanced / metastatic, rare-condition status yes, surrogate-based evidence.
- Previous scores: Completion Outlook `45.5`, Design Confidence `-1.5`, Total Scenario Score `44.0`.

Edited scenario:

- `pathway_profile_ml`: changed from enzyme modulator to GPCR target.
- Structured treatment fields remained aligned with a small-molecule oral intervention:
  - `therapeutic_modality_ml`: small molecule.
  - `delivery_profile_ml`: simple oral pill/tablet.
- `text_context.interventions_ui` added a deliberately contradictory Trial description:
  - cell-based immunotherapy.
  - individualized manufacturing.
  - infusion-site coordination.

Observed scores:

- Completion Outlook decreased slightly from `45.5` to `45.2`.
- Design Confidence decreased from `-1.5` to `-7.5`.
- Total Scenario Score decreased from `44.0` to `37.7`.

What worked:

- Design Confidence appropriately became much more negative when the scenario contained incoherent intervention information.
- The clinical operations question picked up the manufacturing-coordination issue, which is relevant only if the contradictory Trial description detail were true.
- Total Scenario Score appropriately reflected the negative Design Confidence pressure.

Issues to correct in the prompt:

- The review did not appear to show the required `scenario_consistency_note` despite a clear contradiction between `structured_features` and `text_context`.
- The review treated the contradictory `text_context.interventions_ui` cell-therapy/manufacturing description as if it were true scenario evidence, even though `structured_features` still said small molecule and simple oral delivery.
- The prompt must more strongly state that `structured_features` values prevail when they conflict with Trial description fields.
- A contradictory Trial description field may be used as a scenario-readiness concern in Design Confidence, but it must not replace `structured_features` as the analysis source of truth.
- Completion Outlook must not use contradictory `text_context.interventions_ui` manufacturing, site coordination, infusion burden, or retention burden as Completion Outlook score drivers when the selected score inputs remain small molecule / simple oral.
- The narrative introduced an apparent "female-only population" restriction. This was not part of the intended Iteration 4 change and should not be invented unless supported by changed structured fields or text evidence.
- The two participant questions again repeated earlier evidence-standard and assessment-bias patterns too closely.
- Design Confidence again used categorical language such as "typically required for registration-enabling studies" and "can provide the necessary evidence." This should be softened.

Preferred prompt wording direction:

- If `text_context` and `structured_features` conflict, start with the approved participant consistency note:

```text
Some scenario details are not fully aligned across Trial description fields and structured fields. In this case the value in the structured fields drives the analysis, while the Trial description fields are used as supporting context. (Intervention text, Therapeutic Modality)
```

- In Completion Outlook, prefer:

```text
The Completion Outlook changes only slightly. Because the selected intervention fields still describe a small-molecule oral treatment, the contradictory cell-therapy wording should not be treated as a model driver. The main risk-pattern signal is therefore limited to the changed structured fields, while the text mismatch mainly raises a design-coherence question.
```

- In Design Confidence, prefer:

```text
Design Confidence falls because the intervention description is internally inconsistent: `structured_features` describe an oral small molecule, while `text_context.interventions_ui` describes cell-based manufacturing and infusion logistics. `structured_features` should drive the score interpretation, but the mismatch makes the scenario harder to defend unless clarified.
```

Candidate one-shot lesson:

- Use this scenario as one intervention-specific example for the general `structured_features` / `text_context` contradiction rule.
- The one-shot should demonstrate a visible `scenario_consistency_note`.
- The one-shot should show that `structured_features` prevail in Completion Outlook.
- The one-shot should allow Design Confidence to penalize scenario readiness, but only as a `text_context` / `structured_features` mismatch concern, not as if the cell-therapy Trial description had replaced selected values.
- The one-shot should generate new participant questions that focus on clarification and governance of inconsistent scenario evidence rather than repeating the earlier evidence-standard question.
- Later automated waves should add at least one non-intervention contradiction example, such as endpoint or population, to verify that the same rule applies across all Trial description fields.

## UI Review Notes From Live Testing

These notes should be implemented after the narrative-quality review is complete.

Implementation status - 2026-06-13:

- Pending-review visibility and previous-value/delta display policy have been implemented in `frontend/views/trial_simulator.py`.
- The prompt has been tightened for `structured_features` / `text_context` conflict handling, fresh key questions, conditional regulatory/evidence language, and non-duplicative Design Confidence reasoning in `src/narratives/prompt_builder.py`.
- `scripts/check_narrative_prompt_builder.py` now protects those prompt requirements.
- Live Gemini behavior after these wording changes has not yet been evaluated; the first-wave automated eval harness is the next validation step.

## Automated First-Wave Eval Loop

Implementation status - 2026-06-13:

- `scripts/run_narrative_eval_suite.py` builds automated Scenario Review eval packets without manual Streamlit input.
- Reports are written under ignored `reports/narrative_evals/` as both `.md` and `.json`.
- The first-wave scenario path covers four visible iterations per trial:
  1. model-favorable simplification with weaker evidence value;
  2. lower Completion Outlook but potentially stronger patient relevance / design confidence;
  3. operational-burden-only change with no Completion Outlook movement expected;
  4. `structured_features` / `text_context` contradiction where structured fields must prevail.
- The harness records exact UI-label field changes from `models/taxonomy_01.json`, expected behavior, provider output, validated narratives, Design Confidence scoring, key questions, deterministic gap checks, and Codex comments.
- JSON reports preserve the full reviewed-trace audit payload: input packet, provider prompt, raw provider JSON, and validated review. Markdown reports remain shorter and oriented to user review.
- The harness separates deterministic checks from human-review expectations so narrative realism, design-confidence judgment quality, and question usefulness are reviewed explicitly rather than hidden inside brittle automatic labels.
- `python scripts/run_narrative_eval_suite.py --success-smoke` validates the successful reviewed path locally with fixture-backed mock output before sending any first-wave packet to Gemini.
- Current limitation: Completion Outlook movements are planned/synthetic for prompt-quality evaluation rather than produced by live `/predict`; this keeps the first wave fast and focused on narrative/design-confidence behavior.

Working loop:

1. Run a small live Gemini pass first, for example `python scripts/run_narrative_eval_suite.py --provider configured --max-trials 1`.
2. Review the Markdown report with the user.
3. Summarize repeated quality gaps in this plan before changing prompt wording.
4. Apply prompt changes only for repeated or high-confidence failures.
5. Re-run the same eval set and compare the JSON/Markdown outputs.

First live shakedown - 2026-06-13:

- Run: `python scripts/run_narrative_eval_suite.py --provider configured --max-trials 1 --run-id first_wave_live_trial1`.
- Trial: `NCT06989437` / Oncology / Pfizer / borderline baseline Completion Score `45.0`.
- Provider: Gemini `gemini-3.1-flash-lite`; all four visible iterations returned validated Scenario Reviews.
- Deterministic result: `1` failed check and `1` warning across four visible iterations.
- Report artifacts: `reports/narrative_evals/first_wave_live_trial1.md` and `.json`.

Observed quality notes:

- Good: Iteration 1 correctly counterweighted a model-favorable simplification. Completion Outlook improved from `45.0` to `50.0`, while Design Confidence fell to `-3.5` because pivotal intent was not supported by single-group, non-randomized, open-label design.
- Good: The `structured_features` / `text_context` contradiction produced a visible consistency note and acknowledged selected `Small Molecule` / `Simple Oral` fields.
- Issue: The operational-only iteration kept score delta at `0.0`, but the Completion Outlook narrative still used comments derived from planned enrollment, planned site count, and planned total duration. The prompt/checker should force those three planning-assumption fields into Design Confidence, not Completion Outlook, unless the Completion Outlook score inputs changed.
- Issue: Iteration 4 repeated the same medical development question from iteration 3. Question-freshness guidance should be reinforced or the eval should require a scenario-specific question when a new `structured_features` / `text_context` contradiction appears.
- Eval-design correction implemented: Iteration 2 no longer requires total Design Confidence `>= 0.0`. Because the scenario is cumulative and still inherits the prior single-group, non-randomized, open-label pivotal-design weakness, the check now focuses on positive `target_population_alignment` / patient-relevance recognition while allowing total Design Confidence to remain negative.
- Design-state decision implemented: keep eval iterations cumulative, but treat Design Confidence as a fresh current-scenario recalculation. Prior reviews are continuity/delta context only, not accumulated penalties or bonuses.
- Added prompt/checker protection against using broad phrases such as `operational footprint`, `operational scale`, `site footprint`, or `recruitment footprint` as proxies for the three planning-assumption fields in Completion Outlook.

Future eval scenario to add after the first wave:

- Add a "resolved weakness" iteration or separate fixture where a prior single-group/open-label/non-randomized pivotal weakness is fixed by returning to controlled/randomized/blinded evidence. Expected behavior: Design Confidence should recover if current fields support recovery, and the narrative should acknowledge the prior concern as resolved rather than continuing to penalize it.

Live rerun after cumulative/current-state prompt fix - 2026-06-13:

- Run: `python scripts/run_narrative_eval_suite.py --provider configured --max-trials 1 --run-id first_wave_live_trial1_after_cumulative_fix`.
- Result: all four visible iterations returned validated Scenario Reviews; deterministic result changed to `2` failed checks and `0` warnings.
- Improvement: the iteration-2 expectation correction worked. The eval no longer requires total Design Confidence to become non-negative; it checks positive `target_population_alignment`, which Gemini satisfied.
- Improvement: question freshness improved for iteration 4. The final medical question changed from the prior endpoint-standard question to a modality-specific small-molecule/GPCR evidence question.
- Remaining issue after `first_wave_after_audit_trial1`: planning-assumption comments still leaked into Completion Outlook in iteration 3 through planned enrollment, site count, total duration, and operational-footprint wording. This is a real prompt adherence failure, not just a checker issue.
- Corrected `structured_features` / `text_context` rule: structured scenario fields prevail for Completion Outlook and Design Confidence analysis. A Trial description mismatch does not automatically cap Design Confidence; it requires a clear warning so users know the intervention description should be updated before relying on the scenario.
- Calibration note: the new cumulative/current-state wording may have overcorrected by making the provider drop inherited pivotal-design concerns too aggressively in iteration 2 and iteration 4. Future prompt refinement should say "recalculate fresh from current fields" while still preserving unresolved current weaknesses that remain present in the packet.

Agreed narrow guardrail plan before next live run:

- Preserve what worked: cumulative scenario state, fresh current-scenario Design Confidence, subcategory-level evaluation, counterweight behavior, structured-field source-of-truth behavior, and question freshness.
- Clarify the question split rather than broadening the prompt:
  - Completion Outlook answers only: did Completion Outlook score inputs or early-termination risk-pattern evidence move, and why?
  - Design Confidence answers: is the current full design interpretable, coherent, proportionate, and strategically defensible using all relevant packet evidence?
- Make duration wording explicit:
  - `primary_duration_months_ml` / Max Endpoint Duration is a Completion Outlook score input and may support Completion Outlook when present in `model_interpretation` evidence.
  - `planned_duration_months` / Planned Total Duration is operational-only and may support Design Confidence proportionality, but must not explain Completion Outlook movement.
- Add the operational-only zero-delta rule: if `score_delta` is `0.0` and only the three planning assumptions changed, Completion Outlook should use participant-friendly boundary wording: `The Completion Outlook remains unchanged because planning assumptions such as enrollment, site count, and total duration do not directly feed the score. They still matter for whether the scenario feels operationally proportionate and executable. Therefore, the impact of changes in these variables is reflected in Design Confidence instead.`
- Add the non-score-input stable rule: if the latest change is limited to Trial description fields, or combines Trial description fields with planning assumptions, and no structured Completion Outlook score input changed, Completion Outlook should use: `The Completion Outlook score remains stable because the latest changes are not directly used to calculate the Completion Outlook score. Nevertheless, the updated scenario details are considered in Design Confidence.`
- Implementation: the two deterministic Completion Outlook boundary modes are owned by `src/narratives/review_controls.py` and are applied by both the eval harness and the live Scenario Review storage path. The override changes only `completion_outlook_analysis`; it preserves provider-generated Design Confidence, subcategory scoring, rationales, and questions.
- Strengthen the automated operational-only check to fail movement/risk-change language and extra planning-assumption detail in Completion Outlook for zero-delta operational-only scenarios, not only exact forbidden terms.
- Add a stale-carryover check: later non-operational iterations must not describe prior planned enrollment, site count, or planned-duration changes as if they were adjusted in the latest iteration.
- Enforce the approved `structured_features` / `text_context` warning without capping Design Confidence: when controlled fields and intervention description conflict, require the approved participant consistency sentence followed by participant-readable field labels in parentheses, such as `(Intervention text, Therapeutic Modality)`.
- Adjust the `structured_features` / `text_context` eval expectation from "Design Confidence must be negative" to "`structured_features` drive analysis; exact warning required; no automatic Design Confidence cap."

Implementation status:

- Implemented in `src/narratives/prompt_builder.py`, `scripts/check_narrative_prompt_builder.py`, and `scripts/run_narrative_eval_suite.py`. Operational-only zero-delta wording was updated to avoid internal model vocabulary in participant narratives.
- Local verification passed: prompt-builder checker, py_compile, eval success-smoke, mock one-trial smoke, and `git diff --check`.

Live rerun after participant wording fix - 2026-06-13:

- Run: `python scripts/run_narrative_eval_suite.py --provider configured --max-trials 1 --run-id first_wave_live_trial1_after_warning_fix_live`.
- Result: all four visible iterations returned validated Scenario Reviews; deterministic result was `1` failed check and `2` warnings.
- Improvement: the operational-only Completion Outlook sentence was followed exactly. The review stated that planning assumptions such as enrollment, site count, and total duration do not directly feed the score, and that they are reflected in Design Confidence instead.
- Improvement: the `structured_features` / `text_context` warning used the approved participant wording and appended field labels in parentheses, for example `(Intervention text, Therapeutic Modality, Administration Complexity)`.
- Improvement: the `structured_features` / `text_context` contradiction did not automatically cap Design Confidence, which matches the source-of-truth rule for structured fields.
- Remaining issue: the operational-only iteration failed because the harness expected total Design Confidence `<= 0.0`, but the scenario still inherited patient/evidence strengths from prior cumulative edits. The better expectation is subcategory-level: total Design Confidence may remain positive, but `operational_burden_balance` should be neutral or negative when enrollment, sites, or planned duration increase without matching evidence gain.
- Remaining issue: the medical/development question repeated verbatim across later iterations. The prompt needs a stronger rule that unresolved concerns can persist, but the exact participant question must be rewritten around the newest scenario change.
- Remaining issue: in the `structured_features` / `text_context` contradiction, the mismatch was discussed strongly enough that it could be read as driving the scenario judgment. When `structured_features` prevail, the mismatch may be explained in the consistency note and narrative, but it should remain scenario-readiness context and should not override structured-feature evidence or become the main Design Confidence score driver.

Implemented calibration after this rerun:

- Updated `scripts/run_narrative_eval_suite.py` so the operational-only iteration checks `operational_burden_balance <= 0.0` instead of total Design Confidence `<= 0.0`.
- Added deterministic failure for verbatim repeated medical or operational questions; similar but non-identical questions remain warnings.
- Kept `structured_features` / `text_context` mismatch handling as a human-review focus rather than a hard Completion Outlook narrative failure: narratives may discuss the mismatch, while `structured_features` remain the scoring source of truth.
- Updated `src/narratives/prompt_builder.py` to state that operational-burden-only increases without matching evidence gain should not improve `operational_burden_balance`.
- Updated `src/narratives/prompt_builder.py` to state that prior visible questions must not be repeated verbatim; if the concern remains, rewrite around the newest scenario change.
- Updated `src/narratives/prompt_builder.py` to state that `structured_features` / `text_context` mismatch may be discussed in narratives, but should remain scenario-readiness context and should not become the main Design Confidence score driver.

Live rerun after audit hardening - 2026-06-13:

- Run: `python scripts/run_narrative_eval_suite.py --provider configured --max-trials 1 --run-id first_wave_after_audit_trial1`.
- Result: all four visible iterations returned validated Scenario Reviews; deterministic result was `3` failed checks and `2` warnings.
- What worked: shortcut scenario was correctly penalized by Design Confidence; patient-relevance scenario credited `target_population_alignment`; operational-only scenario kept `operational_burden_balance` neutral rather than positive; `structured_features` / `text_context` contradiction used the approved warning wording and structured fields prevailed.
- Remaining issue: participant-facing Completion Outlook used internal wording such as `model-facing risk profile`. This remains a quoted bad example; the replacement is plain language such as `Completion Outlook score inputs` or simply `score inputs`.
- Remaining issue: operational-only Completion Outlook still added comments derived from the three planning-assumption fields: planned enrollment, planned site count, and planned total duration. These fields should not explain Completion Outlook; they belong in Design Confidence as proportionality/executability context.
- Clarification from user: `operational details` in this boundary means only the three planning-assumption fields derived outside the Completion Outlook score: planned enrollment, planned site count, and planned total duration. Other operational trial features may still be discussed in Completion Outlook when they are actual Completion Outlook score inputs and `model_interpretation` evidence supports them.
- Remaining issue: later iterations can carry stale wording from prior changes, such as describing planning assumptions as adjusted in the current iteration when the latest change is actually modality/text. The eval should fail this stale prior-change wording.
- Remaining issue: questions can remain too similar across later iterations. If the newest change is operational-only, at least one question should focus on operational proportionality or executability. If the newest change creates a `structured_features` / `text_context` conflict, at least one question should focus on resolving that contradiction before relying on the scenario.

Implemented adaptation after this run:

- Updated `src/narratives/prompt_builder.py` to avoid participant-facing internal model vocabulary in the main question split and use `Completion Outlook score inputs`.
- Narrowed the Completion Outlook boundary to the three planning-assumption fields: planned enrollment, planned site count, and planned total duration. Broad phrases such as `operational footprint`, `operational scale`, `site footprint`, and `recruitment footprint` are forbidden only when used as proxies for those three planning assumptions in Completion Outlook.
- Strengthened the operational-only zero-delta instruction: after the agreed boundary sentence, do not add extra Completion Outlook commentary derived from those three planning assumptions; discuss them only in Design Confidence.
- Strengthened question freshness instructions for operational-only and `structured_features` / `text_context` contradiction iterations.
- Updated `scripts/run_narrative_eval_suite.py` to fail extra planning-assumption detail after the agreed operational-only boundary sentence and to fail stale planning-assumption carryover in later non-operational iterations.

Robust control-layer refinement after planning review:

- Root cause: the provider has to evaluate the current cumulative scenario while also explaining the latest iteration. That creates a legitimate tension: prior concerns may still be true, but they must not be described as newly changed.
- Root cause: the packet necessarily includes planned enrollment, planned site count, and planned total duration for Design Confidence, so prose-only prompt bans are fragile for Completion Outlook.
- Root cause: question freshness cannot rely only on "be fresh"; the provider tends to repeat the dominant unresolved evidence-standard question even when the newest change is operational or a `structured_features` / `text_context` contradiction.
- Strategy: keep Gemini's useful expert Design Confidence reasoning mostly unchanged, but add a small app-owned control layer for hard product boundaries.
- Added `review_controls` to eval packets with `completion_outlook_mode`, `latest_change_focus`, `completion_outlook_forbidden_latest_fields`, and `question_controls`.
- For the operational planning-assumption zero-delta mode, `completion_outlook_mode=fixed_planning_assumption_boundary` and `required_completion_outlook_sentence` is the agreed participant-facing sentence.
- The eval harness now applies a narrow deterministic override before storing traces for this hard-boundary mode: `completion_outlook_analysis.risk_pattern_summary` and `movement_explanation` are normalized to the agreed sentence. This is intentionally limited to the three planning-assumption fields and does not modify Design Confidence, subcategory rationale, questions, or scoring.
- The provider prompt now tells Gemini to follow `packet.review_controls` when present and treats those controls as product-level logic, not clinical reasoning.
- The eval checker now verifies latest-change question focus for planning-only and `structured_features` / `text_context` contradiction steps.
- The eval checker now exposes `completion_outlook_operational_extra_detail`, `completion_outlook_stale_prior_change`, and `question_latest_change_focus` labels in the Markdown report.

Implementation files:

- `scripts/run_narrative_eval_suite.py`
- `src/narratives/prompt_builder.py`
- `scripts/check_narrative_prompt_builder.py`

Verification:

- `python scripts/check_narrative_prompt_builder.py`
- `python scripts/run_narrative_eval_suite.py --success-smoke`
- `python -m py_compile src/narratives/prompt_builder.py scripts/check_narrative_prompt_builder.py scripts/run_narrative_eval_suite.py`
- `git diff --check`

Audit follow-up implemented before next live run:

- Preserve raw provider output before the operational planning-assumption override. JSON summaries now include `pre_control_output_json`, `pre_control_validated_review`, and `pre_control_scoring`; Markdown reports show the pre-control Completion Outlook when an override was applied.
- Replace remaining active prompt wording that could leak internal phrasing into participant narratives: internal model vocabulary became `Completion Outlook score inputs`, and generic score-evidence wording became `Completion Outlook score evidence` in the active provider prompt.
- Extend `python scripts/run_narrative_eval_suite.py --success-smoke` so it exercises `review_controls`, verifies the packet hash changes when controls are attached, confirms the fixed operational-boundary sentence is applied, and confirms the pre-control Completion Outlook is preserved.

Cautious pre-rerun strengthening:

- Added a narrow participant-language rule to avoid internal phrases such as `model-facing`, `model-supported`, `the model says`, or `model reflects`; these are quoted bad examples. Preferred replacements are `Completion Outlook score inputs`, `score pattern`, or `score-driving fields`.
- Added a narrow `review_controls` continuity rule: explain the latest change without re-labeling older cumulative issues as newly changed.
- No additional clinical/domain constraints were added before the next live run, to avoid degrading the currently good Design Confidence reasoning.

Live rerun after control-layer hardening - 2026-06-13:

- Run: `python scripts/run_narrative_eval_suite.py --provider configured --max-trials 1 --run-id first_wave_control_layer_rerun_1`.
- Result: all four visible iterations returned validated Scenario Reviews; deterministic result was `1` failed check and `0` warnings.
- What worked: operational-only Completion Outlook used the approved boundary sentence before and after the deterministic control; shortcut scenario penalized weaker evidence despite Completion Outlook improvement; patient relevance credited Target Population Alignment; `structured_features` / `text_context` contradiction used the approved warning and selected structured fields in Completion Outlook.
- Eval calibration issue: the operational-only question check was too narrow. The generated questions addressed enrollment, duration, expanded site network, oversight, and data quality, but the checker required a smaller vocabulary such as `planned enrollment`, `site count`, `proportionate`, or `executable`.
- Remaining participant-language issue: Gemini still used phrases such as `in the model` and `model signals`. This should be caught by deterministic checks and lightly discouraged in the prompt without adding broad clinical rules.
- Remaining `structured_features` / `text_context` quality issue: the mismatch can still dominate Design Confidence too strongly across multiple subcategories. The prompt should keep one compact principle: only the conflicting Trial description detail is stale/superseded by the structured value; non-conflicting Trial description details and latest `text_context` changes remain useful context. The conflict should be framed as a scenario-readiness warning, not evidence that the selected structured design itself has the contradicted feature.

Implemented adaptation after this run:

- Broadened the operational-only question-focus checker to accept realistic wording around `enrollment`, `duration`, `scale`, `expanded network`, `oversight`, `data quality`, `site-level`, and `governance`.
- Added deterministic participant-language failure for internal model vocabulary such as `model signals`, `model suggests`, `model indicates`, `model predicts`, `model implies`, `model-driven`, `according to the model`, `in the model`, and related terms.
- Added `structured_features` / `text_context` scenario-readiness dominance warnings when a mismatch strongly penalizes Endpoint Evidence Strength without endpoint evidence support, or when mismatch language appears to drive multiple strong negative Design Confidence subcategory penalties.
- Kept prompt additions minimal: expanded the participant-language avoidance list and added one scenario-readiness principle. No new clinical/domain rule block was added.

Minimal question-quality adjustment:

- Superseded the older three-question contract with a lighter two-question rule: for later visible iterations, the medical/clinical-development question should focus on the medical, evidence, or trial-design implication of the newest material scenario change, while the strategic development question should raise a broader development-path or field-level debate using the trial or latest change as a concrete example.
- Removed the permissive `unless the latest change genuinely reopens the same dilemma` wording. If the same dilemma remains relevant, the provider should reframe it through the newest material change rather than repeating the prior question frame.
- Kept similarity as a warning, not a failure. The warning text now asks the reviewer to check whether the two-question pair has this newest-change / broader-strategic split.
- Added a light question-framing rule: questions should be general debate prompts. Prompt examples now use neutral wording rather than addressing `the team`; after the three-trial rerun, the eval now fails key questions that directly address `the team`.
- Added an operational-only question bridge: when the latest change is limited to planning assumptions, the medical/clinical-development question should connect current evidence ambition to whether the added operational burden is justified rather than repeating the prior endpoint-standard frame.
- Added a targeted three-trial follow-up: avoid repeating the same question opening stem across consecutive visible iterations, especially `What evidence standard would...`; for `structured_features` / `text_context` conflicts, questions should ask how to resolve or reconcile the scenario before relying on it, not how to operationalize stale contradictory Trial description detail.
- Added a shortcut-simplification calibration and refined it after reviewing the final-settings quality wave: when easier execution comes from weaker comparator, masking, allocation, endpoint rigor, or lower evidence ambition, `operational_burden_balance` can still receive feasibility credit, but strong positive credit (`+3` to `+5`) requires bounded rationale, independent operational value, or safety-extension/proportionality context. The eval should flag unjustified strong shortcut credit rather than all positive shortcut credit.
- Live rerun `first_wave_operational_shortcut_cap_3trials_1` returned 12/12 reviewed visible iterations, 0 failed checks, and 3 warning checks. The shortcut cap held: shortcut-driven simplification received only limited Operational Burden Balance credit. Remaining warnings were limited to question opening-frame repetition and one scenario-readiness dominance review item; no immediate prompt change is required from this run.
- Broader live run `first_wave_broader_trials_5_1` returned 20/20 reviewed visible iterations, 3 failed checks, and 10 warnings. Light follow-up only: replace `model signals` with score-pattern wording, make the patient-relevance expectation conditional when a prevention/vaccine-style trial objective conflicts with refractory/metastatic structured edits, and require operational-only medical questions to mention planning burden, scale, or proportionality so inherited endpoint/population questions cannot repeat verbatim.
- Three-question live run `first_wave_three_question_contract_5_1` returned 19/20 reviewed visible iterations, 4 failed checks, and 20 warnings. The third question rendered consistently. One fail was a transient Gemini `ServerError`; one was an eval expectation issue for a preventative vaccine trial where refractory/metastatic structured edits correctly lowered Target Population Alignment; one was a participant-language leak using `in the model`; and one operational-only medical question did not explicitly connect evidence standard to planning burden. Follow-up changes should stay light: strengthen score-pattern replacement language for `in the model`, broaden the prevention/vaccine expectation skip using Trial description context, and vary the strategic/field question lens across evidence standard, access, governance, data reliability, representativeness, feasibility, and interpretability to reduce repeated openings.
- Do not add post-narrative deterministic cleanup for participant wording or question rewriting for now. Remaining internal-language or question-freshness issues should be visible as eval findings and addressed through light prompt/eval refinement unless future evidence shows a narrow production control is necessary.
- Temperature comparison plan: run the same 5 Gemini 3.1 Flash-Lite trials at explicit `0`, explicit `0.3`, and omitted/default temperature. Compare JSON/validation success, Design Confidence score calibration, Completion Outlook and Design Confidence narrative quality, question quality, prompt adherence, and robustness of reasoning. Add a one-trial duplicate run at the most informative setting to check whether outputs reproduce exactly, drift only in wording, or drift in scoring/reasoning. Store the aggregate comparison with `scripts/compare_narrative_temperature_reports.py`.
- Thinking-level comparison result: omitted/default temperature plus primary Gemini `high` thinking is the current final-settings candidate. It improved the five-trial quality profile versus default medium thinking but did not make duplicate outputs deterministic, so reproducibility remains measured through duplicate traces and drift inspection rather than temperature/seed assumptions.
- Final-settings quality/reproducibility plan: run a 10-trial Gemini quality wave to detect adherence and narrative-quality patterns, then run a 3-trial duplicate wave to inspect exact and qualitative reproducibility under the same settings. The helper `scripts/run_final_narrative_quality_plan.py` prints or executes this sequence with explicit omitted/default temperature and high thinking, then stores the aggregate comparison as `final_settings_quality_repro_comparison`.
- Final-settings qualitative follow-up: keep the expert narrative level, but make the Design Confidence opening sentence state the main cross-functional decision tension; broaden participant-language cleanup for `model signal`, `model-score inputs`, and `the model reflects`; broaden question direct-address checks beyond `the team`; and report duplicate-run Design Confidence/subcategory drift without adding post-narrative deterministic cleanup.
- Final validation wave: use `scripts/run_final_narrative_validation_plan.py` after the latest prompt/checker refinements. This wave separates three goals. First, non-cumulative `--scenario-plan boundary` tests isolated unusual latest-change combinations: structured-only, Trial description-only, planning-assumption-only, structured + Trial description, Trial description + planning assumptions, structured + planning assumptions, all three input types together, `structured_features` / `text_context` contradiction, aligned non-conflict structured/text, and shortcut simplification. Second, cumulative `--scenario-plan storyline` runs 12 credible multi-iteration candidates intended for later one-shot example selection, targeting Oncology plus UCB-relevant therapeutic areas and preferring UCB-sponsored candidates when available. Third, duplicate storyline runs test reproducibility when the same baseline and scenario path are reused.
- One-shot example policy: do not embed one-shot examples into the live provider prompt yet. Candidate examples should first be selected from the 12-trial storyline wave and tagged as `presentation_ready`, `good_after_light_edit`, `useful_for_stress_test_only`, or `discard`. A one-shot example is useful only if it demonstrates expert cross-functional tension, Completion Outlook versus Design Confidence separation, coherent 2-4 iteration storyline, robust subcategory scoring rationale, and strong non-prescriptive questions. It should then be A/B tested against the current prompt to verify that it improves quality and adherence without making output formulaic or over-specific to one therapeutic area.
- Boundary pilot `final_validation_boundary_pilot_1` showed the safety matrix works structurally but still needs narrow prompt adherence cleanup before the bigger run: avoid actor-addressed questions such as `How should sponsors...` or `How will the sponsor...`; avoid participant-facing internal wording such as `model indicates`; keep Trial description + planning-assumption-only Completion Outlook stable without discussing enrollment, site count, total duration, resource allocation, or operational footprint; and keep `structured_features` / `text_context` mismatch as a scenario-readiness warning that usually affects the single most relevant Design Confidence subcategory unless independent structured evidence supports broader penalties.
- Follow-up simplification: remove the forbidden actor examples from the live prompt wording and use only positive impersonal question examples; add `model interpretation` / possessive variants to the internal-language avoid list; and add a narrow `stable_non_score_input_context` review-control mode for Trial description + planning-assumption changes with no structured Completion Outlook score-input change. This keeps the prompt lighter than adding another broad rule block.
- Boundary pilot `final_validation_boundary_pilot_2` improved to 11/11 reviewed iterations with 2 failed checks. The remaining narrow fixes are to make `stable_non_score_input_context` avoid naming planning-assumption details in Completion Outlook, including `planned enrollment, site count, and primary duration`, and to add the remaining internal-language variants `model suggests`, `model-derived`, and possessive `model's...` / `model’s...` wording to the participant-language avoid/check lists.
- Boundary pilot `final_validation_boundary_pilot_3` returned 11/11 reviewed iterations but 7 failed checks. The increase came from Gemini variance plus stricter checks, not from provider validation failure. Narrow follow-up only: use a short stable-mode Completion Outlook target sentence for Trial description + planning-assumption changes with no structured score-input change; forbid `team`, `sponsor`, `sponsors`, `investigator`, `investigators`, `stakeholder`, `stakeholders`, `you`, and `your` in participant questions; add `model registers` to participant-language checks; and sharpen shortcut calibration so removing randomization, masking, comparator structure, arms, or endpoint rigor is not independent Operational Burden Balance value by itself.
- Boundary pilot `final_validation_boundary_pilot_4` improved to 11/11 reviewed iterations, 4 failed checks, and 1 warning. The mixed Trial description + planning-assumption boundary and shortcut overreward issues were resolved. Remaining follow-up should stay narrow: add final-output self-check wording for internal model-language replacement and forbidden question words, and calibrate the no-score-input Completion Outlook eval so stable narratives that say an existing risk pattern persists are not treated as score movement.

Completion Outlook and scenario-readiness boundary refinement:

- Planning-assumption boundary: if the latest change is limited to planned enrollment, planned site count, and/or planned total duration, the Completion Outlook score is unchanged because these fields do not feed the Completion Outlook score. If other Completion Outlook score inputs also changed, explain the Completion Outlook narrative using those score-input changes only; planning assumptions remain Design Confidence context for operational proportionality and executability.
- The fixed planning-assumption Completion Outlook sentence is exclusive to the planning-assumption-only boundary mode. The eval now fails if a non-planning-only scenario reuses the exact fixed sentence.
- Trial description field boundary: `text_context` Trial description fields may support the Completion Outlook narrative only when they align with or clarify selected Completion Outlook score inputs. A conflicting Trial description field detail is stale scenario text superseded by the corresponding `structured_features` value and should not be used as Completion Outlook evidence.
- Scenario-readiness boundary: `structured_features` / `text_context` conflict remains visible and may affect Design Confidence as a readiness issue, but should not be treated as evidence that the selected structured design itself has the contradicted feature. The eval warns when the conflict appears to drive multiple strong negative subcategory ratings without independent non-conflicting structured support.

Qualitative resource and budget burden:

- Added one compact Operational Burden Balance principle: the provider may discuss qualitative resource, staffing, and budget implications when packet fields imply added burden, but must not estimate monetary cost, affordability, or financial feasibility without explicit financial evidence.
- Resource and budget burden should influence Design Confidence through proportionality, not automatically. More burden is negative when unjustified by evidence, patient-relevance, governance, or interpretability value gained; it may be neutral or supportive when proportionate to a meaningful design benefit.
- No new cost category or hard cost eval scenario was added.

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
- Pass if Completion Outlook may cite `primary_duration_months_ml` / maximum primary endpoint duration when it is present as Completion Outlook score-input evidence. Verification: automated packet/field check.
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

- Pass if the two questions differ materially from the previous visible iteration, with the medical/clinical-development question anchored to the newest material scenario change and the strategic development question raising a broader development-path or field-level challenge. Verification: golden examples plus manual review.
- Pass if each question is linked to the latest value changes, current narrative tension, or a high-value clinical-development dilemma relevant to the trial. Verification: manual or LLM-review.
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
- Pass if `text_change_evidence` is available when material Trial description fields changed. Verification: automated packet check.

Lifecycle acceptance checks:

- Pass if hidden baseline stores qualitative context while suppressing participant-visible Design Confidence and Total Scenario Score. Verification: automated trace check.
- Pass if first visible iteration can compare Completion Outlook to visible original Completion Score. Verification: manual or LLM-review.
- Pass if first visible iteration evaluates Design Confidence as current-scenario design defensibility, not hidden-score variance. Verification: manual or LLM-review.
- Pass if later visible iterations use prior visible review for continuity without overusing Design Confidence score-to-score storytelling. Verification: manual or LLM-review.
- Pass if pending feature changes preserve previous visible Design Confidence and Total Scenario Score while clearly marking those views as pending review. Verification: focused UI smoke.
- Pass if first visible Total Scenario Score delta uses baseline Completion Outlook as the previous value, then later iterations use previous visible Total Scenario Score. Verification: automated snapshot/delta test plus UI smoke.

## Implementation Sequence

1. Update narrative architecture docs.
   - Add the simplified participant output model: Completion Outlook, Design Confidence, Main Tension, and two Key Questions.
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
   - Add `text_change_evidence` for material Trial description field edits.
   - Preserve missing-pack trace without failing.

4. Update prompt builder and response schema.
   - Reshape participant-facing output into concise Completion Outlook, Design Confidence, Main Tension, and two-question sections.
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
   - Store the participant-facing summary blocks, main tension, and two questions.
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
- Trial description fields in `text_context` are treated as context, not instruction.
- Hidden baseline context exists and is not participant-visible by default.
- Provider traces keep model/settings/validation metadata without secrets.
- Fallback/retry behavior is bounded.
- Output style requires conditional language and avoids direct optimization instructions.

Target enhancements to add:

- Four concise participant-facing sections: Completion Outlook Analysis, Design Confidence Analysis, Main Tension, and two Key Questions.
- Top-level `review_metadata` with `review_mode` and `visible`, while accepting `participant_visible` only as a legacy fallback.
- Three prompt modes: `hidden_baseline`, `first_visible_iteration`, and `later_visible_iteration`.
- Completion Outlook framed as early-termination risk-pattern interpretation.
- Completion Outlook interpretive hypotheses with model signal, possible pattern, context modifiers, and boundary.
- Planned enrollment, planned site count, planned total duration, and operational benchmark metadata excluded from Completion Outlook.
- `primary_duration_months_ml` / maximum primary endpoint duration explicitly allowed as XGBoost Completion Outlook evidence.
- Design Confidence as a quality-of-design adjustment and challenge layer, not a second completion predictor.
- Design Confidence subcategory objects with rating, evidence fields, rationale, short rationale, optional lenses, and optional regulatory/finance note.
- Optional TA `.md` lookup by XGBoost canonical therapeutic-area value.
- `scenario_consistency_note` for clear `text_context` / `structured_features` mismatch.
- `text_change_evidence` for material Trial description field edits.
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

### Split Implementation Plan For Major Changes

Implement the major prompt/schema migration in small pieces. The contract shape, continuity anchors, prompt wording, scoring diagnostics, UI, exporters, and eval checks should not all change in one commit.

Recommended order:

1. Schema and compatibility.
   - Add `main_tension` and the two-question target while temporarily accepting old question fields as compatibility aliases if the current simulator or reports still need them.
   - Do not change prompt behavior yet.

2. Packet continuity anchors.
   - Add `design_confidence_continuity` for later visible iterations.
   - Define the field-to-subcategory relevance map used to decide whether a Design Confidence subcategory had relevant changed evidence.

3. Prompt simplification.
   - Move to the lean participant output shape.
   - Prefer positive `do` instructions and keep only short hard red lines.
   - Keep current-state Design Confidence scoring plus continuity-anchor guidance.
   - Current working label: `Prompt implementation point 3 / Phase 3`. In this plan, "Phase 3" means this prompt-simplification point, not the older scoring Phase 3 in `implementation_plan.md`.
   - Optional prompt-volume reductions may be added here when they preserve packet evidence, reproducibility, scoring boundaries, and auditability.

4. Validation, scoring diagnostics, and storage.
   - Preserve app-owned scoring.
   - Add warnings for large unexplained Design Confidence subcategory direction flips when no relevant field changed.
   - Keep these diagnostics as warnings first unless live validation shows they should block scoring.

5. UI, Markdown reports, Word export, and report-pack templates.
   - Render `main_tension` and the two questions everywhere participant review output is displayed or exported.
   - Remove assumptions that three questions always exist.

6. Eval harness update.
   - Update checks to expect two questions, continuity metadata, and flip diagnostics.
   - Keep historical report compatibility only where needed for old archived runs.

7. Small live validation.
   - Run 1-2 trials first to confirm schema stability, continuity behavior, and obvious prompt adherence before a full validation wave.

8. Final five-trial validation.
   - Run the intended five trials with baseline plus four visible iterations only after the previous steps pass.

Do not add the one-shot example until steps 1-6 are working. Otherwise the validation will not show whether improvement came from better structure or from the example.

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

- Create or update one curated external golden example before deciding whether to embed any example in the live prompt. Add more examples only if the first one is insufficient after A/B testing.
- Update `src/narratives/contract_fixtures.py` after the golden examples clarify expected behavior.

Update `src/narratives/contract_fixtures.py` to reflect:

- `review_metadata`.
- Simplified participant narrative with Completion Outlook, Design Confidence, Main Tension, and two Key Questions.
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
- storage of `review_metadata`, participant summary blocks, `main_tension`, the two-question object, `scenario_consistency_note`, `text_change_evidence`, TA context, and short rationales
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
- key_questions includes only the new two-question target: one medical/clinical-development question and one strategic development question

### Step 7A: Continuity Anchors And Prompt Simplification Target

Before further prompt expansion, shift the next migration toward simpler participant output plus deterministic continuity support.

Decisions:

- Keep rich packet evidence. Do not remove feature-level, pillar-level, variance, score movement, changed-field, operational-assumption, baseline, previous-review, or text-change evidence from packets in the name of simplifying output. This evidence is still needed for Completion Outlook interaction analysis, Design Confidence critical judgment, traceability, and facilitator review.
- Simplify participant-facing output instead. Target: `Completion Outlook` as one concise paragraph plus two key points; `Design Confidence` as one concise paragraph plus two key points; one `main_tension` sentence; two participant questions.
- Preserve detailed internal structure for scoring and audit: four Design Confidence subcategory ratings, evidence fields, score materiality, short rationale, full rationale, consistency note, continuity metadata, and trace fields.
- Add deterministic `design_confidence_continuity` to the review packet. For each Design Confidence subcategory, include previous visible rating, previous visible points, changed relevant fields, and a short continuity instruction.
- Continuity rule: scenario fields are cumulative and Design Confidence is recalculated from the current full scenario state, but unchanged subcategory evidence should preserve prior rating direction unless current packet evidence justifies a reversal. Previous visible review is the main continuity anchor. Hidden baseline remains qualitative original-trial context, not a hard numeric comparator.
- Add eval diagnostics for large unexplained Design Confidence subcategory direction flips when no relevant field changed. The first implementation may flag these in eval reports rather than blocking scoring.
- Keep operational assumptions in Design Confidence. If planning assumptions changed, at least one Design Confidence key point should address operational proportionality unless another current-scenario issue is clearly more material. Planning assumptions still must not explain Completion Outlook movement.
- Move fragile examples out of long prompt prose where possible. Use one curated one-shot/golden example only after the continuity anchors are implemented and A/B tested. The first example should demonstrate an evidence-standard upgrade, later population refinement, operational-only planning update, and structured/text contradiction while preserving endpoint/phase credit when those fields do not change.
- Prefer positive `do` instructions over long negative lists. Keep hard prompt red lines short: no planning assumptions as Completion Outlook drivers; structured fields prevail over conflicting text; Design Confidence uses current-state scoring with continuity anchors; non-neutral ratings cite supported evidence fields; questions are open-ended, non-prescriptive, and tied to the current scenario.

Pre-implementation gap checklist:

- Update the response schema/contract from three questions to two questions. The old `clinical_operations_question` field should either be removed or kept only behind a compatibility alias during migration.
- Update validation, storage, report exporters, Word export, UI rendering, and eval summaries so they do not assume three questions.
- Remove current prompt machinery for clinical-operations question split, question opening-frame examples, direct-address forbidden-word lists, operational-only question special cases, and structured/text contradiction question special cases. The prompt should keep only the two-question goal and freshness/non-prescription rule.
- Remove style micro-rules that have not reliably worked, including long conditional-word lists and `because/however/therefore` instructions. The prompt should say conclusions are hypotheses for discussion; the one-shot example should teach tone.
- Consolidate repeated Completion Outlook operational-boundary prose into positive instructions plus the existing deterministic `review_controls` where needed.
- Add `design_confidence_continuity` before adding the one-shot example, so the example teaches style and edge behavior rather than compensating for missing packet structure.
- Add continuity-flip diagnostics to the eval harness before live validation, so changes like `phase_intent_alignment +4 -> -4` without relevant changed fields are visible in reports.
- Update prompt exports and current generated report pack templates so human reviewers see the new output shape and can compare old versus new behavior.

Verification:

- Packet fixture/checker proves `design_confidence_continuity` is present on later visible iterations and absent or minimal on hidden baseline.
- Prompt export shows the lean participant-facing contract and short hard rules.
- Eval checker flags unexplained subcategory direction flips when no relevant field changed.
- A/B run compares current prompt versus continuity-anchor plus one curated example before embedding any one-shot into the live provider prompt.

Implementation status - 2026-06-15:

- Phase 1 schema and compatibility is complete. `main_tension`, `review_metadata.visible`, and the two-question target are in the provider schema/contract, validation, fixtures, storage, compact continuity context, simulator display, eval-report extraction, review-pack export, and temperature-report comparison. Legacy question aliases and `participant_visible` remain accepted only for migration compatibility.
- Point 3 / Phase 3 prompt simplification has moved beyond the original simplification scope into targeted post-smoke consistency reinforcement. The provider contract uses the simplified visible output shape: `completion_outlook_analysis`, `design_confidence_analysis`, top-level `main_tension`, and two key questions (`medical_clinical_development_question`, `strategic_development_question`).
- App-owned scoring is preserved. The provider still returns ratings, evidence fields, rationale, questions, continuity, and trace fields; the application calculates Design Confidence, Total Scenario Score, and subcategory points.
- The prompt was simplified toward positive instructions. Provider-facing examples now use preferred examples only; risky negative examples were removed from the provider contract.
- Long style-control lists were reduced where they were likely to prime the model with unwanted wording. Model-explanation fields remain available as packet evidence, while visible review wording is steered toward score-input and score-pattern language.
- Added an unplanned but accepted Phase 3 prompt-volume reduction after the main simplification: repeated per-subcategory continuity instruction was collapsed to one continuity-object instruction, structured/text field meanings were compacted, and duplicate prose already present in the response contract was shortened. This was not part of the original point-3 plan, but it supports the same goal without removing feature-level, pillar-level, changed-field, operational-assumption, baseline, previous-review, text-change, scoring, or trace evidence.
- Current prompt-volume checkpoint after that reduction: hidden baseline is about `11.1k` prompt tokens; first-visible cases are about `11.8k-12.0k`; the later-visible continuity inspection case is about `13.4k`, down from about `15.1k`.
- Domain/source vocabulary was not globally removed from dictionaries, packet evidence, or structured field names. The cleanup targets provider instructions, visible examples, visible fixtures, and visible UI labels.
- The old three-question output was reduced to two questions in the schema and simulator display. Legacy aliases remain temporarily in validation/scoring for cached or older reviews.
- `review_metadata.visible` was added while keeping legacy `participant_visible` fallback.
- `main_tension` now propagates through validation, review storage, compact continuity context, and simulator display.
- Visible fixture prose and simulator helper text were cleaned where they used `XGBoost`, `model score`, `model-facing`, or `model movement` wording in places where score-input language is safer.
- Prompt-builder checks now guard against reintroducing risky negative-example terms such as `weak_comment`, `bad_example`, `The score went up and the design is better`, and `change the endpoint and population this way next`.
- Packet-builder checks now guard the storyline review-pack exporter and temperature comparison helper against silently returning to the old three-question visible output.
- Phase 2 continuity anchors are implemented. Later visible review packets now include `iteration_context.design_confidence_continuity` with prior visible subcategory rating, prior visible app-calculated points, current relevant changed fields, previous evidence fields/rationale, and a short continuity instruction for each Design Confidence subcategory. First-visible or no-prior packets mark this object unavailable.
- Eval diagnostics now warn on large unexplained Design Confidence subcategory shifts or sign flips when no current changed field maps to that subcategory.
- After the first Phase 3 live 5x4 run, the continuity relevance map was calibrated as a many-to-many explanation-permission map, not a scoring map. Population fields such as rare-disease status, line of therapy, severity, and biomarker strategy may legitimately explain Operational Burden or Phase/Intent movement as well as Target Population movement. Endpoint/comparator/masking/allocation/adaptive fields may explain Operational Burden movement. Planned Total Timeline, planned enrollment, and planned sites remain Operational Burden context rather than Endpoint Evidence continuity evidence.
- Duration naming decision after the first Phase 3 live 5x4 run: keep `Max Endpoint Duration` for `primary_duration_months_ml`, and call `operational_assumptions.planned_duration_months` `Planned Total Timeline` in provider-facing prompt/report context. This avoids the LLM confusing the endpoint assessment horizon with the operational planning timeline. The internal JSON key remains `planned_duration_months`.
- Post-smoke consistency lock additions are implemented. The prompt now demotes `top_positive_feature_drivers` and `top_negative_feature_drivers` to current-state Completion Outlook support/risk context; latest movement should come from `field_changes`, `score_delta`, changed score inputs, `top_feature_impact_changes`, and pillar/subcategory `xgboost_impact_changes`. `xgboost_impact_changes` is explicitly pillar/subcategory movement context, not field-identity evidence.
- Completion Outlook now has a lightweight storyline lock: compare previous/current score, `score_delta`, changed score inputs, `xgboost_impact_changes`, and prior visible Completion Outlook summary; keep the storyline stable when score is stable and no structured score input changed; reverse the storyline only when score direction and score-input evidence support it.
- Design Confidence now has a continuity/resolution lock: compare prior rating, prior points, prior evidence fields, prior rationale, current relevant changed fields, and current `field_changes`; classify movement as unchanged, prior weakness unresolved/resolved/offset/worsened, new strength, or new weakness before choosing rating/materiality. If a field returns closer to baseline or offsets a prior weakness, the prior penalty should reduce rather than carry forward mechanically.
- Structured/text conflicts now have a stale-conflict rule: unchanged conflicts remain visible as consistency warnings and unresolved prior concerns, but they should not create a new or expanded Design Confidence penalty in later unrelated iterations.
- Completion Outlook and Design Confidence definitions were added to the prompt for final validation. Completion Outlook explains the estimated likelihood that the scenario reaches completion or faces early termination based on previously observed trial patterns; Design Confidence evaluates whether the scenario is coherent, interpretable, patient-relevant, and operationally proportionate. Compact Completion Outlook pillar definitions and Design Confidence subcategory definitions are included.
- Language reinforcement was added with positive replacements rather than a long negative list. Visible wording should use `score pattern reflects`, `Completion Outlook score reflects`, or `current score inputs suggest` for score interpretation, and should frame unresolved concerns as discussion tensions rather than direct redesign instructions. Eval checks now catch concrete leak phrases such as `the model continues`, `model continues`, `must be updated`, and `requires careful re-evaluation`.
- New diagnostics now warn on Completion Outlook direction/storyline inconsistencies and material Design Confidence movement without cited current changed evidence or strong resolution/restoration/offset/worsening reasoning. These remain warnings, not deterministic caps.
- One-trial live smoke `phase3_consistency_lock_live_smoke_1` completed with `0` failed checks, `4/4` visible iterations reviewed, and `1` warning. Manual review found the original consistency issues materially improved: planning-only updates stayed out of Completion Outlook and Endpoint Evidence, Planned Total Timeline no longer drove endpoint scoring, target-population re-credit did not recur, and structured/text contradictions behaved as scenario-readiness warnings.

Prompt export status:

- Prompt exports and Word inspection files have been regenerated for representative fixtures plus a schema-only synthetic later-visible continuity case.
- A compact manual triage summary is now generated alongside the full prompt pack. Human review should start from the compact summary and open full prompt files only for flagged cases or spot checks.
- The synthetic later-visible case is for packet/prompt shape inspection only. It should not be used to judge clinical realism, narrative quality, score behavior, or final validation readiness.
- No new live validation wave should be treated as final until the regenerated prompt exports are inspected with the new continuity anchors.

Verification completed for the partial implementation:

- `python scripts/check_narrative_prompt_builder.py`
- `python scripts/check_narrative_scoring.py`
- `python scripts/check_narrative_contract_fixtures.py`
- `python scripts/check_narrative_mock_reviewer.py`
- `python scripts/check_narrative_packet_builder.py`
- `python scripts/check_narrative_review_store.py`
- `python scripts/check_narrative_live_snapshot_flow.py`
- targeted `python -m py_compile` for changed narrative/UI modules and checkers
- `git diff --check`

Current position and next step:

- Point 3 / Phase 3 should be treated as locally ready for broader live validation. The prompt export inspection and one-trial smoke have served their purpose; do not keep refining wording before seeing broader live behavior.
- Next run the fresh `5`-trial baseline + `4` visible-iteration live validation wave and export the Markdown/Word review pack. Review the results for recurring rather than one-off issues: Completion Outlook/Design Confidence separation, iteration consistency, stale structured/text conflict handling, question freshness, model-language leakage, and prescriptive wording.
- Defer one-shot example embedding until after the 5x4 review. Use examples to teach narrative style, question freshness, and non-prescriptive tone; do not use examples to compensate for missing packet evidence or unresolved prompt contradictions.

### Step 8: UI Integration

Update `frontend/views/trial_simulator.py` only after the provider/mock path is stable:

- Completion Outlook narrative box under Completion Outlook score/plot.
- Design Confidence narrative box under Design Confidence score/plot.
- Key Questions area with one medical/clinical-development question and one strategic development question.
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
- for mixed structured score-input plus planning-assumption changes, Completion Outlook uses changed structured Completion Outlook score inputs and aligned Trial description field context only; planning assumptions remain Design Confidence context through `completion_outlook_mode=structured_score_inputs_only`
- for shortcut simplification, `review_controls.shortcut_design_confidence_rule` calibrates Design Confidence only when multiple core evidence-quality controls are removed together. Production auto-detection is direction-aware and should not apply this shortcut calibration to genuinely stronger evidence-control edits.
- Final checkpoint decision: after `final_validation_boundary_3_4`, do not rerun the boundary wave immediately. The run confirmed shortcut calibration and deterministic stable/operational boundaries. Accepted residual failures are question direct-address style and internal model-language wording. The only remaining substantive mixed structured-plus-planning issue was a proxy phrase (`operational footprint`-style wording), fixed surgically in `structured_score_inputs_only` without broadening the global prompt or changing deterministic overrides.
- Final validation wave `scripts/run_final_narrative_validation_plan.py --execute` completed on 2026-06-15. Reports are:
  - `final_validation_boundary_10_1`: 110/110 reviewed visible iterations, 10 failed checks, 12 warnings.
  - `final_validation_storyline_candidates_12_1`: 48/48 reviewed visible iterations, 5 failed checks, 19 warnings.
  - `final_validation_repro_3_a` and `final_validation_repro_3_b`: each 12/12 reviewed visible iterations, 1 failed check, 3 warnings.
  - `final_validation_repro_3_comparison`: duplicate reproducibility was 12/12 exact iteration matches, 12/12 normalized matches, and 12/12 score matches.
- Current quality readout: boundary controls appear stable at 10-trial scale. Remaining boundary failures are mostly accepted residual categories: participant direct-address question wording and internal model-language wording, plus one operational-only question focus failure. Storyline output is usable for qualitative one-shot selection, but do not select or embed examples before reviewing the full visible storylines by eye.
- Storyline qualitative-review workflow: for each candidate, inspect all 4 visible iterations plus hidden baseline context, exact changed fields, Completion Outlook movement, Design Confidence narrative, subcategory ratings/rationales, questions, and whether the storyline would make sense to a cross-functional team under time pressure. Then classify each candidate as `presentation_ready`, `good_after_light_edit`, `useful_for_stress_test_only`, or `discard`.
- Early candidates to inspect first: `NCT04393298` UCB Oncology (0 fails, 1 warning; strong evidence-standard trade-off), `NCT05590793` Oncology (0 fails, 0 warnings; clean but not UCB), `NCT06315322` UCB Neurology (0 fails, 3 warnings), `NCT04009499` UCB Musculoskeletal (0 fails, 1 warning), and `NCT04643457` UCB Dermatology (0 fails, 2 warnings; more negative/stress-test style).
- Candidates needing caution before one-shot use: `NCT05124925` has a verbatim repeated strategic question, `NCT05076617` has a direct-address strategic question, and `NCT03650400` has an operational-only question-focus failure. These may still be useful for stress testing but should not be selected as presentation examples without review/editing.
- Implemented UI coherence behavior, not a prompt rule: red structured-field flags now highlight validated impossible structured Trial Feature combinations, while keeping the Design Confidence scoring-calibration discussion separate. The UI red-highlights involved controls on field change until the incompatibility is resolved, without amber warnings, compact warning cards, new auto-correction beyond existing placebo sync, or blocking `Review Scenario` / Scenario Review generation. Validated red flags are: parallel with one arm; randomized single-group; randomized with one arm; placebo comparator with no placebo control; placebo control with no control group; placebo control with one arm; single-group with placebo control; active comparator with single-group; active comparator with one arm; double/triple/quadruple blind with single-group/no comparator/no placebo; double/triple/quadruple blind with one arm/no comparator/no placebo; factorial with one arm; crossover with one arm; sequential with one arm. Do not red-flag unusual-but-possible combinations such as Phase 3 single-group, confirmatory single-group, hard clinical endpoint with short endpoint duration, advanced/metastatic plus adjuvant/neoadjuvant, prevention in patients only, healthy-volunteer mismatch, or no DMC in high-risk/pivotal trials in this first pass. Implementation artifacts: `frontend/utils/structured_incompatibility.py`, `frontend/views/trial_simulator.py`, and `scripts/check_structured_incompatibility.py`.
- Implemented Phase 3 consistency reinforcement after live storyline review: provider prompts now demote `top_positive_feature_drivers` and `top_negative_feature_drivers` to current-state Completion Outlook support/risk context; add a Completion Outlook consistency lock around `score_delta`, changed score inputs, `xgboost_impact_changes`, and prior visible storyline; add compact Design Confidence subcategory definitions; add a Design Confidence continuity/resolution lock that recognizes unchanged, unresolved, resolved, offset, worsened, new-strength, and new-weakness effects; and require Design Confidence summary / `main_tension` to synthesize cross-pillar effects. Eval reports now warn on Completion Outlook direction/storyline inconsistencies and on material Design Confidence movement without cited current changed evidence or continuity-resolution reasoning. These are diagnostics, not deterministic caps.
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
