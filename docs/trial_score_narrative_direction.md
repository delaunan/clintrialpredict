# Trial Score Narrative Direction

## Document Role

Architecture scope: `architecture_narratives`

This document is the active planning source for the next narrative/scoring direction. It supersedes the older `Design Confidence / Total Scenario Score` plans and the first `Strategic Review / Trial Score` migration contract in `docs/strategic_review_phase1.md`.

Use this document before changing prompts, provider schemas, scoring code, simulator UI labels, eval scripts, or narrative memory behavior.

## Product Goal

Simplify the participant experience and give the LLM more useful clinical-development judgment while keeping the existing XGBoost score protected.

The participant should ultimately reason about one assessed score:

```text
Trial Score
```

The internal stack may still keep component layers for traceability and facilitator/debug review, but the participant-facing narrative should assess the total score, not repeat separate mini-essays for each component.

## Score Stack

Target direction:

```text
Trial Score = Completion Outlook + Operational Fit + Reality Check
```

Definitions:

- `XGBoost Completion Outlook`: protected model-derived completion / early-termination resemblance score, with existing SHAP and pillar dynamics.
- `Completion Outlook`: development UI component view. After the first iteration, this view should show `XGBoost Completion Outlook + Operational Fit` so the first operational interpretation layer is already included.
- `Operational Fit`: additive operational interpretation under `Execution Framework`, derived from the current XGBoost dynamics plus operational assumptions and scenario coherence.
- `Reality Check`: second interpretation layer where the LLM judges how the scenario evolved and may constructively reinforce, penalize, moderate, or offset the score.
- `Trial Score`: final serious-game score assessed in the participant narrative.

Operational Fit uses a simple app-owned `rating + materiality -> points` mapping recorded in `docs/operational_fit_scoring.md`. Reality Check uses a percentage of the pre-Reality movement, with default `0` when the movement is coherent and realistic.

## Completion Outlook Boundary

`XGBoost Completion Outlook` remains the protected model layer.

It must continue to use:

- existing `/predict` behavior;
- existing XGBoost artifacts;
- existing SHAP decomposition;
- existing therapeutic-area calibration;
- existing four model pillars.

The LLM may interpret XGBoost Completion Outlook dynamics as clinical-development hypotheses, but it must not rewrite model outputs, SHAP values, calibration, or prediction payloads.

Completion Outlook should remain visible during development so users can still understand the historical-pattern anchor. Over time, the participant view may make `Trial Score` primary while moving detailed XGBoost mechanics into drilldown/facilitator context.

The new narrative workflow may interpret total score state and movement, pillar state and movement, subpillar state and movement, feature-level SHAP/evidence, and residual strengths or weaknesses. It must not modify the XGBoost score, SHAP values, model artifacts, therapeutic-area calibration, `/predict` payload or behavior, or the existing four model-derived pillar structure.

## Operational Fit

`Operational Fit` is the first additive interpretation layer.

It should be treated as a separate additive subpillar of `Execution Framework`, not as a separate broad design-quality score and not as a duplicate of XGBoost / SHAP movement.

Detailed Operational Fit logic is recorded in `docs/operational_fit_scoring.md`.

Its role is to review the operational plan's proportionality against the current scenario and model-visible Completion Outlook context. It can use broader model, text, population, endpoint, and benchmark context to interpret whether the operational assumptions are coherent, but the direct Operational Fit score is driven by the combined operational plan.

Relevant inputs include:

- Execution Framework model movements and feature contributions;
- overall Completion Outlook movement;
- changed structured trial features;
- planned enrollment;
- planned site count;
- planned total duration;
- benchmark metadata for those operational assumptions;
- text-context evidence when it affects operational plausibility;
- prior scenario state when continuity matters.

The direct operational inputs are:

- planned enrollment;
- planned site count;
- planned total duration.

At scenario start, `Operational Fit = 0`. The opening operational values are neutral references for the selected trial state, not automatically good or bad.

Pass 1 must distinguish the source of each opening operational value, because moving away from a completed actual has a different interpretation than moving away from a cohort estimate or lower-bound observed value. The source taxonomy should include `completed_actual`, `registered_planned`, `cohort_p50_estimate`, `observed_floor_over_estimate`, and `terminated_observed_floor`.

Site count matters directly, but the most important proportionality lens is often patient-per-site:

```text
patients_per_site = planned_enrollment / planned_site_count
```

Pass 1 should inspect both site count and patients per site. More sites can improve Operational Fit when they reduce patient-per-site burden for an ambitious enrollment target, but can also be excessive or weakly useful when the population is small, the evidence goal is focused, or enrollment/duration assumptions remain incoherent.

Text context may inform Operational Fit proportionality when operational assumptions are being assessed, especially population, endpoint, intervention, and disease-context text. However, text-only changes do not create Operational Fit movement by themselves. Description-only changes route through Reality Check, while Operational Fit remains unchanged unless planned enrollment, planned site count, or planned duration changed.

Operational Fit can add or subtract points in the Trial Score stack when the operational story is meaningfully better or worse than the Completion Outlook alone suggests.

Operational Fit should not be presented as a standalone participant narrative unless needed for facilitator/debug review. In participant-facing language, it should appear inside the total score explanation.

## Reality Check

`Reality Check` is the second interpretation layer.

It assesses the change, the trajectory, and the emerging central discussion topic. The LLM should have more judgment freedom here than in the prior rigid Strategic Review implementation, but the judgment still needs evidence, structure, and guardrails.

Reality Check may:

- reinforce a Trial Score movement when the change is coherent, realistic, and fit for purpose;
- penalize a score gain when it looks one-sided, shortcut-driven, contradictory, unrealistic, or weakly justified;
- penalize increased rigor or robustness when it creates a real issue for operations, credibility, or fit for purpose;
- soften a score decline when the added burden appears necessary, realistic, and scientifically or clinically justified;
- offset part of a movement when a prior gain is reopened or a new unresolved development issue dominates;
- surface a new central discussion topic for the next iteration.

Reality Check should not become another visible component score story. Its purpose is to shape the final Trial Score interpretation and the next debate question.

Reality Check is not a fifth pillar. It is a high-level adjustment layer whose effects can be allocated down to existing pillars/subpillars for traceability and visuals.

### Reality Check Scoring Philosophy

Reality Check starts from:

```text
pre_reality_score = Completion Outlook + Operational Fit
pre_reality_delta = pre_reality_score - previous visible Trial Score or baseline Completion Outlook
```

Default behavior:

```text
If the pre-Reality movement is coherent, realistic, and well explained:
  Reality Check = 0
```

Do not automatically adjust every large move. Reality Check activates only when there is a clear review reason:

- shortcut simplification;
- unrealistic operational or scientific assumptions;
- incoherent answer to the central discussion topic;
- increased rigor that is not operationally credible or fit for purpose;
- negative movement caused by justified robustness or necessary complexity;
- prior development issue resolved, worsened, bypassed, or reopened.

When Reality Check activates, use a percentage of the pre-Reality movement:

```text
slight      20%
moderate    40%
strong      70%
reversal    125-150%
```

Direction is determined by effect:

```text
positive pre-Reality movement:
  reinforce_gain -> positive Reality Check, slight only by default
  offset_gain / reversal -> negative Reality Check

negative pre-Reality movement:
  soften_decline / reversal -> positive Reality Check
  reinforce_decline -> negative Reality Check

near-flat pre-Reality movement:
  reward_coherence or penalize_incoherence can be small, evidence-backed adjustments
```

Reality Check does not automatically follow the pre-Reality movement. For positive pre-Reality movement, Completion Outlook plus Operational Fit already gives the main credit. Therefore positive reinforcement is capped at `slight` by default in V1, and `strong` positive reinforcement is not allowed. Positive movements should usually be accepted with `0` or moderated with an offset when the after-review judgment finds simplification, brittleness, unrealistic assumptions, or unresolved development issue.

Reality Check should be conservative and challenging. It defaults to neutral unless there is a clear incremental reason not already captured by Completion Outlook or Operational Fit. It should be more willing to challenge favorable movements than to soften unfavorable movements.

For negative pre-Reality movement, Reality Check should usually stay neutral unless there is a distinct incremental concern that reinforces the decline. It may soften the decline only rarely, when the decline is materially harsh and the changed scenario adds a concrete compensating strength not already captured elsewhere. Unchanged strengths can provide context, but should not be the main basis for a non-neutral adjustment.

For near-flat pre-Reality movement, Reality Check reward or penalty should normally be `slight` only.

No hard absolute cap is locked for now. Percentages are the first control. If live testing produces unstable swings, add caps later.

Reversal is exceptional. It requires an explicit critical label and rationale that the apparent pre-Reality movement is actively misleading, not merely imperfect.

Examples:

```text
Pre-Reality movement: +6
Reality Check strong offset: -4.2
Final movement: +1.8

Pre-Reality movement: -6
Reality Check strong softening: +4.2
Final movement: -1.8

Pre-Reality movement: +6
Reality Check reversal at 125%: -7.5
Final movement: -1.5
```

There is no separate `full_offset` category in V1. If the concern is material but should not reverse the direction, use `strong`. If the apparent movement is actively misleading and should cross through neutral, use `reversal`.

The participant narrative should explain Reality Check as an after-review scoring judgment about whether the pre-Reality score movement is coherent, realistic, and incrementally supported by the scenario evidence. It should not describe Reality Check as a mechanical component essay, and it should not treat Reality Check as the selector of the participant-visible central discussion topic or broader strategic question. Preferred language should connect the adjustment to the score movement, for example whether a simplification made the profile less robust, whether increased robustness made recruitment or execution riskier, or whether the adjustment changes how the Trial Score movement should be read.

### Pillar And Subpillar Allocation

Reality Check defines one overall app-calculated adjustment first, then allocates that adjustment to existing pillars/subpillars for traceability.

The LLM may choose where the Reality Check lands, but it must target existing pillars/subpillars rather than creating a new visible pillar. V1 should keep this deliberately small and readable: allocate to 1-3 subpillars, not a long list of tiny effects.

Reality Check allocations must use canonical `allocation_target_id` values. The application maps those IDs to exact pillar/subpillar display labels, so the provider should not free-type pillar or subpillar names. For V1, use the current taxonomy subpillars plus the new additive `Operational Fit` subpillar:

| allocation_target_id | Pillar | Subpillar | Included Fields / Inputs | Broader Meaning | Reality Check Allocation Guidance |
| --- | --- | --- | --- | --- | --- |
| `therapeutic_context.therapeutic_area_profile` | Therapeutic Context | Therapeutic Area Profile | Therapeutic area, indication, rare-condition status | Disease context, patient relevance, benchmark context, and calibration limits. | Allocate here only when an after-review issue concerns disease-context fit, rare-condition implications, or stale/inconsistent condition context. Hard-locked premise fields should usually start a new scenario rather than receive Reality Check allocation. |
| `therapeutic_context.development_phase_and_goal` | Therapeutic Context | Development Phase and Goal | Clinical phase, regulatory intent / strategic ambition | Whether the evidence standard, endpoint maturity, population scope, and operational scale fit the development decision. | Allocate here when the issue is phase/ambition coherence, such as a pivotal claim without supporting evidence rigor or an exploratory scenario carrying excessive confirmatory burden. |
| `scientific_challenge.biological_profile` | Scientific Challenge | Biological Profile | Target precedent, pathway profile, therapeutic modality, innovation rank | Biological plausibility, novelty, modality risk, and the evidence burden the mechanism creates. | Allocate here when the after-review concern is about mechanism plausibility, novelty, modality-specific risk, delivery/safety implications, or whether biology supports the evidence claim. |
| `scientific_challenge.protocol_architecture` | Scientific Challenge | Protocol Architecture | Intervention model, primary purpose, adaptive design, endpoint rigor, endpoint structure, biomarker patient selection | Whether the trial design architecture can credibly answer the clinical-development question. | Allocate here when the after-review concern is about endpoint interpretability, comparator/design coherence, biomarker logic, evidence robustness, or simplification that weakens the clinical question. |
| `patient_profile.clinical_severity` | Patient Profile | Clinical Severity | Patient severity, line of therapy | Whether patient burden, acceptable risk, endpoint relevance, and unmet need fit the scenario. | Allocate here when the issue is risk tolerance, disease-course fit, endpoint relevance for the target population, or whether burden is justified by severity/unmet need. |
| `patient_profile.population_scope` | Patient Profile | Population Scope | Gender eligibility, healthy-volunteer flag, adult/pediatric/older-adult eligibility | Whether the population definition is credible, generalizable, ethically coherent, and recruitable. | Allocate here when the issue is population breadth/narrowness, representativeness, vulnerable-population safeguards, or whether eligibility choices support the evidence objective. |
| `execution_framework.trial_complexity_footprint` | Execution Framework | Trial Complexity Footprint | Sponsor type proxy, endpoint duration, number of arms, delivery profile | SHAP-derived trial-footprint complexity, follow-up burden, site capability needs, and operational load. | Allocate here when the after-review issue concerns model-visible complexity, duration burden, arm/site complexity, delivery burden, or footprint credibility not already captured by Operational Fit. |
| `execution_framework.methodological_setup` | Execution Framework | Methodological Setup | Masking, allocation method, DMC status, placebo control, comparator benchmark | Bias control, causal interpretability, governance, comparator credibility, and ethical/methodological setup. | Allocate here when the issue is methodological credibility, governance proportionality, comparator adequacy, placebo ethics, or whether bias control fits the endpoint and population. |
| `execution_framework.operational_fit` | Execution Framework | Operational Fit | Planned enrollment, planned sites, planned total duration, patients per site, operational benchmark metadata | Additive operational proportionality of enrollment, site footprint, duration, and benchmark position relative to the current scenario. | Allocate here when the after-review issue concerns non-XGBoost operational support, patient-per-site burden, duration feasibility, or whether operations support the revised evidence ambition. |

Example:

```json
{
  "reality_check": {
    "effect": "offset_gain",
    "strength": "moderate",
    "correction_fraction": 0.4,
    "central_reason": "The score gain depends on simplification that improves completion resemblance but leaves endpoint interpretability less robust.",
    "allocations": [
      {
        "allocation_target_id": "scientific_challenge.protocol_architecture",
        "share": 0.7,
        "movement_label": "Simplification weakens evidence robustness",
        "rationale": "The endpoint simplification supports completion-like movement, but it may reduce the strength of the evidence claim.",
        "incremental_check": "This is not already counted by Operational Fit because it concerns evidence interpretability, not execution feasibility."
      },
      {
        "allocation_target_id": "execution_framework.operational_fit",
        "share": 0.3,
        "movement_label": "Execution support remains incomplete",
        "rationale": "The revised operational plan improves site burden but does not fully support the higher evidence ambition.",
        "incremental_check": "This is incremental to Completion Outlook because it uses non-model operational assumptions."
      }
    ]
  }
}
```

The app calculates the Reality Check points from effect/strength/fraction and distributes them by validated allocation shares. If provider-defined shares are missing, invalid, or do not sum to `1.0`, the app assigns equal shares deterministically across the valid allocation rows.

Each allocation must include:

- `allocation_target_id`;
- `share`;
- `movement_label`;
- `rationale`;
- `incremental_check`.

Validation should stay simple:

- allocation count is 1-4;
- allocation shares are app-owned when provider shares are missing, invalid, or do not sum to `1.0`;
- `allocation_target_id` targets are valid existing targets;
- `movement_label`, `rationale`, and `incremental_check` are present;
- duplicate same-evidence / same-target / same-consequence allocations are rejected or downgraded.

Provider contract repair should be targeted, not a full re-review. If Pass 1 returns a repairable contract issue, the app may send one short correction prompt with the previous JSON, exact validation errors, canonical allocation IDs, allowed Operational Fit and Reality Check enums, and allowed packet evidence references. The provider must change only the invalid fields. If the second attempt still fails, the trace should report the failed level, such as Operational Fit contract, Reality Check contract, packet evidence references, or Pass 1 Trial Score JSON shape.

The Reality Check radio should answer three questions quickly:

```text
What did the after-review judgment do?
Why?
Where did it land?
```

The barchart should show allocated Reality Check rows under existing pillar/subpillar paths with human-readable labels such as `Reality Check: endpoint robustness`, not provider effect codes.

The Trial Score treemap should show Reality Check as allocated leaves inside impacted existing subpillars. It must not render Reality Check as a fifth top-level pillar.

Participant default view should show one final Trial Score story. Drilldown/facilitator view may show where Reality Check was allocated.

### Anti-Double-Counting Rule

Reality Check should only score the incremental concern or credit not already addressed by Completion Outlook or the Pass 1 Operational Fit assessment.

For V1, use `incremental_check` on each allocation rather than a large anti-double-counting ontology. Pass 1 should briefly explain why the allocation is not merely repeating Completion Outlook or Operational Fit. The app should reject or downgrade allocations that repeat the same evidence, same target, and same consequence as an already-scored component.

Related evidence can still be valid when it creates a different consequence. For example, Operational Fit may penalize execution feasibility, while Reality Check may separately allocate to Scientific Challenge if the operational compression undermines endpoint interpretability.

### Premise Shift And Locked Fields

Therapeutic area, indication, and other premise-defining fields can reset the meaning of a scenario. Changing them may transform the scenario into a new trial premise rather than a normal iteration.

V1 should hard-lock the fields that most clearly change the whole scenario context inside a scenario thread:

```text
therapeutic_area_ml
gbd_cause_id_3_ml
sponsor_tier_ml
```

These changes should be blocked or treated as starting a new scenario, because they alter disease domain, indication/disease category, sponsor premise, benchmarks, calibration context, and continuity. The underlying lead sponsor identity (`lead_sponsor_canonical`) is trial identity context and is not exposed as an editable Trial Features field.

V1 should gate the following premise-sensitive fields rather than hard-lock them:

```text
phase_ml
strategic_ambition_ml
therapeutic_modality_ml
target_pathway_class_ml
primary_purpose_ml
```

If any gated field changes, Pass 1 must run a simple Strategy Shift Check:

```text
supported
partly_supported
unsupported_or_incoherent
```

There is no automatic score penalty for changing a gated field. The change affects scoring only through normal XGBoost Completion Outlook movement and through Reality Check if the strategy shift is not adequately supported.

Use `supported` when the changed strategy is coherently supported by endpoint, comparator, operational, governance, population, and text context as relevant. Use `partly_supported` when the direction is plausible but incomplete. Use `unsupported_or_incoherent` only for direct contradiction or multiple missing core supports.

If the Strategy Shift Check is `unsupported_or_incoherent`:

- Reality Check cannot be positive;
- positive pre-Reality movement can only be accepted with `0` or offset downward;
- negative pre-Reality movement cannot be softened;
- material incoherence should produce a challenging Reality Check narrative.

Structured categorical fields prevail over unchanged or stale descriptive text. If text was not edited in the same change, treat mismatch as stale context first. Stale text alone should not create a major penalty unless it materially undermines interpretation.

### Description-Only Changes

Description-only changes do not affect XGBoost Completion Outlook or Operational Fit. If structured categorical/model fields and operational assumptions are unchanged, the XGBoost score, SHAP values, model pillars, and Operational Fit points remain unchanged.

Description-only changes may affect Reality Check only when they materially clarify, support, challenge, or contradict the structured scenario. Positive Reality Check from description-only changes is capped at `slight` by default because the structured/model scenario did not change. Negative Reality Check may be `moderate` or `strong` only for material contradiction, unsupported strategic claim, or text that materially undermines interpretation.

The participant narrative must explicitly distinguish unchanged structured/model score from changed interpretive context. For example, it should state that Completion Outlook is unchanged because structured score inputs did not change, then explain how the revised description clarifies or challenges the interpretation of the same categorical profile.

## Participant Narrative

The visible narrative should assess the `Trial Score` only.

It should not separately narrate:

- one Completion Outlook essay;
- one Operational Fit essay;
- one Reality Check essay;
- repeated component-by-component rationales.

Preferred shape:

1. State the Trial Score reading and direction.
2. Explain the main reason in clinical-development language.
3. Explain current state, movement versus previous/baseline, and why the movement matters.
4. Use pillar-level bullets as the main reading structure.
5. Within each pillar bullet, use relevant subpillar and feature evidence in prose rather than nested subpillar/feature bullet lists.
6. Fold in Completion Outlook, Operational Fit, and Reality Check evidence only where those components explain the Trial Score.
7. Surface one central discussion topic.
8. End with one broader strategic question that helps orient the next scenario discussion.

The narrative should be concise, conditional, and constructive. It should challenge the scenario without prescribing exact field edits.

The broader strategic question may mention adjustable feature families at a high level, such as eligibility breadth, endpoint ambition, comparator/control strategy, enrollment target, site footprint, follow-up duration, population focus, or design complexity. It should not prescribe exact field values or direct optimization instructions.

Optional hidden discussion prompts are deferred out of the main Pass 2 participant-narrative flow. If reintroduced, they should be generated separately after the participant narrative succeeds.

Avoid:

- direct optimization instructions;
- clinical certainty beyond packet evidence;
- repeated score mechanics;
- repeated questions across iterations;
- hidden-baseline numeric leakage.

## LLM Freedom And Guardrails

The LLM should have more room to judge the scenario than in the prior app-owned categorical Strategic Review mapping.

Allowed:

- identify the most important central discussion topic;
- weigh tradeoffs across score movement, operational plausibility, evidence value, population logic, and execution burden;
- decide which existing pillar/subpillar or concern deserves a Reality Check allocation;
- frame a constructive offset or reinforcement;
- introduce a forward-looking broader question.

Required guardrails:

- no modification of XGBoost, SHAP, therapeutic-area calibration, or prediction artifacts;
- no unsupported clinical, regulatory, efficacy, safety, recruitment, or budget certainty;
- no provider-owned hidden baseline score shown to participants;
- no direct instructions telling the participant which field to change next;
- structured evidence references for any material positive or negative adjustment;
- one central discussion topic by default.

## Two-Pass Narrative Architecture

Locked direction: use two LLM passes with deterministic app scoring between them.

The application cannot insert app-calculated scores into a single running LLM response. If the final participant narrative needs exact Operational Fit, Reality Check, and Trial Score values, scoring must happen between provider calls.

Recommended flow:

```text
Pass 1: Analytical Review
App scoring
Pass 2: Participant Narrative
```

### Pass 1: Analytical Review

Pass 1 receives the full scenario packet and prior continuity context. It returns structured, auditable judgments rather than a polished participant narrative.

V1 top-level output sections:

```text
review_metadata
completion_outlook_analysis
strategy_shift_check
operational_fit
reality_check
development_discussion_options
continuity_update
```

Pass 1 should produce:

- XGBoost / Completion Outlook interpretation;
- total SHAP / score movement, pillar movement, subpillar movement, feature-level SHAP movement, and finest-level changed feature evidence;
- raw current state, baseline state, previous state, and residual state for total score, pillars, subpillars, features, and operational assumptions where available;
- feature, subpillar, and pillar reading with definitions and values;
- consistency update versus prior XGBoost interpretation;
- Operational Fit field ratings for enrollment, site footprint, and total duration;
- combined Operational Fit rating and materiality;
- Reality Check effect, strength, allocation targets, and anti-double-counting rationale;
- three complete development discussion options for Pass 2 selection;
- continuity update for later iterations.

`strategy_shift_check` is required when gated premise-sensitive fields changed. Allowed status values are `supported`, `partly_supported`, `unsupported_or_incoherent`, and `not_applicable`.

`operational_fit` should include field-level analysis for enrollment, site footprint / patients per site, and timeline, plus one `combined_operational_fit` object. The app scores only the combined rating/materiality.

`reality_check` should include effect, strength, central reason, and 1-4 allocations. It should not include app-owned Reality Check points. `central_reason` explains the scoring adjustment only; it is not the selected participant discussion point.

`development_discussion_options` is the canonical visible-iteration Pass 1 structure. Hidden baseline should not return this field. Visible iterations should contain two or three complete options, with no main/alternative split. Each option pairs one development discussion topic with one participant-visible wider question assigned to that exact topic.

Each option should use this shape:

```json
{
  "topic": "...",
  "why_it_matters": "...",
  "supporting_evidence": ["..."],
  "participant_wider_question": {
    "question": "...",
    "supporting_evidence": ["..."]
  }
}
```

Pass 1 should focus on analytical substance for each option: the development issue, why it matters, the trial evidence behind it, and final participant-visible wider question text that Pass 2 can select verbatim.

The validated contract does not emit old main/alternative Pass 1 candidate fields. Storage, debug context, and Pass 2 input should carry `development_discussion_options` directly, plus the Pass 2 selected participant-facing `central_tension` and `broader_strategic_question` once available.

The participant-visible `broader_strategic_question` selected by Pass 2 should be stored and passed forward with the participant-visible central discussion topic in `recent_participant_visible_questions`. Later Pass 2 calls should use this history to preserve continuity while avoiding unnecessary repetition: prefer a different wider debate question when the scenario supports a different discussion topic, but reuse or closely echo the previous question when same-state reuse or a clear return to a prior state calls for consistency.

Facilitator questions are no longer part of the Pass 2 participant-narrative contract. If needed later, they should be generated as a separate optional/lazy step after the participant narrative has succeeded, using final score context and the selected development discussion without slowing or destabilizing the main review flow.

`continuity_update` should be compact and carry what changed since the previous visible iteration and what to watch next. Hidden baseline should not define an active participant-visible discussion topic. Visible participant-facing continuity is stored from the Pass 2 selected central development discussion pair, not from hidden baseline.

For visible iterations, Pass 1 proposes two or three analytical development discussion options. Pass 2 selects one supplied pair using priority 1 state reuse/direct continuity, priority 2 latest material change, and priority 3 history diversity. Pass 2 should not introduce a new analytical basis that was absent from the validated Pass 1 options. Provider validation enforces that the selected participant-visible central discussion topic and wider question match one of the supplied Pass 1 option pairs. Repetition avoidance is handled at the prompt-selection level: Pass 2 compares candidate discussion topics against recent participant-visible history. Same-state reuse or direct storyline continuity may keep the same topic. Otherwise, Pass 2 should prefer a supplied option anchored in a newly changed material issue when available, and use history diversity to avoid repeating the same development issue or wider-question framing among similarly relevant options. A prior issue that remains unresolved can stay visible in the Trial Score narrative or Reality Check when material without automatically becoming the selected `Discussion Point`.

Pass 1 should not write the final participant-facing score narrative, because it does not yet know the app-calculated Operational Fit points, Reality Check points, or final Trial Score.

### App Scoring Between Passes

The application validates Pass 1 and calculates:

- Operational Fit points;
- pre-Reality score and pre-Reality movement;
- Reality Check points and allocations to existing pillars/subpillars;
- updated pillar and subpillar contributions;
- final Trial Score;
- score deltas versus previous visible iteration;
- compact continuity state.

The app owns numeric scoring. The LLM provides ratings, materiality, target pillars/subpillars, evidence references, and rationale.

V1 validation/scoring order:

```text
1. Build Pass 1 packet.
2. Run Pass 1.
3. Validate Pass 1 schema.
4. Validate Strategy Shift Check.
5. Score Operational Fit.
6. Build pre-Reality score and movement.
7. Validate Reality Check effect against actual pre-Reality movement.
8. Score Reality Check.
9. Allocate Reality Check.
10. Calculate Trial Score.
11. Build Pass 2 packet.
12. Run Pass 2.
13. Validate Pass 2 schema/prose.
14. Store final trace.
```

Reality Check is scored only after Operational Fit points are calculated:

```text
pre_reality_score = XGBoost Completion Outlook + Operational Fit
pre_reality_delta = pre_reality_score - previous visible Trial Score
Trial Score = pre_reality_score + Reality Check
```

For the first visible iteration, compare `pre_reality_score` against baseline XGBoost Completion Outlook when no previous visible Trial Score exists.

Reality Check effect must be compatible with the actual `pre_reality_delta` after Operational Fit scoring. If Pass 1 returns an incompatible effect, downgrade Reality Check to neutral in V1 unless the mapping is trivial and explicitly safe. This protects against Pass 1 proposing `offset_gain` before app scoring later makes the actual pre-Reality movement negative.

If Pass 1 or app scoring validation fails, do not show a Trial Score narrative. Preserve XGBoost Completion Outlook and show review unavailable / needs rerun behavior instead.

Store diagnostics for transparency:

```text
xgboost_completion_outlook
operational_fit_points
pre_reality_score
pre_reality_delta
reality_check_points
trial_score
delta_vs_previous_trial_score
delta_vs_previous_pre_reality_score
delta_vs_baseline_xgboost
validation_notes
```

### Pass 2: Participant Narrative

Pass 2 receives:

- exact app-calculated score stack;
- Pass 1 structured analysis;
- previous visible iteration context;
- compact continuity state;
- participant-facing guardrails.

V1 top-level output sections:

```text
review_metadata
trial_score_narrative
pillar_reading
central_tension
broader_strategic_question
```

Pass 2 writes:

- one integrated Trial Score narrative explaining state, movement, and interpretation;
- pillar-level bullets that use subpillar and feature evidence inside prose rather than nested bullet lists;
- one central discussion topic;
- one broader strategic question;
- no separate Completion Outlook essay, Operational Fit essay, and Reality Check essay.

`trial_score_narrative` should briefly explain the current Trial Score reading, movement versus previous/baseline, main reason for movement, and how Completion Outlook, Operational Fit, and Reality Check affect the reading when relevant.

`pillar_reading` is the main UI reading structure. It should include one concise reading per pillar. Subpillar and feature evidence should appear inside the prose, not as nested subpillar/feature bullet lists.

`central_tension` should be the final participant-facing discussion topic selected from the visible Pass 1 `development_discussion_options`.

`broader_strategic_question` should be reflective and debate-oriented, but still contextualized to the scenario, condition, population, evidence goal, operational setting, or trial context. It should not be a generic conference question detached from the current scenario, and it should not become a direct instruction to change a specific field value.

Pass 2 should use the validated score explanation from Pass 1 and app scoring. It should not make new scoring decisions, invent a supplied discussion topic outside the supplied options, or introduce a new analytical basis.

Pass 2 repair is separate from Pass 1 repair. If Pass 2 returns invalid participant narrative JSON after Pass 1 and app scoring succeeded, the app may send one targeted Pass 2 correction prompt with the same Pass 2 input, previous Pass 2 JSON, and exact Pass 2 validation errors. The provider must repair only invalid or missing participant-narrative fields and must not rerun Pass 1, change app-calculated scores, or change the analytical basis. If the second Pass 2 attempt still fails, Trial Score remains visible and the trace/UI should show a participant narrative warning instead of failing the scenario score.

This preserves exact scoring, continuity, and participant-readable narrative quality without forcing an extra LLM call on valid Pass 2 responses.

## Development UI Contract

V1 UI should keep the three radio buttons for development and for the current product version:

```text
Completion Outlook
Reality Check
Trial Score
```

For now, these views mean:

- `Completion Outlook`: show `XGBoost Completion Outlook + Operational Fit`. Keep the familiar model pillar rows, and add `Operational Fit` only under `Execution Framework`.
- `Reality Check`: replace the current `Strategic Review` radio behavior with Reality Check. Show the adjustment, short rationale, central discussion topic, and allocation trace.
- `Trial Score`: show `Completion Outlook + Reality Check`. In this view, expose Reality Check as allocated subpillar leaves inside impacted existing subpillars, for example `Scientific Challenge / Protocol Architecture / Reality Check: endpoint robustness` or `Execution Framework / Operational Fit / Reality Check: operational support`. Do not show Reality Check as a fifth pillar.

Future participant-facing UI may eventually show only `Trial Score`, but that is not the V1 target. During implementation and current product use, keep all three radio buttons visible so scoring behavior can be inspected.

The final participant narrative should assess the total Trial Score, not write separate essays for Completion Outlook, Operational Fit, and Reality Check.

Reality Check allocations should appear within existing subpillars, not directly under pillars. In the Reality Check radio, the treemap may use `Reality Check` as the root view, but the visible allocation path should still be existing pillar -> existing subpillar -> Reality Check leaf. In the Trial Score radio, the full composition treemap should embed Reality Check leaves inside impacted existing subpillars. Operational Fit and Reality Check must not render as top-level fifth pillars.

## Migration Plan

Implementation status: V1 contract/schema constants, deterministic Operational Fit and Reality Check scoring, registry-owned Reality Check allocation IDs, active Pass 1 provider prompt/schema, targeted Pass 1 repair retry, Pass 2 participant-narrative prompt/schema, targeted Pass 2 repair retry, real-provider Pass 2 routing, mock-provider adaptation, storage trace fields, facilitator-question collapsed rendering, and initial simulator labels/rendering are implemented in `src/narratives/trial_score_contract.py` and the adjacent narrative modules. The active prompt builder has been simplified to the V1 Trial Score contract only. Remaining work should continue from those files rather than recreating a separate schema plan.

Current implementation also passes `operational_movement_context` into Pass 1 so the provider can compare current enrollment, site count, calculated patients per site, and duration against neutral baseline assumptions and residual cohort percentiles. The prompt explicitly separates movement from baseline, residual percentile/status, and benchmark-context changes caused by cohort-defining field edits. Percentiles may counterbalance movement, and distance from P50 alone must not drive Operational Fit.

Current implementation also passes compact model state and movement evidence into Pass 1. Model state is the fixed snapshot of signed model forces: positive impacts are favorable by definition and negative impacts are unfavorable by definition. Model movement is the delta from baseline and/or previous visible iteration, including whether a pillar, subpillar, or direct feature contribution crossed zero, improved while still negative, worsened while still positive, or reversed sign. Visible-iteration movement ranking is previous-first: latest delta from the immediately prior iteration determines top positive/negative movement when available, while baseline delta remains context; baseline delta is used for ranking only when no previous iteration exists. Feature-level evidence is sourced only from direct XGBoost-backed `feature_level_impacts` exported by the decomposition helper and is capped to the top three positive and top three negative rows for prompt use. Therapeutic-area threshold offsets, residual/clipping adjustments, unmapped internal factors, and non-model registry fields are excluded from feature-level prompt evidence. Baseline reviews should use state evidence even when movement evidence is empty. Pass 2 receives the same compact model evidence as narrative context only, so it can explain the validated analysis without recalculating Completion Outlook.

The packet includes `model_interpretation.model_signal_guidance` so the provider has explicit rules for `completion_outlook_analysis.main_model_signals`. Baseline signals should be state-only. Visible iteration signals should prioritize movement first, then current-state anchors. Signal wording should prefer feature label/value with parent subpillar and pillar, then subpillar, then pillar-only fallback; generic pillar slogans should be avoided.

Hidden-baseline compaction now preserves the baseline Completion Outlook summary when available and carries compact baseline orientation/watch context into the first visible prompt. Hidden baseline output remains qualitative context only: no hidden Trial Score, hidden Operational Fit points, hidden Reality Check points, participant-visible questions, or active discussion topic should be treated as prior visible history.

Pass 1 should explicitly act as a clinical development, trial design, regulatory strategy, and clinical operations expert. Its goal is to review the evidence package, summarize the current design logic, observe scenario dynamics across iterations, and identify weak assumptions or development issues. Pass 1 should prioritize rich analytical material over participant-facing wording; Pass 2 owns final phrasing and participant-facing style constraints.

Hidden-baseline analysis should be deep enough to anchor the later visible storyline. It should use the trial text, structured fields, model evidence, operational context, therapeutic-area context, relevant reference-pack summaries, and general clinical-development expertise to explain the actual development problem: population, intervention, endpoints, follow-up windows, oversight needs, scientific purpose, feasibility, and what development decision the evidence package could credibly support. Reference packs can support clinical, regulatory, or development interpretation, but the provider must not imply a document supports a claim unless the pack actually provides that support; if no reference pack is relevant, Pass 1 should rely on packet evidence and expert interpretation. The Pass 1 `analytical_narrative_draft` should read like a substantive source note for the future storyline, not a short score recap: it should name concrete trial facts, endpoint/follow-up logic, safety or monitoring burden, evidence ambition, similar-trial operational pattern, and the decision the baseline evidence can or cannot support. Across hidden and visible reviews, Pass 1 should use packet evidence to cover the most relevant supported dimensions: population/setting/clinical context, endpoint interpretability, safety governance, comparator or standard-of-care context, development decision supported, evidence completeness risk, and program-level meaning. When immune markers, disease-control measures, clinically confirmed events, long follow-up, vulnerable populations, or special settings are present, Pass 1 should explain why they matter for interpreting safety, response, feasibility, generalizability, or confidence in the next development step. Participant-facing and draft narrative wording should describe operational context as similar-trial patterns or comparable studies, not as benchmark data. Underlying operational benchmark metadata remains available as structured context and audit evidence, but the narrative should translate it into clinical-development language. Validation enforces a minimum total draft depth before Pass 2 receives the draft: at least 320 words for visible reviews and at least 450 words for hidden baseline.

Pass 1 routes evidence by section to avoid prompt overload. `completion_outlook_analysis` should use XGBoost score, pillar/subpillar/feature impacts, model movement, current-state drivers, and changed structured features. `operational_fit` should use planned enrollment, planned sites, planned duration, patients per site, operational movement context, similar-trial operational context, and trial text only when it directly affects feasibility; non-operational edits should not drive Operational Fit scoring unless they directly change enrollment, sites, or duration assumptions. `reality_check` should use pre-Reality movement direction, changed fields, model movement, Operational Fit result, trial text, relevant reference packs, and general expert judgment to assess coherence, realism, shortcut risk, under-support, or justified rigor. If the packet contains an active prior negative Reality Check carryover candidate, Pass 1 should also return `reality_check_carryover_assessment` with `status` (`still_relevant`, `partly_mitigated`, or `resolved_or_superseded`) and `current_issue_relation` (`same_issue`, `new_independent_issue`, or `mixed_or_unclear`). This field decides whether the previous concern remains active after the latest scenario change and whether the latest Reality Check concern is distinct enough to cumulate. `strategy_shift_check` should use gated premise-sensitive changed fields and the development premise. `development_discussion_options` should use material changes, Pass 1 interpretation, score-aligned movement, participant-visible history, and the strongest unresolved development trade-offs.

Reality Check carryover is app-scored, not provider-scored. The app creates a carryover candidate only from the immediately previous visible review when its Reality Check was materially negative. If Pass 1 marks the carryover `still_relevant`, the previous negative adjustment remains as a floor. If it marks `partly_mitigated`, half of the previous negative adjustment remains. If it marks `resolved_or_superseded`, the carryover is released. If the current Reality Check is negative and Pass 1 marks the latest issue as `new_independent_issue`, the app can cumulate the current negative adjustment with the active carryover under a bounded cap. If the issue is the same or unclear, the app uses the more negative of the current adjustment and the carryover floor rather than adding both. This prevents unresolved serious concerns from disappearing while avoiding repeated penalties for the same issue.

Gated premise-sensitive changes do not automatically clear carryover. They are evidence that may support `resolved_or_superseded` if the development premise changed enough to make the previous concern irrelevant. Exact same-state reuse remains separate and higher priority: when the canonical scenario state matches a prior visible review, the app reuses that prior score trace rather than recalculating the carryover.

Hidden baseline should keep baseline development issues inside `analytical_narrative_draft.development_landscape_read` and should not return `development_discussion_options`. Visible Pass 1 should return two or three `development_discussion_options`. They let later iterations continue a prior participant-visible storyline when direct continuity supports it, or shift to another discussion topic when the scenario dynamic changes. Pass 1 should include at least one option anchored in a newly changed material issue when available, while prior unresolved issues can remain visible in Reality Check or the analytical draft. Each option includes its own wider-perspective strategic question so Pass 2 can select and shape a coherent pair without remapping questions after the fact. Topics should be concise title-style labels for display and should prefer analytically specific evidence trade-offs, such as long-term safety confidence versus evidence completeness, over short operational labels when packet evidence supports the richer framing.

Provider repair prompts and Pass 1 validation now use one shared recursive packet-evidence reference helper from the Trial Score contract. This allows provider citations to point to deep movement evidence, such as patients-per-site benchmark position or movement relative to P50 inside `operational_movement_context`, without failing evidence validation.

Current contract refinement: the two-pass role split is asymmetric.

- Pass 1 is the full analyst and rough narrative drafter. It keeps structured outputs for Completion Outlook, Operational Fit, Reality Check, strategy shift, visible-iteration `development_discussion_options`, and continuity, plus `analytical_narrative_draft`.
- The Pass 1 draft is provisional and intentionally richer than the final output. It may explain current model state, model movement, Operational Fit reasoning, pre-Reality direction, Reality Check reasoning, the development landscape, relevant reference-pack implications, and score-aware implications used in the analysis. This draft is not participant-facing, so the final-output score-language rule does not apply here.
- The implemented shape is:

```json
"analytical_narrative_draft": {
  "current_state_read": "...",
  "movement_read": "...",
  "operational_fit_read": "...",
  "reality_check_read": "...",
  "development_landscape_read": "..."
}
```

- Hidden baseline produces the same draft object as qualitative context, and draft fields must be non-empty even though the review remains hidden, `visible=false`, and score-free. The compacted hidden baseline preserves only useful qualitative development discussion summary context, not hidden numeric component scores.
- After Pass 1 validation, the application calculates app-owned scores and adds `score_alignment_notes`. These notes may include internal numeric values for calibration, but they also expose participant-safe labels such as `trial_score_direction`, `required_direction_phrase`, `pre_reality_direction`, `operational_fit_importance`, `operational_fit_wording_instruction`, `reality_check_importance`, `wording_calibration`, and `conflicts`.
- Participant-facing text should prefer direction and importance over numeric points or exact Trial Score values. Pass 2 may use internal values to calibrate wording. This is prompt/style guidance, not a validation blocker.
- Pass 2 is an editor/formatter, not a second analyst. It receives the Pass 1 draft, validated Pass 1 structured analysis, app-calculated scores, score-alignment notes, trajectory/reuse context, and compact model evidence. Its task is to restructure into the final participant-facing sections, align wording with score direction/materiality, remove repetition, and preserve the Pass 1 analytical conclusion.
- The Pass 2 final participant format is:
- `Trial Score`: combines three labeled subparagraphs into one participant read: `trial_score_narrative.summary` as `Overall Evolution`, `movement_reading` as `Completion Outlook`, and `score_interpretation` as `Reality Check`. It must reflect `required_direction_phrase` without exposing exact score values, and the three subparagraphs should not repeat the same central sentence. The `Completion Outlook` paragraph is the pre-Reality read: model-visible completion-likelihood movement plus app-rated execution scale, footprint, duration, size, or operational dimensions when material.
- `What Is Driving The Score`: renders `pillar_reading` as two to four material bullets only. A bullet may cover one pillar or combine related pillars/subpillars. Pass 2 should not mechanically list every pillar, and each bullet should add a distinct driver or evidence angle rather than restating the Trial Score paragraphs.
  - `Discussion Point: <topic>`: renders the selected concise `central_tension.summary` as the section title, followed by `central_tension.why_it_matters` and the paired participant-visible wider `broader_strategic_question.question`.
  - `central_tension.summary`: a stable title-like key for validation/history, not necessarily the full display sentence.
- Operational Fit is app-owned score context. Participant prose presents app-rated operational evidence inside the relevant score driver, usually Execution Framework, using plain wording such as right scale, footprint, duration, size, or operational dimensions. If `operational_fit_importance` is `none`, Pass 2 must keep the operational scale, footprint, duration, size, or fit neutral and may only describe a non-scored execution-burden implication when supported.
- Pass 2 should use `pass1_draft` and `pass1_analysis` as the source for richer interpretation. When Pass 1 contains trial-specific detail, the final narrative should stay concise while carrying forward the most relevant rationale, illustration, or clinical-development implication instead of reducing the read to generic score movement language. Elaboration should appear only where it helps explain the current score movement, Reality Check, or selected discussion topic, and wording should remain conditional.
- Reality Check should appear in the final participant narrative only when it is material, conflict-relevant, or interpretation-changing. When mentioned, describe it as a realism/coherence qualifier, not as points or a standalone component essay.
- Pass 2 should use cautious hypothesis language throughout, such as "may", "might", "could", "appears", "suggests", and "would need support", because the narrative interprets scenario evidence rather than asserting external truth.
- Pass 2 must not re-rate Operational Fit, re-decide Reality Check, reinterpret model movement, introduce new clinical/regulatory claims, or change the central analytical conclusion except to soften/align wording when app-generated alignment notes identify overstatement or inconsistency.
- FDA/EMA/reference packs should primarily enrich Pass 1. Pass 2 should receive reference implications through the Pass 1 draft and validated analysis rather than re-reading or reinterpreting full reference packs by default.

Validation and repair implementation:

- Pass 1 schema is `trial_score_pass1_schema_v3`, Pass 2 schema is `trial_score_pass2_schema_v2`, and prompt template is `trial_score_two_pass_prompt_v1_8`.
- Pass 1 validation checks draft shape and non-empty string fields for hidden baseline and visible reviews. Semantic validation remains light; structured ratings remain the scoring source of truth. Numeric wording inside the draft is allowed so Pass 2 can edit it rather than failing the run.
- Targeted Pass 1 repair prompts can repair missing/malformed draft fields without changing valid Operational Fit, Reality Check, strategy-shift, or evidence-field content.
- App-owned `score_alignment_notes` are generated after scoring. These translate app points into qualitative direction/materiality and identify wording conflicts such as capped Operational Fit, neutral Reality Check despite concern language, slight movement being described as major, or same-state reuse requiring reversion language.
- Pass 2 input includes `pass1_draft` and `score_alignment_notes`.
- Pass 2 prompt and repair prompts say: edit the Pass 1 draft, do not reanalyze, do not introduce new unsupported claims, app-calculated direction/materiality overrides draft intensity, use `required_direction_phrase`, prefer qualitative direction/materiality over numeric score/point language in the final participant-facing prose, return two to four material pillar-reading bullets, express material app-rated operational evidence inside the relevant pillar/subpillar using scale, footprint, duration, size, or operational-dimension language, and mention Reality Check only when it materially changes the interpretation.
- Pass 2 validation mirrors the response schema shape and rejects returned app-owned score fields. It does not reject numeric prose; exact score wording is handled as prompt/style guidance.
- Live simulator debug output shows `pass2_editor_input_debug_summary` with `app_calculated_scores_shared_with_pass2`, Pass 1 draft, selected Pass 1 analytical basis, score-alignment notes, model evidence, and participant guardrails. This is a compact summary, not the full raw Pass 2 input. Full raw Pass 2 input remains available in the audit bundle. Exact app scores are part of Pass 2 input calibration, and the score-language restriction applies only to final participant-facing prose.
- Keep existing bounded retry counts: Pass 1 uses the existing malformed JSON and targeted validation repair path; Pass 2 uses the existing one-shot participant-narrative repair path. Do not add a third LLM pass unless a later live-provider audit proves the editor-only flow is insufficient.

1. Freeze the current uncommitted narrative/scoring implementation until the new contract is specified.
2. Use this document as the active source of truth.
3. Use `docs/operational_fit_scoring.md` as the detailed Operational Fit scoring source of truth.
4. Before reusing any existing code, inspect it against this contract and keep only contract-compatible behavior.
5. Reuse generic reviewed-snapshot and consistency-warning plumbing only when it matches the new contract.
6. Preserve structured/text consistency behavior: structured categorical fields prevail over unchanged or stale descriptive text, with warnings before penalties.
7. Rewrite obsolete `Strategic Review`, `Design Confidence`, `Total Scenario Score`, and visible `Quality Review` concepts rather than carrying them forward as implementation names.
8. Update old operational-only behavior: operational assumptions remain outside XGBoost, but they now contribute to displayed `Completion Outlook` through Operational Fit.
9. Define the new provider schema around:
   - total-score assessment;
   - operational-fit assessment;
   - reality-check judgment;
   - central discussion topic;
   - broader strategic question;
   - evidence references;
   - optional facilitator/debug trace fields.
10. Rework deterministic scoring tests around the new score stack before changing the UI.
11. Add consistency/field-change checks for hard-locked fields, gated Strategy Shift Check behavior, and structured-text conflict warnings.
12. Update packet builder and mock/provider outputs after deterministic scoring checks pass.
13. Update simulator UI labels and narrative rendering after the schema/scoring contract is stable.
14. Rebuild live evals only after the new prompt shape is testable on one or two trials.

Implementation should proceed autonomously in small slices with a bounded audit after each slice before moving on. Each audit should compare the current diff against this document and `docs/operational_fit_scoring.md`, identify gaps, apply targeted fixes, and repeat until there are no material findings for that slice.

High-risk implementation points and required mitigations:

- **Old contract leakage**: active new code must not emit visible `Strategic Review`, `Design Confidence`, `Quality Review`, or `Total Scenario Score` fields. Old aliases are allowed only in clearly marked legacy-cache adapters. Add a checker that fails if new scoring/provider outputs use obsolete visible fields.
- **Reality Check movement mismatch**: score Operational Fit first, calculate actual `pre_reality_delta`, then validate Reality Check effect compatibility. Incompatible effects downgrade to neutral in V1 unless the mapping is trivial and explicitly safe.
- **Operational Fit over-crediting**: score only combined Operational Fit; enforce materiality guardrails from `docs/operational_fit_scoring.md`; require coherent support across at least two operational fields for a positive `+5.0`.
- **Noisy Reality Check allocations**: require 1-4 allocations, canonical `allocation_target_id` values, and present `movement_label`, `rationale`, and `incremental_check`. If provider shares are missing, invalid, or do not sum to `1.0`, assign equal app-owned shares rather than neutralizing an otherwise valid Reality Check. Do not invent fallback subpillars.
- **Over-punitive Strategy Shift Check**: use `unsupported_or_incoherent` only for direct contradiction or multiple missing core supports. Use `partly_supported` for plausible but incomplete shifts.
- **Structured/text conflict over-penalty**: structured categorical fields prevail, but unchanged or stale text should create warning/context first. Penalize only when the conflict materially undermines interpretation.

Recommended slice order:

```text
1. Contract/schema constants and old-name boundary cleanup.
2. Deterministic scoring and validation checkers.
3. Packet builder and prompt/provider schemas.
4. Mock provider and storage trace updates.
5. Simulator UI rendering and score-view data.
6. Narrow command verification and manual-test scenario handoff.
```

Do not run Playwright/browser automation for this migration by default. If visual or interaction confidence requires browser validation, stop and provide a concise manual scenario for the user to test instead.

## Obsolete Direction

The following directions are superseded:

- `Quality Review / Quality Adjustment / Final Candidate Score`;
- `Design Confidence / Total Scenario Score`;
- per-pillar visible Design Confidence scoring;
- rigid Strategic Review as the only additive movement-aware modifier;
- participant narrative split into separate component essays.

Historical implementation details may remain useful when recycling code, but they should not guide the next product behavior.

## Implementation Record

### 2026-06-18 Trial Score V1 Narrative Production Cleanup

Main goal: fix and enhance Scenario Review / Trial Score narrative production while simplifying the active implementation around the current contract.

Implemented direction:

- `Trial Score = Completion Outlook + Operational Fit + Reality Check`.
- `Completion Outlook` remains the protected XGBoost/SHAP model anchor.
- `Operational Fit` is app-scored and shown under `Execution Framework`, including feature-value detail lines for planned enrollment, planned sites, and planned duration.
- `Reality Check` is app-scored from validated classifications and allocated to canonical existing pillar/subpillar targets.
- The participant-facing output is generated in Pass 2 as one integrated Trial Score narrative, one central discussion topic, and one broader strategic question. Facilitator questions are not part of the Pass 2 contract.

Provider and prompt flow:

- Pass 1 is the analytical scoring/classification pass.
- App scoring runs between Pass 1 and Pass 2 and owns all numeric score values.
- Pass 2 receives exact app-calculated scores plus validated Pass 1 analysis and writes participant narrative only.
- The active prompt builder is simplified to the current Trial Score V1 contract.
- OpenAI and Gemini provider paths use the same staged behavior conceptually: Pass 1, targeted repair when needed, then Pass 2 narrative generation.

Retry and validation behavior:

- Gemini malformed JSON / max-token Pass 1 retry remains reserved for parse failure or provider truncation.
- Pass 1 validation repair retries target invalid classifications, invalid allocation targets, anti-double-counting failures, and invalid scoring structure.
- Pass 2 repair is separate and targets only invalid participant-narrative fields; it does not rerun Pass 1 or change app-owned scores.
- Retry history records stage, attempt, validation messages, parse status, latency, response length, and remaining errors.
- After retries are exhausted, failure messages identify the failed level clearly.

Operational Fit and same-state behavior:

- Operational Fit is deterministic/app-owned once Pass 1 classifications validate.
- A provider cannot keep Operational Fit credit when the current operational state returns to a prior identical scenario state.
- Same-state deterministic app scoring is reused; Pass 2 may regenerate narrative with explicit reversion/path context.
- This applies to deterministic app scoring state, not to replaying old participant narrative.

Reality Check behavior:

- Reality Check defaults to zero when the pre-Reality movement is coherent and realistic.
- Reality Check is not a fifth pillar.
- Reality Check allocations use canonical allocation target IDs, not free-typed subpillar labels.
- Invalid or invented allocation labels are repaired through targeted validation; after repair exhaustion, scoring fails clearly rather than accepting arbitrary text.

UI behavior:

- Development UI keeps three score views: `Completion Outlook`, `Reality Check`, and `Trial Score`.
- `Completion Outlook` displays XGBoost Completion Outlook plus Operational Fit after visible scenario review.
- Operational Fit appears under `Execution Framework` in the bar chart and treemap.
- Reality Check zero-state behavior should remain visible and interpretable without inventing fake visual movement.
- Overlay progress text is tied to real staged behavior:
  - `Evaluating Scenario Impact...` for Pass 1;
  - `Refining Score...` for Pass 1 repair;
  - `Generating Analysis...` for Pass 2.
- Duplicate non-overlay spinner/status labels were removed.
- Locked premise fields are disabled and greyed with lighter value text.

Storage and diagnostics:

- Review storage was reset to `narrative_review_store_v2` for a clean current-contract start.
- Old diagnostics were deleted so future local checks start clean.
- Current run diagnostics and audit bundles are for inspecting prompt content, provider responses, retry history, score decisions, and UI trace propagation.

Deleted legacy paths:

- The active scoring facade now delegates only to `validate_pass1_review()` and `score_pass1_review()` in `src/narratives/trial_score_contract.py`.
- Old `Strategic Review`, `Design Confidence`, `Quality Review`, and `Total Scenario Score` active scoring paths were removed.
- Obsolete Strategic Review / Design Confidence check scripts were deleted.
- Disabled legacy live-eval harness stubs were deleted rather than kept as warning-only compatibility code.

Verification gate:

```bash
bash scripts/check_trial_score_v1_migration.sh
```

That gate validates the active Trial Score contract, obsolete-field guard, prompt builder/provider schema, provider normalization and repair behavior, deterministic mock reviewer behavior, review storage/cache behavior, packet assembly, live-style snapshot flow, participant-facing failure formatting, visual data composition, py_compile, and `git diff --check`.

Remaining validation priority:

- Run a local UI scenario with live Gemini for `NCT02741128`, changing planned enrollment and planned sites, then reverting them, to confirm same-state scoring reuse, Pass 2 reversion narrative, Operational Fit treemap details, Reality Check zero-state behavior, and staged overlay progress text.
- If a new batch/live eval harness is needed later, rebuild it around this current Trial Score V1 contract instead of restoring deleted Strategic Review or Design Confidence harnesses.
