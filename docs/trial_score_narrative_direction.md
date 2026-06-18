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

It assesses the change, the trajectory, and the emerging central tension. The LLM should have more judgment freedom here than in the prior rigid Strategic Review implementation, but the judgment still needs evidence, structure, and guardrails.

Reality Check may:

- reinforce a Trial Score movement when the change is coherent, realistic, and fit for purpose;
- penalize a score gain when it looks one-sided, shortcut-driven, contradictory, unrealistic, or weakly justified;
- penalize increased rigor or robustness when it creates a real issue for operations, credibility, or fit for purpose;
- soften a score decline when the added burden appears necessary, realistic, and scientifically or clinically justified;
- offset part of a movement when a prior gain is reopened or a new unresolved tension dominates;
- surface a new central tension for the next iteration.

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
- incoherent answer to the central tension;
- increased rigor that is not operationally credible or fit for purpose;
- negative movement caused by justified robustness or necessary complexity;
- prior tension resolved, worsened, bypassed, or reopened.

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

Reality Check does not automatically follow the pre-Reality movement. For positive pre-Reality movement, Completion Outlook plus Operational Fit already gives the main credit. Therefore positive reinforcement is capped at `slight` by default in V1, and `strong` positive reinforcement is not allowed. Positive movements should usually be accepted with `0` or moderated with an offset when the after-review judgment finds simplification, brittleness, unrealistic assumptions, or unresolved tension.

For negative pre-Reality movement, Reality Check may soften the decline when added burden, robustness, or complexity appears necessary and credible, or reinforce the decline when the scenario is also incoherent, unrealistic, or operationally under-supported. Reversal remains exceptional and must explain why the apparent movement is actively misleading.

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

The participant narrative should explain Reality Check as an after-review judgment about realism, robustness, simplification, and emerging tension. It should not describe Reality Check as a mechanical component essay. Preferred language should connect the adjustment to the strategic review of the scenario, for example whether a simplification made the profile less robust, whether increased robustness made recruitment or execution riskier, or whether an emerging tension changes how the Trial Score movement should be read.

### Pillar And Subpillar Allocation

Reality Check defines one overall app-calculated adjustment first, then allocates that adjustment to existing pillars/subpillars for traceability.

The LLM may choose where the Reality Check lands, but it must target existing pillars/subpillars rather than creating a new visible pillar. V1 should keep this deliberately small and readable: allocate to 1-3 subpillars, not a long list of tiny effects.

Reality Check allocations must target existing subpillars. The pillar is the parent path, not the direct allocation target. For V1, use the current taxonomy subpillars plus the new additive `Operational Fit` subpillar:

| Pillar | Subpillar | Included Fields / Inputs | Broader Meaning | Reality Check Allocation Guidance |
| --- | --- | --- | --- | --- |
| Therapeutic Context | Therapeutic Area Profile | Therapeutic area, indication, rare-condition status | Disease context, patient relevance, benchmark context, and calibration limits. | Allocate here only when an after-review issue concerns disease-context fit, rare-condition implications, or stale/inconsistent condition context. Hard-locked premise fields should usually start a new scenario rather than receive Reality Check allocation. |
| Therapeutic Context | Development Phase and Goal | Clinical phase, regulatory intent / strategic ambition | Whether the evidence standard, endpoint maturity, population scope, and operational scale fit the development decision. | Allocate here when the issue is phase/ambition coherence, such as a pivotal claim without supporting evidence rigor or an exploratory scenario carrying excessive confirmatory burden. |
| Scientific Challenge | Biological Profile | Target precedent, pathway profile, therapeutic modality, innovation rank | Biological plausibility, novelty, modality risk, and the evidence burden the mechanism creates. | Allocate here when the after-review concern is about mechanism plausibility, novelty, modality-specific risk, delivery/safety implications, or whether biology supports the evidence claim. |
| Scientific Challenge | Protocol Architecture | Intervention model, primary purpose, adaptive design, endpoint rigor, endpoint structure, biomarker patient selection | Whether the trial design architecture can credibly answer the clinical-development question. | Allocate here when the after-review concern is about endpoint interpretability, comparator/design coherence, biomarker logic, evidence robustness, or simplification that weakens the clinical question. |
| Patient Profile | Clinical Severity | Patient severity, line of therapy | Whether patient burden, acceptable risk, endpoint relevance, and unmet need fit the scenario. | Allocate here when the issue is risk tolerance, disease-course fit, endpoint relevance for the target population, or whether burden is justified by severity/unmet need. |
| Patient Profile | Population Scope | Gender eligibility, healthy-volunteer flag, adult/pediatric/older-adult eligibility | Whether the population definition is credible, generalizable, ethically coherent, and recruitable. | Allocate here when the issue is population breadth/narrowness, representativeness, vulnerable-population safeguards, or whether eligibility choices support the evidence objective. |
| Execution Framework | Trial Complexity Footprint | Sponsor type proxy, endpoint duration, number of arms, delivery profile | SHAP-derived trial-footprint complexity, follow-up burden, site capability needs, and operational load. | Allocate here when the after-review issue concerns model-visible complexity, duration burden, arm/site complexity, delivery burden, or footprint credibility not already captured by Operational Fit. |
| Execution Framework | Methodological Setup | Masking, allocation method, DMC status, placebo control, comparator benchmark | Bias control, causal interpretability, governance, comparator credibility, and ethical/methodological setup. | Allocate here when the issue is methodological credibility, governance proportionality, comparator adequacy, placebo ethics, or whether bias control fits the endpoint and population. |
| Execution Framework | Operational Fit | Planned enrollment, planned sites, planned total duration, patients per site, operational benchmark metadata | Additive operational proportionality of enrollment, site footprint, duration, and benchmark position relative to the current scenario. | Allocate here when the after-review issue concerns non-XGBoost operational support, patient-per-site burden, duration feasibility, or whether operations support the revised evidence ambition. |

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
        "pillar": "Scientific Challenge",
        "subpillar": "Protocol Architecture",
        "share": 0.7,
        "direction": "down",
        "movement_label": "Simplification weakens evidence robustness",
        "rationale": "The endpoint simplification supports completion-like movement, but it may reduce the strength of the evidence claim.",
        "incremental_check": "This is not already counted by Operational Fit because it concerns evidence interpretability, not execution feasibility."
      },
      {
        "pillar": "Execution Framework",
        "subpillar": "Operational Fit",
        "share": 0.3,
        "direction": "down",
        "movement_label": "Execution support remains incomplete",
        "rationale": "The revised operational plan improves site burden but does not fully support the higher evidence ambition.",
        "incremental_check": "This is incremental to Completion Outlook because it uses non-model operational assumptions."
      }
    ]
  }
}
```

The app calculates the Reality Check points from effect/strength/fraction and distributes them by validated allocation shares.

Each allocation must include:

- `pillar`;
- `subpillar`;
- `share`;
- `direction`;
- `movement_label`;
- `rationale`;
- `incremental_check`.

Validation should stay simple:

- allocation count is 1-3;
- allocation shares sum to `1.0`, allowing only small rounding tolerance;
- pillar/subpillar targets are valid existing targets;
- direction matches the signed Reality Check effect;
- `movement_label`, `rationale`, and `incremental_check` are present;
- duplicate same-evidence / same-target / same-consequence allocations are rejected or downgraded.

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
lead_sponsor_canonical
```

These changes should be blocked or treated as starting a new scenario, because they alter disease domain, indication/disease category, sponsor identity, benchmarks, calibration context, and continuity.

V1 should gate the following premise-sensitive fields rather than hard-lock them:

```text
sponsor_tier_ml
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
7. Surface one central tension.
8. End with one broader strategic question that helps orient the next scenario discussion.

The narrative should be concise, conditional, and constructive. It should challenge the scenario without prescribing exact field edits.

The broader strategic question may mention adjustable feature families at a high level, such as eligibility breadth, endpoint ambition, comparator/control strategy, enrollment target, site footprint, follow-up duration, population focus, or design complexity. It should not prescribe exact field values or direct optimization instructions.

Optional facilitator questions may be generated for discussion support, but they should be hidden from the main participant UI by default and exposed only in a collapsed facilitator/debug section. Facilitator questions should include why the question matters and related adjustable feature families when useful.

Avoid:

- direct optimization instructions;
- clinical certainty beyond packet evidence;
- repeated score mechanics;
- repeated questions across iterations;
- hidden-baseline numeric leakage.

## LLM Freedom And Guardrails

The LLM should have more room to judge the scenario than in the prior app-owned categorical Strategic Review mapping.

Allowed:

- identify the most important central tension;
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
- one central tension by default.

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
central_tension_candidate
broader_strategic_question_candidate
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
- Reality Check effect, strength, central tension candidate, allocation targets, and anti-double-counting rationale;
- continuity update for later iterations.

`strategy_shift_check` is required when gated premise-sensitive fields changed. Allowed status values are `supported`, `partly_supported`, `unsupported_or_incoherent`, and `not_applicable`.

`operational_fit` should include field-level analysis for enrollment, site footprint / patients per site, and timeline, plus one `combined_operational_fit` object. The app scores only the combined rating/materiality.

`reality_check` should include effect, strength, central reason, and 1-3 allocations. It should not include app-owned Reality Check points.

`central_tension_candidate` should identify the main scenario tension, why it matters, supporting evidence, and related adjustable feature families where useful.

`broader_strategic_question_candidate` should be a reflective, topic-related debate question. It should not always be a direct question about changing a specific field value. It may mention related adjustable feature families as orientation, but the question should feel like a conference or strategy discussion prompt that helps the audience debate the next direction.

`continuity_update` should be compact and carry only the active tension, what changed since the previous visible iteration, whether a prior tension was resolved/reopened/worsened, and what to watch next.

Pass 1 proposes the central tension analytically. The application validates that candidate against the scored evidence, and Pass 2 writes the final participant-facing central tension. Pass 2 should not introduce a new analytical basis that was absent from the validated Pass 1 analysis.

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
facilitator_questions
```

Pass 2 writes:

- one integrated Trial Score narrative explaining state, movement, and interpretation;
- pillar-level bullets that use subpillar and feature evidence inside prose rather than nested bullet lists;
- one central tension;
- one broader strategic question;
- optional facilitator questions for collapsed facilitator/debug display;
- no separate Completion Outlook essay, Operational Fit essay, and Reality Check essay.

`trial_score_narrative` should briefly explain the current Trial Score reading, movement versus previous/baseline, main reason for movement, and how Completion Outlook, Operational Fit, and Reality Check affect the reading when relevant.

`pillar_reading` is the main UI reading structure. It should include one concise reading per pillar. Subpillar and feature evidence should appear inside the prose, not as nested subpillar/feature bullet lists.

`central_tension` should be the final participant-facing version of the validated Pass 1 central tension.

`broader_strategic_question` should be reflective and debate-oriented, but still contextualized to the scenario, condition, population, evidence goal, operational setting, or trial context. It should not be a generic conference question detached from the current scenario, and it should not become a direct instruction to change a specific field value.

`facilitator_questions` are optional, hidden from the main participant UI by default, and limited to concise discussion prompts with why they matter and related feature families when useful.

Pass 2 should use the validated score explanation from Pass 1 and app scoring. It should not make new scoring decisions, invent a different central tension, or introduce a new analytical basis.

This preserves exact scoring, continuity, and participant-readable narrative quality without forcing three LLM calls.

## Development UI Contract

V1 UI should keep the three radio buttons for development and for the current product version:

```text
Completion Outlook
Reality Check
Trial Score
```

For now, these views mean:

- `Completion Outlook`: show `XGBoost Completion Outlook + Operational Fit`. Keep the familiar model pillar rows, and add `Operational Fit` only under `Execution Framework`.
- `Reality Check`: replace the current `Strategic Review` radio behavior with Reality Check. Show the adjustment, short rationale, central tension, and allocation trace.
- `Trial Score`: show `Completion Outlook + Reality Check`. In this view, expose Reality Check as allocated subpillar leaves inside impacted existing subpillars, for example `Scientific Challenge / Protocol Architecture / Reality Check: endpoint robustness` or `Execution Framework / Operational Fit / Reality Check: operational support`. Do not show Reality Check as a fifth pillar.

Future participant-facing UI may eventually show only `Trial Score`, but that is not the V1 target. During implementation and current product use, keep all three radio buttons visible so scoring behavior can be inspected.

The final participant narrative should assess the total Trial Score, not write separate essays for Completion Outlook, Operational Fit, and Reality Check.

Reality Check allocations should appear within existing subpillars, not directly under pillars. In the Reality Check radio, the treemap may use `Reality Check` as the root view, but the visible allocation path should still be existing pillar -> existing subpillar -> Reality Check leaf. In the Trial Score radio, the full composition treemap should embed Reality Check leaves inside impacted existing subpillars. Operational Fit and Reality Check must not render as top-level fifth pillars.

## Migration Plan

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
   - central tension;
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
- **Noisy Reality Check allocations**: require 1-3 allocations, existing subpillar targets only, valid shares, signed direction compatibility, and present `movement_label`, `rationale`, and `incremental_check`. Do not invent fallback subpillars.
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

## Next Step

Start implementation slice 1: contract/schema constants and old-name boundary cleanup. Inspect the current uncommitted code against this document and `docs/operational_fit_scoring.md`, keep only contract-compatible reviewed-snapshot/consistency plumbing, and rewrite or discard obsolete Strategic Review / Design Confidence / Quality Review scoring paths.
