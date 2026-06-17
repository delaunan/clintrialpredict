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

## Operational Fit

`Operational Fit` is the first additive interpretation layer.

It should be treated as a subpillar of `Execution Framework`, not as a separate broad design-quality score.

Detailed Operational Fit logic is recorded in `docs/operational_fit_scoring.md`.

Its role is to review all relevant XGBoost dynamics and operational assumptions together, then decide whether the scenario is operationally coherent with the model-visible design pattern.

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
full_offset 100%
reversal    125-150%
```

Direction is determined by effect:

```text
positive pre-Reality movement:
  reinforce_gain -> positive Reality Check
  offset_gain / full_offset / reversal -> negative Reality Check

negative pre-Reality movement:
  soften_decline / reversal -> positive Reality Check
  reinforce_decline -> negative Reality Check

near-flat pre-Reality movement:
  reward_coherence or penalize_incoherence can be small, evidence-backed adjustments
```

No hard absolute cap is locked for now. Percentages are the first control. If live testing produces unstable swings, add caps later.

Reversal is exceptional. It requires an explicit critical label and rationale that the apparent pre-Reality movement is actively misleading, not merely imperfect.

Examples:

```text
Pre-Reality movement: +6
Reality Check full_offset: -6
Final movement: 0

Pre-Reality movement: +6
Reality Check reversal at 125%: -7.5
Final movement: -1.5

Pre-Reality movement: -6
Reality Check reversal at 125%: +7.5
Final movement: +1.5
```

### Pillar And Subpillar Allocation

Reality Check defines the overall correction at score-move level first, then allocates that correction to existing pillars/subpillars.

The LLM may choose where the Reality Check lands, but it must target existing pillars/subpillars rather than creating a new visible pillar.

Example:

```json
{
  "reality_check": {
    "effect": "offset_gain",
    "strength": "moderate",
    "correction_fraction": 0.4,
    "central_tension": "Completion resemblance vs evidence value",
    "pillar_allocations": [
      {
        "pillar": "Scientific Challenge",
        "subpillar": "Endpoint Evidence",
        "share": 0.7,
        "direction": "down",
        "incremental_reason": "The simplified endpoint structure may weaken interpretability."
      },
      {
        "pillar": "Execution Framework",
        "subpillar": "Operational Fit",
        "share": 0.3,
        "direction": "down",
        "incremental_reason": "The execution plan does not fully support the revised evidence claim."
      }
    ]
  }
}
```

The app calculates the Reality Check points from effect/strength/fraction and distributes them by validated allocation shares.

Participant default view should show one final Trial Score story. Drilldown/facilitator view may show where Reality Check was allocated.

### Anti-Double-Counting Rule

Reality Check should only score the incremental concern or credit not already addressed by Completion Outlook or the Pass 1 Operational Fit assessment.

For each Reality Check adjustment, Pass 1 should provide:

```json
{
  "already_addressed_by": ["completion_outlook", "operational_fit_assessment"],
  "remaining_uncaptured_effect": "...",
  "incremental_basis": "not_captured_by_completion_outlook | not_captured_by_operational_fit | cross_pillar_consequence | consistency_resolution | prior_tension_resolution | prior_tension_regression",
  "not_double_counted_because": "..."
}
```

Important wording: Pass 1 can know what was addressed by its own Operational Fit assessment, but it cannot know final app-calculated Operational Fit points until after the app scoring step. Therefore the double-counting check refers to what was addressed by the assessment, not final scored points.

App validation should downgrade or reject Reality Check adjustments when:

- `incremental_basis` is missing;
- `not_double_counted_because` is empty;
- the adjustment repeats the same evidence, same target, and same consequence as Operational Fit;
- the adjustment repeats a strongly moving XGBoost subpillar without adding a distinct consequence;
- allocation shares do not sum to `1.0`;
- target pillar/subpillar names are not allowed.

Related evidence can still be valid when it creates a different consequence. For example, Operational Fit may penalize execution feasibility, while Reality Check may separately allocate to Scientific Challenge if the operational compression undermines endpoint interpretability.

### Premise Shift And Locked Fields

Therapeutic area, indication, and other premise-defining fields can reset the meaning of a scenario. Changing them may transform the scenario into a new trial premise rather than a normal iteration.

Before implementation, decide which premise fields should be locked in the simulator to avoid accidental resets. Candidate fields include therapeutic area, indication, and fields that define the disease/population premise strongly enough that the prior tension history may no longer apply.

If premise fields remain editable, Pass 1 should classify the change:

```text
same_premise_refinement
adjacent_indication_shift
new_trial_premise
incoherent_switch
```

Reality Check should not penalize a TA/indication change simply because it changed. It should penalize or offset only when the rest of the scenario does not adapt, score movement is no longer comparable, structured/text fields conflict materially, operational assumptions no longer match the new patient pool, or endpoint/comparator/governance no longer fit the new disease context.

Structured categorical fields prevail over free text when they conflict. The conflict should produce a scenario-consistency warning, not a strong penalty by itself, unless it materially undermines interpretation.

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
3. Fold in the key Completion Outlook / Operational Fit dynamic only where it changes the interpretation.
4. Surface one central tension.
5. Explain whether the Reality Check reinforces, penalizes, moderates, or offsets the score only when that is necessary for the Trial Score story.
6. End with one broader strategic question.

The narrative should be concise, conditional, and constructive. It should challenge the scenario without prescribing exact field edits.

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

Pass 1 should produce:

- XGBoost / Completion Outlook interpretation;
- feature, subpillar, and pillar reading with definitions and values;
- consistency update versus prior XGBoost interpretation;
- Operational Fit field ratings for enrollment, site footprint, and total duration;
- combined Operational Fit rating and materiality;
- Reality Check effect, strength, central tension candidate, allocation targets, and anti-double-counting rationale;
- continuity update for later iterations.

Pass 1 should not write the final participant-facing score narrative, because it does not yet know the app-calculated Operational Fit points, Reality Check points, or final Trial Score.

### App Scoring Between Passes

The application validates Pass 1 and calculates:

- Operational Fit points;
- Reality Check points and allocations to existing pillars/subpillars;
- updated pillar and subpillar contributions;
- final Trial Score;
- score deltas versus previous visible iteration;
- compact continuity state.

The app owns numeric scoring. The LLM provides ratings, materiality, target pillars/subpillars, evidence references, and rationale.

### Pass 2: Participant Narrative

Pass 2 receives:

- exact app-calculated score stack;
- Pass 1 structured analysis;
- previous visible iteration context;
- compact continuity state;
- participant-facing guardrails.

Pass 2 writes:

- one integrated Trial Score narrative;
- one central tension;
- one broader strategic question;
- no separate Completion Outlook essay, Operational Fit essay, and Reality Check essay.

This preserves exact scoring, continuity, and participant-readable narrative quality without forcing three LLM calls.

## Development UI Contract

Near-term UI should keep the three radio buttons while the new strategy is developed:

```text
Completion Outlook
Reality Check
Trial Score
```

For now, these views mean:

- `Completion Outlook`: show `XGBoost Completion Outlook + Operational Fit`. Keep the familiar model pillar rows, and add `Operational Fit` only under `Execution Framework`.
- `Reality Check`: replace the current `Strategic Review` radio behavior with Reality Check. Show the adjustment, short rationale, central tension, and allocation trace.
- `Trial Score`: show `Completion Outlook + Reality Check`. In this view, expose Reality Check as allocated subpillar leaves inside the impacted existing pillars, for example `Reality Check` under `Scientific Challenge` or `Execution Framework`. Do not show Reality Check as a fifth pillar.

Future participant-facing UI may eventually show only `Trial Score`. That is not the development target yet; during implementation, keep all three radio buttons visible so scoring behavior can be inspected.

The final participant narrative should assess the total Trial Score, not write separate essays for Completion Outlook, Operational Fit, and Reality Check.

## Migration Plan

1. Freeze the current uncommitted narrative/scoring implementation until the new contract is specified.
2. Use this document as the active source of truth.
3. Remove obsolete root planning files that preserve old Design Confidence implementation momentum.
4. Mark older Strategic Review docs as superseded rather than active.
5. Define the new provider schema around:
   - total-score assessment;
   - operational-fit assessment;
   - reality-check judgment;
   - central tension;
   - broader strategic question;
   - evidence references;
   - optional facilitator/debug trace fields.
6. Decide the exact app-owned Operational Fit mapping and Reality Check percentage mapping.
7. Define Pass 1 analytical-review schema and Pass 2 participant-narrative schema.
8. Rework scoring tests around the new score stack before changing the UI.
9. Update simulator UI labels and narrative rendering after the schema/scoring contract is stable.
10. Rebuild live evals only after the new prompt shape is testable on one or two trials.

## Obsolete Direction

The following directions are superseded:

- `Quality Review / Quality Adjustment / Final Candidate Score`;
- `Design Confidence / Total Scenario Score`;
- per-pillar visible Design Confidence scoring;
- rigid Strategic Review as the only additive movement-aware modifier;
- participant narrative split into separate component essays.

Historical implementation details may remain useful when recycling code, but they should not guide the next product behavior.

## Next Step

Before implementation resumes, formalize the Pass 1 analytical-review schema, app scoring contract, and Pass 2 participant-narrative schema in this document or a directly linked schema note. Then inspect the current uncommitted code and decide which pieces should be kept, simplified, or discarded under the new direction.
