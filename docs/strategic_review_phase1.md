# Strategic Review Phase 1 Contract

## Document Role

This document is the clean Phase 1 source for the next narrative/scoring direction. It is intentionally separate from older narrative plans that contain unfinished or superseded `Design Confidence`, `Total Scenario Score`, `Quality Review`, and calibration work.

Use this document to align terminology and product behavior before changing prompts, scoring code, UI labels, eval harnesses, or architecture history.

## Superseded Direction

The previous working direction was:

```text
Completion Score + Design Confidence = Total Scenario Score
```

That direction is superseded for the next migration by:

```text
Completion Outlook + Strategic Review = Trial Score
```

Preserved from the existing system:

- XGBoost remains the completion / early-termination resemblance anchor.
- SHAP and existing four pillars remain the explanation layer for model movement.
- Hidden baseline context remains useful.
- First visible participant review remains separate from hidden baseline review.
- Iteration memory remains useful for continuity.
- Existing bar chart and treemap concepts should be reused where possible.
- LLM providers remain reviewers only; application code owns numeric scoring.

Changed:

- `Strategic Review` replaces broad `Design Confidence` as the participant-facing modifier concept.
- `Trial Score` replaces `Total Scenario Score` as the final serious-game score label.
- `Strategic Review` is one single numeric modifier, not four visible mini-scores.
- `Strategic Review` is movement-aware and proportional to the latest Completion Outlook movement.
- The LLM classifies the move and explains the tradeoff; the application calculates the numeric modifier.

## Score Stack

```text
Trial Score = Completion Outlook + Strategic Review
```

Definitions:

- `Completion Outlook`: XGBoost-derived historical resemblance to completed versus early-terminated trials. It is supported by SHAP values, score movement, and the four existing pillars.
- `Strategic Review`: a single movement-aware modifier that evaluates whether the participant's latest move resolves, worsens, or reopens the main strategic tradeoff behind the Completion Outlook movement.
- `Trial Score`: final serious-game score shown to participants.

The LLM must not calculate the final `Trial Score`, rewrite XGBoost outputs, rewrite SHAP values, or modify therapeutic-area calibration.

The LLM should not return a suggested numeric `Strategic Review` value. It should return categorical effect labels and rationale. The application calculates the numeric Strategic Review modifier from the agreed budget and percentage mapping.

## Completion Outlook

Completion Outlook remains the model layer.

It is explained through the existing four pillars:

- `Therapeutic Context`
- `Scientific Challenge`
- `Execution Framework`
- `Patient Profile`

The model layer answers:

```text
How does the current scenario resemble historical completed or early-terminated trials, and what drove that movement?
```

SHAP, pillar deltas, changed fields, score deltas, and model drivers are used to explain the Completion Outlook movement. They are not sufficient by themselves to decide whether the participant made a strategically good move.

## Strategic Review

Strategic Review is the tradeoff layer.

It answers:

```text
Should the latest Completion Outlook movement be trusted, softened, offset, or lightly reinforced from a strategic tradeoff perspective?
```

Strategic Review evaluates:

- whether the move exposed the main tension;
- whether the move created or resolved a contradiction;
- whether the move improved one dimension by damaging another;
- whether the move preserved previous gains;
- whether the move reopened a previously improved tension;
- whether a strong positive model movement looks like oversimplification or model-gaming;
- whether a negative model movement is strategically justified by improved evidence value, population logic, governance, or tradeoff balance.

Strategic Review should be one visible numeric value. Its sublevels are qualitative explanation labels, not visible mini-scores.

There should be no Strategic Review score per Completion Outlook pillar and no visible Strategic Review mini-score per qualitative sublevel. The Strategic Review score is one combined modifier.

Qualitative sublevels:

- `Current Tension`
- `Carryover Check`
- `Tradeoff Resolution`

`New Tension` is not a scored Strategic Review sublevel. It may be introduced in the narrative as a forward-looking `Next Consideration`, but it should not affect the current Strategic Review score because participants were not yet asked to solve it.

## Movement-Aware Scoring Principle

Strategic Review should be proportional to the latest Completion Outlook movement.

The LLM classifies the move. The application maps that classification into points.

General behavior:

- Good positive move: Strategic Review slightly reinforces the Completion Outlook gain.
- Bad or one-sided positive move: Strategic Review offsets part of the Completion Outlook gain.
- Good negative move: Strategic Review softens the Completion Outlook decline.
- Bad negative move: Strategic Review reinforces the decline.
- Flat move: Strategic Review can be small positive or negative if the move resolves or creates an important tension not reflected in the model score.

Locked Phase 1 direction:

```text
movement_size = abs(current_completion_outlook - previous_completion_outlook)
strategic_review_budget = max(2, 0.40 * movement_size)
```

The budget gives Strategic Review enough room to matter when Completion Outlook barely moves, while keeping the modifier proportional when Completion Outlook moves materially.

Use effect labels that make the sign clear relative to the Completion Outlook movement.

For positive Completion Outlook movement:

```text
supports_score_gain:         +25% of budget
lightly_supports_score_gain: +10% of budget
neutral:                       0
partly_offsets_score_gain:   -50% of budget
strongly_offsets_score_gain:-100% of budget
critical_reversal:          -150% of budget
```

For negative Completion Outlook movement:

```text
softens_score_decline:       +25% of budget
lightly_softens_decline:     +10% of budget
neutral:                       0
reinforces_score_decline:    -50% to -100% of budget
critical_negative_review:   -150% of budget
```

The positive Strategic Review side is intentionally small to avoid double-counting a model-favorable move. The negative side has more room because Strategic Review exists partly to catch misleading score gains, unresolved contradictions, and model-favorable oversimplification.

Carryover and residual tension effects must also use percentages of the same Strategic Review budget rather than fixed point additions. This keeps latest-move effects and memory effects proportional and avoids uncontrolled cumulative scoring.

When carryover is relevant, calculate Strategic Review from one combined factor:

```text
combined_review_factor = latest_move_factor + tension_status_factor
strategic_review = strategic_review_budget * combined_review_factor
```

Do not apply a separate carryover factor when the same unresolved or regressed tension is already the main reason for the latest-move effect label. In that case, use one stronger latest-move label instead.

Prior tension status mapping:

```text
resolved:                 0%
obsolete:                 0%
superseded:               0% or -10%
partially_active:        -15%
still_active_secondary:  -25%
still_active_primary:    -50%
regressed:              -100%
```

Positive carryover is allowed, but only when triggered by current scenario evidence:

```text
newly_resolved:                 positive effect may apply through the latest-move classification
protected_gain_preserved:       +10%
further_improved:               +15% to +25%
stable_background_strength:       0%
```

Positive carryover should not stack every iteration. A solved tension becomes a protected gain only while the latest move could plausibly disturb it. If the gain remains true but is not directly at issue, it becomes a stable background strength with no score effect.

Avoid double counting: do not apply a separate still-active or regression factor if that same unresolved tension is already the main reason for the latest-move effect label.

Operational-only changes use their own materiality budget because planned enrollment, planned site count, and planned duration do not feed Completion Outlook.

For operational-only changes:

```text
Completion Outlook delta = 0
strategic_review_budget = operational_materiality_budget
```

Operational materiality budget:

```text
minor:    2
moderate: 3
major:    4
extreme:  5
```

Use the largest materiality across changed operational fields:

- planned enrollment;
- planned site count;
- planned duration.

Implementation ownership:

- The LLM classifies operational materiality as `minor`, `moderate`, `major`, or `extreme` using the operational field changes, benchmark context, and active/carryover tensions.
- The application maps that categorical materiality to the numeric budget above.
- Phase 3 may replace or constrain the LLM classification with deterministic thresholds if playtesting shows inconsistent materiality labels.

Then apply percentage factors to the operational materiality budget:

```text
supports_tradeoff_balance:          +25%
lightly_supports_tradeoff_balance:  +10%
neutral:                              0
worsens_active_tension:             -50%
strongly_worsens_active_tension:   -100%
reopens_protected_tension:         -150%
```

Operational-only changes should be sized by operational materiality itself, not by borrowing an unrelated previous Completion Outlook delta.

No absolute hard clamp is locked in Phase 1. Hard clamps can be added later if playtesting or eval traces show extreme Strategic Review behavior that needs a safety bound.

Display precision:

- Store exact Strategic Review values internally.
- Display `Completion Outlook`, `Strategic Review`, and `Trial Score` with one decimal place.
- Do not round Strategic Review to whole points in the participant UI.
- Show an explicit sign for Strategic Review values, including positive values such as `+1.4`.
- Clamp visible `Trial Score` to the `0.0` to `100.0` scale.
- Display the actual `Strategic Review` modifier value even when the final `Trial Score` is clamped.

## First Iteration Role

The first visible iteration is the diagnostic review.

The hidden baseline remains mostly invisible as an assessment object. Participants should not receive a full baseline strategic critique before acting.

Participant-facing baseline behavior:

- Show the initial `Completion Outlook` view only.
- Keep the existing baseline gauge, bar chart, treemap, trial features, and model-driver visuals available as the exercise already supports.
- Keep the numeric `Completion Outlook` visible to participants rather than reducing it to a band only.
- Keep the current `Completion Outlook` radio/view behavior mostly unchanged so participants can understand how the model-facing score and drivers moved.
- Do not show baseline `Strategic Review`.
- Do not show baseline `Trial Score`.
- Do not show a baseline main-tension narrative or direct challenge beyond the general exercise instructions.

Hidden baseline behavior:

- The LLM system may fully analyze the existing trial for later continuity.
- Hidden baseline analysis may identify up to five candidate tensions suggested by the trial profile before any Strategic Review adjustment.
- Candidate tensions are preparation context only. They must not lock the storyline, because the first participant move determines the first visible active tension.
- Candidate tensions may be useful prompt context if they help the first visible review interpret the participant's move.
- During development, baseline analysis and candidate tensions should be visible in a facilitator/debug expandable area, such as a hidden object behind the narrative zone. This should not be part of the default participant-facing view.
- A future supporting prompt document may list typical clinical-trial tension families, but the first visible review should still be grounded in the actual baseline-to-current change.

Flow:

1. The system stores baseline trial context, Completion Outlook, pillars, and optional hidden review memory.
2. The participant makes the first scenario change.
3. The first visible review compares baseline versus first scenario.
4. The LLM identifies the first main tension or contradiction revealed by the move.
5. The LLM classifies whether the move resolved, worsened, or exposed that tension.
6. The app calculates Strategic Review from the movement-aware scoring rules.
7. The first identified tension becomes the storyline anchor for later iterations.

The first iteration can offset a positive Completion Outlook gain when the gain appears to come from oversimplification or a one-sided shortcut. It should not be forced to offset every positive move. If the first move is balanced, Strategic Review can be neutral or mildly positive while introducing a stress-test consideration for the next iteration.

For the first visible review, Completion Outlook delta is calculated against the original participant-visible baseline Completion Outlook. The Strategic Review budget for that first review is based on this baseline-to-current delta. This allows a large first model improvement to be meaningfully offset when needed to introduce the first active tension.

For the first visible `Trial Score`, the displayed Trial Score delta should also compare against the original participant-visible baseline Completion Outlook, because no baseline Trial Score is shown before the first participant move.

## Later Iteration Logic

Later iterations use focused wave assessment.

Each visible review should answer:

1. Did the latest move address the active tension?
2. Did it preserve previous gains?
3. Did it regress a previously improved tension?
4. Did it create a larger new contradiction?
5. Should the active tension remain, sharpen, or be replaced by a new one?

Wave behavior:

- Wave 1: identify the main tension revealed by the first move.
- Wave 2: assess whether the participant resolved that tension.
- Wave 3: introduce or reframe the next tension even if the prior tension is unresolved, while keeping the unresolved tension in memory and carryover checks.
- Wave 4: assess final balance and regression.

An unresolved tension does not block a new visible tension. For short exercises with only a few participant iterations, the system should avoid trapping participants on one repeated question. If a prior tension remains unresolved, keep it in the storyline state as active, partially active, or still-active carryover, but allow the visible narrative to introduce another relevant tension or a reframed challenge.

Participant-facing tension hierarchy:

- Show one main tension in the default participant narrative.
- If a prior tension remains unresolved, do not forget it; mention it as secondary context when relevant.
- A later move may fail to resolve the newest main tension but still resolve an earlier secondary tension. The tension-status update step should detect and credit that.
- A scenario change can make any prior tension obsolete at any time if it changes the premise that created the tension. Obsolete tensions should have no carryover score effect and should not be repeated as active questions.

## Tension Portfolio

A solved tension does not disappear. It becomes a protected constraint.

Later moves are judged by whether they:

- address the active tension;
- preserve solved tensions;
- avoid reopening old contradictions;
- avoid creating a larger contradiction than the one being solved.

Example:

```text
Iteration 1 tension:
Feasibility vs Evidence Strength.

Iteration 2 resolves it:
Evidence credibility improves while execution burden remains below baseline.

Iteration 3 introduces population narrowing:
The review should check whether the population move solves a new tension without reopening the earlier feasibility/evidence tension.
```

## Move Classification Rubric

The LLM should classify moves using a constrained rubric. It should not make unsupported claims that a trial is objectively good or bad.

Useful move classes:

- `oversimplification`: large Completion Outlook gain from reducing control, blinding, endpoint rigor, follow-up, population breadth, enrollment, duration, or strategic ambition.
- `proportionate_governance`: DMC, masking, comparator, and randomization changes that make sense in later-phase, larger, riskier, hard-endpoint trials.
- `evidence_strengthening`: harder endpoint, better comparator, clearer endpoint structure, stronger masking/control, or more appropriate follow-up.
- `execution_burden`: larger enrollment, more sites, longer duration, complex arms, severe population, or high administration complexity.
- `population_narrowing`: biomarker selection, rare-disease focus, age/gender restrictions, high-severity subgroup, or other narrowing that may improve signal clarity while pressuring recruitment or generalizability.
- `strategic_mismatch`: phase/intent, endpoint, comparator, population, governance, or operational assumptions do not fit together.
- `balanced_improvement`: Completion Outlook improves and the tradeoff remains coherent.
- `productive_negative_move`: Completion Outlook declines, but the move plausibly improves evidence value, strategic coherence, or protected prior tension balance.
- `unresolved_complexity`: Completion Outlook declines because burden increased without a clear strategic payoff.

## Tension Families

The active tension should usually come from one of these families:

- `Feasibility vs Evidence Strength`
- `Operational Burden vs Strategic Ambition`
- `Population Focus vs Generalizability`
- `Endpoint Practicality vs Clinical Meaningfulness`
- `Comparator Rigor vs Recruitment Acceptability`
- `Innovation Ambition vs Development Uncertainty`

The LLM can write the participant-facing wording, but the underlying family should be structured for continuity and evaluation.

## UI Direction

Top-level score labels:

```text
Completion Outlook
Strategic Review
Trial Score
```

`Strategic Review` should reuse the current `Design Confidence` display treatment wherever possible. The intended UI migration is primarily a label/name change plus underlying score-contract change, not a visual redesign.

`Trial Score` should reuse the current `Total Scenario Score` display treatment wherever possible. The intended UI migration is primarily a rename from `Total Scenario Score` to `Trial Score`.

Bar chart:

```text
Completion Outlook
Strategic Review
Trial Score
```

View-specific chart behavior:

- `Completion Outlook` radio/view: keep the current plots unchanged. Do not add the `Strategic Review` bar, and do not group the four existing Completion Outlook pillars under a new overarching branch in this view.
- `Strategic Review` radio/view: show `Strategic Review` using the current `Design Confidence` visual treatment, with the new Strategic Review score contract.
- `Trial Score` radio/view: include one additional bar for `Strategic Review` in the bar chart so the score stack is visible as `Completion Outlook`, `Strategic Review`, and `Trial Score`.
- Do not show previous-value variance for `Strategic Review`. In particular, do not show `+/- pts` or a previous-value card next to the Strategic Review gauge, and do not show a `+/- pts` annotation for the Strategic Review bar in the bar chart.
- Keep score-movement variance for `Completion Outlook` elements and for the overall `Trial Score` view where appropriate.

Treemap:

```text
Trial Score
├── Completion Outlook
│   ├── Therapeutic Context
│   ├── Scientific Challenge
│   ├── Execution Framework
│   └── Patient Profile
└── Strategic Review
    ├── Current Tension
    ├── Carryover Check
    ├── Tradeoff Resolution
```

Strategic Review subitems should be qualitative labels. They should explain why the one Strategic Review value offsets, softens, or reinforces the Completion Outlook movement.

Treemap behavior:

- `Completion Outlook` radio/view: keep the current treemap unchanged.
- `Strategic Review` and `Trial Score` radio/views: show two overarching branches: `Completion Outlook` and `Strategic Review`. Under `Completion Outlook`, group the four existing model pillars. Under `Strategic Review`, show only active/relevant qualitative subitems from `Current Tension`, `Carryover Check`, and `Tradeoff Resolution`.
- Show `Carryover Check` only when a prior tension is still relevant to the current review, a protected gain was preserved, or a prior improvement regressed. Do not show it only for historical completeness.
- Hide `Carryover Check` by default in the first visible review because there is no prior participant-facing tension yet.

## Participant Narrative

Each visible review should be concise, focused, and integrated.

The narrative should be one combined participant-facing review of the `Trial Score`, not separate sections that independently explain `Completion Outlook` and then `Strategic Review`. The review should discuss the overall score dynamic in one flow so the model movement, pillar interactions, tension status, and Strategic Review modifier do not become repetitive.

Recommended flow:

1. Start with the Trial Score reading.
   - State whether the scenario improved, declined, or stayed broadly stable.
   - Name the main reason in plain trial-development language.
   - Avoid opening with the raw mechanics of `Completion Outlook + Strategic Review`.

2. Explain the Completion Outlook movement inside the score story.
   - Identify the main score driver or pillar interaction.
   - Translate it into participant language such as lower/higher early-termination risk, simpler execution, stronger/weaker completion resemblance, or changed evidence burden.
   - Avoid listing every pillar unless the interaction itself matters.

3. Explain the Strategic Review modifier as the critical interpretation of that movement.
   - State whether the modifier supports the movement, softens a decline, offsets a gain, or reverses part of the apparent improvement.
   - Tie the modifier to the current tension and tradeoff, not to a generic design-quality judgment.

4. State the tension status.
   - Say whether the current tension is resolved, partly resolved, still unresolved, obsolete, or superseded.
   - Mention carryover only when a previous tension remains relevant, a protected gain was preserved, or a prior improvement regressed.

5. End with one broad strategic question for debate.
   - This may introduce a new tension for the next iteration.
   - The new tension is forward-looking and should not affect the current Strategic Review score.
   - Ask one high-level question tied to the current or emerging tension.
   - Prefer wording such as "Broadly speaking, how can..." so participants debate the strategic direction rather than optimize a specific field.
   - Do not generate three questions, separate medical/development questions, or direct field-change instructions.

Use clear points inside the integrated review when helpful, but avoid isolated score-component narratives that repeat the same evidence.

The narrative should use conditional language:

- "may suggest"
- "could indicate"
- "appears"
- "raises the question"
- "needs scrutiny"

Avoid overclaiming clinical truth, power, ethics, recruitment feasibility, or regulatory adequacy unless directly supported by provided fields.

Migration note: reuse the prior Scenario Review architecture where useful, including hidden baseline context, prompt modes, validation, provider abstraction, review storage, and continuity state. Remove pieces that are specific to per-subcategory `Design Confidence` scoring, provider-returned score materiality, per-pillar Strategic Review scoring, or visible Design Confidence contribution rows. The new consistency mechanism should track tension status and categorical Strategic Review effect labels rather than carrying forward old design-subcategory movement ratings.

## Consistency State

Store compact storyline state after each review.

Suggested structure:

```json
{
  "active_tension": "Feasibility vs Evidence Strength",
  "active_tension_status": "unresolved",
  "last_move_classification": "one_sided_simplification",
  "protected_gains": ["reduced execution burden"],
  "regression_watch": ["do not weaken endpoint credibility further"],
  "next_consideration": "Restore evidence credibility without returning fully to baseline burden."
}
```

This state should be passed into later reviews so the provider evaluates continuity before introducing new tensions.

## Strategic Review Unavailable State

If the provider or validation path cannot produce a valid Strategic Review for the current scenario:

- Show the current `Completion Outlook`.
- Mark `Strategic Review` as unavailable.
- Do not calculate or display `Trial Score` for that scenario.
- Do not reuse stale Strategic Review from a previous scenario.

Participant-facing wording:

```text
Strategic Review is unavailable for this scenario. Completion Outlook is still shown.
```

## Migration Preparation Decisions

Provider contract:

- Reuse the existing Scenario Review provider contract, packet inputs, and review plumbing where they still serve the new flow.
- Preserve useful inputs such as hidden baseline context, current/previous snapshots, Completion Outlook score movement, pillar impacts, changed fields, operational assumptions, text context, provider metadata, validation, and continuity state.
- Simplify the output contract around one `Strategic Review` modifier instead of per-subcategory or per-pillar `Design Confidence` scoring.
- Avoid carrying forward old output fields only for compatibility if they would keep `Design Confidence` assumptions alive.

Storage and backward compatibility:

- Avoid leftover `Design Confidence` traces or compatibility paths polluting the new Strategic Review logic.
- Old traces can be treated as obsolete for the migrated Strategic Review flow.
- Files or helpers that are harder to recycle cleanly than to delete and rebuild may be removed during implementation, provided the removal is scoped and verified.
- Do not mix old `Design Confidence` storage with new Strategic Review scoring or continuity state.

Mock reviewer:

- The deterministic mock reviewer is not mandatory as the first migration target.
- Prefer validating the new flow directly with Gemini when prompt/content expectations are ready, unless a small mock path is needed to keep local contract tests deterministic.
- If a mock reviewer is retained, it should be minimal and should not preserve old Design Confidence scoring behavior.

Eval harness:

- The current eval harness may be deleted or rebuilt if adapting it would preserve too much obsolete Design Confidence logic.
- The first new eval should be defined after reviewing an actual first trial review and listing the prompt, flow, and content expectations for the new Strategic Review behavior.

UI transition:

- `Design Confidence` text should disappear from the migrated UI rather than remain behind a feature flag.
- The current UI structure should be adapted, not thrown away wholesale: `Strategic Review` reuses the current Design Confidence display treatment, and `Trial Score` reuses the current Total Scenario Score display treatment.

Documentation cleanup:

- Keep history useful, but prevent stale plans from steering implementation.
- Prefer marking old Design Confidence / Total Scenario Score directions as superseded where they conflict with this document.
- Preserve general flow history when still valid, especially hidden baseline, first visible review, provider abstraction, validation, storage, and continuity concepts.
- Avoid broad doc rewrites until implementation decisions are stable enough to promote into `docs/architecture_narratives.md`.

## Phase Roadmap

This roadmap records the intended migration path. It is not approval to edit code; each phase should be planned and verified separately.

### Phase 1: Product Contract

Status: active in this document.

Goals:

- Define the new score stack: `Completion Outlook + Strategic Review = Trial Score`.
- Define Strategic Review as one movement-aware modifier, not a second predictive model.
- Define the first visible iteration as diagnostic review.
- Define later reviews as focused wave assessments with tension continuity.
- Define the high-level UI direction for the bar chart, treemap, and narrative.
- Resolve open product questions before code migration.

Expected output:

- This document updated with locked decisions.
- Older unfinished plans left untouched until Phase 7 cleanup.

### Phase 2: Prompt And Schema Migration

Goal: migrate provider-facing language and structured output from the old `Design Confidence` contract to the new `Strategic Review` contract.

Expected changes:

- Replace `Design Confidence` / `Total Scenario Score` prompt language with `Strategic Review` / `Trial Score`.
- Add structured move classification.
- Add structured fields for `Current Tension`, `Carryover Check`, `Tradeoff Resolution`, tension status, and forward-looking `Next Consideration`.
- Keep the LLM responsible for classification and rationale only.
- Keep numeric Strategic Review calculation app-owned.

### Phase 3: Scoring Engine Migration

Goal: replace per-subcategory design scoring with one movement-aware Strategic Review modifier.

Expected changes:

- Use Completion Outlook delta to size the Strategic Review adjustment budget.
- Use LLM move classification to choose reinforcement, offset, softening, or negative reinforcement.
- Apply agreed proportional budget, percentage factors, operational-only materiality budget, and flat-movement handling.
- Calculate `Trial Score = Completion Outlook + Strategic Review`.
- Keep XGBoost, SHAP, and calibration untouched.

### Phase 4: Storage And Storyline Migration

Goal: store enough state to keep reviews consistent across iterations.

Expected changes:

- Store active tension and status.
- Store protected gains and regression watch.
- Store latest move classification and Strategic Review mapping.
- Store whether a tension is unresolved, improved, resolved, or reopened.
- Pass this compact state into later reviews.

### Phase 5: UI Migration

Goal: reuse the existing visual language while changing the score meaning.

Expected changes:

- Rename visible score stack to `Completion Outlook`, `Strategic Review`, and `Trial Score`.
- Add `Strategic Review` as one additional bar in the score bar chart.
- Add `Strategic Review` as one top-level treemap box.
- Keep `Completion Outlook` pillar details under the model side of the treemap.
- Keep `Strategic Review` sublevels qualitative only: `Current Tension`, `Carryover Check`, and `Tradeoff Resolution`.

### Phase 6: Evaluation Harness Migration

Goal: verify that the new mechanic behaves correctly before live UI reliance.

Expected checks:

- First iteration identifies a main tension from the first participant move.
- Positive Completion Outlook movement can be offset when it is one-sided or contradictory.
- Positive Completion Outlook movement receives only small reinforcement when it is balanced.
- Negative Completion Outlook movement can be softened when strategically justified.
- Negative Completion Outlook movement can be reinforced when it worsens tradeoff balance.
- Solved tensions become protected constraints.
- Later moves are checked for regression against protected gains.
- Flat Completion Outlook movement can still receive a small Strategic Review adjustment when meaningful.

### Phase 7: Documentation Cleanup

Goal: prevent stale plans from steering future implementation.

Expected changes:

- Mark old `Design Confidence` / `Total Scenario Score` sections as superseded.
- Promote durable Strategic Review decisions into `docs/architecture_narratives.md`.
- Update or retire `implementation_plan.md` and `prompt_enhancement_plan.md`.
- Keep historical details only where they explain implementation provenance.
