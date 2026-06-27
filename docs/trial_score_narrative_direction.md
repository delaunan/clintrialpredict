# Trial Score Narrative Direction

## Document Role

Architecture scope: `architecture_narratives`

This document is the active planning source for the current narrative/scoring direction. It supersedes the older `Design Confidence / Total Scenario Score` plans and the first `Strategic Review / Trial Score` migration notes, which were removed during the deployment-focused documentation cleanup.

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

Active implementation note, 2026-06-22: the earlier app-owned `rating + materiality -> points` Operational Fit mapping and mechanical Reality Check fraction mapping are superseded for the live Scenario Review path. The application now protects boundaries and arithmetic, while the LLM owns the judgmental Operational Fit and Reality Check point values inside hard rails.

## Active Three-Pass Workflow

The active Scenario Review workflow is:

```text
Pass 1: Evolution and Evidence
Pass 2: Score Adjudication
Pass 3: Participant Narrative
```

Pass 1 receives the scenario packet and prior context. It does not score. It generates:

- protected Completion Outlook interpretation;
- latest meaningful changes;
- model movement evidence;
- operational movement evidence;
- new, persistent, mitigated, or resolved issues;
- one strongest current development tension and one paired wider question;
- an analytical draft for later scoring and narrative shaping.

Pass 1 should identify the strongest current development tension by comparing the current scenario with the previous visible scenario first. The original baseline is background context. A persistent old issue should not become the current discussion point unless the latest change materially affects it or no newer issue is more important.

Pass 2 receives Pass 1 evidence, current Completion Outlook, previous visible score trace, previous Operational Fit and Reality Check values/assessments, previous score-evolution read, up to five compact recent score traces when available, carryover candidate, changed fields, compact operational context, Operational Fit hash/match continuity, structured-feature continuity, and compact Reality Check memory. It does not need the full duplicated Operational Fit state payload in the LLM-facing prompt; that payload can remain diagnostic/audit material. It directly assigns:

- `operational_fit.points`, from `-5` to `+5`;
- `reality_check.points`, from `-15` to `+15`;
- Reality Check allocation rows using canonical `allocation_target_id` values;
- carryover/new-issue relationship text;
- a compact score-evolution read.

The app validates Pass 2 rather than reinterpreting it. App-owned rails remain:

- XGBoost Completion Outlook, SHAP, calibration, artifacts, and `/predict` are unchanged;
- Operational Fit is a current-state score. If the current operational assumptions, operational benchmark/movement context, and structured scenario context match a previous accepted trace, the app requires the same Operational Fit points. If they do not match, Pass 2 may assess the current operational plan inside the `-5/+5` rail;
- a full return to hidden baseline neutralizes Operational Fit and Reality Check;
- point ranges, evidence references, allocation targets, and arithmetic must validate;
- same-state reuse preserves the prior accepted score trace instead of asking the LLM to rescore;
- compact score continuity retains the latest 5 accepted traces for component continuity, structured-feature interpretation continuity, and compact Reality Check memory without sending bulky raw prompts or narratives back to the scoring pass. Full same-state replay remains separate and uses visible trace history keyed by scenario state.

This intentionally gives up deterministic reproducibility for new states. Reproducibility is preserved only for identical same-state replay/cache behavior and deterministic app rails.

Reality Check should be more aggressive on the negative side when a favorable or neutral-looking movement depends on simplification that weakens evidence robustness, endpoint credibility, comparator strength, population relevance, governance, interpretability, decision fitness, or leaves a prior issue materially unresolved. This is scoring calibration inside the hard `-15` to `+15` range, not a mechanical formula or movement-relative cap.

Pass 3 receives the accepted score trace and writes participant-facing narrative only. It must not re-score Operational Fit, re-decide Reality Check, reinterpret model movement, or select among multiple hidden alternatives. The participant flow uses one discussion point, not three candidate questions.

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
If the pre-reality check movement is coherent, realistic, and well explained:
  Reality Check = 0
```

Do not automatically adjust every large move. Reality Check activates only when there is a clear review reason:

- shortcut simplification;
- unrealistic operational or scientific assumptions;
- incoherent answer to the central discussion topic;
- increased rigor that is not operationally credible or fit for purpose;
- negative movement caused by justified robustness or necessary complexity;
- prior development issue resolved, worsened, bypassed, or reopened.

When Reality Check activates, Pass 2 directly assigns `reality_check.points` inside the hard `-15` to `+15` range. The score should reflect the materiality of the incremental issue, the previous score trace, carryover context, and whether the current move creates, resolves, bypasses, or worsens a development concern. It is not calculated as a percentage of pre-reality check movement.

Direction is determined by the sign of `reality_check.points`:

```text
negative points: the Reality Check penalizes, offsets, or challenges the pre-reality check read
positive points: the Reality Check supports, compensates for, or softens the pre-reality check read
zero points: the pre-reality check movement is accepted as coherent enough without additional adjustment
```

Reality Check does not automatically follow the pre-reality check movement. For positive pre-reality check movement, Completion Outlook plus Operational Fit already gives the main credit. Positive movements should usually be accepted with `0` or moderated with a negative adjustment when the after-review judgment finds simplification, brittleness, unrealistic assumptions, or unresolved development issue. Positive reinforcement should remain rare and well supported.

If pre-reality check movement is already positive, Reality Check must be `0` or negative. Accept the gain with `0`, or challenge it with a negative adjustment when the improved score depends on unsupported simplification, unrealistic burden, unresolved carryover, or under-supported operational/scientific assumptions. Positive Reality Check offsets are reserved for unfavorable pre-reality check movement where the model-plus-operational score appears to under-credit a concrete realism, rigor, or fit-for-purpose improvement.

Reality Check should be conservative and challenging. It defaults to neutral unless there is a clear incremental reason not already captured by Completion Outlook or Operational Fit. It should be more willing to challenge favorable movements than to soften unfavorable movements.

For negative pre-reality check movement, Reality Check should usually stay neutral unless there is a distinct incremental concern that reinforces the decline. It may soften the decline only rarely, when the decline is materially harsh and the changed scenario adds a concrete compensating strength not already captured elsewhere. Unchanged strengths can provide context, but should not be the main basis for a non-neutral adjustment.

For near-flat pre-reality check movement, Reality Check should usually remain neutral unless a material simplification, contradiction, carryover issue, or newly resolved concern justifies a non-zero adjustment.

The hard active range is `-15` to `+15`. There is no movement-relative percentage control in the live workflow.

A large Reality Check is exceptional. It requires explicit rationale that the apparent pre-reality check movement is materially misleading, under-supported, shortcut-driven, or missing an important resolved/compensating issue, not merely imperfect.

Examples:

```text
Pre-reality check movement: +6
Reality Check negative adjustment: -6
Final movement: 0

Pre-reality check movement: -6
Reality Check positive adjustment: +3
Final movement: -3

Pre-reality check movement: +6
Reality Check large negative adjustment: -7.5
Final movement: -1.5
```

There is no separate `full_offset` category in V1. If the apparent movement is materially misleading, Pass 2 may assign enough negative or positive Reality Check points to cross through neutral, provided the point value remains inside the hard range and the rationale/evidence/allocation fields validate.

The participant narrative should explain Reality Check as an after-review scoring judgment about whether the pre-reality check score movement is coherent, realistic, and incrementally supported by the scenario evidence. It should not describe Reality Check as a mechanical component essay, and it should not treat Reality Check as the selector of the participant-visible central discussion topic or broader strategic question. Preferred language should connect the adjustment to the score movement, for example whether a simplification made the profile less robust, whether increased robustness made recruitment or execution riskier, or whether the adjustment changes how the Trial Score movement should be read.

### Pillar And Subpillar Allocation

Reality Check defines one overall accepted adjustment first, then allocates that adjustment to existing pillars for traceability.

The LLM may choose where the Reality Check lands by using canonical target IDs, but the visible subgroup/subcategory is always `Reality Check` under the affected pillar. The source subpillar remains stored for audit, while the participant-facing visual gets one concise `Reality Check` row with an app-owned deterministic short explanation. V1 should keep this deliberately small and readable: allocate to 1-3 rows, not a long list of tiny effects.

Reality Check allocations must use canonical `allocation_target_id` values. The application maps those IDs to exact pillars and source subpillars, so the provider should not free-type pillar or subpillar names. For V1, use the current taxonomy subpillars plus the new additive `Operational Fit` subpillar as source targets:

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
    "points": -6,
    "relationship_to_previous": "new_issue",
    "carryover_status": "none",
    "new_issue_status": "new_independent_issue",
    "reason": "The score gain depends on simplification that improves completion resemblance but leaves endpoint interpretability less robust.",
    "incremental_check": "This is incremental beyond Completion Outlook because the score gain comes with evidence loss that the completion-likelihood movement does not price directly.",
    "evidence_fields": ["endpoint_rigor_ml"],
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

Pass 2 returns the accepted Reality Check point proposal directly. The app validates range, evidence references, allocation target IDs, and arithmetic, then distributes accepted points by validated allocation shares. If provider-defined shares are missing, invalid, or do not sum to `1.0`, the app assigns equal shares deterministically across the valid allocation rows.

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

The barchart should show allocated Reality Check rows under the affected pillar with the visible subcategory `Reality Check` and a short human-readable explanation, not provider effect codes.

The Trial Score treemap should show Reality Check as allocated leaves inside impacted existing pillars. It must not render Reality Check as a fifth top-level pillar or as a separate Reality Check row under every source subpillar.

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
- positive pre-reality check movement can only be accepted with `0` or offset downward;
- negative pre-reality check movement cannot be softened;
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

Optional hidden discussion prompts are deferred out of the main participant-narrative flow. If reintroduced, they should be generated separately after the participant narrative succeeds.

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

## Three-Pass Narrative Architecture

Locked direction: use three targeted LLM passes with app validation rails between them.

The application cannot insert accepted score context into a single running LLM response. The active flow separates evolution/evidence generation, score adjudication, and participant-facing narrative shaping so the scoring LLM can judge Operational Fit and Reality Check against previous score trace, carryover, and current evolution without overloading the final narrative pass.

Recommended flow:

```text
Pass 1: Evolution and Evidence
Pass 2: Score Adjudication
Pass 3: Participant Narrative
```

### Pass 1: Evolution and Evidence

Pass 1 receives the full scenario packet and prior continuity context. It returns structured, auditable evidence and interpretation rather than score decisions or polished participant narrative. Its evidence arrays should be bullet-first and compact; `analytical_narrative_draft` should remain short source-note prose that Pass 2 and Pass 3 can use without making Pass 1 write the final participant narrative.

V1 top-level output sections:

```text
review_metadata
completion_outlook_analysis
strategy_shift_check
evolution_evidence
development_discussion_options
continuity_update
analytical_narrative_draft
```

Pass 1 should produce:

- XGBoost / Completion Outlook interpretation;
- total SHAP / score movement, pillar movement, subpillar movement, feature-level SHAP movement, and finest-level changed feature evidence;
- raw current state, baseline state, previous state, and residual state for total score, pillars, subpillars, features, and operational assumptions where available;
- feature, subpillar, and pillar reading with definitions and values;
- consistency update versus prior XGBoost interpretation;
- current-vs-previous and current-vs-baseline evolution evidence;
- new, persistent, resolved, or mitigated issues for the scoring pass;
- one strongest current development tension and one wider participant question for Pass 3;
- continuity update for later iterations.

`strategy_shift_check` is required when gated premise-sensitive fields changed. Allowed status values are `supported`, `partly_supported`, `unsupported_or_incoherent`, and `not_applicable`.

Pass 1 must not return `operational_fit`, `reality_check`, score points, allocation rows, or carryover assessment fields. Those belong to Pass 2.

`development_discussion_options` is the canonical visible-iteration Pass 1 structure. Hidden baseline should not return this field. Visible iterations should contain exactly one complete option, with no main/alternative split. The option pairs the strongest current development tension with one participant-visible wider question assigned to that exact topic.

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

Pass 1 should focus on analytical substance for the option: the development issue, why it matters now, the trial evidence behind it, how it compares with the previous scenario and original baseline, and final participant-visible wider question text that Pass 3 can use verbatim.

The validated contract does not emit old main/alternative Pass 1 candidate fields. Storage, debug context, and Pass 3 input should carry `development_discussion_options` directly, plus the Pass 3 selected participant-facing `central_tension` and `broader_strategic_question` once available.

The participant-visible `broader_strategic_question` selected by Pass 3 should be stored and passed forward with the participant-visible central discussion topic in `recent_participant_visible_questions`. Later Pass 1 and Pass 3 calls should use this history to preserve continuity while avoiding unnecessary repetition: prefer a different wider debate question when the scenario supports a different discussion topic, but reuse or closely echo the previous question when same-state reuse or a clear return to a prior state calls for consistency.

Facilitator questions are no longer part of the main participant-narrative contract. If needed later, they should be generated as a separate optional/lazy step after the participant narrative has succeeded, using final score context and the selected development discussion without slowing or destabilizing the main review flow.

`continuity_update` should be compact and carry what changed since the previous visible iteration and what to watch next. Hidden baseline should not define an active participant-visible discussion topic. Visible participant-facing continuity is stored from the Pass 3 selected central development discussion pair, not from hidden baseline.

For visible iterations, Pass 1 proposes one analytical development discussion option. Pass 3 must use that supplied pair unless same-state reuse requires direct continuity with a prior reviewed state. Provider validation enforces that the selected participant-visible central discussion topic and wider question match the supplied Pass 1 option pair. Repetition avoidance is handled at the prompt-selection level: Pass 1 and Pass 3 compare the current tension against recent participant-visible history and should avoid unnecessary repetition when the scenario supports a different strongest tension. A prior issue that remains unresolved can stay visible in the Trial Score narrative or Reality Check when material without automatically becoming the selected `Discussion Point`.

Pass 1 should not write the final participant-facing score narrative, because it does not yet know the accepted Operational Fit points, Reality Check points, or final Trial Score.

### Pass 2: Score Adjudication

The application validates Pass 1 and then sends a compact scoring input to Pass 2. Pass 2 calculates:

- Operational Fit points;
- pre-reality check score and pre-reality check movement;
- Reality Check points and allocations to existing pillars/subpillars;
- score-evolution read and active issue to carry forward.

The LLM owns new-state Operational Fit and Reality Check points. The app owns hard validation rails, evidence-reference validation, canonical allocation targets, baseline-return neutralization, same-state replay, arithmetic, storage, and UI rendering.

V1 validation/scoring order:

```text
1. Build Pass 1 packet.
2. Run Pass 1.
3. Validate Pass 1 schema.
4. Validate Strategy Shift Check.
5. Build Pass 2 scoring input for visible scenarios.
6. Run Pass 2 Score Adjudication.
7. Validate scoring JSON, point ranges, evidence references, allocation target IDs, same-state/baseline rails, and arithmetic.
8. If needed, run one targeted scoring repair.
9. Calculate accepted Trial Score from Completion Outlook + Operational Fit + Reality Check.
10. Build Pass 3 narrative input.
11. Run Pass 3 Participant Narrative.
12. Validate Pass 3 schema/prose.
13. Store final trace.
```

Reality Check is scored only after Operational Fit points are calculated:

```text
pre_reality_score = XGBoost Completion Outlook + Operational Fit
pre_reality_delta = pre_reality_score - previous visible Trial Score
Trial Score = pre_reality_score + Reality Check
```

For the first visible iteration, compare `pre_reality_score` against baseline XGBoost Completion Outlook when no previous visible Trial Score exists.

If Pass 1 or Pass 2 scoring validation fails after targeted repair, do not show a Trial Score narrative. Preserve XGBoost Completion Outlook and show review unavailable / needs rerun behavior instead.

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

### Pass 3: Participant Narrative

Pass 3 receives:

- accepted score stack;
- Pass 1 structured analysis;
- Pass 2 scoring review;
- previous visible iteration context;
- compact continuity state;
- selected model evidence from Pass 1 rather than broad raw model evidence;
- participant-facing guardrails.

V1 top-level output sections:

```text
review_metadata
trial_score_narrative
pillar_reading
central_tension
broader_strategic_question
```

Pass 3 writes:

- one integrated Trial Score narrative explaining state, movement, and interpretation;
- pillar-level bullets that use subpillar and feature evidence inside prose rather than nested bullet lists;
- one central discussion topic;
- one broader strategic question;
- no separate Completion Outlook essay, Operational Fit essay, and Reality Check essay.

`trial_score_narrative` should briefly explain the current Trial Score reading, movement versus previous/baseline, main reason for movement, and how Completion Outlook, Operational Fit, and Reality Check affect the reading when relevant.

`pillar_reading` is the main UI reading structure. It should include one concise reading per pillar. Subpillar and feature evidence should appear inside the prose, not as nested subpillar/feature bullet lists.

`central_tension` should be the final participant-facing discussion topic from the visible Pass 1 `development_discussion_options`.

`broader_strategic_question` should be reflective and debate-oriented, but still contextualized to the scenario, condition, population, evidence goal, operational setting, or trial context. It should not be a generic conference question detached from the current scenario, and it should not become a direct instruction to change a specific field value.

Pass 3 should use the validated evidence from Pass 1 and accepted score review from Pass 2. It should not make new scoring decisions, invent a supplied discussion topic outside the supplied option, or introduce a new analytical basis.

Pass 3 repair is separate from Pass 1 and Pass 2 scoring repair. If Pass 3 returns invalid participant narrative JSON after scoring succeeded, the app may send one targeted correction prompt with the same narrative input, previous narrative JSON, and exact validation errors. The provider must repair only invalid or missing participant-narrative fields and must not rerun Pass 1, change accepted scores, or change the analytical basis. If the second narrative attempt still fails, Trial Score remains visible and the trace/UI should show a participant narrative warning instead of failing the scenario score.

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
- `Trial Score`: show `Completion Outlook + Operational Fit + Reality Check`. In this view, expose Reality Check as a `Reality Check` subgroup/subcategory inside impacted existing pillars, using `source_subpillar` only for audit and an app-owned deterministic short explanation for display. Do not show Reality Check as a fifth pillar.

Future participant-facing UI may eventually show only `Trial Score`, but that is not the V1 target. During implementation and current product use, keep all three radio buttons visible so scoring behavior can be inspected.

The final participant narrative should assess the total Trial Score, not write separate essays for Completion Outlook, Operational Fit, and Reality Check.

Reality Check allocations should appear within existing pillars as one visible `Reality Check` subgroup/subcategory per allocation row, not as provider-created subpillar labels. In the Reality Check radio, the treemap may use `Reality Check` as the root view, but the visible allocation path should still be existing pillar -> `Reality Check`. In the Trial Score radio, the full composition treemap should embed Reality Check leaves inside impacted existing pillars. Operational Fit and Reality Check must not render as top-level fifth pillars.

## Migration Plan

Implementation status: the active flow is implemented as three provider-facing stages in `src/narratives/trial_score_contract.py`, `src/narratives/prompt_builder.py`, `src/narratives/provider.py`, and adjacent narrative modules:

- Pass 1 Evidence/Evolution: validates `completion_outlook_analysis`, `evolution_evidence`, `strategy_shift_check`, exactly one visible `development_discussion_options` item, `continuity_update`, and `analytical_narrative_draft`.
- Pass 2 Score Adjudication: the LLM assigns direct Operational Fit and Reality Check points; the app validates point ranges, evidence refs, allocation target IDs, baseline-return neutralization, same-state replay, and arithmetic. One targeted scoring repair retry is allowed for invalid scoring JSON.
- Pass 3 Participant Narrative: the LLM shapes the accepted score trace into `trial_score_narrative`, two to four `pillar_reading` bullets, `central_tension`, and `broader_strategic_question`; it must not re-score or introduce unsupported claims. One targeted narrative repair retry is allowed.

Current packet and prompt rules:

- `operational_movement_context` is supplied as evidence for Pass 1 and Pass 2. Planned enrollment, planned sites, planned duration, patients per site, and the operational benchmark/context define Operational Fit continuity. Non-operational edits do not erase a previous Operational Fit assessment when the operational state remains equivalent; Reality Check may still move for non-operational coherence changes.
- Compact model state and movement evidence are supplied so `completion_outlook_analysis.main_model_signals` can use concrete score-pattern evidence without treating every model movement as a direct causal claim.
- Hidden baseline remains compact qualitative context only: no hidden Trial Score, hidden Operational Fit points, hidden Reality Check points, participant-visible questions, or active discussion topic are treated as prior visible history. Hidden baseline uses a bounded fast provider profile and deterministic compact fallback so Simulation Mode is not blocked by rich baseline generation.
- Pass 1 should act as the clinical development, trial design, regulatory strategy, and clinical operations analyst. It should produce substantive compact source-note material, but no scoring objects.
- Pass 2 should compare the current evidence against previous score trace, carryover candidate, new issues, resolved issues, and persistent issues. Carryover is now a scoring-LLM judgment inside validated rails, not the old app formula.
- Pass 3 should use cautious hypothesis language, avoid exact score/point wording in participant-facing prose, use the supplied Pass 1 discussion pair, and describe Reality Check only when material, conflict-relevant, or interpretation-changing.

Validation and repair implementation:

- Active versions are `trial_score_evidence_pass_schema_v4`, `trial_score_scoring_pass_schema_v1`, `trial_score_narrative_pass_schema_v1`, and `trial_score_three_pass_prompt_v2_2`.
- Pass 1 repair can fix schema/evidence/draft scaffolding but must not introduce Pass 2 scoring objects.
- Pass 2 scoring repair can fix schema, required fields, ranges, evidence refs, and allocation target IDs without rerunning interpretation.
- Pass 3 repair can fix participant-narrative fields without rerunning interpretation or changing accepted scores.

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

Historical note: this subsection records the 2026-06-18 cleanup direction and is superseded by the active 2026-06-22 three-pass workflow above where it conflicts.

Main goal: fix and enhance Scenario Review / Trial Score narrative production while simplifying the active implementation around the current contract.

Implemented direction:

- `Trial Score = Completion Outlook + Operational Fit + Reality Check`.
- `Completion Outlook` remains the protected XGBoost/SHAP model anchor.
- Superseded: `Operational Fit` was app-scored in this earlier direction; the active workflow gives Pass 2 LLM ownership of `operational_fit.points` inside app validation rails.
- Superseded: `Reality Check` was app-scored from validated classifications in this earlier direction; the active workflow gives Pass 2 LLM ownership of `reality_check.points` and validated allocation targets.
- Superseded: participant-facing output was generated in Pass 2 in this earlier direction; the active workflow uses Pass 3 for one integrated Trial Score narrative, one central discussion topic, and one broader strategic question. Facilitator questions are not part of the main participant-narrative contract.

Provider and prompt flow:

- Superseded: Pass 1 was the analytical scoring/classification pass in this earlier direction. The active Pass 1 is evolution/evidence only.
- Superseded: app scoring ran between Pass 1 and Pass 2 in this earlier direction. The active Pass 2 scoring call owns new-state Operational Fit and Reality Check point proposals inside app validation rails.
- Superseded: Pass 2 wrote participant narrative in this earlier direction. The active Pass 3 receives accepted scores plus validated Pass 1 and Pass 2 context and writes participant narrative only.
- The active prompt builder is simplified to the current Trial Score V1 three-pass contract.
- OpenAI and Gemini provider paths use the same staged behavior conceptually: Pass 1, targeted repair when needed, Pass 2 scoring, targeted scoring repair when needed, then Pass 3 narrative generation.

Retry and validation behavior:

- Gemini malformed JSON / max-token Pass 1 retry remains reserved for parse failure or provider truncation.
- Pass 1 validation repair retries target invalid classifications, invalid allocation targets, anti-double-counting failures, and invalid scoring structure.
- Pass 2 scoring repair is separate and targets only invalid scoring fields; it does not rerun Pass 1. Pass 3 narrative repair targets only invalid participant-narrative fields and does not change accepted scores.
- Retry history records stage, attempt, validation messages, parse status, latency, response length, and remaining errors.
- After retries are exhausted, failure messages identify the failed level clearly.

Operational Fit and same-state behavior:

- Superseded: Operational Fit was deterministic/app-owned once Pass 1 classifications validated in this earlier direction. The active workflow accepts Pass 2 `operational_fit.points` after app validation.
- A provider cannot keep Operational Fit credit when the current operational state returns to a prior identical scenario state.
- Same-state accepted scoring is reused; Pass 3 may regenerate narrative with explicit reversion/path context.
- This applies to accepted scoring state, not to replaying old participant narrative.

Reality Check behavior:

- Reality Check defaults to zero when the pre-reality check movement is coherent and realistic.
- Positive Reality Check on an already positive pre-reality check movement is not allowed; use `0` to accept the gain or negative points to challenge it. Positive offsets are for unfavorable pre-reality check moves that appear to under-credit rigor, realism, or fit-for-purpose.
- Reality Check is not a fifth pillar.
- Reality Check allocations use canonical allocation target IDs, not free-typed subpillar labels.
- Reality Check allocation display uses the subcategory `Reality Check` with a deterministic short explanation; the canonical target's source subpillar remains audit metadata.
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
