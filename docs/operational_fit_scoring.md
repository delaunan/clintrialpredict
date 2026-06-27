# Operational Fit Scoring Logic

## Document Role

Architecture scope: `architecture_narratives`

This document records the planning decisions for adding `Operational Fit` as a subpillar of `Execution Framework` in the Trial Score direction.

It supports `docs/trial_score_narrative_direction.md` and should be read before changing narrative packet fields, provider prompts, scoring code, barcharts, treemaps, or simulator UI behavior for planned enrollment, planned site count, and planned total duration.

Active implementation note, 2026-06-25: the detailed `rating + materiality -> points` table below is retained as historical calibration guidance, not the live scoring algorithm. The active Scenario Review flow gives the LLM direct responsibility for `Operational Fit` points in the Pass 2 Score Adjudication call, while the app validates hard rails: `-5/+5`, evidence references, baseline-return neutralization, arithmetic, and deterministic reuse when the current operational assumptions, operational benchmark/movement context, and structured scenario context match a previous accepted trace. The app retains the latest 5 compact accepted score traces for score/component continuity checks, structured-feature interpretation continuity, and Reality Check memory; full same-state replay uses the review store's visible trace history keyed by scenario state.

## Core Decision

`Operational Fit` is a new additive subpillar under `Execution Framework`.

It should be separate from existing XGBoost / SHAP-derived subpillars such as `Trial Complexity Footprint`, because those existing subpillars already have model-derived SHAP values.

Do not mix non-XGBoost operational assumptions into the same scored leaf as SHAP-derived fields unless provenance is explicitly separated.

Recommended visual hierarchy:

```text
Execution Framework
├── Methodological Setup
├── Trial Complexity Footprint
└── Operational Fit
    ├── Planned enrollment
    ├── Planned site count
    └── Planned total duration
```

`Operational Fit` can be shown in barcharts and treemaps as its own additive contribution. It should not be treated as a fifth model pillar and should not alter XGBoost, SHAP, therapeutic-area calibration, or `/predict`.

## Baseline Neutrality

At scenario start, `Operational Fit = 0`.

The opening operational values are treated as the neutral reference for that trial. They are not assumed to be perfect, clinically optimal, or executable in an absolute sense. They are the best available neutral operational assumption for the selected trial state.

Operational Fit is assessed as a current-state operational proportionality score. It must be reassessed when the participant changes one or more of:

- planned enrollment;
- planned site count;
- planned total duration;
- structured scenario fields that can alter the operational meaning, benchmark context, execution burden, duration proportionality, patient-per-site interpretation, or evidence ambition attached to the same operational estimates.

If the full scenario state matches a previous accepted scenario, same-state replay reuses the whole prior score trace, including Operational Fit and Reality Check. If only the Operational Fit state matches a previous accepted trace, the app reuses that prior Operational Fit points value while Reality Check may still reassess incremental coherence or shortcut concerns. If the operational estimates are unchanged but the structured scenario context changes, Pass 2 may reassess Operational Fit inside the `-5/+5` rail.

The LLM should not ask:

```text
Are these absolute values good or bad?
```

It should ask:

```text
Given the current trial scenario and model movement, did the operational plan become more or less proportionate than the neutral baseline?
```

## Baseline Value Sources

The provider packet should distinguish how each opening value was obtained. `actual` is not a single interpretation.

Use a source taxonomy like:

```text
completed_actual
registered_planned
cohort_p50_estimate
observed_floor_over_estimate
terminated_observed_floor
```

### completed_actual

The trial is completed and the value reflects final or near-final realized conduct.

Interpretation:

- strongest neutral baseline;
- movement away from it is a hypothetical redesign away from known completed conduct;
- do not treat it as universally optimal, but give it higher baseline confidence.

### registered_planned

The value appears planned and has not been overwritten by final actuals.

Interpretation:

- reasonable neutral baseline;
- less certain than completed actual;
- participant movement is a plan modification.

### cohort_p50_estimate

No reliable trial-specific value exists, so the opening value is filled from the closest valid cohort median.

Interpretation:

- neutral by construction;
- participant movement should be judged through percentile shift and scenario fit;
- do not over-trust it as a trial-specific plan.

### observed_floor_over_estimate

An active or non-completed trial has progressively recorded actuals that are higher than the cohort estimate, so the opening value uses `max(observed_actual, cohort_estimate)`.

Interpretation:

- not a final actual;
- lower-bound or floor from observed-to-date conduct;
- movement below this floor is suspect or under-supported because the trial has already recorded at least that much;
- movement above this floor may represent added ambition beyond observed-to-date conduct.

### terminated_observed_floor

A terminated trial has actuals recorded at stopping or termination.

Interpretation:

- not a completed-trial neutral actual;
- stopped-state or partial execution footprint;
- should not be treated as the right planned operational scale;
- movement away may be defensible if the scenario is redesigning away from the terminated pattern.

Recommended packet fields:

```json
{
  "baseline_value_source": "completed_actual | registered_planned | cohort_p50_estimate | observed_floor_over_estimate | terminated_observed_floor",
  "baseline_confidence": "high | medium | low",
  "baseline_interpretation": "final_actual | planned_reference | cohort_neutral_estimate | observed_lower_bound | terminated_partial_actual"
}
```

## Field-Level Questions

Each field should be inspected independently, but only the combined Operational Fit should produce the visible numeric contribution.

### Planned Enrollment

Primary conceptual question:

```text
Is the revised enrollment target proportionate to the target population, eligibility breadth, disease context, endpoint ambition, and evidence goal?
```

This field relates to population ambition and recruitment reality, but for visual simplicity it remains inside `Operational Fit` before Reality Check. Cross-pillar interpretation can later discuss how it interacts with `Patient Profile`.

### Planned Site Count

Primary conceptual question:

```text
Does the revised site footprint create a patient-per-site burden that is more or less proportionate than the neutral baseline, given the closest cohort and current trial scenario?
```

Site count should usually be interpreted through:

```text
patients_per_site = planned_enrollment / planned_site_count
```

The site count is especially meaningful because it positions the scenario against the closer cohort's patient-per-site distribution.

### Planned Total Duration

Primary conceptual question:

```text
Is the revised total duration proportionate to enrollment pace, endpoint follow-up needs, disease trajectory, and operational complexity?
```

Duration should not be treated as automatically good or bad. Longer duration can support adequate follow-up or create operational drag. Shorter duration can improve execution or under-support recruitment and endpoint observation.

### Combined Operational Fit

Final Operational Fit question:

```text
Do the revised enrollment target, site footprint, and total duration form a coherent execution plan for the current trial scenario?
```

The combined answer is the one that should map to points.

## Percentiles And Movement

Operational Fit should separate:

1. movement effect;
2. residual state.

Movement effect asks:

```text
Did the participant move closer to or farther from a plausible cohort pattern?
```

Residual state asks:

```text
Even after the move, does the scenario remain stretched, unusual, overbuilt, or typical?
```

Example for site footprint:

```text
Baseline patient-per-site percentile: P2
Current patient-per-site percentile: P10
Cohort median: P50
```

Interpretation:

```text
The participant improved Operational Fit because the site footprint now reduces patient burden per site and moves from an extreme cohort position toward a less extreme one. However, it remains below the cohort median, so the positive adjustment should be capped or moderated.
```

This should not be penalized simply because P10 remains below median. Direction from baseline matters.

Counter-example:

```text
Baseline patient-per-site percentile: P2
Current patient-per-site percentile: P0.5
```

Interpretation:

```text
The change moves farther from the cohort norm and increases site-level recruitment pressure. If the current value remains close to observed actuals, the penalty should be softened, but the direction is still operationally more stretched.
```

Scoring rule:

```text
If current moves toward P50:
  positive or neutral

If current moves away from P50:
  negative or neutral

If current remains extreme after moving toward P50:
  cap positive score

If current is close to observed actuals:
  soften negative score

If current improves percentile but conflicts with enrollment or duration:
  mixed or slight positive only
```

### V1 Prompt-Facing Movement Context

The V1 narrative packet now carries `operational_movement_context` for the four operational fields that can drive Operational Fit reasoning:

- `planned_enrollment`;
- `planned_sites`;
- `patients_per_site`;
- `planned_duration_months`.

For each field, the packet separates the neutral opening assumption from the current scenario value. It preserves the baseline value, current value, value source, baseline confidence, benchmark position, movement direction, movement magnitude, movement relative to P50, and whether the benchmark cohort/context changed between baseline and current scenario.

Opening operational assumptions are neutral references. A completed actual, registered planned value, cohort estimate, or observed floor may be more or less reliable as evidence, but it is not automatically a positive or negative Operational Fit judgment. Pass 1 should rate the current scenario by combining:

- movement from the neutral baseline;
- residual percentile/status against the closest available cohort;
- benchmark cohort changes caused by updated structured fields;
- source confidence and study evidence.

Percentiles are contextual, not a standalone grade. A large movement from baseline can be acceptable when the current value lands near a plausible cohort percentile, and a small movement can still be weak if it leaves the scenario operationally incoherent. Distance from P50 alone must not determine the rating.

`patients_per_site` is calculated from enrollment and sites, not edited directly. It should be interpreted cautiously as site-footprint proportionality evidence. Completed actual site counts remain valid observed values for completed trials and should not be penalized as estimates, but patient-per-site context should still be shown so the LLM can assess whether the site footprint and enrollment ambition are proportionate.

## Materiality And Rating

Use categorical ratings first, then map them to app-owned points.

Recommended ratings:

```text
strongly_improves_fit
moderately_improves_fit
slightly_improves_fit
neutral_or_unclear
slightly_worsens_fit
moderately_worsens_fit
strongly_worsens_fit
```

Recommended materiality bands:

```text
minor
moderate
major
extreme
```

For the first implementation, lock the app scoring as:

```text
Operational Fit points = lookup(rating, materiality)
Operational Fit points = clamp(points, -5.0, +5.0)
```

The LLM does not return points. It returns rating, materiality, evidence, direction, and rationale. The app owns the numeric mapping.

V1 mapping:

```text
minor:
  slightly_improves_fit       +0.3
  moderately_improves_fit     +0.7
  strongly_improves_fit       +1.0
  neutral_or_unclear           0.0
  slightly_worsens_fit        -0.3
  moderately_worsens_fit      -0.7
  strongly_worsens_fit        -1.0

moderate:
  slightly_improves_fit       +0.7
  moderately_improves_fit     +1.4
  strongly_improves_fit       +2.0
  neutral_or_unclear           0.0
  slightly_worsens_fit        -0.7
  moderately_worsens_fit      -1.4
  strongly_worsens_fit        -2.0

major:
  slightly_improves_fit       +1.2
  moderately_improves_fit     +2.4
  strongly_improves_fit       +3.5
  neutral_or_unclear           0.0
  slightly_worsens_fit        -1.2
  moderately_worsens_fit      -2.4
  strongly_worsens_fit        -3.5

extreme:
  slightly_improves_fit       +1.8
  moderately_improves_fit     +3.5
  strongly_improves_fit       +5.0
  neutral_or_unclear           0.0
  slightly_worsens_fit        -1.8
  moderately_worsens_fit      -3.5
  strongly_worsens_fit        -5.0
```

This mapping makes Operational Fit comparable to a meaningful subpillar when operational changes are major or extreme, while preventing small edits from generating peer-size moves.

Keep the proportionality rule simple:

```text
Small operational edit -> small Operational Fit move.
Large Operational Fit move -> requires a material operational edit.
```

Operational Fit is not budgeted from XGBoost pillar deltas, because enrollment, sites, and total duration are intended to add information not directly captured by XGBoost. However, tiny XGBoost movement plus tiny operational edits should not produce a large Operational Fit adjustment because tiny operational edits should be classified as `minor`.

Simple guardrail:

```text
If all operational changes are minor:
  Operational Fit stays within +/-1.0 through the minor materiality mapping.

If operational changes are moderate:
  Operational Fit stays within +/-2.0 through the moderate materiality mapping.

Use +/-3.5 or more only when materiality is major/extreme because at least one operational field has a major/extreme change, an observed-floor issue, or a large cohort-percentile shift.
```

This avoids a small scenario change producing a `+3` or `-3` Operational Fit move when the XGBoost pillar/subpillar movements are also minimal.

Guardrails:

- `+/-5.0` should require at least two of the three fields to move materially, or one field to become extreme versus cohort.
- Positive `+5.0` should require coherent support across enrollment, site count, and duration, not just one better-looking field.
- If field-level directions conflict, score the combined Operational Fit, not each field independently.
- Do not let Operational Fit dominate the entire Execution Framework unless operational assumptions are central to the scenario.

## Interaction With XGBoost Movements

XGBoost pillar or subpillar movement should not be interpreted as automatically easier, harder, better, or worse.

Correct framing:

```text
XGBoost movement = historical completion-pattern movement
Operational Fit = operational proportionality of the current scenario
Reality Check = coherence, realism, and fit-for-purpose check of the combined pattern
```

A positive XGBoost move can reflect a profile that resembles completed trials more, but it does not automatically mean the design is scientifically stronger or operationally easier.

A negative XGBoost move can reflect a profile that resembles terminated trials more, but it does not automatically mean the design is lower quality or strategically worse.

Operational Fit should ask:

```text
Does the operational plan make the current completion-pattern movement more credible, less credible, or materially unresolved?
```

For unchanged operational assumptions after a non-operational scenario change, Operational Fit should remain score-neutral. The narrative may still explain that the operational plan now looks more or less proportionate for the revised scenario, and Reality Check may correct the total score if that mismatch makes the pre-reality check movement unrealistic or under-supported.

Useful interaction labels:

```text
aligned
under_supported
overbuilt
unmodeled_support
mixed
```

Interpretation:

- `aligned`: operational assumptions support the current scenario pattern.
- `under_supported`: operational assumptions look too thin for the current scenario ambition.
- `overbuilt`: operational assumptions look excessive relative to the scenario evidence goal or scale.
- `unmodeled_support`: operational assumptions add feasibility support that XGBoost does not directly capture.
- `mixed`: operational signals point in different directions or remain ambiguous.

Avoid:

```text
XGBoost up = easier, therefore positive Operational Fit
XGBoost down = harder, therefore negative Operational Fit
```

That is too causal and can double count.

## Provenance And Visual Rules

The barchart and treemap should preserve provenance.

Existing `Trial Complexity Footprint` fields such as delivery profile, number of arms, sponsor type, and maximum primary endpoint duration already have SHAP values. They should remain inside the model-derived XGBoost subpillar.

Operational assumptions should appear as a separate additive subpillar:

```text
Execution Framework
  Methodological Setup              SHAP-derived
  Trial Complexity Footprint        SHAP-derived
  Operational Fit                   Added operational assessment
```

Do not place planned enrollment, planned site count, or planned total duration as children of `Trial Complexity Footprint`, because they did not generate that SHAP contribution.

Do not add automatic cross-pillar adjustment points before Reality Check. For example, do not automatically penalize `Patient Profile` for high enrollment and also penalize `Execution Framework` for the same high enrollment. That double counts the same operational fact.

Before Reality Check:

```text
Operational fields create one direct contribution under Execution Framework.
```

During Reality Check:

```text
The LLM can explain cross-pillar tension qualitatively and decide whether the total Trial Score deserves additional reinforcement, moderation, offset, or reversal.
```

## Suggested Provider Output Shape

The Pass 1 LLM should inspect field-level ratings, but the app should score only the combined Operational Fit before Pass 2. Field-level ratings are explanatory and validation support; they are not summed into separate field scores.

Example:

```json
{
  "operational_fit": {
    "enrollment_fit": {
      "rating": "slightly_worsens_fit",
      "materiality": "moderate",
      "baseline_value": 480,
      "current_value": 700,
      "baseline_value_source": "cohort_p50_estimate",
      "baseline_percentile": 50,
      "current_percentile": 82,
      "movement_direction": "away_from_cohort_median",
      "residual_state": "high_enrollment_ambition",
      "rationale": "..."
    },
    "site_footprint_fit": {
      "rating": "moderately_improves_fit",
      "materiality": "moderate",
      "baseline_sites": 20,
      "current_sites": 40,
      "baseline_patients_per_site": 30,
      "current_patients_per_site": 15,
      "baseline_percentile": 2,
      "current_percentile": 10,
      "movement_direction": "toward_cohort_median",
      "residual_state": "still_stretched",
      "rationale": "..."
    },
    "timeline_fit": {
      "rating": "neutral_or_unclear",
      "materiality": "minor",
      "baseline_value": 36,
      "current_value": 38,
      "baseline_value_source": "registered_planned",
      "baseline_percentile": 55,
      "current_percentile": 60,
      "movement_direction": "near_neutral",
      "residual_state": "typical",
      "rationale": "..."
    },
    "combined_operational_fit": {
      "rating": "slightly_improves_fit",
      "materiality": "moderate",
      "interaction_with_completion_outlook": "unmodeled_support",
      "central_reason": "The larger site footprint reduces patient-per-site pressure enough to improve executability, although enrollment remains ambitious.",
      "evidence_fields": [
        "operational_assumptions.planned_enrollment",
        "operational_assumptions.planned_sites",
        "operational_assumptions.planned_duration_months"
      ]
    }
  }
}
```

The app should calculate points from `combined_operational_fit.rating`, `combined_operational_fit.materiality`, and validated evidence. Field-level ratings explain and validate the combined judgment.

## V1 Locked Decisions And Monitoring Questions

Locked for V1:

- use the V1 rating/materiality point mapping above;
- score only `combined_operational_fit`, not the field-level ratings;
- clamp Operational Fit to `-5.0/+5.0`;
- keep field-level ratings as explanatory and validation support;
- expose field-level ratings only in facilitator/debug context by default.

Monitor during implementation and live testing:

- whether a small positive Operational Fit should apply when only one field improves but the other two remain neutral;
- whether completed actual baselines should cap positive redesign credit more strongly than estimated baselines;
- whether percentile movement should be calculated against P50 only, or against an accepted neutral band such as P25-P75;
- whether the `+/-5.0` guardrails are too permissive or too restrictive.
