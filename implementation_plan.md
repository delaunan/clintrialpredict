# Implementation Plan: Planned Duration Operational Benchmark

## Status

Planning document for the next `estimation` branch increment.

No runtime duration implementation is approved yet. This plan should be reviewed and approved before editing benchmark builders, runtime utilities, or the Simulation Mode UI.

## Current Baseline

The active operational benchmark layer already supports:

- `planned_enrollment`
- `planned_sites`
- deterministic benchmark metadata
- pending operational-assumption updates
- snapshot-bound metadata
- no XGBoost, SHAP, parity, API, or Completion Score changes

Verification run on 2026-06-03:

```bash
python scripts/check_operational_benchmarks.py
python scripts/build_operational_benchmarks.py
python scripts/check_operational_benchmarks.py
```

Result:

- benchmark checks passed before rebuild
- benchmark builder regenerated `frontend/data/operational_benchmarks_v1.csv`, `frontend/data/operational_benchmarks_v1_report.json`, and `frontend/data/operational_benchmarks_v1.xlsx`
- benchmark checks passed after rebuild

## Current Timeline Data Foundation

The prep pipeline now makes the timeline fields needed for duration planning available after `data/data_clinpred.csv` is rebuilt.

Fields available in rebuilt `data/data_clinpred.csv` and exportable through `frontend/data/search_registry.csv`:

```text
primary_completion_date
primary_completion_date_type
primary_completion_duration_months
completion_date
completion_date_type
completion_duration_months
```

Implementation state:

- `src/prep/data_loader_clinpred.py` imports `primary_completion_date_type` and `completion_date_type` from `studies.txt`.
- `src/prep/data_loader_clinpred.py` derives `primary_completion_duration_months` and `completion_duration_months` from `start_date` using `365.25 / 12` days per month.
- Invalid, missing, or non-positive date differences become missing duration values.
- `src/prep/pipeline.py` registers the six fields as Timeline metadata so they can be exported to `search_registry.csv` and `models/taxonomy_01.json`.
- The six fields are intentionally not model-facing fields and do not enter `FEATURE_REGISTRY`, the preprocessor, XGBoost, SHAP, or parity-sensitive model inputs.
- `notebooks/validation_clinpred.ipynb` and `notebooks/production_01.ipynb` now expose `REBUILD_DATA_CLINPRED`; set it to `True` to rebuild and overwrite `data/data_clinpred.csv`, or `False` to load the existing CSV.

Validated coverage from the filtered trial universe before CSV rebuild:

```text
rows: 34,066
primary_completion_date: 34,046
primary_completion_date_type: 34,066
primary_completion_duration_months: 33,939
completion_date: 33,748
completion_date_type: 34,066
completion_duration_months: 33,672
```

## Goal

Add a third active operational assumption:

```text
planned_duration_months
```

This should estimate and benchmark total operational trial duration in months for Simulation Mode.

It must remain separate from:

```text
primary_duration_months_ml
```

`primary_duration_months_ml` is the model-facing maximum primary endpoint duration already used by XGBoost. `planned_duration_months` is an operational scenario assumption and must stay outside the XGBoost prediction path unless a later architecture decision explicitly changes that.

## Duration Target Definition

Recommended v1 target:

```text
planned_duration_months = total operational trial duration from start_date to completion_date
```

Use `primary_completion_date` to derive `planned_primary_completion_months` as readout-timing context and as a consistency floor for active/planned trials. It is not the same target as total study duration because it measures the primary endpoint milestone rather than full operational completion.

Source columns available in `data/data_clinpred.csv`:

- `start_date`
- `completion_date`
- `primary_completion_date`
- `primary_duration_months`
- `primary_duration_months_ml`
- `is_duration_unknown`
- `is_duration_unknown_ml`

## Scope

In scope:

- derive historical total-duration targets from completed trials with valid `start_date` and `completion_date`
- add duration percentiles to the compact operational benchmark artifact
- add runtime lookup, defaulting, classification, and metadata for `planned_duration_months`
- add a Simulation Mode numeric input and assumption card
- preserve the existing operational-assumption pending/update pattern
- update checker coverage and schema checks
- update `docs/architecture_estimation.md` with the final accepted duration contract after implementation

Out of scope:

- XGBoost retraining
- score adjustment from duration
- SHAP or pillar attribution for duration
- API contract changes
- LLM narrative generation
- Coherence Score implementation
- country, cost, market, spend, or development-commitment estimation

## Source Rules

Date-type interpretation:

```text
ESTIMATED = planned candidate
ACTUAL = observed milestone
```

Status groups:

```text
completed = COMPLETED
stopped = TERMINATED, WITHDRAWN
active_noncompleted = RECRUITING, ACTIVE_NOT_RECRUITING, ENROLLING_BY_INVITATION, NOT_YET_RECRUITING, SUSPENDED
```

Historical benchmark target:

1. Use completed trials only.
2. Require parseable `start_date` and `completion_date`.
3. Require `completion_date_type = ACTUAL`.
4. Require `completion_date >= start_date`.
5. Convert elapsed days to months using a fixed deterministic conversion.
6. Exclude zero or implausibly tiny durations.
7. Do not mix terminated/withdrawn early-stop durations into the completed-trial planned-duration benchmark.

Historical readout benchmark target:

1. Use completed trials only.
2. Require parseable `start_date` and `primary_completion_date`.
3. Require `primary_completion_date_type = ACTUAL`.
4. Require `primary_completion_date >= start_date`.
5. Convert elapsed days to months using the same fixed deterministic conversion.

Initial scenario default priority:

1. Derive `endpoint_duration_months` from `primary_duration_months_ml` when positive.
2. Derive `raw_primary_completion_months` from `start_date -> primary_completion_date` when valid.
3. Derive `raw_total_duration_months` from `start_date -> completion_date` when valid.
4. If the trial is completed and `completion_date_type = ACTUAL`, use `raw_total_duration_months` as `final_observed_total_duration`.
5. If the trial is not completed and `completion_date_type = ESTIMATED`, use `raw_total_duration_months` as `estimated_planned_total_duration`.
6. If the trial is not completed and `completion_date_type = ACTUAL`, treat `raw_total_duration_months` only as an observed lower bound / early-stop duration, not as the planned total duration.
7. If no usable planned total duration exists, use completed-trial benchmark P50 as `model_default`.
8. If the participant edits the value, source becomes `user_scenario`.

Primary completion source priority:

1. If `primary_completion_date_type = ESTIMATED`, use `raw_primary_completion_months` as `estimated_primary_completion_candidate`.
2. If `primary_completion_date_type = ACTUAL`, use `raw_primary_completion_months` as `actual_primary_completion_lower_bound`.
3. If no usable primary completion candidate exists, use completed-trial primary-readout benchmark P50.
4. Always floor by `endpoint_duration_months` when available.

Important non-completed-trial rule:

```text
For terminated, withdrawn, and other non-completed trials, actual dates help establish a minimum duration already reached.
They must not become the planned-duration target by themselves, because the simulation is trying to recreate the duration
that could have been planned at study start without knowing the study would be shortened.
```

Use actual non-completed dates only inside a `max(...)` consistency rule:

```text
planned_primary_completion_months =
max(
  estimated_primary_completion_candidate,
  benchmark_primary_completion_p50,
  endpoint_duration_months,
  actual_primary_completion_lower_bound_if_available
)

planned_duration_months =
max(
  estimated_total_duration_candidate,
  benchmark_total_duration_p50,
  planned_primary_completion_months,
  endpoint_duration_months,
  actual_total_duration_lower_bound_if_available
)
```

If no usable date-derived value exists yet:

```text
planned_primary_completion_months =
max(
  benchmark_primary_completion_p50,
  endpoint_duration_months
)

planned_duration_months =
max(
  benchmark_total_duration_p50,
  planned_primary_completion_months,
  endpoint_duration_months
)
```

If no benchmark exists either:

```text
planned_primary_completion_months = endpoint_duration_months
planned_duration_months = endpoint_duration_months
```

and mark benchmark confidence as low or not available.

Detailed status/date-type handling:

| Case | Use |
| ---- | --- |
| `COMPLETED + ACTUAL completion_date` | Final observed total duration. |
| `non-completed + ESTIMATED completion_date` | Planned total duration candidate. |
| `TERMINATED/WITHDRAWN + ACTUAL completion_date` | Early-stop lower bound only. |
| `active_noncompleted + ACTUAL completion_date` | Lower-bound/context only; likely status lag or administrative inconsistency. |
| `non-completed + ACTUAL primary_completion_date` | Real primary-readout lower bound. |
| `non-completed + ESTIMATED primary_completion_date` | Planned primary-readout candidate. |

Do not use `today - start_date`, `verification_date - start_date`, `last_update_posted_date - start_date`, or extract/update timestamps to create elapsed-duration lower bounds. The simulation default should not change merely because the database is refreshed later.

## Benchmark Hierarchy

Reuse the current operational benchmark hierarchy:

```text
Level 1: same phase + same indication + rare disease flag
Level 2: same phase + same therapeutic area + rare disease flag
Level 3: same phase + same therapeutic area
Level 4: same phase only
```

Duration should initially avoid same-level modality refinement unless source analysis shows stable sample sizes and a real operational-duration signal. This keeps duration less granular than enrollment and patients-per-site until proven otherwise.

## Metadata Contract

Add:

```text
operational_assumptions.planned_duration_months
```

Also derive and store readout timing context:

```text
operational_assumptions.planned_primary_completion_months
```

`planned_primary_completion_months` may be metadata/context in v1 rather than a separate editable input.

Recommended fields:

- `value`
- `source`
- `benchmark_level_used`
- `benchmark_n`
- `benchmark_p25`
- `benchmark_p50`
- `benchmark_p75`
- `benchmark_p90`
- `duration_status`
- `support_level`
- `supporting_signals`
- `conflicting_signals`
- `benchmark_snapshot_id`
- `is_benchmark_stale`
- `low_confidence_flag`
- `interpretation_hint`
- `total_duration_months_observed`
- `primary_completion_duration_months_context`
- `endpoint_duration_months_floor`
- `actual_total_duration_lower_bound`
- `actual_primary_completion_lower_bound`
- `estimated_total_duration_candidate`
- `estimated_primary_completion_candidate`
- `benchmark_total_duration_p50`
- `benchmark_primary_completion_p50`

Classification should mirror enrollment/sites:

```text
below_benchmark
typical
ambitious
above_benchmark_high
not_available
```

## UI Behavior

Simulation Mode should show three operational assumptions:

- Planned Enrollment
- Planned Sites
- Planned Duration

`planned_duration_months` behavior should match the existing operational assumptions:

- numeric input
- pending previous-value marker
- operational-only updates do not call `/predict`
- model-facing Trial Feature edits still require `Predict Trial Completion`
- assumption card shows source, benchmark position, percentiles, confidence, and interpretation hint
- clear text that it does not enter the XGBoost Completion Score
- readout timing can be shown as context, for example `Primary readout timing`, without making it a separate editable field in v1
- if an actual non-completed stop date exists, show it as lower-bound/early-stop context, not as the planned duration source

## Stale-State Rule

For v1, use the same benchmark cohort stale fields as enrollment/sites:

- `phase_ml`
- `gbd_cause_id_3_ml`
- `therapeutic_area_ml`
- `is_rare_disease_ml`
- `therapeutic_modality_ml`

If duration does not use modality refinement in the benchmark hierarchy, decide during implementation whether modality should still trigger stale state for consistency or be removed for duration-specific staleness.

## Implementation Phases

### Phase D0: Analysis

- Count completed trials with valid `start_date` and `completion_date`.
- Count completed trials with `completion_date_type = ACTUAL`.
- Count completed trials with valid `start_date` and `primary_completion_date`.
- Count completed trials with `primary_completion_date_type = ACTUAL`.
- Count non-completed trials where `completion_date_type = ESTIMATED`.
- Count non-completed trials where `completion_date_type = ACTUAL`, grouped by terminated/withdrawn/other status.
- Count non-completed trials where `primary_completion_date_type = ESTIMATED`.
- Count non-completed trials where `primary_completion_date_type = ACTUAL`, grouped by stopped versus active non-completed status.
- Count valid durations by phase, therapeutic area, indication, and rare flag.
- Inspect duration distribution for impossible or extreme values.
- Decide the deterministic month conversion and outlier policy.
- Validate that non-completed actual completion dates are treated as lower bounds / early-stop context only.
- Validate that `endpoint_duration_months = primary_duration_months_ml` acts as a lower bound for both readout and total duration defaults.

### Phase D1: Artifact Builder

Files:

- `scripts/build_operational_benchmarks.py`

Work:

- add total-duration target derivation
- add primary-readout target derivation
- add duration percentile summaries
- add primary-readout percentile summaries or metadata support if approved after D0
- add duration report counts and coverage
- keep artifact compact and deterministic

### Phase D2: Runtime Utility

Files:

- `src/operational_benchmarks.py`

Work:

- add `duration_months` metric support
- add `primary_completion_months` context support
- add default lookup helper
- add `classify_duration_months`
- add `planned_duration_months_metadata`
- add metadata fields for endpoint floors, estimated candidates, and actual lower bounds
- enforce `endpoint_duration_months <= planned_primary_completion_months <= planned_duration_months` for active/planned simulation defaults
- keep behavior outside model-facing prediction code

### Phase D3: Checker

Files:

- `scripts/check_operational_benchmarks.py`

Work:

- add schema checks for duration columns
- add lookup coverage checks
- add defaulting safety checks
- confirm model-boundary checks still prove duration does not enter XGBoost

### Phase D4: Simulation Mode UI

Files:

- `frontend/views/edit_trial.py`

Work:

- activate `planned_duration_months`
- add state/source/baseline/widget helpers
- add numeric input
- include metadata in snapshots
- add assumption card
- include pending operational update behavior

### Phase D5: Documentation

Files:

- `docs/architecture_estimation.md`

Work:

- promote accepted duration target definition
- document source rules, metadata, stale-state behavior, and exclusions
- update active scope from enrollment/sites to enrollment/sites/duration

## Verification

Narrow checks:

```bash
python scripts/build_operational_benchmarks.py
python scripts/check_operational_benchmarks.py
```

If runtime UI files are edited, run the narrowest available app or smoke check for Simulation Mode.

If any model-facing features, preprocess scripts, scoring functions, coefficients, categories, or parity-sensitive paths are touched, run:

```bash
python refresh_registry.py
python audit_parity.py
```

Expected parity rule:

```text
100% Perfect Parity before deployment.
```

## Approval Checkpoint

Before implementation, confirm:

1. `planned_duration_months` means planned total operational trial duration from `start_date` to `completion_date`.
2. `planned_primary_completion_months` is derived as readout-timing context and consistency support, not a separate editable v1 input.
3. For non-completed trials, `ACTUAL` completion/readout dates are lower bounds or early-stop context only; they do not directly define the planned-duration target.
4. Duration remains outside XGBoost and Completion Score.
5. Duration uses the current operational benchmark hierarchy without modality refinement unless D0 proves refinement is warranted.
