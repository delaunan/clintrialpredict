# Implementation Plan: Planned Duration Operational Benchmark

## Status

Implementation document for the current `estimation` branch duration increment.

D0-D3 are implemented for the non-UI duration foundation: artifact builder, runtime utility, checker, and notebook checks. Simulation Mode UI activation remains pending and should be handled separately.

## Current Baseline

The active operational benchmark layer supports:

- `planned_enrollment`
- `planned_sites`
- non-UI `planned_duration_months` runtime/checker/notebook support
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

Current artifact summary:

- Source records loaded: 34,066.
- Completed ACTUAL enrollment targets: 20,526.
- Completed positive site-count proxy targets: 19,880.
- Completed patients-per-site targets: 19,689.
- Completed ACTUAL primary-completion timing targets: 20,681.
- Completed ACTUAL total-duration targets: 20,476.
- Artifact rows: 7,406.
- Duplicate benchmark keys: 0.
- Source data version: `41bcafd6226cd999`.
- Strict runtime duration scan over `frontend/data/search_registry.csv`: 0 `not_available`, 285 unique selected duration cohorts, selected `duration_months_n` minimum 50, median 129, mean about 210.

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

Add a third benchmarked operational-assumption foundation:

```text
planned_duration_months
```

This estimates and benchmarks total operational trial duration in months. It is ready for later Simulation Mode UI wiring, but the UI is not activated in this step.

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

Use `primary_completion_date` to derive `planned_primary_completion_months` as readout-timing context. It is not the same target as total study duration because it measures the primary endpoint milestone rather than full operational completion. The editable duration cohort is selected by full completion duration support, not by primary-completion support.

Source columns available in `data/data_clinpred.csv`:

- `start_date`
- `completion_date`
- `completion_date_type`
- `completion_duration_months`
- `primary_completion_date`
- `primary_completion_date_type`
- `primary_completion_duration_months`
- `primary_duration_months`
- `primary_duration_months_ml`
- `is_duration_unknown`
- `is_duration_unknown_ml`

## Scope

In scope:

- derive historical total-duration targets from completed trials with valid `start_date` and `completion_date`
- add duration percentiles to the compact operational benchmark artifact
- add runtime lookup, defaulting, classification, and metadata for `planned_duration_months`
- update checker coverage and schema checks
- update `notebooks/estimation.ipynb` with duration checks

Out of scope:

- XGBoost retraining
- score adjustment from duration
- SHAP or pillar attribution for duration
- API contract changes
- Simulation Mode numeric input or assumption card
- operational-assumption pending/update UI behavior
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
stopped_interrupted = TERMINATED, WITHDRAWN, SUSPENDED
active_nonstopped = RECRUITING, ACTIVE_NOT_RECRUITING, ENROLLING_BY_INVITATION, NOT_YET_RECRUITING
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

Initial scenario default priority separates source priority from floor constraints. Use a trusted direct date-derived value first. Use benchmark defaults only when direct dates are missing or untrustworthy. Apply floors only in fallback / untrusted cases so the estimate is not shorter than known lower-bound evidence.

Primary completion source priority:

1. Derive `raw_primary_completion_months` from `start_date -> primary_completion_date` when valid.
2. Derive `endpoint_duration_months` from `primary_duration_months_ml` when positive.
3. If the trial is active/non-stopped and `primary_completion_date_type = ACTUAL`, use `raw_primary_completion_months` directly as `actual_primary_completion`.
4. If the trial is active/non-stopped and `primary_completion_date_type = ESTIMATED`, use `raw_primary_completion_months` directly as `estimated_primary_completion`.
5. If the trial is completed and `primary_completion_date_type = ACTUAL`, use `raw_primary_completion_months` directly as completed actual readout context.
6. If the trial is completed, `primary_completion_date_type` is missing, and `raw_primary_completion_months` is valid, use `raw_primary_completion_months` directly for the individual trial default with warning metadata. Do not include this row in the completed benchmark distribution.
7. Otherwise, use completed-trial primary-readout benchmark P50 from the selected context when available and apply fallback floors only in fallback/untrusted cases.

Primary completion fallback / floor rule:

```text
planned_primary_completion_months =
same selected full-duration cohort primary_completion_months_p50
if primary_completion_months_n >= 50
else not_available
```

Primary-completion timing is context for the duration card, not a separate editable input and not an independent cohort selector. The runtime should not choose a different, stronger primary-completion cohort if the selected full-duration cohort has weak primary-completion support.

Total duration source priority:

1. Select the best full-duration benchmark cohort first, using `duration_months_n >= 50`.
2. Read `duration_months_p50` from that selected row for the editable duration benchmark/default.
3. Read `primary_completion_months_p50` from the same selected row only when `primary_completion_months_n >= 50`; otherwise mark primary-completion context as unavailable.
4. Derive `raw_total_duration_months` from `start_date -> completion_date` when valid.
5. If the trial is completed and `completion_date_type = ACTUAL`, use `raw_total_duration_months` directly as `final_observed_total_duration`.
6. If the trial is completed, `completion_date_type` is missing, and `raw_total_duration_months` is valid, use `raw_total_duration_months` directly for the individual trial default with warning metadata. Do not include this row in the completed benchmark distribution.
7. If the trial is active/non-stopped and `completion_date_type = ACTUAL`, use `raw_total_duration_months`, but mark a lower-confidence source such as `actual_completion_noncompleted_status_lag`.
8. If the trial is active/non-stopped and `completion_date_type = ESTIMATED`, use `raw_total_duration_months` directly as `estimated_planned_total_duration`.
9. Otherwise, use completed-trial total-duration benchmark P50 as the candidate and apply stopped/interrupted total-duration floors.
10. If the participant edits the value, source becomes `user_scenario`.

Important non-completed-trial rule:

```text
For stopped/interrupted non-completed trials, actual dates help establish a minimum duration already reached.
They must not become the planned-duration target by themselves, because the simulation is trying to recreate the duration
that could have been planned at study start without knowing the study would be shortened.
```

Total duration fallback / floor rule:

```text
planned_duration_months =
max(
  benchmark_total_duration_p50,
  planned_primary_completion_months_same_cohort if available,
  actual_total_completion_duration_if_stopped_interrupted_and_available,
  estimated_total_completion_duration_if_stopped_interrupted_and_available,
  missing_type_total_completion_duration_if_stopped_interrupted_and_available
)
```

Primary-completion context should not make a sparse full-duration cohort acceptable. It also should not independently pull the editable duration value to a different cohort. In fallback/untrusted cases, primary-completion context may be used only as a same-cohort floor candidate so the total-duration default is not shorter than the readout timing context.

If no usable date-derived value exists and no benchmark exists either:

```text
planned_primary_completion_months = not_available
planned_duration_months = not_available
```

and mark benchmark confidence as not available. `primary_duration_months_ml` remains endpoint context and should not become the editable total-duration default by itself.

Detailed status/date-type handling:

| Case | Use |
| ---- | --- |
| `COMPLETED + ACTUAL completion_date` | Final observed total duration. |
| `COMPLETED + missing completion_date_type + valid completion_date` | Direct individual-trial total duration with warning metadata; excluded from benchmark construction. |
| `active/non-stopped + ACTUAL completion_date` | Direct total-duration candidate with status-lag / lower-confidence metadata. |
| `active/non-stopped + ESTIMATED completion_date` | Direct planned total-duration candidate. |
| `stopped/interrupted + ACTUAL completion_date` | Early-stop lower bound only. |
| `stopped/interrupted + ESTIMATED completion_date` | Floor candidate if still present in the registry, not a trusted direct value. |
| `stopped/interrupted + missing completion_date_type + valid completion_date` | Lower-bound floor/context only, not a trusted direct value. |
| `COMPLETED + ACTUAL primary_completion_date` | Completed actual primary-readout context. |
| `COMPLETED + missing primary_completion_date_type + valid primary_completion_date` | Direct individual-trial primary completion timing with warning metadata; excluded from benchmark construction. |
| `active/non-stopped + ACTUAL primary_completion_date` | Direct reached primary-readout milestone. |
| `active/non-stopped + ESTIMATED primary_completion_date` | Direct planned primary-readout candidate. |
| `stopped/interrupted + ACTUAL primary_completion_date` | Primary-readout lower bound only. |
| `stopped/interrupted + ESTIMATED primary_completion_date` | Floor candidate if still present in the registry, not a trusted direct value. |
| `stopped/interrupted + missing primary_completion_date_type + valid primary_completion_date` | Lower-bound floor/context only, not a trusted direct value. |

Do not use `today - start_date`, `verification_date - start_date`, `last_update_posted_date - start_date`, or extract/update timestamps to create elapsed-duration lower bounds. The simulation default should not change merely because the database is refreshed later.

## Benchmark Hierarchy

Build separate completed-only percentile distributions for:

```text
benchmark_total_duration_months
benchmark_primary_completion_months
```

Use a duration-specific hierarchy that keeps the existing clinical fallback backbone but tries coarse endpoint-duration similarity first:

```text
Level 1: same phase + same indication + rare disease flag + endpoint duration bin
Level 2: same phase + same therapeutic area + rare disease flag + endpoint duration bin
Level 3: same phase + same therapeutic area + endpoint duration bin
Level 4: same phase + endpoint duration bin
Level 5: same phase + same indication + rare disease flag
Level 6: same phase + same therapeutic area + rare disease flag
Level 7: same phase + same therapeutic area
Level 8: same phase only
```

`endpoint duration bin` is derived from `primary_duration_months_ml`, for example:

```text
<=3
3-6
6-12
12-18
18-24
24-36
36-60
>60 months
```

Use a cohort for editable duration only when `duration_months_n >= 50`. If a cohort is too sparse for full completion duration, fall back to the next broader level. Primary-completion context is read from the same selected full-duration row only when `primary_completion_months_n >= 50`.

Placeholder cohort values must not become specific benchmark identities:

```text
If gbd_cause_id_3_ml is 0, missing, or otherwise invalid:
  skip indication-level duration cohorts and continue to TA-level cohorts if TA is valid.

If therapeutic area is missing, UNKNOWN, UNCLASSIFIED, OTHER, or OTHER/UNCLASSIFIED:
  skip TA-level duration cohorts and continue to phase-level cohorts.

If primary_duration_months_ml is missing, non-positive, or cannot be assigned to a valid endpoint duration bin:
  skip endpoint-duration-bin cohorts and continue to the same clinical hierarchy without the endpoint-duration bin.
```

Do not use therapeutic modality, sponsor tier, administration complexity, endpoint rigor, masking, placebo, number of arms, allocation, or comparator benchmark as primary duration benchmark keys in v1. They may later become support/conflict signals after a separate calibration pass.

The D0 temporary analysis showed endpoint-duration bins were the strongest duration predictor and improved cross-validated median/default error. Therapeutic modality had signal but did not improve the tested duration fallback variant enough to justify using it as a primary v1 benchmark key.

## Metadata Contract

Add:

```text
operational_assumptions.planned_duration_months
```

Also derive and store readout timing context:

```text
operational_assumptions.planned_primary_completion_months
```

`planned_primary_completion_months` is metadata/context in v1, not a separate editable input.

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
- `date_type_used`
- `status_group`
- `trusted_direct_date_used`
- `benchmark_default_used`
- `floors_applied`
- `warning_flags`
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

Expected warning flags include:

```text
primary_completion_shorter_than_primary_duration_ml
total_duration_shorter_than_primary_completion
noncompleted_actual_completion_date_used
stopped_actual_date_used_as_lower_bound
completed_missing_completion_date_type_assumed_actual
completed_missing_primary_date_type_assumed_actual
stopped_missing_completion_date_type_used_as_lower_bound
stopped_missing_primary_date_type_used_as_lower_bound
benchmark_low_confidence
```

Classification should mirror enrollment/sites:

```text
below_benchmark
typical
ambitious
above_benchmark_high
not_available
```

## Pending D4 UI Behavior

D4 should make Simulation Mode show three operational assumptions:

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
- opening labels should use no suffix for direct AACT-backed values and `(est.)` for system-filled benchmark/default values
- after participant edit, remove the visible suffix while storing `source = user_scenario` in metadata

## Stale-State Rule

For v1, duration benchmark stale-state should track the fields that define the approved duration benchmark cohort:

- `phase_ml`
- `gbd_cause_id_3_ml`
- `therapeutic_area_ml`
- `is_rare_disease_ml`
- `primary_duration_months_ml`

`therapeutic_modality_ml` should not trigger duration benchmark staleness in v1 unless a later implementation explicitly adds modality to the duration benchmark hierarchy or support/conflict layer.

## Implementation Phases

### Phase D0: Analysis

- Count completed trials with valid `start_date` and `completion_date`.
- Count completed trials with `completion_date_type = ACTUAL`.
- Count completed trials with valid `start_date` and `primary_completion_date`.
- Count completed trials with `primary_completion_date_type = ACTUAL`.
- Count non-completed trials where `completion_date_type = ESTIMATED`.
- Count non-completed trials where `completion_date_type = ACTUAL`, grouped by stopped/interrupted versus active/non-stopped status.
- Count non-completed trials where `primary_completion_date_type = ESTIMATED`.
- Count non-completed trials where `primary_completion_date_type = ACTUAL`, grouped by stopped/interrupted versus active/non-stopped status.
- Count completed and stopped/interrupted trials with valid duration values but missing date types.
- Count valid durations by phase, therapeutic area, indication, and rare flag.
- Inspect duration distribution for impossible or extreme values.
- Decide the deterministic month conversion and outlier policy.
- Validate that non-completed actual completion dates are treated as lower bounds / early-stop context only.
- Validate that `endpoint_duration_months = primary_duration_months_ml` acts as a fallback/floor only when trusted direct primary-completion dates are unavailable or untrusted.

### Phase D1: Artifact Builder - complete

Files:

- `scripts/build_operational_benchmarks.py`

Work:

- add total-duration target derivation
- add primary-readout target derivation
- add duration percentile summaries
- add primary-readout percentile summaries or metadata support if approved after D0
- add duration report counts and coverage
- keep artifact compact and deterministic

### Phase D2: Runtime Utility - complete

Files:

- `src/operational_benchmarks.py`

Work:

- add `duration_months` metric support
- add `primary_completion_months` context support
- add default lookup helper
- add `classify_duration_months`
- add `planned_duration_months_metadata`
- add metadata fields for endpoint floors, estimated candidates, and actual lower bounds
- preserve trusted direct primary-completion dates even when they are shorter than `primary_duration_months_ml`, with warning metadata
- enforce fallback/default consistency without silently overwriting trusted direct date-derived values
- keep behavior outside model-facing prediction code

### Phase D3: Checker and Notebook - complete

Files:

- `scripts/check_operational_benchmarks.py`
- `notebooks/estimation.ipynb`

Work:

- add schema checks for duration columns
- add lookup coverage checks
- add defaulting safety checks
- confirm model-boundary checks still prove duration does not enter XGBoost
- add notebook cells for duration artifact checks, source-priority examples, metadata shape, and single-checker execution

### Phase D4: Simulation Mode UI - pending

Files:

- `frontend/views/edit_trial.py`

Work:

- Activate `planned_duration_months` as the third active operational assumption, alongside `planned_enrollment` and `planned_sites`.
- Add `planned_duration_months` to `ACTIVE_OPERATIONAL_ASSUMPTION_KEYS` and remove it from the future-reserved tuple.
- Import and use `planned_duration_default_from_operational_benchmark(...)` and `planned_duration_months_metadata(...)` from `src/operational_benchmarks.py`.
- Add state/source/baseline/widget helpers mirroring the exact pattern used by Planned Enrollment and Planned Sites:
  - `get_planned_duration_state_key(...)`
  - `get_planned_duration_source_state_key(...)`
  - `get_planned_duration_baseline_state_key(...)`
  - `ensure_planned_duration_state(...)`
  - `get_current_planned_duration_assumption(...)`
  - `get_current_planned_duration_source(...)`
  - `_sync_planned_duration_widget(...)`
  - `get_previous_planned_duration_assumption(...)`
- Add a numeric input labeled `Planned Duration (months)` or `Duration (months)`.
- Apply the same lightweight source-label convention:
  - no suffix when the opening value comes from usable AACT-backed duration data,
  - `(est.)` when the opening value is system-filled from benchmark/default logic,
  - remove the suffix after participant edit while storing `source = user_scenario`.
- Keep the same pending behavior:
  - a modified duration value turns the operational assumption container blue,
  - the label shows `:blue[(previous: X)]`,
  - `Predict Trial Completion` turns blue,
  - operational-only duration changes create/update a snapshot via `simulation_operational_update`,
  - operational-only duration changes do not call `/predict` and do not change Completion Score, SHAP, impact bar, treemap, TA calibration, model artifacts, taxonomy artifacts, or API payloads.
- Include `planned_duration_months_metadata(...)` output in `operational_assumptions.planned_duration_months` for baseline snapshots, simulation prediction snapshots, operational-only update snapshots, and simulation history.
- Add a Duration Assumption card matching the Enrollment/Sites card pattern:
  - current value in months,
  - benchmark position,
  - reference cohort and `n`,
  - P25/P50/P75/P90,
  - primary readout timing context from the same selected full-duration cohort when available,
  - source/context line only when the opening value is system-estimated,
  - explicit copy that it does not enter the XGBoost Completion Score.
- Preserve the strict duration benchmark rule already implemented in runtime:
  - editable duration cohort requires `duration_months_n >= 50`,
  - primary-completion context comes from the same selected full-duration row only if `primary_completion_months_n >= 50`,
  - duration does not use modality refinement or non-vaccine Infections fallback.
- Add duration stale-state tracking for `phase_ml`, `gbd_cause_id_3_ml`, `therapeutic_area_ml`, `is_rare_disease_ml`, and `primary_duration_months_ml`; do not include therapeutic modality for duration stale-state in v1.
- Extend reset/off-on behavior so duration widget/source/baseline keys are cleared and recreated exactly like enrollment/sites.
- Extend static checks to prove `planned_duration_months` is not in `SIMULATION_FEATURE_IDS`, is not in prediction payloads, and is not sent to `/predict`.

### Phase D5: Documentation - pending for UI activation

Files:

- `docs/architecture_estimation.md`
- `docs/architecture_edit.md`
- `docs/architecture_narratives.md`
- `implementation_plan.md`

Work after D4 is implemented:

- Mark `planned_duration_months` as active in Simulation Mode alongside Planned Enrollment and Planned Sites.
- Record the final UI label chosen for the input, currently `Duration (months)` or `Planned Duration (months)`, with `(est.)` only for system-filled opening values.
- Confirm the source convention remains consistent:
  - no suffix for direct AACT-backed opening values,
  - `(est.)` for benchmark/default opening values,
  - no suffix after user edits, with metadata storing `source = user_scenario`.
- Document that the Duration Assumption card follows the Enrollment/Sites card behavior: blue pending state, previous value marker, blue Predict button, operational-only snapshot update, and no `/predict` call unless model-facing Trial Features also changed.
- Document that duration remains outside XGBoost, SHAP, Completion Score, impact bar, treemap, TA calibration, model artifacts, taxonomy artifacts, and API prediction payloads.
- Update the narrative architecture so the future LLM receives `operational_assumptions.planned_duration_months` metadata and treats direct AACT values, benchmark assumptions, and user scenarios according to source metadata rather than visible UI suffixes.
- Add the verification results from D4: py_compile, operational checker, static boundary scans, and browser smoke test scenarios.

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

## D4 Approval Checkpoint

Before UI activation, preserve these already-approved rules:

1. `planned_duration_months` means planned total operational trial duration from `start_date` to `completion_date`.
2. `planned_primary_completion_months` is derived as readout-timing context and consistency support, not a separate editable v1 input.
3. For stopped/interrupted trials, `ACTUAL` completion/readout dates are lower bounds or early-stop context only; they do not directly define the planned-duration target.
4. Duration remains outside XGBoost and Completion Score.
5. Duration uses the approved endpoint-duration-bin + clinical fallback hierarchy without modality refinement.
