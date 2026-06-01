# ClinTrialPredict Enrollment Estimation Architecture

## Purpose

This document defines the v1 enrollment estimation / benchmarking architecture for ClinTrialPredict serious-game mode. It now also records the Phase 1 implementation and QA state for the deterministic planned-enrollment benchmark layer.

The purpose is not to estimate all missing operational quantities. The purpose is to provide a simple, deterministic, auditable benchmark for `planned_enrollment_assumption`.

The enrollment benchmark helps the narrative layer assess whether the selected patient number is coherent with the current simulated trial profile. It does not enter the XGBoost model, does not directly modify the Completion Score, and feeds the `Coherence Score` only.

```text
V1 estimation scope = planned enrollment assumption only.
Sites, countries, total duration, cost, market potential, downstream commitment, and full operational-scale estimation are postponed.
```

This document should be read alongside [docs/architecture_narratives.md](/home/delaunan/code/delaunan/clintrialpredict/docs/architecture_narratives.md), which defines how the benchmark is interpreted through the serious-game narrative layer.

## Phase 1 Implementation Status - 2026-06-01

Phase 1 is implemented as a deterministic, offline-built enrollment benchmark foundation. It does not add Simulation Mode UI, does not call an LLM, does not implement Coherence Score, and does not touch XGBoost, SHAP, therapeutic-area calibration, audit/demo parity, model artifacts, taxonomy files, API contracts, or deployment configuration.

Implemented files:

- `scripts/build_enrollment_benchmarks.py`: offline artifact builder and calibration report generator.
- `scripts/check_enrollment_benchmarks.py`: lightweight validation for artifact schema, strict lookup, fallback lookup, missing artifact behavior, and classification boundaries.
- `src/enrollment_benchmarks.py`: runtime lookup and metadata utility.
- `frontend/data/enrollment_benchmarks_v1.csv`: compact production-friendly benchmark artifact.
- `frontend/data/enrollment_benchmarks_v1_report.json`: practical audit/calibration report.
- `notebooks/estimation.ipynb`: central reproducible analytical notebook for the v1 planned-enrollment benchmark layer.
- `notebooks/archive/estimation_legacy_before_enrollment_benchmark.ipynb`: archived broad-estimation notebook retained for history.

Reproducibility workflow:

```bash
python scripts/build_enrollment_benchmarks.py
python scripts/check_enrollment_benchmarks.py
```

Then open/run:

```text
notebooks/estimation.ipynb
```

The notebook inspects and validates the artifact and also reproduces the main calculations in memory in the `BENCHMARK_COHORTS`, `BENCHMARK_PERCENTILES`, and `VALIDATION` sections. The build script remains the source of truth for writing the production-facing artifact.

Current artifact summary:

```text
source records loaded: 34,066
completed positive ACTUAL enrollment benchmark targets: 20,526
artifact rows: 876
minimum confident cohort threshold: n >= 50
low-confidence benchmark rows: 667
duplicate benchmark keys: 0
```

Rows by benchmark level:

```text
phase_indication_rare: 658
phase_ta_rare:        138
phase_ta:              76
phase_only:             4
```

Phase-only fallback rows are all confident:

```text
PHASE1/PHASE2:  n=1,907, P25=24.0,  P50=48.0,  P75=108.0, P90=225.0
PHASE2:         n=8,987, P25=41.0,  P50=92.0,  P75=190.0, P90=332.0
PHASE2/PHASE3:  n=511,  P25=89.5,  P50=222.0, P75=454.0, P90=974.0
PHASE3:         n=9,121, P25=150.0, P50=326.0, P75=606.0, P90=1,077.0
```

Coverage QA across all 34,066 source snapshots found no benchmark lookup gaps:

```text
phase_indication_rare: 23,299 rows
phase_ta_rare:         7,361 rows
phase_ta:              2,270 rows
phase_only:            1,136 rows
not_available:             0 rows
low-confidence matches:    0 rows
```

When using each row's current enrollment as the planned value, `not_available` classifications came only from invalid enrollment input values:

```text
typical:               15,494
below_benchmark:        9,088
ambitious:              4,775
above_benchmark_high:   3,463
not_available:          1,246

missing enrollment:        14
zero/non-positive:       1,232
```

Therefore, with a valid phase, valid planned enrollment value, and present/correct artifact, runtime should return a benchmark. The broadest fallback is `phase_only`. `not_available` should normally mean missing/invalid planned enrollment, missing/corrupt artifact, unrecognized phase outside the artifact, or incomplete percentile values.

Validation commands run during Phase 1 and QA:

```bash
python scripts/build_enrollment_benchmarks.py
python scripts/check_enrollment_benchmarks.py
python -m py_compile scripts/build_enrollment_benchmarks.py scripts/check_enrollment_benchmarks.py src/enrollment_benchmarks.py
git diff --check
```

Audit parity was previously confirmed at `4,423/4,423` perfect parity and was not rerun during the final QA pass because no prediction, audit, preprocessing, model, SHAP, therapeutic-area calibration, API, taxonomy, or deployment files were touched.

## Current Foundation

ClinTrialPredict currently provides:

- An AACT / ClinicalTrials.gov-based trial database.
- Industry-led Phase II / Phase III trial data.
- Existing structured Trial Features used in simulation mode.
- An XGBoost completion / early-termination prediction engine.
- SHAP-derived completion-score explanation artifacts.
- Existing audit, parity, and therapeutic-area calibration behavior.

The enrollment benchmark must consume existing trial fields and simulation snapshots without changing the completion model path.

## Core Separation Principle

Never mix design-stage inputs, observed-to-date lower bounds, final observed values, and synthetic estimates.

For enrollment v1, use these concepts:

```text
planned_value = planned/estimated enrollment available at design stage or in the record.
final_observed_value = actual enrollment from completed trials, usable as historical benchmark data or completed-trial display value.
observed_lower_bound = actual enrollment for ongoing/actionable trials, not final truth.
model_default = benchmark-derived default used when no planned value is available.
user_scenario = participant-edited enrollment assumption.
benchmark_distribution = historical comparable-trial distribution used for P25/P50/P75/P90.
```

Clarifications:

- Completed-trial actual enrollment can be used to build benchmark distributions.
- Ongoing actual enrollment must not be treated as final total.
- Estimated/planned enrollment can be used as the starting scenario value when available.
- User scenario value is the participant's current assumption and should be compared against the benchmark.
- The benchmark position is not clinical truth; it is a reference point.

## Field-Use Contract

Each enrollment-related value should be classified before use.

| Field class | Meaning | Example | V1 use |
| ----------- | ------- | ------- | ------ |
| Design-stage / planned value | Known or intended near trial design time. | Estimated enrollment from the record. | Initial `planned_enrollment_assumption` when available. |
| Final observed value | Known only after trial completion. | Completed-trial actual enrollment. | Historical benchmark data or completed-trial display context. |
| Observed-to-date lower bound | Current partial operational value, not necessarily final. | Ongoing actual enrollment from current extract. | Lower-bound context only; not final truth. |
| Model default | Synthetic benchmark-derived default. | P50 or other selected cohort default when no planned value exists. | Neutral starting assumption. |
| User scenario | Participant-edited assumption. | User changes enrollment to 600. | Current scenario value evaluated against benchmark and design profile. |
| Benchmark distribution | Historical comparable-trial enrollment values. | Cohort P25/P50/P75/P90. | Deterministic benchmark reference. |

Rule: every stored enrollment benchmark output must preserve source, cohort, percentile, status, and snapshot metadata so future reviewers can tell which value was planned, observed, lower-bound, defaulted, or user-edited.

## V1 MVP Definition

The v1 estimation MVP is not a full missing-operational-value workbench. It is an enrollment-benchmark layer for serious-game simulation. It provides a neutral, auditable benchmark against which a user's planned enrollment assumption can be assessed.

MVP includes:

- `planned_enrollment_assumption`.
- Enrollment source priority.
- Deterministic benchmark hierarchy.
- P25/P50/P75/P90 benchmark percentiles.
- `enrollment_status` classification.
- `support_level` classification.
- Support/conflict signals.
- Benchmark stale-state logic.
- Snapshot metadata for the narrative payload.
- Integration with Coherence Score through the narrative architecture.

Explicitly excluded from MVP:

- Full operational-size estimation.
- Site-count estimation.
- Country-count estimation.
- Total-duration estimation.
- Cost engine.
- Spend curve.
- Calendar spend model.
- Future phase modelling.
- Future development commitment model.
- Market potential.
- Model training beyond optional later calibration.
- Direct Completion Score adjustment.
- Pillar-level Coherence attribution.
- Pseudo-SHAP from LLM.

## Planned Enrollment Assumption Source Priority

Use this source priority:

```text
1. If planned/estimated enrollment is available, use it as the initial `planned_enrollment_assumption`.
2. If the trial is completed and actual enrollment represents final observed value, it can be used as final observed context.
3. If no usable planned value exists, use a benchmark-derived `model_default`.
4. If the participant edits the value, source becomes `user_scenario`.
```

The enrollment assumption is assessed against the current design choices. The enrollment assumption must be supported by the selected trial profile.

Practical behavior:

```text
model_default inside benchmark = neutral.
user_scenario inside benchmark = usually neutral or lightly supportive if consistent with the design.
user_scenario outside benchmark = discussion signal, not automatic penalty.
user_scenario outside benchmark + conflicting design signals = possible Coherence Score penalty.
```

If planned enrollment is missing and the system uses a benchmark-derived default, that default is neutral. It should not create a positive Coherence Score effect simply because it sits inside the benchmark range. It becomes an evaluated scenario assumption only when the user keeps it as the current assumption for a prediction snapshot or actively edits it.

## Benchmark Snapshot Logic

The enrollment benchmark belongs to the current prediction snapshot, not permanently to the original trial.

If the participant changes indication, therapeutic area, rare disease flag, phase, modality, patient profile, endpoint design, or other relevant design features, the old enrollment benchmark becomes stale. The benchmark should refresh only after the user clicks `Predict Trial Completion`, consistent with the existing simulation snapshot workflow.

During editing, the UI may show:

```text
Enrollment benchmark will refresh after prediction
```

Core rule:

```text
Current design snapshot -> benchmark cohort -> P25/P50/P75/P90 -> enrollment_status -> Coherence Score input.
```

## Benchmark Cohort Hierarchy

Use this exact v1 hierarchy:

```text
Level 1: same phase + same indication + rare disease flag
Level 2: same phase + same therapeutic area + rare disease flag
Level 3: same phase + same therapeutic area
Level 4: same phase only
```

Rules:

```text
Use the strictest level with enough historical trials.
Use the Phase 1 minimum sample threshold: n >= 50.
If n is too small, relax one level.
If all levels are sparse, return a low-confidence benchmark and avoid overinterpreting the enrollment status.
```

Therapeutic modality, sponsor tier, administration complexity, line of therapy, patient subtype, endpoint rigor, and endpoint duration should not define the primary benchmark cohort in v1. They can be used as support/conflict signals for Coherence Score. This avoids excessive stratification and unstable percentiles.

## Enrollment Benchmark Calibration Gate

Before implementation, run an Enrollment Benchmark Calibration Gate to confirm that the selected cohort hierarchy provides stable percentile benchmarks and that excluded fields are better used as support/conflict signals rather than primary matching fields.

The goal is not primarily to prove statistical significance with p-values. The goal is to build stable and useful benchmark cohorts.

P-values can be misleading for this use case: with very large cohorts, tiny differences can become statistically significant; with small cohorts, useful operational patterns may not reach significance. The benchmark should therefore prioritize stability, coverage, effect size, and interpretability.

Calibration target:

```text
log1p(actual_enrollment) for completed trials with reliable final actual enrollment.
```

Completed actual enrollment can be used for historical benchmark calibration. Ongoing actual enrollment should not be treated as final truth. Estimated/planned enrollment may be useful as a design-stage context field but should not replace completed actual enrollment when calibrating final historical benchmark distributions.

Candidate benchmark or support-signal fields:

- Phase.
- Indication / GBD L3 indication.
- Therapeutic area.
- Rare disease flag.
- Therapeutic modality.
- Sponsor tier.
- Administration complexity.
- Line of therapy.
- Patient severity.
- Adult / child / older adult flags.
- Healthy volunteer status.
- Endpoint rigor.
- Endpoint structure.
- Primary endpoint duration.
- Number of arms.
- Comparator benchmark.
- Placebo control.
- Allocation.
- Masking.

Primary benchmark fields define the historical comparison cohort. Support/conflict signals help the Coherence Score interpret whether the selected enrollment is supported by the current design.

Fields such as therapeutic modality, sponsor tier, administration complexity, line of therapy, patient subtype, endpoint rigor, endpoint duration, comparator, and number of arms may influence enrollment feasibility, but in v1 they should normally remain support/conflict signals unless the calibration gate proves they are stable enough for primary cohort matching.

A field can enter the primary benchmark hierarchy only if it has:

- Meaningful effect size on log enrollment.
- Good coverage across the dataset.
- Enough samples per group.
- Stable P25/P50/P75/P90 percentiles.
- Acceptable bootstrap confidence interval width.
- Limited outlier sensitivity.
- Stable `enrollment_status` labels.
- Simple interpretation for users and facilitators.

If a field has signal but weak coverage or unstable percentiles, it should remain a support/conflict signal for the Coherence Score rather than a primary benchmark matching field.

Calibration checks:

1. Coverage check:
   - How many trials have non-missing values for each candidate field?

2. Group size check:
   - For each candidate field and candidate field combination, how many benchmark groups have `n >= 50`?
   - How often would the system need to fall back to a broader cohort level?

3. Distribution separation check:
   - Compare median and interquartile range of `log1p(actual_enrollment)` across groups.
   - Prefer effect sizes and distribution separation over p-values alone.

4. Percentile stability check:
   - Bootstrap P25/P50/P75/P90 for candidate cohorts.
   - Flag cohorts where percentile estimates are unstable.

5. Outlier sensitivity check:
   - Test whether very large trials dominate the mean or distort percentiles.
   - Prefer median and percentile logic over mean-only logic.

6. Label stability check:
   - Check how often the same trial would move between `below_benchmark`, `typical`, `ambitious`, and `above_benchmark_high` under small bootstrap variations.

7. Fallback behavior check:
   - Confirm that the four-level hierarchy produces a usable benchmark for most trials.
   - Confirm that sparse groups fall back cleanly to broader levels.

The default v1 hierarchy remains:

```text
Level 1: same phase + same indication + rare disease flag
Level 2: same phase + same therapeutic area + rare disease flag
Level 3: same phase + same therapeutic area
Level 4: same phase only
```

The calibration gate can recommend changes later, but the first implementation should stay simple unless the analysis clearly shows that another field materially improves benchmark stability and relevance without creating sparse cohorts.

### Phase 1 Calibration Findings

The Phase 1 calibration gate used the benchmark-eligible population:

```text
overall_status == COMPLETED
enrollment_type == ACTUAL
enrollment > 0
N = 20,526
```

The gate checked candidate-field coverage, group sample sizes, effect range on `log1p(actual_enrollment)`, percentile spread, outlier sensitivity, fallback behavior, and label stability. It did not use p-values as the main decision basis.

Key group-size findings:

```text
phase:
  groups=4, min=511, median=5,447, max=9,121, groups n>=50=4/4

therapeutic_area:
  groups=19, min=97, median=1,059, max=3,304, groups n>=50=19/19

gbd_cause_id_3_ml:
  groups=152, min=1, p25=14, median=52, p75=172, max=1,380, groups n>=50=77/152

is_rare_disease_ml:
  groups=2, group sizes 18,317 and 2,209, groups n>=50=2/2
```

Interpretation:

- Phase is stable and belongs in every benchmark level.
- Therapeutic area is stable and is a strong fallback dimension.
- Indication / GBD L3 has useful clinical specificity and signal, but many groups are sparse. It is appropriate for the strictest level only when `n >= 50`; otherwise fallback is required.
- Rare-disease flag is stable and clinically important for strict matching.

Other candidate-field findings:

```text
primary_duration_months_ml:
  groups=851, min=1, p25=1, median=1, p75=6, max=2,160, groups n>=50=59/851

sponsor_tier_ml:
  Top-Tier Pharma:              n=8,263,  P50=213, P90=921.8
  Mid-Cap Pharma:               n=1,943,  P50=162, P90=675.6
  Biotech and Emerging Pharma:  n=10,320, P50=120, P90=545

biomarker_stratification_ml:
  groups=2, group sizes 17,159 and 3,367, groups n>=50=2/2
```

Interpretation:

- Exact primary duration is a continuous/high-cardinality value and fragments the data too heavily. It should not be used as an exact primary cohort key in v1. Future calibration may test duration bins, but adding duration bins to indication-level matching may still create sparse cohorts.
- Sponsor tier clearly shifts enrollment distributions, but adding it to the primary hierarchy would over-stratify already sparse indication groups. It should be a Phase 2 support/conflict signal.
- Biomarker stratification is stable as a binary field and should be explicitly included in future support/conflict-signal calibration, but it is not a v1 primary benchmark key.
- Therapeutic modality and other design features may have meaningful signal, but they remain support/conflict candidates unless a future calibration proves they improve relevance without harming coverage and label stability.

Outlier finding:

```text
median completed ACTUAL enrollment: 154
P99 completed ACTUAL enrollment:    3,548.75
max completed ACTUAL enrollment:    90,116
```

This supports percentile-based benchmarking rather than mean-based benchmarking. Large outlier trials exist, and P25/P50/P75/P90 are more robust and auditable for v1.

## Deterministic Enrollment Classification

Calculate benchmark percentiles:

```text
P25 = low benchmark
P50 = typical benchmark
P75 = high benchmark
P90 = very high benchmark
```

Classification:

```text
if enrollment < P25:
    enrollment_status = "below_benchmark"

if P25 <= enrollment <= P75:
    enrollment_status = "typical"

if P75 < enrollment <= P90:
    enrollment_status = "ambitious"

if enrollment > P90:
    enrollment_status = "above_benchmark_high"
```

The deterministic enrollment label is a benchmark position, not a clinical judgment. Clinical interpretation belongs to the Coherence Score layer.

Examples:

```text
below_benchmark + Phase III + rigorous endpoint may suggest possible evidence-size concern.
below_benchmark + rare disease + Phase II may be acceptable.
above_benchmark_high + common adult Phase III may be plausible.
above_benchmark_high + rare pediatric late-line therapy may signal recruitment burden.
```

## Enrollment Support And Conflict Signals

The benchmark label alone is not enough. The system should check whether the current design profile supports the selected enrollment assumption.

Supportive signals:

```text
- common indication,
- non-rare disease,
- adult population,
- broad patient profile,
- earlier line of therapy,
- larger sponsor tier,
- simpler therapeutic modality,
- simpler administration,
- simple endpoint structure,
- sufficient primary endpoint duration,
- Phase III context when a large confirmatory trial is expected.
```

Conflicting signals:

```text
- rare disease,
- pediatric population,
- severe or fragile population,
- later-line population,
- complex therapeutic modality,
- complex administration,
- strict endpoint or hard clinical endpoint,
- short endpoint duration,
- niche indication.
```

Support level values:

```text
support_level = "supported_by_current_design | partly_supported_by_current_design | weakly_supported_by_current_design"
```

Examples:

```text
Common adult Phase III disease + 1,200 patients:
above benchmark but potentially supported.

Rare pediatric Phase II gene therapy + 1,200 patients:
above benchmark high and weakly supported by the design.
```

## Relationship To Coherence Score

Enrollment is one input into Coherence Score, not the Coherence Score itself. It should mainly influence:

- Operational feasibility.
- Population relevance.
- Change integrity.

Enrollment must not dominate the broader rubric.

```text
Enrollment should normally have a maximum standalone effect of about -10 to +4 points inside the Coherence Score logic.
```

Interpretation principle:

```text
One weak enrollment signal creates a discussion point.
Several weak or conflicting design signals create a Coherence Score penalty.
A difficult design can still receive a positive Coherence Adjustment if the participant strengthens it coherently.
```

Alignment with the narrative architecture:

```text
Completion Score = untouched XGBoost score.
Coherence Score = design defensibility and risk-mitigation quality.
Coherence Adjustment = deterministic application calculation.
Adjusted Trial Value Score = Completion Score + Coherence Adjustment.
```

The full serious-game scoring and narrative contract belongs in [docs/architecture_narratives.md](/home/delaunan/code/delaunan/clintrialpredict/docs/architecture_narratives.md). This document defines only the enrollment benchmark metadata that feeds that layer.

## Enrollment Benchmark Metadata Object

Planning JSON example:

```json
{
  "planned_enrollment": {
    "value": 600,
    "source": "planned_value | final_observed_value | observed_lower_bound | model_default | user_scenario",
    "benchmark_level_used": "phase_indication_rare | phase_ta_rare | phase_ta | phase_only | not_available",
    "benchmark_n": 123,
    "benchmark_p25": 120,
    "benchmark_p50": 220,
    "benchmark_p75": 420,
    "benchmark_p90": 750,
    "enrollment_status": "below_benchmark | typical | ambitious | above_benchmark_high | not_available",
    "support_level": "not_evaluated",
    "supporting_signals": [],
    "conflicting_signals": [],
    "benchmark_snapshot_id": "...",
    "is_benchmark_stale": false,
    "low_confidence_flag": false,
    "interpretation_hint": "Enrollment is above the usual benchmark and is only partly supported by the current design choices."
  }
}
```

This object should be assembled after the latest prediction snapshot and passed to the narrative layer. It should be stored with the serious-game prediction snapshot so later iterations can compare the user's scenario path without recomputing historical context ambiguously.

Phase 1 runtime returns `support_level = "not_evaluated"` and empty `supporting_signals` / `conflicting_signals`. Support/conflict logic is deferred to Phase 2.

## Production Runtime Artifact Strategy

The production app should not need to load the full historical `data_clinpred.csv` dataset in order to calculate enrollment benchmarks at runtime.

The full historical dataset is used offline in the analytical notebook or build script to run the Enrollment Benchmark Calibration Gate and precompute a compact benchmark artifact.

Phase 1 artifact path:

```text
frontend/data/enrollment_benchmarks_v1.csv
```

Phase 1 report path:

```text
frontend/data/enrollment_benchmarks_v1_report.json
```

This location was selected because `frontend/data/` is already copied into the app image and already holds compact app-loaded artifacts such as `search_registry.csv` and `gbd_l3_indication_lookup.csv`.

The artifact should contain one row per benchmark cohort and fallback level.

Recommended artifact fields:

- `benchmark_version`
- `source_data_version`
- `benchmark_key`
- `phase`
- `indication_or_therapeutic_area`
- `rare_disease_flag`
- `benchmark_level_used`
- `benchmark_n`
- `benchmark_p25`
- `benchmark_p50`
- `benchmark_p75`
- `benchmark_p90`
- `low_confidence_flag`
- `created_at`
- `outlier_policy`
- `calibration_notes`, optional

Phase 1 artifact schema:

```text
benchmark_version
source_data_version
benchmark_key
phase
indication_or_therapeutic_area
gbd_cause_id_3_ml
therapeutic_area
rare_disease_flag
benchmark_level_used
benchmark_n
benchmark_p25
benchmark_p50
benchmark_p75
benchmark_p90
low_confidence_flag
created_at
outlier_policy
calibration_notes
```

At runtime, the app uses the current prediction snapshot to look up the relevant benchmark row, apply deterministic fallback logic if needed, classify the current `planned_enrollment_assumption`, and pass the resulting metadata to the narrative layer.

Runtime should require only:

- The current trial prediction snapshot.
- The compact enrollment benchmark artifact.
- Deterministic fallback logic.

Runtime should not require:

- The full raw historical trial database.
- Notebook-only calibration data.
- Model retraining.
- Recomputing all benchmark percentiles from scratch.

This keeps the production app lighter, faster, more auditable, and less dependent on raw historical data.

Current runtime utility:

```text
src/enrollment_benchmarks.py
```

Main functions:

```text
load_enrollment_benchmarks(...)
lookup_enrollment_benchmark(...)
classify_enrollment(...)
planned_enrollment_metadata(...)
```

Runtime fallback behavior:

```text
1. Try phase + indication + rare disease.
2. If missing or low confidence (n < 50), try phase + therapeutic area + rare disease.
3. If missing or low confidence, try phase + therapeutic area.
4. If missing or low confidence, try phase only.
5. If no confident row exists, allow an available low-confidence row as a last-resort benchmark.
6. Return not_available if the artifact is missing/corrupt, phase cannot be matched, planned enrollment is invalid/missing, or percentile values are incomplete.
```

## First Implementation Path

Focused v1 path and Phase 1 status:

1. Audit enrollment fields and enrollment_type values in `data/data_clinpred.csv`. Completed in notebook and report.
2. Create enrollment source flags:
   - `is_completed_actual_enrollment_target`.
   - `is_estimated_planned_enrollment`.
   - `is_ongoing_actual_enrollment_lower_bound`.
3. Run the Enrollment Benchmark Calibration Gate:
   - evaluate candidate fields,
   - check coverage and sample size,
   - test percentile and label stability proxies,
   - confirm that the default hierarchy is acceptable,
   - decide which fields remain support/conflict signals.
4. Precompute and save a compact enrollment benchmark artifact:
   - one row per benchmark cohort and fallback level,
   - P25/P50/P75/P90,
   - cohort size,
   - benchmark level,
   - confidence flag,
   - benchmark version and source data version.
5. Build a runtime benchmark lookup function using the approved v1 hierarchy and compact artifact.
6. Calculate or retrieve P25/P50/P75/P90 for the selected current snapshot.
7. Classify the current assumption as `below_benchmark`, `typical`, `ambitious`, `above_benchmark_high`, or `not_available`.
8. Keep `support_level = "not_evaluated"` and support/conflict lists empty until Phase 2.
9. Keep XGBoost, SHAP, therapeutic-area calibration, audit mode, and parity behavior unchanged.

Do not require model training in v1. The first implementation should use deterministic cohort percentiles.

Not yet implemented:

- Simulation Mode Planned Enrollment UI field.
- Initial planned-enrollment assumption selection in the UI/session state.
- Storage of benchmark metadata in prediction snapshots.
- Support/conflict signal generation from structured Trial Features.
- Narrative / Coherence layer call.

## Notebook Plan And Current Workbook

The central notebook is:

```text
notebooks/estimation.ipynb
```

The previous broad missing-operational-value notebook was archived to:

```text
notebooks/archive/estimation_legacy_before_enrollment_benchmark.ipynb
```

The current notebook proves and documents the enrollment benchmark contract before UI or API integration. It is an analytical workbook, not the source of truth for writing the production-facing artifact. The source-of-truth artifact build command is:

```bash
python scripts/build_enrollment_benchmarks.py
```

That rebuilds:

```text
frontend/data/enrollment_benchmarks_v1.csv
frontend/data/enrollment_benchmarks_v1_report.json
```

Then validate with:

```bash
python scripts/check_enrollment_benchmarks.py
```

Recommended v1 notebook / implementation blocks:

```text
<REF:DATA_LOAD> Load data/data_clinpred.csv.
<REF:ENROLLMENT_AUDIT> Audit enrollment and enrollment_type.
<REF:ENROLLMENT_FLAGS> Build enrollment source and target-readiness flags.
<REF:CALIBRATION_GATE> Evaluate candidate enrollment benchmark fields for coverage, effect size, sample size, percentile stability, outlier sensitivity, fallback behavior, and label stability.
<REF:BENCHMARK_COHORTS> Build phase/indication/TA/rare-disease benchmark cohorts.
<REF:BENCHMARK_PERCENTILES> Calculate P25/P50/P75/P90.
<REF:BENCHMARK_ARTIFACT> Load and inspect the compact enrollment benchmark artifact with cohort keys, fallback levels, P25/P50/P75/P90, cohort size, confidence flags, benchmark version, and source data version.
<REF:ENROLLMENT_CLASSIFICATION> Classify enrollment assumptions.
<REF:SNAPSHOT_METADATA> Define output metadata object for narrative payload.
<REF:VALIDATION> Check sample sizes, sparse groups, outliers, and label stability.
```

Notebook blocks focused on cost, sites, countries, total duration, calendar spend, future market commitment, and portfolio views are not part of v1. They belong to later roadmap work.

The notebook intentionally does not automatically overwrite the artifact when run end-to-end. It reproduces the calculations in memory and validates the already-built artifact. This prevents accidental production-facing artifact churn during analytical review.

## Validation And Audit Checks

Validation for v1 should focus on deterministic benchmark stability rather than predictive-model performance.

Recommended checks:

- Enrollment field availability by `enrollment_type`.
- Completed actual enrollment counts by phase, indication, therapeutic area, and rare disease flag.
- Candidate-field coverage.
- Candidate-field effect size on `log1p(actual_enrollment)`.
- Benchmark group size by candidate hierarchy.
- Bootstrap stability of P25/P50/P75/P90.
- Bootstrap stability of `enrollment_status` labels.
- Fallback frequency from strict to broader benchmark levels.
- Outlier sensitivity of benchmark percentiles.
- Comparison between default hierarchy and any candidate refined hierarchy.
- Sparse cohort frequency under the four-level hierarchy.
- P25/P50/P75/P90 stability by cohort.
- Outlier sensitivity for very large trials.
- Frequency of each `enrollment_status`.
- Consistency of source assignment across `planned_value`, `final_observed_value`, `observed_lower_bound`, `model_default`, and `user_scenario`.
- Snapshot stale-state behavior when design fields change before the next prediction.

Do not report deterministic benchmark labels as clinical recommendations.

These checks are used to make the benchmark stable and auditable, not to turn v1 into a predictive enrollment model.

Phase 1 QA checks already completed:

- Artifact regenerated successfully at `frontend/data/enrollment_benchmarks_v1.csv`.
- Report regenerated successfully at `frontend/data/enrollment_benchmarks_v1_report.json`.
- Artifact schema matches the implementation and this document.
- Artifact row count is 876.
- Duplicate benchmark keys: 0.
- Report values are consistent with the notebook summaries.
- Runtime utility tested for strict lookup, fallback lookup, phase-only fallback, no-match behavior, missing artifact behavior, corrupt artifact behavior, missing/invalid planned enrollment, boundary behavior at P25/P75/P90, missing percentile behavior, all-low-confidence synthetic fallback behavior, and `not_available` metadata behavior.
- Notebook contains all required `<REF:...>` sections and executed successfully with notebook magics stripped in the local QA harness.
- Production runtime does not require full `data/data_clinpred.csv`; only the compact artifact and the current trial snapshot are needed.

## V1 Non-Goals

- No site-count estimator in v1.
- No country-count estimator in v1.
- No total-duration estimator in v1.
- No cost layer in v1.
- No calendar spend model in v1.
- No future development commitment model in v1.
- No market layer in v1.
- No full operational-scale engine in v1.
- No model retraining for XGBoost in v1.
- No use of enrollment as an XGBoost feature in v1 unless it already exists safely in the current model path.
- No direct adjustment of Completion Score from enrollment.
- No pillar-level Coherence attribution in v1.
- No LLM pseudo-SHAP in v1.

## Future Estimation Roadmap, Not V1

The previous broader estimation architecture remains useful as later roadmap thinking, but it is not part of the v1 serious-game enrollment benchmark.

Future estimation roadmap topics:

- Site-count estimation.
- Country-count estimation.
- Total-duration estimation.
- Cost translation.
- Calendar spend.
- Future development commitment.
- Market potential.
- Full operational-scale estimation.
- Reconciliation of enrollment/sites/countries/duration.

Future versions may also explore model-based enrollment calibration, including feature importance analysis, an enrollment proxy model, SHAP on the proxy model, correlation analysis, percentile segmentation, and clustering of similar trials. These are optional hardening steps and should not block the deterministic v1 benchmark.

## Future Coding Assistant Instructions

- Use this document as the source of truth for v1 enrollment benchmarking.
- Keep v1 implementation deterministic and auditable.
- Before changing benchmark cohorts or adding primary benchmark fields, run/update the Enrollment Benchmark Calibration Gate.
- Do not require the full historical `data_clinpred.csv` dataset at production runtime for enrollment benchmarking.
- Precompute enrollment benchmark percentiles offline and ship a compact benchmark artifact.
- Production runtime should use lookup plus deterministic fallback logic, not recompute historical percentiles from the full database.
- Treat the benchmark artifact as versioned, auditable input to the narrative layer.
- Do not train a model for v1 unless explicitly requested later.
- Do not add sites, countries, total duration, cost, or market logic to v1.
- Do not modify XGBoost, SHAP, therapeutic-area calibration, or audit/demo parity.
- Do not add new primary benchmark fields merely because they sound clinically plausible.
- Only add primary benchmark fields if they improve relevance while preserving sample size and percentile stability.
- Prefer support/conflict signals over over-granular benchmark matching when sample size is limited.
- Do not treat p-values alone as sufficient evidence for benchmark field selection.
- Treat planned enrollment as a scenario assumption, not as clinical truth.
- Treat benchmark percentiles as reference values, not recommendations.
- Treat deterministic `enrollment_status` as benchmark position, not clinical judgment.
- Let the narrative architecture interpret the benchmark through Coherence Score.
- Preserve the separation between planned values, final observed values, observed-to-date lower bounds, model defaults, and user scenarios.

## Next Phase Entry Point

The next phase should start from the Phase 1 artifact and runtime utility, not from the full historical dataset.

Expected Phase 2 sequence:

1. Add the Planned Enrollment field to Simulation Mode without changing the XGBoost prediction payload or model-facing feature set.
2. Select the initial planned enrollment assumption using the source-priority rules:
   - estimated/planned value when available,
   - completed final observed value only as completed-trial context,
   - benchmark-derived model default when no usable planned value exists,
   - `user_scenario` after participant edit.
3. Attach `planned_enrollment_metadata(...)` output to the latest prediction snapshot.
4. Mark benchmark metadata stale when relevant design fields change before the next `Predict Trial Completion`.
5. Add support/conflict signal logic using structured Trial Features, starting with fields already identified as plausible Phase 2 signals: sponsor tier, biomarker stratification, therapeutic modality, administration complexity, line of therapy, patient profile, endpoint rigor/structure, duration bins if calibrated, number of arms, comparator/placebo, allocation, and masking.
6. Only after snapshot metadata and support/conflict logic are stable, pass the metadata into the narrative / Coherence layer.

Phase 2 must still preserve the completion model boundary:

```text
Completion Score = existing XGBoost/SHAP/TA-calibrated score.
Enrollment benchmark = deterministic metadata for narrative/coherence reasoning.
No enrollment benchmark value modifies XGBoost, SHAP values, TA calibration, audit parity, or the existing prediction API contract.
```
