# ClinTrialPredict Enrollment Estimation Architecture

## Purpose

This document defines the v1 enrollment estimation / benchmarking architecture for ClinTrialPredict serious-game mode. It now also records the Phase 1 implementation and QA state for the deterministic planned-enrollment benchmark layer.

The purpose is not to estimate all missing operational quantities. The purpose is to provide simple, deterministic, auditable benchmarks for active operational assumptions.

The enrollment and site-count benchmarks help the future narrative layer assess whether selected operational assumptions are coherent with the current simulated trial profile. They do not enter the XGBoost model, do not directly modify the Completion Score, and should feed only structured narrative / Coherence payloads after those layers are implemented.

```text
Current deterministic operational-assumption scope = planned enrollment + planned site count.
Countries, total duration, cost, market potential, downstream commitment, and full operational-scale estimation are postponed.
```

This document should be read alongside [docs/architecture_narratives.md](/home/delaunan/code/delaunan/clintrialpredict/docs/architecture_narratives.md), which defines how the benchmark is interpreted through the serious-game narrative layer.

## Active V1 Implementation Scope

The active implementation scope is narrow by design:

- Planned Enrollment.
- Planned Site Count, using registry-derived aggregate facility-count proxy benchmarks.
- Deterministic benchmark metadata.
- Compact production artifact runtime.
- No XGBoost changes.
- No SHAP changes.
- No therapeutic-area calibration changes.
- No audit/demo parity changes.
- No API contract changes.
- No Coherence Score implementation.
- No LLM narrative generation.

The broader architecture may reserve names for future operational assumptions, but active implementation remains limited to planned enrollment and planned site count. Reserved future keys must not imply active country, duration, cost, market, spend, or development-commitment estimates.

## Current Implementation Status - 2026-06-03

The active estimation layer is the combined deterministic operational benchmark. Historical enrollment-only and site-only benchmark scripts, runtime utilities, artifacts, and reports have been removed to keep one source of truth.

Active files:

- `scripts/build_operational_benchmarks.py`: the only benchmark builder.
- `scripts/check_operational_benchmarks.py`: the only benchmark checker. It includes schema checks, lookup checks, registry-wide coverage checks, defaulting safety checks, and model-boundary checks.
- `src/operational_benchmarks.py`: the only benchmark runtime utility.
- `frontend/data/operational_benchmarks_v1.csv`: active compact runtime artifact.
- `frontend/data/operational_benchmarks_v1_report.json`: active build report.
- `frontend/data/operational_benchmarks_v1.xlsx`: analyst inspection export only; the app does not read it.
- `notebooks/estimation.ipynb`: current explanatory notebook aligned to the combined operational artifact.

Reproducibility workflow:

```bash
python scripts/build_operational_benchmarks.py
python scripts/check_operational_benchmarks.py
```

The production runtime uses only `frontend/data/operational_benchmarks_v1.csv` through `src/operational_benchmarks.py`. It must not load `data/data_clinpred.csv`.

## Phase 2A / 2B Implementation Status - 2026-06-01

The narrow Operational Assumptions foundation is implemented in Simulation Mode.

Implemented behavior:

- Planned Enrollment appears only in Simulation Mode as a separate Operational Assumption mini-card.
- Planned Enrollment uses the compact benchmark artifact through `src/enrollment_benchmarks.py`.
- Planned Enrollment does not enter `SIMULATION_FEATURE_IDS`, taxonomy, the `/predict` payload, XGBoost, SHAP impacts, pillar impacts, impact bar, treemap, therapeutic-area calibration, or audit/demo parity behavior.
- Baseline simulation snapshots, successful model-facing simulation predictions, operational-only updates, and simulation history records store `operational_assumptions`.
- `operational_assumptions.planned_enrollment` stores deterministic metadata from `planned_enrollment_metadata(...)`.
- `planned_sites`, `planned_countries`, and `planned_duration_months` were present only as inactive reserved keys before S3. S3 later activated `planned_sites` in Simulation Mode only.
- Operational-only changes refresh snapshot metadata through a generic `simulation_operational_update` path without calling `/predict` and without changing Completion Score or XGBoost chart data.
- The Predict button and gauge-side `Click Predict to update` prompt now respond consistently to model-facing Trial Feature changes and active operational-assumption changes.
- Planned Enrollment pending state shows a previous-value marker and clears after the operational update.
- The Enrollment Assumption card shows the assumption source, for example `planned value`, `final observed enrollment`, `benchmark default`, or `user scenario`.
- Benchmark stale-state is limited to the active lookup cohort fields: `phase_ml`, `gbd_cause_id_3_ml`, `therapeutic_area_ml`, `is_rare_disease_ml`, and `therapeutic_modality_ml`.

Current active operational assumption keys:

```text
planned_enrollment
```

Reserved inactive operational assumption keys before S3:

```text
planned_sites
planned_countries
planned_duration_months
```

Still intentionally not implemented:

- Site-count estimation.
- Country-count estimation.
- Duration estimation.
- Cost, market potential, spend curve, or future development commitment logic.
- LLM narrative generation.
- Coherence Score.
- Adjusted Trial Value Score.
- XGBoost retraining or score adjustment from Planned Enrollment.
- SHAP or pillar attribution for Planned Enrollment.
- API contract changes.
- Deployment changes.

Planned Enrollment Phase 2A / 2B is implemented. S1B `planned_sites` benchmark-feasibility audit, S2 `planned_sites` compact benchmark artifact/runtime utility, and S3 `planned_sites` Simulation Mode integration are now implemented. Any future active operational assumption must follow the generic operational-assumption pending/update pattern and remain outside XGBoost until a separate architecture decision says otherwise.

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
| Observed-to-date lower bound | Current partial operational value, not necessarily final. | Ongoing or terminated actual enrollment from current extract. | Lower-bound candidate when no planned/estimated value exists; not final truth. |
| Model default | Synthetic benchmark-derived default. | P50 or other selected cohort default when no planned value exists. | Neutral starting assumption. |
| User scenario | Participant-edited assumption. | User changes enrollment to 600. | Current scenario value evaluated against benchmark and design profile. |
| Benchmark distribution | Historical comparable-trial enrollment values. | Cohort P25/P50/P75/P90. | Deterministic benchmark reference. |

Rule: every stored enrollment benchmark output must preserve source, cohort, percentile, status, and snapshot metadata so future reviewers can tell which value was planned, observed, lower-bound, defaulted, or user-edited.

## V1 MVP Definition

The v1 estimation MVP is not a full missing-operational-value workbench. It is an enrollment-benchmark layer for serious-game simulation. It provides a neutral, auditable benchmark against which a user's planned enrollment assumption can be assessed.

Active v1 scope includes:

- `planned_enrollment_assumption`.
- Enrollment source priority.
- Deterministic benchmark hierarchy.
- P25/P50/P75/P90 benchmark percentiles.
- `enrollment_status` classification.
- `support_level = "not_evaluated"` until a later implementation phase.
- Empty support/conflict signal lists until a later implementation phase.
- Benchmark stale-state logic.
- Snapshot metadata for the narrative payload.
- A future path for narrative / Coherence consumption after the structured payload is stable.

Explicitly excluded from active v1:

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
- LLM narrative generation.
- Coherence Score implementation.

## Planned Enrollment Assumption Source Priority

Use this source priority:

```text
1. If planned/estimated enrollment is available, use it as the initial `planned_enrollment_assumption`.
2. If the trial is completed and actual enrollment represents final observed value, it can be used as final observed context.
3. If no usable planned value exists and the trial is not completed, treat actual/current enrollment as `observed_lower_bound` when available.
4. If no usable planned value exists, look up benchmark P50 as `model_default`.
5. For non-completed trials with both observed lower-bound and benchmark P50 available, initialize planned enrollment as `max(observed_lower_bound, model_default)`.
6. If the participant edits the value, source becomes `user_scenario`.
```

This prevents an active, terminated, or otherwise non-completed trial from defaulting to a benchmark P50 that is smaller than the enrollment already observed in the registry extract.

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

For the active Planned Enrollment runtime, benchmark stale-state should be triggered only by fields that change the active benchmark lookup cohort:

- `phase_ml`
- `gbd_cause_id_3_ml`
- `therapeutic_area_ml`
- `is_rare_disease_ml`
- `therapeutic_modality_ml`

If the participant changes one of those cohort fields, the old operational benchmark becomes stale. The benchmark should refresh only after the user clicks `Predict Trial Completion`, consistent with the existing simulation snapshot workflow.

Other design fields such as patient profile, endpoint design, sponsor tier, administration complexity, comparator logic, endpoint duration, number of arms, and population profile are future support/conflict-signal candidates. They may later influence narrative or Coherence interpretation, but they should not trigger benchmark-cohort refresh until that support/conflict layer exists.

Later, once support/conflict signal generation exists, additional design fields may affect interpretation and may require a more nuanced stale-state rule.

During editing, the UI may show:

```text
Enrollment benchmark will refresh after prediction
```

Core rule:

```text
Current design snapshot -> benchmark cohort -> P25/P50/P75/P90 -> enrollment_status -> future narrative / Coherence payload input.
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

Therapeutic modality, sponsor tier, administration complexity, line of therapy, patient subtype, endpoint rigor, and endpoint duration should not define the primary benchmark cohort in v1. They can be evaluated later as support/conflict signals for narrative / Coherence reasoning. This avoids excessive stratification and unstable percentiles.

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

Primary benchmark fields define the historical comparison cohort. Future support/conflict signals can help narrative / Coherence logic interpret whether the selected enrollment is supported by the current design.

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

If a field has signal but weak coverage or unstable percentiles, it should remain a future support/conflict signal rather than a primary benchmark matching field.

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

The completed v1 implementation stays simple unless later analysis clearly shows that another field materially improves benchmark stability and relevance without creating sparse cohorts.

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

Coherence Score is not implemented. When a future Coherence Score exists, enrollment should be one input into it, not the score itself. It should mainly influence:

- Operational feasibility.
- Population relevance.
- Change integrity.

Enrollment must not dominate the future broader rubric.

```text
Enrollment should normally have a maximum standalone effect of about -10 to +4 points inside a future Coherence Score rubric.
```

Interpretation principle:

```text
One weak enrollment signal creates a discussion point.
Several weak or conflicting design signals may create a future Coherence Score penalty.
A difficult design can still receive a positive future Coherence Adjustment if the participant strengthens it coherently.
```

Future alignment with the narrative architecture:

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

Current Planned Enrollment runtime returns `support_level = "not_evaluated"` and empty `supporting_signals` / `conflicting_signals`. Support/conflict logic remains deferred until a separate architecture decision and implementation phase.

## Revised Operational Assumptions Roadmap

The estimation architecture evolves in staged, decision-gated increments. Planned Enrollment is active. S1B, S2, and S3 for `planned_sites` are complete, and `planned_sites` is active in Simulation Mode only. `planned_duration_months` remains inactive. `planned_countries` remains excluded.

Current roadmap:

1. Superseded: Phase 1 enrollment-only benchmark artifact and runtime utility.
   - The historical enrollment-only implementation was merged into the combined operational benchmark.
   - Standalone enrollment benchmark scripts, runtime utility, artifact, and report have been removed.

2. Completed: Phase 2A / 2B Planned Enrollment Simulation Mode integration and `operational_assumptions` container.
   - `planned_enrollment` was the first active benchmarked operational assumption.
   - `planned_sites`, `planned_countries`, and `planned_duration_months` were initially stored only as inactive reserved metadata keys.
   - Planned Enrollment remains outside XGBoost, SHAP, therapeutic-area calibration, audit/demo parity, API contracts, model artifacts, and taxonomy artifacts.

3. Completed: S1B `planned_sites` benchmark-feasibility audit.
   - Confirmed that `number_of_facilities` can support a cautious registry-derived aggregate facility-count proxy benchmark.
   - Confirmed that `number_of_facilities` is not true planned sites, not true actual activated sites, and not true estimated sites.
   - The site-specific exploratory notebook was removed after consolidation into `notebooks/estimation.ipynb`.

4. Superseded: S2 `planned_sites` compact benchmark artifact and runtime utility.
   - The historical site-only implementation was merged into the combined operational benchmark.
   - Standalone site benchmark scripts, runtime utility, artifact, and report have been removed.

5. Completed: S2 QA checkpoint before S3.
   - Verify the S2 artifact, report, runtime utility, and check script cleanly before any UI/runtime integration.
   - S2 QA returned GO and authorized the narrow S3 Simulation Mode integration prompt.

6. Completed: S3 `planned_sites` Simulation Mode integration.
   - `planned_sites` is active in Simulation Mode only.
   - Active runtime uses `src/operational_benchmarks.py` and `frontend/data/operational_benchmarks_v1.csv`.
   - Operational-only `planned_sites` changes update snapshot metadata without calling `/predict`.
   - Must preserve the current completion-model boundary unless a separate architecture decision changes it.

7. Implemented: Post-S3 `planned_sites` defaulting revision.
   - Treat non-completed `number_of_facilities` as current registry facility-count proxy context and a lower-bound candidate.
   - Add deterministic patients-per-site benchmark support from completed trials with positive enrollment and positive facility-count proxy values.
   - Initialize non-completed editable `planned_sites` from the maximum of current registry facility-count proxy and the enrollment-coherent patients-per-site candidate when patients-per-site P50 is available.
   - Use pure site-count P50 only as a fallback when the enrollment-coherent patients-per-site candidate is unavailable.
   - Keep the revision outside XGBoost, SHAP, `/predict`, prediction payloads, calibration, audit parity, model artifacts, taxonomy artifacts, LLM narratives, Coherence Score, and Adjusted Trial Value Score.

8. Next: Browser smoke testing and deployment readiness review for Planned Site Count.
   - Confirm the operational-assumption card shows the revised lower-bound context and enrollment-coherent candidate.
   - Confirm planned-sites operational-only updates still do not call `/predict`.

9. Next architecture analysis before changing cohorts: modality-aware operational benchmark evaluation.
   - Completed analysis showed that full modality-first fallback is too risky because it can trade disease relevance for broad modality relevance.
   - Implemented same-level `therapeutic_modality_ui` refinement for Planned Enrollment and patients-per-site only.
   - Current four-level clinical fallback remains the backbone; modality can refine a selected clinical level but cannot replace it.

10. Later: D1 `planned_duration_months` duration-definition validation and data audit.
   - Validate duration definitions and source quality before any artifact or runtime work.
   - Must join back to `data/studies.txt` for date-type qualifiers.

11. Later, only if D1 passes: D2 duration compact benchmark artifact and runtime utility.
   - Requires D1 decision gates to pass first.

12. Later, only if D2 passes: D3 duration Simulation Mode integration.
   - May activate `planned_duration_months` only after D2 passes and a separate implementation prompt authorizes D3.

13. Future after stable deterministic payloads: narrative payload builder, LLM narrative, then Coherence Score.
   - Narrative and scoring work must consume deterministic payloads.
   - LLM text must not invent operational ranges or create hidden score changes.
   - Coherence Score must remain separate from the existing XGBoost Completion Score.
   - `planned_countries` remains excluded.

Next required step: validate S3 behavior in browser smoke testing before any deployment decision. Duration work remains separate and starts with D1 only after a separate authorization prompt.

The detailed source contracts, methodology, metadata shapes, staged roadmap, and decision gates for S1B/S2/S3/D1/D2/D3 are defined in `Next Operational Assumptions - Sites And Duration Planning`.

## Next Operational Assumptions - Sites And Duration Planning

This section is the source-of-truth plan for the next operational-assumption families after active Planned Enrollment. S1B `planned_sites` audit, S2 `planned_sites` artifact/runtime utility, and S3 `planned_sites` Simulation Mode integration are complete. `planned_duration_months` remains inactive, and `planned_countries` remains excluded.

### Current Boundary

- `planned_enrollment` and `planned_sites` are the active operational assumptions.
- `planned_sites` is active only in Simulation Mode and remains outside the prediction model.
- `planned_duration_months` remains an inactive reserved key.
- `planned_countries` remains explicitly excluded from this implementation plan.
- The active runtime uses `frontend/data/operational_benchmarks_v1.csv` and `src/operational_benchmarks.py` for both `operational_assumptions.planned_enrollment` and `operational_assumptions.planned_sites`.
- Future operational assumptions must preserve the current boundary unless a separate architecture decision changes it: no XGBoost changes, no `/predict` or API contract changes, no SHAP changes, no pillar impact changes, no impact bar or treemap changes, no therapeutic-area calibration changes, no audit/demo parity changes, no model artifact changes, and no taxonomy artifact changes.
- Production runtime should continue to use compact benchmark artifacts and current trial snapshots, not the full raw historical database.
- LLM narratives, Coherence Score, and Adjusted Trial Value Score remain out of scope.

### Planned Site Count - Source Contract

The confirmed candidate field for a future site-count benchmark is `number_of_facilities`.

Source contract:

- `data/data_clinpred.csv` has 34,066 rows and 157 fields.
- The final CSV does not contain detailed per-site records.
- The final CSV does not contain site names, site addresses, cities, investigators, facility-level rows, per-site status, planned-vs-actual site labels, or estimated-vs-actual site labels.
- Site-relevant final CSV fields are `number_of_facilities`, `includes_us`, `includes_us_ml`, and `includes_us_ui`.
- `number_of_facilities` comes from `data/calculated_values.txt`.
- `data/calculated_values.txt` also includes fields such as `actual_duration`, `were_results_reported`, `months_to_report_results`, `has_us_facility`, and `has_single_facility`.
- The current prep loader merges the candidate field through `_engineer_facilities(...)` with `['nct_id', 'number_of_facilities']`.
- There is no accompanying `facility_type`, `site_type`, `number_of_facilities_type`, `planned_facilities`, `actual_facilities`, or `estimated_facilities` field.

Interpretation:

- `number_of_facilities` is a registry-derived aggregate facility-count proxy.
- It must not be described as true planned sites.
- It must not be described as true actual activated sites.
- It must not be described as true estimated sites.
- No detailed site-level production data is available locally.
- `includes_us` is a derived geography/facility-presence flag and is not a primary site-count benchmark input.
- The current loader may derive `includes_us` without filtering the `countries.txt` `removed` flag, so it should not be used as a primary benchmark key for `planned_sites`.
- `planned_countries` remains excluded.

Previously observed site-count audit values for `data/data_clinpred.csv`:

- `number_of_facilities` present: 34,066 rows.
- Positive facility count: 31,824 rows.
- Zero facility count: 2,242 rows.
- Minimum: 0.
- Median: 12.
- Maximum: 1,745.

### Planned Site Count - First-Estimate Methodology

The first benchmark target should be completed trials with `number_of_facilities > 0`.

Methodology rules:

- Completed trials can support historical facility-count benchmark distributions, but only as completed registry facility-count proxy values.
- Ongoing trials should be treated as current registry facility-count proxy values or lower-confidence lower-bound information, not final site counts.
- The deterministic benchmark pattern mirrors Planned Enrollment because S1B confirmed source stability, target readiness, cohort coverage, and fallback reliability.
- Initial confidence threshold should be `n >= 50`, mirroring enrollment unless later validation changes it.
- S2/S3 first implemented a standalone site-count benchmark using completed registry facility-count proxy distributions.
- Post-S3 product review concluded that non-completed `number_of_facilities` should not initialize the editable `planned_sites` assumption directly, because ongoing or planned registry facility counts may be incomplete.
- For ongoing and planned trials, `number_of_facilities` should become a current registry facility-count proxy context value and lower-bound candidate.
- The next narrow revision should add an enrollment-coherent default for `planned_sites` using a patients-per-site benchmark derived from completed trials with positive enrollment and positive facility-count proxy values.
- Patients-per-site for this revision should be computed as `enrollment / number_of_facilities`, not as a model prediction.
- The enrollment-coherent default should remain deterministic benchmark metadata and must not modify XGBoost, SHAP, `/predict`, prediction payloads, calibration, audit parity, model artifacts, or taxonomy artifacts.

Proposed benchmark hierarchy:

1. `phase + indication + rare disease flag`
2. `phase + therapeutic area + rare disease flag`
3. `phase + therapeutic area`
4. `phase only`

### Planned Site Count - Phase S1B Audit Result

S1B was completed as an audit, notebook, and documentation phase only. It did not create production benchmark artifacts, runtime utilities, UI, API changes, model changes, SHAP changes, therapeutic-area calibration changes, audit/demo parity changes, taxonomy changes, or deployment changes.

Audit materials:

- Site-specific notebook: removed after consolidation into the current `notebooks/estimation.ipynb`.
- Non-production audit report: `notebooks/outputs/site_count_s1b_audit.json`.

Source contract result:

- Source field confirmed: `number_of_facilities`.
- Source origin confirmed: `data/calculated_values.txt`, merged by `src/prep/data_loader_clinpred.py::_engineer_facilities(...)`.
- Numeric parsing confirmed for all 34,066 `data/data_clinpred.csv` rows.
- No planned/actual/estimated qualifier exists for `number_of_facilities`.
- No detailed local production facility table was found for site names, site addresses, cities, investigators, facility-level rows, per-site status, planned-vs-actual site labels, or estimated-vs-actual site labels.
- The field remains a registry-derived aggregate facility-count proxy, not true planned sites, true actual activated sites, or true estimated sites.
- `includes_us` remains excluded from the primary site-count benchmark hierarchy.

Overall `number_of_facilities` quality statistics:

```text
total rows: 34,066
present:    34,066
missing:         0
zero:        2,242
negative:        0
positive:   31,824
min:             0
p25:             1
median:         12
p75:            40
p90:            93
p95:           149
p99:           302.35
max:         1,745
```

Benchmark target result:

- Provisional target: completed trials with `number_of_facilities > 0`.
- Target population: 19,880 rows.
- Interpretation: completed registry facility-count proxy values, not true actual activated site count.
- Ongoing trials were treated only as descriptive current registry facility-count proxy / lower-confidence lower-bound values.

Exploratory hierarchy result with `min_n >= 50`:

```text
phase_indication_rare: rows=659, confident=105, low_confidence=554, duplicate_keys=0, p50 range=1-277
phase_ta_rare:         rows=138, confident=53,  low_confidence=85,  duplicate_keys=0, p50 range=1-78
phase_ta:              rows=76,  confident=48,  low_confidence=28,  duplicate_keys=0, p50 range=1-75
phase_only:            rows=4,   confident=4,   low_confidence=0,   duplicate_keys=0, p50 range=5-28
```

Fallback coverage across all 34,066 source rows:

```text
phase_indication_rare: 23,439
phase_ta_rare:          7,260
phase_ta:               2,231
phase_only:             1,136
not_available:              0
low_confidence matches:     0
```

Outlier caveat:

- The largest observed value was 1,745 facilities.
- The top observed values are large multicenter/global trials, including cardiovascular and respiratory Phase III programs.
- This upper tail supports percentile-based benchmarks and argues against mean-based site-count benchmarks.

Decision gate result:

S2 is recommended, narrowly, because:

- `number_of_facilities` is available and numeric.
- Completed positive facility-count rows are sufficient.
- Phase-only fallback is robust and confident.
- Duplicate benchmark keys are zero.
- Low-confidence behavior can be flagged.
- Outliers are documented.
- Architecture wording clearly prevents overclaiming true site-count precision.

Historical S2 scope authorized after S1B:

- Create a site-count benchmark builder, checker, runtime utility, compact artifact, and report.
- These standalone site-only files were later superseded by the combined operational benchmark and deleted.
- Keep `planned_sites` inactive in UI until S3.
- Do not touch `/predict`, XGBoost, SHAP, therapeutic-area calibration, audit mode, taxonomy, model artifacts, `planned_duration_months`, or `planned_countries`.

Post-S1B status:

- `planned_sites` remains inactive.
- `planned_duration_months` remains inactive and outside S1B scope.
- `planned_countries` remains excluded.
- No runtime, UI, API, model, SHAP, therapeutic-area calibration, audit/demo parity, taxonomy, or deployment behavior changed.

### Planned Site Count - Phase S2 Implementation Status

S2 originally created a standalone site-count benchmark builder, checker, runtime utility, CSV artifact, and report. That standalone implementation has been superseded by the combined operational benchmark. The historical site-only files were deleted during the 2026-06-03 cleanup.

Active site-count benchmark support now lives in:

- `scripts/build_operational_benchmarks.py`
- `scripts/check_operational_benchmarks.py`
- `src/operational_benchmarks.py`
- `frontend/data/operational_benchmarks_v1.csv`
- `frontend/data/operational_benchmarks_v1_report.json`

### Planned Site Count - Phase S3 Implementation Status

S3 is implemented as a narrow Simulation Mode integration for `planned_sites`. `planned_sites` is now an active operational assumption in Simulation Mode only.

Implemented behavior:

- `planned_sites` appears in the Operational Assumption mini-card area near Planned Enrollment.
- `planned_sites` originally used a standalone site-only benchmark utility and artifact, now deleted after consolidation.
- After the single-artifact refactor, active Simulation Mode uses `src/operational_benchmarks.py` and `frontend/data/operational_benchmarks_v1.csv`.
- `planned_sites` is stored under `operational_assumptions.planned_sites`.
- `planned_sites` metadata is stored in baseline simulation snapshots, successful model-facing simulation prediction snapshots, operational-only updates, and simulation history.
- Operational-only `planned_sites` changes refresh snapshot metadata through the generic `simulation_operational_update` path without calling `/predict`.
- Changing only `planned_sites` does not change Completion Score, SHAP impacts, pillar impacts, impact bar, or treemap.
- Changing `planned_sites` plus model-facing Trial Features calls `/predict` only because the model-facing fields changed; `planned_sites` is not sent as a prediction payload field.
- Benchmark stale-state follows the same cohort fields as Planned Enrollment: `phase_ml`, `gbd_cause_id_3_ml`, `therapeutic_area_ml`, `is_rare_disease_ml`, and `therapeutic_modality_ml`.

Source and wording:

- Completed trials may use positive `number_of_facilities` as completed registry facility-count proxy context.
- Ongoing and planned trials treat positive `number_of_facilities` as current registry facility-count proxy context and a lower-bound candidate, not as the editable planned-sites estimate by itself.
- Active Simulation Mode now initializes non-completed `planned_sites` from the post-S3 enrollment-coherent defaulting rule documented below.
- Missing or invalid site-count proxy values use `benchmark_default` from site-count P50 only when the enrollment-coherent patients-per-site candidate is unavailable.
- Participant edits use `user_scenario`.
- UI and metadata use conservative registry-derived facility-count proxy wording.
- `number_of_facilities` remains a registry-derived aggregate facility-count proxy, not true planned sites, not true actual activated sites, and not true estimated sites.

S3 boundary confirmations:

- `planned_sites` remains outside XGBoost.
- `planned_sites` remains outside `/predict`.
- `planned_sites` remains outside SHAP, pillar impacts, impact bar, treemap, and Completion Score.
- Therapeutic-area calibration, audit mode, model artifacts, taxonomy artifacts, API contracts, and prediction payload contracts were not changed.
- `planned_duration_months` remains inactive.
- `planned_countries` remains excluded.
- No LLM narratives, Coherence Score, or Adjusted Trial Value Score were implemented.

### Planned Site Count - Post-S3 Defaulting Revision Implementation Status

This section records the implemented revision after S3 manual review.

Problem statement:

- For completed trials, `number_of_facilities` can remain useful as a completed registry facility-count proxy.
- For ongoing, active-not-recruiting, recruiting, not-yet-recruiting, or otherwise non-completed trials, `number_of_facilities` may reflect currently listed or observed registry facilities rather than the final intended operating scale.
- Therefore, current non-completed facility count should be shown as context and lower-bound evidence, not automatically treated as the planned-sites assumption.
- The editable `planned_sites` assumption should be initialized from a simple deterministic default that stays coherent with the current planned enrollment assumption.

Implemented rule:

1. If the selected trial is completed and has positive `number_of_facilities`, initialize `planned_sites` from that value with source `completed_registry_facility_count`.
2. If the selected trial is not completed and has positive `number_of_facilities`, store and display that value as `current_registry_facility_count_proxy` context and as a lower-bound candidate.
3. Look up the combined operational benchmark row and use `site_count_p50` as the standalone site benchmark default candidate when available.
4. Use compact patients-per-site benchmarks derived from completed trials with positive ACTUAL enrollment and positive `number_of_facilities`.
5. Use the same fallback hierarchy as the site-count benchmark: `phase_indication_rare`, `phase_ta_rare`, `phase_ta`, then `phase_only`.
6. Compute an enrollment-coherent candidate as `planned_enrollment / patients_per_site_p50` when both values are available.
7. For non-completed trials with patients-per-site P50 available, initialize the editable `planned_sites` value as:

```text
max(
    current_registry_facility_count_proxy,
    planned_enrollment / patients_per_site_p50
)
```

8. Use pure `site_count_benchmark_p50` only as a fallback when the patients-per-site candidate is unavailable.

Round the selected default to a practical positive integer. If no candidate is available, leave the value unavailable.

Recommended source labels for the revised behavior:

- `completed_registry_facility_count`
- `current_registry_facility_count_proxy`
- `benchmark_default`
- `enrollment_coherent_benchmark_default`
- `user_scenario`

Example: `NCT02615184`

- Current registry facility-count proxy: 25.
- Planned enrollment assumption: 76.
- Matched site-count benchmark: `phase_indication_rare`, `n=58`, P50 10.
- Matched completed-trial patients-per-site P50 observed during analysis: approximately 7.31.
- Enrollment-coherent site candidate: `76 / 7.31 = 10.4`.
- Revised non-completed default: `max(25, 10.4) = 25`.
- The resulting value is still 25, but the interpretation changes: 25 is the current registry facility-count proxy lower-bound/context that already exceeds the benchmark and enrollment-coherent default candidates, not a claim that 25 is the true planned or final site count.

Example: planned-stage trial with 2 listed registry facilities and 200 planned enrollment:

- Current registry facility-count proxy: 2.
- If patients-per-site P50 for the matched cohort is approximately 8, the enrollment-coherent candidate is `200 / 8 = 25`.
- The revised default becomes `max(2, 25)`, so the editable planned-sites assumption would not default to only 2 unless the user explicitly chooses that scenario.

Implemented files and artifacts:

- Builder: `scripts/build_operational_benchmarks.py`.
- Checker: `scripts/check_operational_benchmarks.py`.
- Runtime utility: `src/operational_benchmarks.py`.
- Compact artifact: `frontend/data/operational_benchmarks_v1.csv`.
- Report: `frontend/data/operational_benchmarks_v1_report.json`.
- Simulation Mode integration: `frontend/views/edit_trial.py`.
- Active UI/runtime source: the single combined operational artifact. The older enrollment-only and site-only artifacts were removed after consolidation.

Combined artifact summary:

- Source records loaded: 34,066.
- Completed ACTUAL enrollment targets: 20,526.
- Completed positive site-count proxy targets: 19,880.
- Completed patients-per-site targets: 19,689.
- Artifact rows: 4,006.
- Duplicate benchmark keys: 0.
- Source data version: `0a97519bd78f561a`.
- Rows by benchmark level: `phase_indication_rare` 656, `phase_indication_rare_modality` 1,948, `phase_indication_rare_non_vaccine_infections` 81, `phase_ta_rare` 133, `phase_ta_rare_modality` 660, `phase_ta_rare_non_vaccine_infections` 8, `phase_ta` 72, `phase_ta_modality` 440, `phase_ta_non_vaccine_infections` 4, `phase_only` 4.
- Low-confidence rows by metric: enrollment 3,530, site_count 656, patients_per_site 3,520.
- Coverage QA not available: enrollment 0, site_count 0, patients_per_site 0.
- Coverage QA low-confidence matches: enrollment 0, site_count 0, patients_per_site 0.

Single-artifact runtime rule:

- `frontend/views/edit_trial.py` imports benchmark metadata and defaulting from `src/operational_benchmarks.py`.
- Planned Enrollment and Planned Site Count both use `frontend/data/operational_benchmarks_v1.csv`.
- The runtime keeps metric-specific evidence fields: `enrollment_n`, `site_count_n`, and `patients_per_site_n`.
- The runtime keeps metric-specific confidence flags.
- If a matched cohort has at least one relevant operational metric with `n >= 50`, the app keeps that specific cohort row and does not jump to a broader fallback only because another metric is near-threshold.
- Metrics with `30 <= n < 50` are usable but low confidence.
- Metrics with `n < 30` are considered too sparse and fall back when needed.
- Same-level therapeutic modality refinement can override enrollment and patients-per-site percentiles when the matching refined metric has `n >= 50`.
- Non-vaccine Infections fallback can override vaccine-heavy clinical fallback rows for enrollment and patients-per-site when the non-vaccine Infections row has `n >= 50`.
- Raw site-count benchmark rows do not use modality refinement.
- There are no `phase_only_modality` rows in the active artifact.
- Placeholder cohort values are not allowed to become specific benchmark identities: unclassified indication id `0` is skipped for indication-level rows, unclassified/unknown therapeutic areas are skipped for TA-level rows, and unknown/unclassified modality is skipped for modality-refinement rows.

Current cohort confidence decision:

- The operational benchmark row is selected once, then metric-specific evidence is evaluated separately for enrollment, site count, and patients-per-site.
- `n >= 50` means high-confidence evidence for that metric.
- `30 <= n < 50` means usable low-confidence evidence for that metric.
- `n < 30` is too sparse for that metric and should fall back when a broader row is available.
- If one metric is high confidence and another is just below 50 on the same specific cohort, keep the specific row and expose the weaker metric as low confidence. Do not automatically broaden the whole cohort only because one metric is near-threshold.
- The current known near-threshold mismatch is small: the specific `PHASE1/PHASE2` Leukemia non-rare cohort has `enrollment_n=49`, `site_count_n=51`, and `patients_per_site_n=49`. This is a low-confidence warning, not a reason to discard the clinically specific row.

Current defaulting rules:

- Planned Enrollment uses a positive planned/estimated enrollment value when present.
- For completed trials without a planned/estimated value, Planned Enrollment may use the final observed enrollment value.
- For non-completed trials without a planned/estimated value, current/observed enrollment is treated as a lower bound and the default is `max(observed_enrollment_lower_bound, enrollment_p50)`.
- Completed trials with positive `number_of_facilities` initialize Planned Sites from the completed registry facility-count proxy.
- Non-completed trials treat positive `number_of_facilities` as current registry facility-count proxy context and lower-bound evidence.
- For non-completed trials, if planned enrollment and patients-per-site P50 are available, the default Planned Sites value is `max(current_registry_facility_count_proxy, planned_enrollment / patients_per_site_p50)`.
- Pure site-count P50 is used only when the enrollment-coherent patients-per-site candidate cannot be calculated.
- User edits override defaults and use `user_scenario` source labels.

### Modality-Aware Cohort Refinement Implementation

Recent review showed that therapeutic modality can materially change operational scale, especially for vaccines. Full modality-first fallback was rejected because it caused material clinical-specificity loss. Same-level modality refinement is now implemented in the active combined operational artifact and runtime utility.

Observed directional findings from completed trials with positive enrollment and positive facility-count proxy values:

- Vaccine trials have much higher enrollment and patients-per-site medians than the all-trial baseline.
- In the RSV/lower respiratory Phase 3 cohort, vaccine-like trials had a much higher patients-per-site median than non-vaccine trials.
- Other modalities also differ: monoclonal antibodies, cell/gene therapy, peptide hormones, RNA therapy, ADCs, and small molecules have materially different enrollment, site-count, and patients-per-site profiles.

Implemented decision:

- Add same-level modality refinement to the active benchmark artifact for Planned Enrollment and patients-per-site only.
- Add narrow non-vaccine Infections fallback rows for Planned Enrollment and patients-per-site only.
- Do not create or use modality-refinement rows for unknown/unclassified modality. Unknown modality can contribute to the broad non-vaccine Infections exclusion pool when vaccine classification is reliable, but it must not become its own modality benchmark.
- Keep raw site-count P50 clinical-only. It remains useful reference/fallback evidence, but site count is also affected by geography, registry completeness, and operational logistics.
- Because the main non-completed site default uses `planned_enrollment / patients_per_site_p50`, modality-aware patients-per-site improves the site default without making raw site-count P50 the primary driver.
- Protect smaller modalities by using modality only when the exact same selected clinical level plus modality has `n >= 50`.
- If modality-specific `n < 50`, keep the original clinical benchmark unless the narrow non-vaccine Infections rule applies.
- For non-vaccine Infections trials, if same-level modality refinement is unavailable, try non-vaccine Infections cohorts up the clinical hierarchy before using all-modality Infections cohorts.
- Do not use `phase + modality`.
- Do not use `phase_only_modality`.
- Do not let modality replace clinical context.

Implemented same-level refinement sequence:

```text
1. Select the current clinical benchmark exactly as before:
   phase + indication + rare disease
   phase + therapeutic area + rare disease
   phase + therapeutic area
   phase only

2. If the selected clinical level is phase + indication + rare disease:
   use phase + indication + rare disease + therapeutic modality only if that metric n >= 50.

3. If the selected clinical level is phase + therapeutic area + rare disease:
   use phase + therapeutic area + rare disease + therapeutic modality only if that metric n >= 50.

4. If the selected clinical level is phase + therapeutic area:
   use phase + therapeutic area + therapeutic modality only if that metric n >= 50.

5. If the selected clinical level is phase only:
   do not apply modality refinement.
```

Narrow non-vaccine Infections fallback sequence:

```text
For trials where therapeutic area is Infections and therapeutic modality is not Vaccine:

1. Try same-level modality refinement first, using n >= 50.
2. If unavailable and the selected clinical level is phase + indication + rare disease:
   try phase + indication + rare disease excluding Vaccine, then phase + TA + rare disease excluding Vaccine, then phase + TA excluding Vaccine.
3. If unavailable and the selected clinical level is phase + TA + rare disease:
   try phase + TA + rare disease excluding Vaccine, then phase + TA excluding Vaccine.
4. If unavailable and the selected clinical level is phase + TA:
   try phase + TA excluding Vaccine.
5. Use a non-vaccine Infections row only if n >= 50.
6. If no non-vaccine Infections row is usable, keep the original all-modality clinical fallback.
```

Placeholder fallback safeguards:

```text
1. If gbd_cause_id_3_ml is 0 or missing, skip indication-level cohorts and continue with TA-level cohorts when TA is valid.
2. If therapeutic area is OTHER/UNCLASSIFIED, UNCLASSIFIED, UNKNOWN, OTHER, or missing, skip TA-level cohorts and keep any valid indication-level cohort before falling back to phase only.
3. If therapeutic modality is UNKNOWN, UNCLASSIFIED, or missing, do not create or use same-level modality-refinement rows.
4. For Infections only, the non-vaccine fallback remains INFECTIONS excluding VACCINE; unknown modality may be included in that broad non-vaccine pool, but not as an UNKNOWN modality row.
```

Analysis result before implementation:

- Full modality-first fallback selected modality for approximately 97% of rows but caused clinical-specificity loss for approximately 34-36% of rows.
- Earlier constrained fallback selected modality for approximately 75-76% of rows but still caused clinical-specificity loss for approximately 16-17% of rows.
- Same-level refinement selected modality for 60.3% of enrollment rows and 58.1% of patients-per-site rows with 0% clinical-specificity loss.
- Same-level refinement changed enrollment P50 by more than 2x for approximately 1.1% of rows.
- Same-level refinement changed patients-per-site P50 by more than 2x for approximately 3.3% of rows.
- Follow-up vaccine-dominance review showed that some Infections clinical fallback cohorts were vaccine-heavy and materially inflated patients-per-site for non-vaccine trials. The approved implementation keeps modality refinement at `n >= 50` and adds the narrow non-vaccine Infections fallback above.
- Current `search_registry` audit coverage: enrollment modality-refined 2,542 rows, enrollment non-vaccine Infections fallback 140 rows, patients-per-site modality-refined 2,421 rows, patients-per-site non-vaccine Infections fallback 163 rows, site-count non-vaccine Infections fallback 0 rows.

Confidence rules preserve the current thresholds:

- `n >= 50`: confident.
- `30 <= n < 50`: usable low confidence for clinical fallback rows, but too sparse for modality override and too sparse for non-vaccine Infections fallback.
- `n < 30`: too sparse; fallback.

Implementation boundaries:

- Do not load `data/data_clinpred.csv` at production runtime to compute patients-per-site.
- Patients-per-site distributions are precomputed into the compact combined operational benchmark artifact.
- Keep `planned_sites` outside XGBoost, `/predict`, SHAP, pillar impacts, impact bar, treemap, Completion Score, therapeutic-area calibration, audit mode, model artifacts, taxonomy artifacts, API contracts, and prediction payloads.
- Keep `planned_duration_months` inactive.
- Keep `planned_countries` excluded.
- Do not implement LLM narratives, Coherence Score, or Adjusted Trial Value Score as part of this revision.

Next required step: browser-smoke-test the revised planned-sites initialization and card metadata before deployment readiness review.

### Current Operational Benchmark Handoff

This is the current source-of-truth handoff for the active operational benchmark layer.

Active files:

- Builder: `scripts/build_operational_benchmarks.py`.
- Checker: `scripts/check_operational_benchmarks.py`.
- Runtime utility: `src/operational_benchmarks.py`.
- Active artifact: `frontend/data/operational_benchmarks_v1.csv`.
- Active report: `frontend/data/operational_benchmarks_v1_report.json`.
- Simulation Mode integration: `frontend/views/edit_trial.py`.

Current benchmark sources:

- Benchmark statistics are built from `data/data_clinpred.csv`, not from `frontend/data/search_registry.csv`.
- `frontend/data/search_registry.csv` is used for UI trial selection and selectable-trial audit coverage.
- Production runtime uses only the compact artifact `frontend/data/operational_benchmarks_v1.csv`; it must not load `data/data_clinpred.csv`.

Current operational benchmark rules:

- Planned Enrollment and Planned Site Count are active in Simulation Mode only.
- Planned Enrollment and Planned Site Count both use the combined operational artifact through `src/operational_benchmarks.py`.
- Planned Enrollment defaulting uses a positive planned/estimated value when present.
- Non-completed trials without a planned/estimated enrollment use `max(observed_lower_bound, enrollment_p50)`.
- Completed trials with positive `number_of_facilities` use completed registry facility-count proxy for Planned Sites.
- Non-completed trials treat `number_of_facilities` as current registry facility-count proxy lower-bound/context.
- Non-completed Planned Sites default to `max(current_registry_facility_count_proxy, planned_enrollment / patients_per_site_p50)` when patients-per-site P50 is available.
- Pure site-count P50 is only a fallback when the enrollment-coherent patients-per-site candidate is unavailable.
- Raw site-count percentiles remain clinical-only. They do not use modality refinement or non-vaccine Infections fallback.

Current cohort and refinement rules:

- First select the clinical cohort: `phase_indication_rare`, then `phase_ta_rare`, then `phase_ta`, then `phase_only`.
- Invalid indication disables only indication-level cohorts; invalid therapeutic area disables only TA-level cohorts. The runtime still uses the strongest remaining valid cohort before falling back to `phase_only`.
- Keep a selected clinical row if at least one relevant operational metric has `n >= 50`; do not discard the row only because another metric is near-threshold low confidence.
- Same-level therapeutic modality refinement applies only to enrollment and patients-per-site.
- Same-level modality refinement requires `n >= 50`.
- Same-level modality refinement does not use unknown/unclassified modality.
- No `phase + modality` fallback exists.
- No `phase_only_modality` rows exist.
- For non-vaccine Infections trials only, if same-level modality refinement is unavailable, the runtime tries non-vaccine Infections fallback rows up the clinical hierarchy.
- Non-vaccine Infections fallback applies only to enrollment and patients-per-site and requires `n >= 50`.
- If no non-vaccine Infections row is usable, the runtime keeps the original all-modality clinical fallback.

Current artifact and audit values:

- Artifact rows: 4,006.
- Duplicate benchmark keys: 0.
- Source data version: `0a97519bd78f561a`.
- Rows by level: `phase_indication_rare` 656, `phase_indication_rare_modality` 1,948, `phase_indication_rare_non_vaccine_infections` 81, `phase_ta_rare` 133, `phase_ta_rare_modality` 660, `phase_ta_rare_non_vaccine_infections` 8, `phase_ta` 72, `phase_ta_modality` 440, `phase_ta_non_vaccine_infections` 4, `phase_only` 4.
- `search_registry` modality assignment: 4,423 assigned, 0 not assigned.
- Audit lookup coverage: enrollment `not_available=0`, patients-per-site `not_available=0`, site-count `not_available=0`.
- Audit modality/non-vaccine coverage: enrollment modality-refined 2,542 and non-vaccine Infections fallback 140; patients-per-site modality-refined 2,421 and non-vaccine Infections fallback 163; site-count modality-refined 0 and non-vaccine Infections fallback 0.
- Audit defaulting safety: site defaults below current proxy 0; completed site proxy not used 0.

Validation commands:

```bash
python scripts/build_operational_benchmarks.py
python scripts/check_operational_benchmarks.py
python -m py_compile scripts/build_operational_benchmarks.py scripts/check_operational_benchmarks.py src/operational_benchmarks.py frontend/views/edit_trial.py
git diff --check
```

Manual/browser validation still needed before deployment readiness:

- Open Simulation Mode in the edit-trial app.
- Confirm Enrollment and Site Count cards render.
- Confirm vaccine examples such as `NCT05035212` still use modality-refined enrollment and patients-per-site.
- Confirm non-vaccine Infections examples such as `NCT04938830` use non-vaccine Infections fallback for enrollment and patients-per-site but not for raw site-count.
- Confirm operational-only changes still do not call `/predict`.
- Confirm Completion Score, SHAP, impact bar, treemap, pillar impacts, therapeutic-area calibration, audit behavior, API contracts, model artifacts, and taxonomy artifacts are unchanged.

### Planned Site Count - Future Metadata Shape

Future `operational_assumptions.planned_sites` metadata should use cautious source labels:

- `registry_facility_count_proxy`
- `completed_registry_facility_count`
- `current_registry_facility_count`
- `current_registry_facility_count_proxy`
- `benchmark_default`
- `enrollment_coherent_benchmark_default`
- `user_scenario`

Avoid source labels that imply unsupported precision:

- `actual_sites`
- `final_actual_sites`
- `planned_sites_from_registry`

Suggested future metadata shape:

```json
{
  "planned_sites": {
    "value": 80,
    "source": "registry_facility_count_proxy | completed_registry_facility_count | current_registry_facility_count | current_registry_facility_count_proxy | benchmark_default | enrollment_coherent_benchmark_default | user_scenario",
    "current_registry_facility_count_proxy": 25,
    "site_default_basis": "completed_registry_facility_count | current_registry_facility_count_proxy | enrollment_coherent_benchmark_default | benchmark_default | user_scenario | not_available",
    "patients_per_site_p50": 7.31,
    "enrollment_coherent_site_candidate": 10.4,
    "benchmark_level_used": "phase_indication_rare | phase_ta_rare | phase_ta | phase_only | not_available",
    "benchmark_n": 123,
    "benchmark_p25": 12,
    "benchmark_p50": 35,
    "benchmark_p75": 90,
    "benchmark_p90": 160,
    "site_count_status": "below_benchmark | typical | ambitious | above_benchmark_high | not_available",
    "support_level": "not_evaluated",
    "supporting_signals": [],
    "conflicting_signals": [],
    "benchmark_snapshot_id": "...",
    "is_benchmark_stale": false,
    "low_confidence_flag": false,
    "interpretation_hint": "Site count is above the usual registry facility-count benchmark for similar completed trials."
  }
}
```

### Planned Duration - Source Contract

`planned_duration_months` must not be implemented without date-type-aware source rules.

Source contract:

- Rebuilt `data/data_clinpred.csv` contains `start_date`, `completion_date`, `primary_completion_date`, `completion_date_type`, `primary_completion_date_type`, `primary_completion_duration_months`, `completion_duration_months`, and `start_year`.
- `data/studies.txt` contains date fields and matching `*_date_type` qualifiers.
- Important date/type pairs include `start_date -> start_date_type`, `completion_date -> completion_date_type`, `primary_completion_date -> primary_completion_date_type`, `study_first_posted_date -> study_first_posted_date_type`, `results_first_posted_date -> results_first_posted_date_type`, `disposition_first_posted_date -> disposition_first_posted_date_type`, and `last_update_posted_date -> last_update_posted_date_type`.
- `data/studies.txt` also contains `start_month_year`, `verification_month_year`, `completion_month_year`, and `primary_completion_month_year`.
- `*_date_type` fields are current registry qualifiers with values such as `ACTUAL`, `ESTIMATED`, or blank/missing. They are not historical planned-vs-actual pairs.
- The raw `studies.txt` file does not preserve both original estimated dates and later actual dates side by side. The current date value is replaced over time as the registry changes.
- If future source fields are needed beyond the exported CSV columns, duration validation must join back to `data/studies.txt` by `nct_id`.
- Do not use `actual_duration` blindly unless the architecture first documents how it is computed and whether it is reliable for the selected duration definition.

### Planned Duration - First-Estimate Methodology

The safest first deterministic definition is:

```text
completed full trial duration = completion_date - start_date, in months
```

Initial benchmark population:

- Completed trials.
- Valid `start_date`.
- Valid `completion_date`.
- Positive duration.
- `completion_date_type = ACTUAL`.

D1 must quantify whether requiring `start_date_type = ACTUAL` is too restrictive, because many completed trials may have blank `start_date_type`.

Endpoint-focused duration based on `primary_completion_date` must be audited separately and must not be mixed with full completion duration without a clear documented rule. `primary_completion_date - start_date` should serve as readout-timing context and consistency support because primary endpoint milestones often occur before full administrative study completion.

Endpoint-focused duration should therefore be interpreted as:

- Primary readout timing context.
- Not equivalent to final study duration.
- Potentially more reflective of protocol burden and interim milestone timing than administrative closeout duration.

Benchmark comparisons using endpoint-focused duration must remain separated from full completion-duration benchmarks unless a deterministic normalization rule is validated in D1.

Additional methodology rules:

- Ongoing trials should not use `today - start_date`, `verification_date - start_date`, `last_update_posted_date - start_date`, or extract/update timestamps as elapsed-duration lower bounds.
- Non-completed `ESTIMATED` completion and primary completion dates can be direct planned candidates when the trial is active/non-stopped.
- Non-completed `ACTUAL` dates from stopped/interrupted trials are lower-bound or early-stop context only.
- No survival model or censoring model is part of this staged rollout.
- Deterministic historical medians and percentiles should be tested first.

Status groups:

```text
completed = COMPLETED
stopped_interrupted = TERMINATED, WITHDRAWN, SUSPENDED
active_nonstopped = RECRUITING, ACTIVE_NOT_RECRUITING, ENROLLING_BY_INVITATION, NOT_YET_RECRUITING
```

Validated source-priority logic:

1. Use trusted direct date-derived values when available.
2. Use completed-trial benchmark P50 when direct dates are missing or untrustworthy.
3. Apply floors only in fallback / untrusted cases so the estimate is not shorter than known lower-bound evidence.
4. Preserve metadata explaining which source was used and which values acted only as floors.
5. Completed trials with missing date types and valid date-derived durations may use those durations directly for the individual trial default with warning metadata, but must remain excluded from completed benchmark construction.

Primary completion logic:

```text
If active/non-stopped and primary_completion_date_type = ACTUAL:
    use actual primary completion duration directly

Else if active/non-stopped and primary_completion_date_type = ESTIMATED:
    use estimated primary completion duration directly

Else if completed and primary_completion_date_type = ACTUAL:
    use actual primary completion duration directly

Else if completed and primary_completion_date_type is missing and duration is valid:
    use primary completion duration directly for this individual trial
    exclude from benchmark construction
    add warning metadata

Else:
    planned_primary_completion_months =
    max(
      benchmark_primary_completion_p50,
      primary_duration_months_ml if available,
      actual_primary_completion_duration if stopped/interrupted and available,
      estimated_primary_completion_duration if stopped/interrupted and available,
      missing_type_primary_completion_duration if stopped/interrupted and available
    )
```

Do not force `primary_duration_months_ml` as a floor when a trusted active/non-stopped or completed `ACTUAL`, `ESTIMATED`, or completed missing-type primary completion date is available. If the trusted date-derived primary completion duration is shorter than `primary_duration_months_ml`, preserve the trusted value and add a warning flag.

Total duration logic:

```text
First calculate planned_primary_completion_months.

If completed and completion_date_type = ACTUAL:
    use actual total completion duration directly

Else if completed and completion_date_type is missing and duration is valid:
    use total completion duration directly for this individual trial
    exclude from benchmark construction
    add warning metadata

Else if active/non-stopped and completion_date_type = ACTUAL:
    use actual total completion duration, but mark status-lag / lower-confidence source

Else if active/non-stopped and completion_date_type = ESTIMATED:
    use estimated total completion duration directly

Else:
    planned_duration_months =
    max(
      benchmark_total_duration_p50,
      planned_primary_completion_months,
      actual_total_completion_duration if stopped/interrupted and available,
      estimated_total_completion_duration if stopped/interrupted and available,
      missing_type_total_completion_duration if stopped/interrupted and available,
      primary_duration_months_ml if no trusted primary completion exists
    )
```

If a trusted total duration is shorter than `planned_primary_completion_months`, preserve this as a warning/QA condition rather than silently overwriting the direct date-derived value.

Approved benchmark hierarchy:

1. `phase + indication + rare disease flag + endpoint duration bin`
2. `phase + therapeutic area + rare disease flag + endpoint duration bin`
3. `phase + therapeutic area + endpoint duration bin`
4. `phase + endpoint duration bin`
5. `phase + indication + rare disease flag`
6. `phase + therapeutic area + rare disease flag`
7. `phase + therapeutic area`
8. `phase only`

Use a cohort only when `n >= 50`. `endpoint duration bin` is derived from `primary_duration_months_ml`, for example `<=3`, `3-6`, `6-12`, `12-18`, `18-24`, `24-36`, `36-60`, and `>60` months.

Placeholder cohort values must not become specific duration benchmark identities:

```text
If gbd_cause_id_3_ml is 0, missing, or otherwise invalid:
  skip indication-level duration cohorts and continue to TA-level cohorts if TA is valid.

If therapeutic area is missing, UNKNOWN, UNCLASSIFIED, OTHER, or OTHER/UNCLASSIFIED:
  skip TA-level duration cohorts and continue to phase-level cohorts.

If primary_duration_months_ml is missing, non-positive, or cannot be assigned to a valid endpoint duration bin:
  skip endpoint-duration-bin cohorts and continue to the same clinical hierarchy without the endpoint-duration bin.
```

Do not use therapeutic modality, sponsor tier, administration complexity, endpoint rigor, masking, placebo, number of arms, allocation, or comparator benchmark as primary duration benchmark keys in v1. These may later become support/conflict signals after separate calibration.

Important data-quality risks:

- Missing dates.
- Partial dates.
- Blank date types.
- Estimated completion dates.
- Future dates.
- Negative durations.
- Implausibly short durations.
- Administrative lag.
- Terminated trials.
- Completed trials with stale estimated labels.
- Confusion between primary completion and full completion.

### Planned Duration - Future Metadata Shape

Suggested future metadata shape:

```json
{
  "planned_duration_months": {
    "value": 36.0,
    "source": "final_observed_total_duration | completed_missing_completion_date_type_duration | actual_completion_noncompleted_status_lag | estimated_planned_total_duration | benchmark_default_with_floors | user_scenario",
    "duration_definition": "start_date_to_completion_date_months",
    "date_basis": {
      "start_date_type": "ACTUAL | ESTIMATED | blank | unknown",
      "completion_date_type": "ACTUAL | ESTIMATED | blank | unknown"
    },
    "status_group": "completed | active_nonstopped | stopped_interrupted",
    "trusted_direct_date_used": true,
    "benchmark_default_used": false,
    "floors_applied": [],
    "warning_flags": [],
    "planned_primary_completion_months": 24.0,
    "primary_completion_source": "actual_primary_completion | estimated_primary_completion | completed_actual_primary_completion | completed_missing_primary_date_type_duration | benchmark_default_with_floors | not_available",
    "primary_completion_duration_months_context": 24.0,
    "endpoint_duration_months_context": 18.0,
    "actual_total_duration_lower_bound": null,
    "actual_primary_completion_lower_bound": null,
    "estimated_total_duration_candidate": null,
    "estimated_primary_completion_candidate": null,
    "benchmark_level_used": "phase_indication_rare_endpoint_bin | phase_ta_rare_endpoint_bin | phase_ta_endpoint_bin | phase_endpoint_bin | phase_indication_rare | phase_ta_rare | phase_ta | phase_only | not_available",
    "benchmark_n": 123,
    "benchmark_p25": 18.0,
    "benchmark_p50": 30.0,
    "benchmark_p75": 48.0,
    "benchmark_p90": 72.0,
    "duration_status": "shorter_than_benchmark | typical | long | very_long | not_available",
    "support_level": "not_evaluated",
    "supporting_signals": [],
    "conflicting_signals": [],
    "benchmark_snapshot_id": "...",
    "is_benchmark_stale": false,
    "low_confidence_flag": false,
    "interpretation_hint": "Duration is longer than the usual completed-trial benchmark for similar trials."
  }
}
```

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

### Staged Roadmap

Recommended order:

```text
S1B completed -> S2 implemented -> S2 QA checkpoint GO -> S3 implemented -> post-S3 planned_sites defaulting revision implemented -> browser QA -> D1 -> D2 -> D3
```

Site-count stages:

- S1B: `planned_sites` architecture formalization and benchmark-feasibility audit. Completed.
- S2: `planned_sites` compact benchmark artifact and runtime utility. Implemented.
- S2 QA checkpoint: completed with GO before S3.
- S3: `planned_sites` Simulation Mode integration. Implemented.
- Post-S3 defaulting revision: implemented. It treats non-completed registry facility counts as lower-bound context and initializes editable `planned_sites` from an enrollment-coherent deterministic benchmark default.

Duration stages:

- D1: `planned_duration_months` duration-definition validation and data audit.
- D2: `planned_duration_months` compact benchmark artifact and runtime utility.
- D3: `planned_duration_months` Simulation Mode integration.

Each stage should be implemented separately. Completing S1B authorized only S2 after a separate prompt. S2 QA authorized S3 through a separate prompt, and S3 is now implemented. The post-S3 planned-sites defaulting revision is implemented. D1 does not authorize duration artifact creation or runtime activation unless a later prompt explicitly authorizes D2/D3.

### Decision Gates

S1B completed and authorized S2 because it confirmed:

- `number_of_facilities` source contract is confirmed.
- Completed positive facility-count population is large enough.
- Benchmark hierarchy has acceptable fallback coverage.
- Extreme outliers are documented.
- Phase-only fallback is reliable.
- There are no duplicate benchmark keys.
- Low-confidence rows are acceptable and flagged.
- Wording clearly avoids "actual site count" overclaiming.

S2 QA completed with GO before S3 by verifying:

- The historical site-only artifact existed, was non-empty, and had the required schema.
- The historical site-only report matched the artifact/report summary values.
- The historical site-only checker passed.
- The historical site-only runtime returned schema-safe missing-artifact metadata and deterministic lookup behavior.
- Boundary classification at P25/P75/P90 matches the S2 check script.
- Before S3, `planned_sites` remained inactive, with no UI, API, model, SHAP, calibration, taxonomy, audit/demo parity, or prediction-payload changes.

D1 must confirm:

- rebuilt `data/data_clinpred.csv` date/date-type fields are reliable for duration planning, with `studies.txt` as the fallback source if more qualifiers are needed.
- Completed actual completion-date population is quantified.
- `start_date_type` restrictiveness is quantified.
- Duration definition is selected.
- Duration outliers are documented.
- `completion_date` versus `primary_completion_date` boundary is documented.
- Primary completion timing remains clearly separated as readout context and consistency support, not the total-duration target.
- Trusted active/non-stopped or completed date-derived values are not silently overwritten by `primary_duration_months_ml` floors.
- Stopped/interrupted actual dates are lower-bound or early-stop context only.
- The endpoint-duration-bin + clinical fallback hierarchy remains acceptable at `n >= 50`.

### Explicit Non-Goals

This staged plan explicitly excludes:

- `planned_countries`.
- Cost estimation.
- Country-count estimation.
- Site-level modelling.
- Recruitment modelling.
- Survival modelling.
- LLM narratives.
- Coherence Score.
- Adjusted Trial Value Score.
- Model retraining.
- Prediction-pipeline changes.
- API changes.
- Frontend activation outside the documented Simulation Mode operational-assumption path.

## Operational Assumptions Snapshot Container

The runtime stores operational assumptions in a structured `operational_assumptions` object. `planned_enrollment` and `planned_sites` are currently the active benchmarked operational assumptions. `planned_duration_months` remains inactive until D3. `planned_countries` remains excluded.

S3 activates `planned_sites` in the `operational_assumptions` runtime container for Simulation Mode only. It remains outside XGBoost, `/predict`, SHAP, therapeutic-area calibration, audit/demo parity behavior, model artifacts, taxonomy artifacts, and API contracts.

Current active/reserved shape:

```python
operational_assumptions = {
    "planned_enrollment": {
        "value": 600,
        "source": "user_scenario",
        "benchmark_level_used": "phase_ta_rare",
        "benchmark_n": 123,
        "benchmark_p25": 120,
        "benchmark_p50": 280,
        "benchmark_p75": 520,
        "benchmark_p90": 900,
        "enrollment_status": "ambitious",
        "support_level": "not_evaluated",
        "supporting_signals": [],
        "conflicting_signals": [],
        "benchmark_snapshot_id": "...",
        "is_benchmark_stale": False,
        "low_confidence_flag": False,
        "interpretation_hint": "..."
    },
    "planned_sites": {
        "value": 80,
        "source": "enrollment_coherent_benchmark_default",
        "current_registry_facility_count_proxy": 25,
        "site_default_basis": "enrollment_coherent_benchmark_default",
        "patients_per_site_p50": 7.31,
        "enrollment_coherent_site_candidate": 10.4,
        "benchmark_level_used": "phase_ta",
        "benchmark_n": 123,
        "benchmark_p25": 12,
        "benchmark_p50": 35,
        "benchmark_p75": 90,
        "benchmark_p90": 160,
        "site_count_status": "typical",
        "support_level": "not_evaluated",
        "supporting_signals": [],
        "conflicting_signals": [],
        "benchmark_snapshot_id": "...",
        "is_benchmark_stale": False,
        "low_confidence_flag": False,
        "interpretation_hint": "..."
    },
    "planned_countries": {
        "status": "future_reserved",
        "value": None,
        "benchmark_status": "not_implemented"
    },
    "planned_duration_months": {
        "status": "future_reserved",
        "value": None,
        "benchmark_status": "not_implemented"
    }
}
```

Current and future container rules:

- `planned_enrollment` is currently active and benchmarked.
- `planned_sites` is currently active and benchmarked in Simulation Mode only.
- `planned_duration_months` remains inactive until D3.
- `planned_countries` remains excluded.
- Reserved future keys must not appear as user-editable assumptions until their own staged implementation phase.
- Reserved future keys should not drive score, narrative, charts, or model behavior.
- Active operational assumptions should not drive Completion Score, SHAP, pillar impacts, impact bar, treemap, therapeutic-area calibration, or `/predict` payloads unless a separate architecture decision changes that boundary.
- Cost, market potential, spend curve, and future development commitment should not be added to this object.

## Why LLM Scoring Comes Later

LLM narrative generation and Coherence Score should wait until the structured operational assumptions are stable. The system needs a deterministic payload before it asks an LLM to explain trial-design coherence.

The future narrative layer should consume benchmarked assumptions. It should not invent operational ranges in free text, infer missing site/country/duration values without validated artifacts, or convert narrative phrasing into hidden score changes.

Order of work:

```text
deterministic assumptions -> structured payload -> narrative explanation -> scoring rubric
```

This keeps the benchmark layer auditable. It also prevents the first LLM implementation from becoming an implicit estimator for sites, countries, duration, cost, market size, or feasibility.

## Deterministic Benchmark Pattern For Future Operational Assumptions

Any future duration layer should follow the enrollment and site-count pattern only after its data validation passes. Planned Site Count is now active in Simulation Mode only, using the combined operational benchmark artifact and runtime utility. Country-count planning is excluded from the current staged roadmap.

Required pattern:

- Offline artifact builder.
- Compact production artifact.
- Runtime lookup utility.
- P25/P50/P75/P90 benchmark percentiles.
- Benchmark classification.
- Support/conflict signals later, after the base benchmark is stable.
- Snapshot metadata.
- No direct XGBoost score modification.
- No SHAP modification.
- No therapeutic-area calibration modification.
- No audit/demo parity behavior modification.

Each future layer should define its own source priority, target-readiness flags, cohort hierarchy, fallback behavior, confidence threshold, and invalid-value handling. Site count and duration should not inherit enrollment assumptions by default.

## Historical Non-Goals For S1B Planned Sites Audit

S1B is a benchmark-feasibility audit and architecture-hardening step only. It should not:

- Activate `planned_sites`.
- Create production site benchmark artifacts.
- Create `src/site_benchmarks.py`.
- Add `planned_sites` UI.
- Implement `planned_duration_months`.
- Implement `planned_countries`.
- Estimate cost, market potential, spend curve, or future development commitment.
- Call an LLM.
- Implement Coherence Score.
- Retrain XGBoost.
- Modify SHAP.
- Modify therapeutic-area calibration.
- Modify audit/demo parity behavior.
- Modify API contracts.
- Deploy.

S1B only validated and documented whether `number_of_facilities` could support a compact `planned_sites` benchmark. S1B is complete, and S2 was later authorized and implemented through a separate prompt.

## Historical Non-Goals Before S3

Before S3 was explicitly authorized:

- Do not activate `planned_sites`.
- Do not add `planned_sites` UI.
- Do not add `planned_sites` to `ACTIVE_OPERATIONAL_ASSUMPTION_KEYS`.
- Do not import or activate `src/site_benchmarks.py` from `frontend/views/edit_trial.py`.
- Do not change `/predict`.
- Do not change XGBoost.
- Do not change SHAP.
- Do not change therapeutic-area calibration.
- Do not change audit/demo parity behavior.
- Do not change model artifacts.
- Do not change taxonomy artifacts.
- Do not change API contracts.
- Do not implement `planned_duration_months`.
- Do not implement `planned_countries`.
- Do not implement LLM narratives.
- Do not implement Coherence Score.
- Do not implement Adjusted Trial Value Score.
- Do not deploy based only on documentation.

S2 QA returned GO and S3 was separately authorized. These historical non-goals were superseded only for narrow `planned_sites` Simulation Mode activation. They remain active boundaries for all other excluded capabilities.

## Explicit Non-Goals After S3

After S3:

- Do not add `planned_sites` to `SIMULATION_FEATURE_IDS`.
- Do not add `planned_sites` to model-facing Trial Features.
- Do not send `planned_sites` to `/predict`.
- Do not include `planned_sites` in SHAP, pillar impacts, impact bar, treemap, therapeutic-area calibration, audit/demo parity behavior, model artifacts, taxonomy artifacts, API contracts, or prediction payload contracts.
- Do not implement `planned_duration_months`.
- Do not implement `planned_countries`.
- Do not implement LLM narratives.
- Do not implement Coherence Score.
- Do not implement Adjusted Trial Value Score.
- Do not deploy based only on S3 integration.

## Production Runtime Artifact Strategy

The production app should not need to load the full historical `data_clinpred.csv` dataset in order to calculate operational benchmarks at runtime.

The full historical dataset is used offline in the analytical notebook or build script to precompute compact benchmark artifacts.

Active runtime artifact path:

```text
frontend/data/operational_benchmarks_v1.csv
```

Active runtime report path:

```text
frontend/data/operational_benchmarks_v1_report.json
```

Active runtime utility:

```text
src/operational_benchmarks.py
```

Main active runtime functions:

```text
load_operational_benchmarks(...)
lookup_operational_benchmark(...)
classify_enrollment(...)
classify_site_count(...)
planned_enrollment_metadata(...)
planned_sites_metadata(...)
planned_sites_default_from_operational_benchmark(...)
```

The active artifact contains one row per benchmark cohort and fallback level with metric-specific evidence:

```text
phase / indication / therapeutic area / rare disease / therapeutic modality where applicable
enrollment_n / enrollment_p25 / enrollment_p50 / enrollment_p75 / enrollment_p90
site_count_n / site_count_p25 / site_count_p50 / site_count_p75 / site_count_p90
patients_per_site_n / patients_per_site_p25 / patients_per_site_p50 / patients_per_site_p75 / patients_per_site_p90
```

The runtime must keep metric-specific counts and confidence flags. It must not collapse them into one shared `benchmark_n`.
Same-level modality rows are available only for enrollment and patients-per-site refinement. Raw site-count evidence remains clinical-only.

Historical standalone enrollment-only and site-only benchmark artifacts have been removed. Current benchmark documentation and examples should use only the combined operational benchmark artifact and runtime:

```text
frontend/data/operational_benchmarks_v1.csv
src/operational_benchmarks.py
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

- Historical enrollment-only artifact regenerated successfully before later consolidation and deletion.
- Historical enrollment-only report regenerated successfully before later consolidation and deletion.
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

Current future estimation roadmap topics:

- Browser smoke testing and deployment readiness review for the implemented Planned Site Count Simulation Mode layer.
- Planned Duration Months benchmark layer, only if duration source quality supports it.
- Reconciliation of enrollment/sites/duration, only after those layers exist.

Cost translation, calendar spend, future development commitment, market potential, and full operational-scale estimation remain out of scope. They should not be added to the current operational assumptions container.

Planned Country Count remains excluded from the current staged roadmap. Reopen it only after a separate architecture decision and source audit.

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
- Do not add active sites, countries, total duration, cost, or market logic to v1.
- For the next operational-assumption work, use `Next Operational Assumptions - Sites And Duration Planning` as the source of truth.
- Treat S1B `planned_sites` audit as complete.
- Treat S2 `planned_sites` compact benchmark artifact/runtime utility as implemented.
- Treat S3 `planned_sites` Simulation Mode integration as implemented.
- Treat the post-S3 planned-sites defaulting revision as implemented.
- For non-completed trials, do not initialize the editable `planned_sites` assumption directly from `number_of_facilities`.
- Treat non-completed `number_of_facilities` as `current_registry_facility_count_proxy` context and a lower-bound candidate.
- Add deterministic patients-per-site benchmark support from completed trials with positive enrollment and positive registry facility-count proxy values.
- Use the enrollment-coherent default rule documented in `Planned Site Count - Post-S3 Defaulting Revision Implementation Status`.
- `planned_sites` integration must follow the existing operational-only update pattern and must not call `/predict` for operational-only changes.
- Do not add `planned_sites` to `SIMULATION_FEATURE_IDS`.
- Do not send `planned_sites` to `/predict`.
- Do not modify XGBoost, SHAP, therapeutic-area calibration, or audit/demo parity.
- Do not add new primary benchmark fields merely because they sound clinically plausible.
- Only add primary benchmark fields if they improve relevance while preserving sample size and percentile stability.
- Prefer support/conflict signals over over-granular benchmark matching when sample size is limited.
- Do not treat p-values alone as sufficient evidence for benchmark field selection.
- Treat planned enrollment as a scenario assumption, not as clinical truth.
- Treat benchmark percentiles as reference values, not recommendations.
- Treat deterministic `enrollment_status` as benchmark position, not clinical judgment.
- Let the future narrative / Coherence architecture interpret the benchmark after the structured payload is stable.
- Preserve the separation between planned values, final observed values, observed-to-date lower bounds, model defaults, and user scenarios.

## Historical Phase 2 Entry Point

This Phase 2 entry point is historical. It describes the already implemented Planned Enrollment foundation and is not the next implementation instruction. The next operational-assumption work must follow `Next Operational Assumptions - Sites And Duration Planning`: S1B is complete, S2 is implemented, S2 QA returned GO, S3 Simulation Mode integration is implemented, and the post-S3 planned-sites defaulting revision is implemented. The next step is browser smoke testing and deployment readiness review for the active operational benchmark layer.

Historical Planned Enrollment implementation sequence:

1. Add the Planned Enrollment field to Simulation Mode without changing the XGBoost prediction payload or model-facing feature set.
2. Select the initial planned enrollment assumption using the source-priority rules:
   - estimated/planned value when available,
   - completed final observed value only as completed-trial context,
   - benchmark-derived model default when no usable planned value exists,
   - `user_scenario` after participant edit.
3. Create the `operational_assumptions` snapshot container.
4. Attach `planned_enrollment_metadata(...)` output to the latest prediction snapshot.
5. Mark benchmark metadata stale when relevant design fields change before the next `Predict Trial Completion`.
6. Show a small Enrollment Assumption note/card that uses the deterministic metadata.
7. Reserve `planned_sites`, `planned_countries`, and `planned_duration_months` as inactive metadata or documentation only.

Do not add support/conflict signal generation, LLM calls, Coherence Score, active sites, countries, active duration, cost, market potential, API contract changes, deployment changes, model retraining, SHAP changes, therapeutic-area calibration changes, or audit/demo parity changes without a separate architecture decision.

Future operational-assumption phases must still preserve the completion model boundary unless a separate architecture decision changes it:

```text
Completion Score = existing XGBoost/SHAP/TA-calibrated score.
Enrollment benchmark = deterministic metadata for narrative/coherence reasoning.
No enrollment benchmark value modifies XGBoost, SHAP values, TA calibration, audit parity, or the existing prediction API contract.
```
