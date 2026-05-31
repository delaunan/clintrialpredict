# CTPredict Estimation Architecture

## 1. Purpose of this document

This document is the planning backbone for the future CTPredict estimation mode. It defines the strategic and technical architecture for turning the current trial prediction product into a missing operational value estimation workflow.

This is not an implementation specification yet. It does not prescribe immediate code edits, model training, database changes, or UI changes. It should guide future discussions with Gemini CLI, Codex CLI, or another coding assistant before implementation begins.

## 2. Product vision

The estimation ambition is to fill missing operational values for clinical trials in a reproducible, auditable way. The immediate focus is final duration, enrollment, site count, and country count, with clear separation between observed values, lower bounds, and modelled estimates.

The workflow should feel like an estimation and data-quality workbench. The core question is: "Given the fields available for this trial, what operational values are missing, what can be estimated, and how reliable are those estimates?"

## 3. Current foundation

CTPredict currently provides:

- An AACT / ClinicalTrials.gov-based trial database.
- A focus on industry-led Phase II / Phase III clinical trials.
- An XGBoost completion / early-termination prediction engine.
- A Streamlit application for trial search, trial detail review, and completion-risk scoring.
- A scoring architecture aligned with the existing API, registry, and parity audit workflow.
- A future product direction toward estimation and strategic forecasting modes.

## 4. Target estimation workflow

The future workflow should help analysts prepare trial-level operational estimates for downstream cost, planning, and forecasting work. Users start from existing trial records, identify missing or incomplete operational quantities, and generate auditable estimates with uncertainty ranges and validation flags.

Example users:

- Data analyst.
- Clinical operations analyst.
- Forecasting analyst.
- Business development analyst.
- Finance partner.

Example tasks:

- Identify missing or unusable operational fields.
- Reconstruct country count from source files.
- Estimate final duration, enrollment, sites, and countries.
- Preserve observed-to-date values as lower bounds.
- Flag implausible operational bundles.
- Export a reusable estimation table.

Example constraints:

- Avoid leakage from final observed values when forecasting from design-stage inputs.
- Keep estimates reproducible from committed source data.
- Label every estimate with its source, target definition, model family, and validation status.
- Avoid writing derived outputs back into primary raw data.

## 5. Main architecture overview

| Module | Purpose | Input | Output | Implementation stage |
| ------ | ------- | ----- | ------ | -------------------- |
| Trial data layer | Provide structured trial records from AACT / ClinicalTrials.gov and existing registry artifacts. | Raw and processed trial data, sponsor data, design data, eligibility data, intervention data. | Trial-level records for search, scoring, and estimation. | Existing |
| Design-stage feature layer | Represent information available at or near trial start. | Trial design, phase, sponsor, therapeutic area, condition, intervention, eligibility, planned dates. | Leakage-safe feature set for prediction and estimation. | Existing / Planned MVP |
| Completion-risk engine | Estimate completion or early-termination risk using the current model. | Design-stage model features from the existing pipeline. | Completion-risk score, probability, and explanation components. | Existing |
| Operational scale estimation engine | Estimate missing operational quantities. | Design-stage features, historical completed trials, active-trial censoring logic. | Predicted duration, enrollment, site count, country count, recruitment burden. | Planned MVP |
| Reconciliation engine | Check whether estimated operational quantities are mutually plausible. | Estimated enrollment, sites, countries, duration, lower bounds, historical ratios. | Flags for impossible or suspicious bundles. | Planned MVP |
| Cost translation layer | Optionally convert operational estimates into transparent cost estimates later. | Operational quantities, phase, therapeutic area, complexity assumptions, cost parameters. | Total trial cost, remaining cost, avoided cost, cost drivers. | Planned later |
| Estimation output layer | Produce reusable trial-level estimation records. | Trial records, model outputs, validation flags, metadata. | Estimation-ready table for later UI, cost, or forecasting work. | Planned MVP |
| Estimation validation engine | Evaluate model stability and estimate quality. | Validation folds, subgroup diagnostics, error metrics, reconciliation checks. | Champion model selection and known limitations. | Planned MVP |
| Explanation layer | Explain estimate sources, target definitions, and uncertainty. | Model outputs, assumptions, diagnostics, source fields. | Human-readable estimation notes and audit metadata. | Planned later |

## 6. Data layers

The future system should distinguish between design-stage data, final observed operational data, and synthetic financial data. These layers should remain explicit in documentation, modelling datasets, and future implementation.

### 6.1 Design-stage data

Design-stage data is information known at or near trial start. It is appropriate for forecasting because it does not depend on future trial execution.

Examples:

- Phase.
- Sponsor.
- Therapeutic area.
- Condition.
- Intervention type.
- Allocation.
- Masking.
- Intervention model.
- Primary purpose.
- Planned enrollment.
- Planned start date.
- Planned completion dates.
- Number of arms.
- Number of outcomes.
- Eligibility complexity proxies.

### 6.2 Final observed operational data

Final observed operational data is known only once a trial has progressed or completed. These variables can be useful as prediction targets, validation data, or historical calibration inputs, but they must not be used incorrectly as design-stage model features.

Examples:

- Actual enrollment.
- Final duration.
- Final number of sites.
- Final number of countries.
- Final trial status.
- Final completion or termination outcome.

Using these variables as inputs to a design-stage forecast would create leakage if they were not available at the decision point.

### 6.3 Synthetic financial data

Synthetic financial data is created by the future cost engine. It is a estimation layer, not observed sponsor truth.

Examples:

- Estimated total cost.
- Estimated cost spent to date.
- Remaining cost.
- Annual cost by year.
- Avoided cost if terminated.
- Downstream development cost.
- Probability-weighted future investment.

Cost outputs should be labelled as estimates derived from assumptions, not validated real-world sponsor costs.

### 6.4 Available cost-driver fields in the current data

The current `data/data_clinpred.csv` file already contains several fields that can support a first cost-reconstruction strategy. These fields should not all be treated as equally reliable. Some are design-stage fields, some are observed-to-date fields, and some are final observed fields only for completed trials.

| Cost need | Available field or source | Initial interpretation |
| --------- | ------------------------- | ---------------------- |
| Trial identity | `nct_id`, `brief_title`, `official_title`, `ui_search_label` | Asset or trial identifier for portfolio display. |
| Current status | `overall_status` | Determines whether operational values are final, partial, planned, or censored. |
| Stage | `phase`, `phase_ui`, `phase_ml` | Core cost and future-commitment driver. |
| Enrollment | `enrollment`, `enrollment_type` | Reliable final target mainly for completed trials with actual enrollment; ongoing actuals are lower bounds, not final totals. |
| Duration | `start_date`, `completion_date`, `primary_completion_date`, `primary_duration_months`, `is_duration_unknown` | Completed trials should define total trial duration from start-to-completion dates. `primary_duration_months` should be treated as an endpoint/follow-up duration proxy, not the total cost-duration target. |
| Sites | `number_of_facilities` | Final-ish for completed trials; observed-to-date lower bound for ongoing trials. |
| Countries | Reconstruct from `data/countries.txt` by `nct_id` | Country count is not directly in `data_clinpred.csv`; ongoing country lists are observed-to-date lower bounds. The cost method should remain US-agnostic. |
| Design complexity | `number_of_arms`, `allocation`, `masking`, `intervention_model`, `has_placebo`, `has_dmc` | Cost multipliers and operational-complexity proxies. |
| Therapeutic context | `therapeutic_area`, `gbd_indication_name`, `gbd_cause_id_3`, `is_rare_disease` | Phase and TA cost calibration, rare-disease recruitment burden, future market logic. |
| Product complexity | `therapeutic_modality`, `administration_complexity`, `primary_purpose` | Per-patient and protocol-complexity assumptions. |
| Endpoint burden | `endpoint_rigor`, `endpoint_structure`, `primary_outcomes_ui` | Endpoint and follow-up complexity assumptions. |
| Sponsor context | `sponsor_tier`, `lead_sponsor_canonical` | Optional calibration feature, not a direct cost claim. |
| Market and pricing-power inputs | `daly_global`, `daly_high_income`, `yld_global`, `yll_global`, `chronic_ratio_global`, `market_skew_index`, `is_rare_disease` | Future value and pricing-power layer, separate from cost. |

The first cost engine should use these fields to reconstruct realistic relative costs, not to claim access to true sponsor accounting data.

## 7. Design-stage versus future-observed leakage principle

Future models must preserve a strict separation between:

- What is known at trial design stage.
- What is observed after trial execution.
- What is estimated synthetically by a model or scenario rule.

This separation is critical because estimation mode may forecast from information available at a specific point in time. Inputs available after that point must not be treated as if they were available to the estimator.

Rule: Any future model must document whether each input feature is available at design stage, observed after trial execution, or synthetically estimated.

This rule applies to completion-risk modelling, operational scale estimation, optional cost translation, market potential estimation, downstream commitment estimation, and estimation validation.

## 8. Estimation date data contract

The estimation mode can be anchored to an estimation date. This is the date at which the model is assumed to forecast missing or incomplete operational values. A trial may later be completed, terminated, withdrawn, or still ongoing in the real dataset, but estimation features must only use information that would be available at or before the estimation date.

For the first estimation version, the objective is not to reconstruct a perfect historical AACT snapshot. The objective is to build a robust modelling contract for missing operational values. Final completion or termination outcomes may be used as training targets for completed trials, but they must not leak into design-stage forecast features.

### 8.1 Scenario date concepts

| Concept | Meaning | Use |
| ------- | ------- | --- |
| Estimation date | Date at which missing or incomplete values are estimated. | Defines what is visible, observed-to-date, and forecast-only. |
| Trial start date | Date trial activity begins. | Determines eligibility and elapsed duration. |
| Known or planned completion date as of estimation date | Completion expectation visible at that point if available. | Input to duration estimate, subject to uncertainty. |
| Final observed completion date | Completion date visible only after trial execution. | Training target or validation data, not an input for leakage-safe forecasts. |
| Current extracted status | Trial status in the current source extract. | Useful for data audit, but not necessarily the status at a historical estimation date. |
| Estimation status | Whether a record is completed, ongoing, censored, lower-bound only, or unsuitable for a target. | Keeps target construction and feature use explicit. |
| Expected completion date | Estimated or planned end date. | Candidate duration feature or later cost input, subject to uncertainty. |

### 8.2 Trial eligibility for estimation

For a trial to appear in an estimation dataset, it should normally satisfy:

- The trial start date is present when duration or elapsed-time features are needed.
- The phase and therapeutic context are available.
- The target field is either observed as a completed-trial final value or explicitly treated as an observed-to-date lower bound.
- The record has enough design-stage features to support the selected estimator.
- The target definition does not require using future-observed values as input features.

The number of selected trials does not need to be fixed at architecture time. The same data contract should work for the full dataset, a filtered modelling cohort, or a manually curated validation subset.

### 8.3 Field-use contract

Each candidate field should be classified before use.

| Field class | Description | Example | Allowed estimation use |
| ----------- | ----------- | ------- | ------------------------------ |
| Directly visible at estimation date | Known by the estimation date. | Phase, sponsor, therapeutic area, trial title, planned enrollment if available. | Yes. |
| Observed-to-date lower bound | Current operational value visible by the estimation date but not necessarily final. | Patients enrolled to date, active/listed sites, listed countries, elapsed duration. | Yes, as lower bound and audit input. |
| Final observed only | Known only after trial execution. | Final actual enrollment, final duration, final site count, final country count, final outcome. | No, except for training, validation, and retrospective audit. |
| Prediction feature | Field allowed as input to an estimator. | Phase, TA, rare disease flag, modality, planned enrollment, number of arms. | Yes if available at estimation date. |
| Synthetic estimate | Modelled or rule-based output. | Predicted final enrollment, site count, country count, or duration. | Yes, if labelled as estimate. |
| Excluded for leakage | Field that reveals future knowledge relative to the estimation date. | Final completion outcome for an active historical trial. | No. |

### 8.4 Precomputation versus runtime calculation

Anchoring to an estimation date does not mean every value must be calculated interactively. Most asset-level estimates can be precomputed in advance. Date-dependent values are lightweight and can be calculated when an estimation date is selected.

Precomputed asset-level estimates:

- Predicted final duration.
- Predicted final enrollment.
- Predicted final site count.
- Predicted final country count.
- Estimated total current-trial cost.
- Estimated future-phase cost to reach market.
- Estimated remaining time to market.
- Completion-risk score from the existing XGBoost engine.
- Market potential category.

Runtime date-dependent calculations:

- Elapsed duration at the missing-value review date.
- Cost incurred by the missing-value review date.
- Remaining current-trial cost after the date.
- Saveable cost this year.
- Saveable cost next year.
- Saveable cost in following years.
- Whether the selected date is plausible for the selected asset.

For a selected estimation cohort, these runtime calculations should be computationally light. They are mainly date arithmetic plus deterministic rules applied to precomputed operational estimates.

### 8.5 First-stage implementation path

The first implementation stage should follow this sequence:

1. Identify and classify variables by field class.
2. Build completed-trial training targets for final duration, enrollment, sites, and countries.
3. Reconstruct observed-to-date and lower-bound fields for selected scenario assets where possible.
4. Estimate missing or incomplete operational quantities for ongoing/actionable trials.
5. Apply transparent cost assumptions to produce current-trial cost outputs.
6. Invent next phases up to market using scenario templates and similar completed trials.
7. Add next-phase cost and risk assumptions to estimate future development commitment.
8. Produce a reusable asset-level table with precomputed estimates.
9. At scenario time, apply the selected missing-value review date to calculate incurred, remaining, and saveable cost.

This sequence should be completed before any estimation UI work beyond a minimal prototype.

## 9. Operational scale estimation plan

Future models should estimate the operational quantities required by the cost engine. These estimates should be separated from the current completion-risk model so that cost assumptions remain interpretable and modular.

Completed trials should be the preferred training base for final operational scale. Active, recruiting, not-yet-recruiting, enrolling-by-invitation, and active-not-recruiting trials require careful treatment because duration, enrollment, site count, and country count may be censored, planned, revised, or incomplete.

Terminated and withdrawn trials are useful for modelling partial spend and termination risk, but they should not be treated as clean examples of the final scale a trial would have reached if completed.

### 9.1 Operational value types

Future datasets should distinguish three operational values per trial and per quantity:

| Value type | Meaning | Main use |
| ---------- | ------- | -------- |
| Observed-to-date | Value currently visible in AACT / ClinicalTrials.gov at the estimation date. | Lower bound for ongoing trials; partial-spend input. |
| Predicted final value | Estimated final operational scale. | Main estimate used for missing-value completion. |
| Partial observed value | Partial scale observed before completion, termination, or extraction. | Lower-bound and censoring audit. |

For ongoing trials, observed patient, site, and country values should constrain the prediction but should not be assumed to be the final total. If an ongoing trial has already reported 120 patients, 20 sites, or 4 countries, the predicted final value should not be lower than those observed-to-date values.

```text
predicted_final_if_continued = max(model_prediction, observed_to_date)
```

The same principle applies to duration:

```text
predicted_final_duration = max(model_prediction, elapsed_duration_to_decision_date)
```

For the first version, exact historical observed-to-date patient, site, and country values may not be available for a past estimation date. In that case, use the current extract as a pragmatic lower-bound approximation where appropriate and clearly label it as an approximation.

Before building model-ready datasets, create explicit target-readiness flags. These flags make the operational modelling step auditable and prevent planned, ongoing, completed, and partial-stop values from being mixed accidentally.

Required MVP flags:

| Flag | Meaning | Use |
| ---- | ------- | --- |
| `is_completed_actual_enrollment_target` | Completed trial with positive `ACTUAL` enrollment. | Training target for final enrollment model. |
| `is_ongoing_actual_enrollment_lower_bound` | Ongoing/actionable trial with positive `ACTUAL` enrollment. | Lower bound when estimating final enrollment. |
| `is_estimated_planned_enrollment` | Trial with positive `ESTIMATED` enrollment. | Planned/expected scale feature, not final actual truth. |
| `is_completed_site_count_target` | Completed trial with positive site/facility count. | Training target for final site-count model. |
| `is_ongoing_site_count_lower_bound` | Ongoing/actionable trial with positive site/facility count. | Lower bound when estimating final site count. |
| `is_completed_country_count_target` | Completed trial with reconstructed positive country count. | Training target for final country-count model. |
| `is_ongoing_country_count_lower_bound` | Ongoing/actionable trial with reconstructed positive country count. | Lower bound when estimating final country count. |
| `is_completed_duration_target` | Completed trial with positive known duration. | Training target for final duration model. |

For duration, the MVP target should be date-derived total trial duration:

```text
total_duration_months_observed =
    (completion_date or primary_completion_date - start_date) / average days per month
```

`primary_duration_months` can remain a useful endpoint-duration or follow-up-duration feature, but it should not be used as the total trial-duration target for cost modelling.

### 9.2 Recommended MVP modelling approach

For the MVP, avoid predicting total cost directly. Instead, estimate operational quantities and convert them into cost with transparent assumptions.

Recommended first models:

- Final duration model trained on completed trials.
- Final enrollment model trained on completed trials with reliable actual enrollment.
- Final site-count model trained on completed trials.
- Final country-count model trained on completed trials after reconstructing country count from `data/countries.txt`.

The preferred first algorithm family is gradient-boosted tabular regression because the project already uses XGBoost, the source data is structured and mixed-type, and the targets are nonlinear and right-skewed. The architecture should still select the best model per indicator using validation, not assume one algorithm always wins.

Fallbacks should remain simple and explainable:

- Phase + therapeutic area median.
- Phase + rare disease flag median.
- Phase + therapeutic modality median.
- Global phase median when sample sizes are sparse.

| Future model | Target variable | Why it matters | Possible features | MVP priority |
| ------------ | --------------- | -------------- | ----------------- | ------------ |
| Final duration model | Date-derived total duration in months for completed trials. | Drives monthly management cost, calendar spend, and time-to-next-decision. | Phase, therapeutic area, intervention type, endpoint-duration proxy, planned enrollment, number of arms, masking, allocation, endpoint proxies. | High |
| Actual enrollment model | Final actual enrolled patients for completed trials. | Drives patient-level cost and recruitment burden. | Planned or observed-to-date enrollment, phase, condition, sponsor type, therapeutic area, rare disease proxy, eligibility complexity. | High |
| Site count model | Final site count for completed trials. | Drives site startup, monitoring, and geography complexity. | Planned or observed-to-date sites, planned enrollment, phase, therapeutic area, sponsor tier, intervention model. | Medium |
| Country count model | Final country count for completed trials. | Drives country startup cost and operational complexity. | Reconstructed country count, sponsor tier, trial size, therapeutic area, phase, planned enrollment. | Medium |
| Recruitment complexity proxy | Relative difficulty of recruiting and retaining patients. | Modifies patient cost, duration risk, and operational burden. | Rare disease proxy, eligibility restrictions, age eligibility, condition severity, intervention type, trial duration. | High |
| Downstream phase size estimate | Expected size of invented next phase. | Supports future commitment estimates beyond the current trial. | Current phase, next-phase template, therapeutic area, condition, historical similar completed trials, market category. | Later |

### 9.3 Model selection protocol

Each operational indicator should use the simplest model that is accurate enough and stable enough for optional cost translation. The first notebook should compare a small set of candidates, then choose a champion per target.

| Target | Candidate models | Recommended first champion | Validation focus |
| ------ | ---------------- | -------------------------- | ---------------- |
| Final duration months | Phase/TA median baseline, `HistGradientBoostingRegressor`, `XGBRegressor` on `log1p(target)`. | `XGBRegressor` if it clearly improves over baselines; otherwise gradient boosting or median fallback. | MAE and RMSE on months after inverse transform; calibration by phase and TA; minimum duration sanity checks. |
| Final enrollment | Phase/TA median baseline, `HistGradientBoostingRegressor`, `XGBRegressor` on `log1p(target)`, optional quantile model later. | `XGBRegressor` on `log1p(enrollment)` for MVP if validation is stable. | Error on log scale and original scale; high-enrollment outlier handling; phase-level calibration. |
| Final site count | Phase/TA median baseline, `HistGradientBoostingRegressor`, `XGBRegressor` on `log1p(target)`. | Best validated gradient-boosting model, with median fallback for sparse groups. | Count plausibility, lower-bound enforcement, calibration by phase and multinational proxy. |
| Final country count | Phase/TA median baseline, `HistGradientBoostingRegressor`, `XGBRegressor` on `log1p(target)`. | Best validated gradient-boosting model after reconstructing country count from `data/countries.txt`. | Count plausibility, US-agnostic geography handling, calibration by phase and sponsor tier. |
| Recruitment complexity proxy | Rule-based score, shallow tree model, or gradient boosting if a target is defined. | Rule-based MVP score unless a strong supervised target is available. | Interpretability and directional plausibility. |
| Future phase size | Similar-trials median, weighted nearest-neighbor median, optional model later. | Similar-trials median for MVP. | Explainability, stability, and reasonable future-phase scale. |

General rules:

- Train final-scale models primarily on completed trials.
- Use `log1p` targets for positive, skewed quantities and inverse-transform predictions before cost calculation.
- Compare every model against a simple grouped-median baseline.
- Prefer the simpler model if performance is similar.
- Enforce lower bounds after prediction for actionable ongoing assets.
- Clip impossible values, such as negative duration, enrollment below 1, or country count below 1 when a country is known.
- Validate by phase, therapeutic area, rare disease flag, and sponsor tier to catch unstable subgroup behavior.
- Save model diagnostics and assumptions before using outputs in cost calculations.

The UI should not expose the model-selection complexity. It should show concise estimates and clear assumption labels.

### 9.4 Updated operational-size modelling direction

Early notebook benchmarks showed that direct point estimates for enrollment, sites, and countries can look much better when the model is allowed to use final observed operational quantities from completed trials as features. Those scores are useful as an upper-bound or known-input benchmark, but they are too optimistic for a forecast unless those quantities are actually known at the estimation date.

The recommended next approach is therefore:

1. Keep three benchmark modes.
   - **Dependency-pruned forecast**: exclude `enrollment_num`, `site_count_num`, and `country_count_num` when those values would have to be guessed. This is the cleanest design-stage forecast baseline.
   - **Sequenced forecast**: predict one operational quantity first, then feed that prediction into the next model. Use out-of-fold upstream predictions during training so downstream models do not learn from true completed values that would not be known in practice.
   - **Known-input mode**: allow enrollment, sites, or countries as features only when they are user-provided, planned, or observed-to-date lower bounds at the scenario date.
2. Prioritize duration and enrollment accuracy.
   - Duration is currently the most stable target because endpoint/follow-up duration, indication, sponsor, phase, and design features carry direct signal.
   - Enrollment remains the most important operational-size target and needs a stronger approach than one direct regressor.
3. Add supervised operational size bands before trying generic clustering.
   - For enrollment, create bands such as `tiny`, `small`, `medium`, `large`, and `mega`.
   - Train a classifier for enrollment band, then use band probabilities or band-specific priors alongside the log-enrollment regressor.
   - Evaluate whether the model first gets the order of magnitude right before judging exact patient count.
   - After this supervised baseline is established, run a controlled clustering/archetype experiment as an optional enhancement. Cluster trials using design, sponsor, indication, endpoint, and complexity features; add the cluster label as a candidate feature or cluster-specific prior; keep it only if it improves enrollment or duration validation metrics, not merely because the clusters look plausible.
4. Use grouped priors as stabilizers.
   - Blend or compare model outputs with historical medians and ranges by phase, therapeutic area, indication group, rare-disease flag, sponsor tier, and modality.
   - This is especially important because median completed enrollment is much smaller than the mean, while a small number of very large trials dominate absolute error.
5. Estimate sites from enrollment logic when appropriate.
   - Site count should be checked against expected patients per site for similar trials.
   - Independent site predictions can still be benchmarked, but the implementation should reconcile enrollment, sites, and countries as a bundle.
6. Return ranges, not only point estimates.
   - The estimation module should show expected value plus low/high scenario values for enrollment and duration.
   - Point estimates alone are misleading for heavily skewed operational targets.
7. Always run reconciliation checks before cost calculation.
   - Flag impossible or suspicious combinations such as countries greater than sites, fewer than one patient per site, or a very low-enrollment trial spread across many sites.
   - Reconciliation should not silently hide model uncertainty; it should make questionable operational bundles auditable.

This direction came from comparing four benchmark families in `notebooks/estimation.ipynb`: grouped-median baselines, enhanced gradient-boosted regressors, dependency-pruned regressors, and sequenced y-to-y models. The key conclusion is that the strongest raw scores were partly driven by operational cross-target features, while dependency-pruned and sequenced results are more credible for forecasting. Future sessions should use the pruned and sequenced scores as the default implementation benchmark, and reserve the stronger known-input/oracle scores for scenarios where the user explicitly provides planned or current operational values.

### 9.5 Required and optional data sources

The primary source for the first notebook should be `data/data_clinpred.csv`.

Recommended supporting source already present:

- `data/countries.txt`: reconstruct country count by `nct_id`.

Potentially useful existing sources if the notebook needs deeper audits:

- `data/design_outcomes.txt`: outcome count or endpoint timing audit if existing derived fields are insufficient.
- `data/interventions.txt`: intervention-type audit if `therapeutic_modality` requires validation.
- `data/conditions.txt`: condition text audit if indication grouping needs review.

Files not currently present but useful only for later versions:

- Historical AACT snapshots by date, if exact observed-to-date reconstruction becomes important.
- External phase-transition / likelihood-of-approval benchmarks, if the estimation workflow needs a true probability of reaching market rather than a scenario assumption.
- External clinical trial cost benchmarks, if assumptions need validation against published or licensed cost data.

The MVP does not require these external files. It can proceed with `data_clinpred.csv`, reconstructed country counts, transparent cost assumptions, and scenario-level future risk assumptions.

## 10. Cost translation engine

The cost model should be a transparent, assumption-driven finance engine. It should convert operational quantities into cost estimates using editable assumptions rather than a single black-box prediction.

Conceptual formula:

```text
Total trial cost =
    fixed setup cost
  + country startup cost x number of countries
  + site startup cost x number of sites
  + patient cost x number of patients
  + monthly trial management cost x duration in months
  + monitoring / data / safety cost
  + complexity multipliers
  + closeout and reporting cost
```

The first version should be interpretable, editable, and easy to explain. It should be clear which costs are driven by phase, geography, trial size, therapeutic area, and protocol complexity.

The cost engine should use `predicted_final_if_continued` values for total cost and remaining commitment. It should use `observed-to-date` values and elapsed duration for incurred cost. This allows the estimation workflow to estimate how much has already been spent, how much is committed if the trial continues, and how much can be saved by stopping or pausing.

The MVP should not assume linear spend. Current-trial cost should be allocated over the expected trial lifecycle using a simple spend curve. This is sufficient for discussion-quality missing value estimation and avoids overcomplicating the UI.

Cost calculation should be deterministic once the operational estimates and assumptions are fixed. This makes the output reproducible in a notebook and explainable in a workshop.

Recommended first calculation order:

1. Build final-if-continued operational estimates.
2. Calculate estimated total current-trial cost.
3. Apply the non-linear lifecycle spend curve to split cost before and after the missing-value review date.
4. Calculate saveable cost by calendar bucket.
5. Estimate future-phase cost to reach market.
6. Combine current remaining cost and future-phase cost into total development commitment.

Do not put every cost driver in the UI. Keep the UI focused on the decision variables; keep detailed assumptions in the notebook, documentation, and optional drill-down views.

| Cost driver | Description | Possible data source or proxy | Adjustable assumption |
| ----------- | ----------- | ----------------------------- | --------------------- |
| Phase | Development phase affects baseline cost and required evidence burden. | AACT phase field, current registry phase feature. | Phase-specific base cost and multiplier. |
| Therapeutic area | Different areas have different operational and endpoint cost structures. | Existing therapeutic area mapping. | Therapeutic-area multiplier. |
| Oncology flag | Oncology trials may require more complex endpoints, monitoring, and site infrastructure. | Therapeutic area or condition mapping. | Oncology complexity multiplier. |
| Rare disease proxy | Rare conditions can increase recruitment cost and duration uncertainty. | Existing rare disease status or indication proxy. | Recruitment and site activation multiplier. |
| Number of patients | Patient count drives per-patient treatment, visit, and data costs. | Planned enrollment or predicted actual enrollment. | Cost per patient. |
| Number of sites | Site count drives startup, monitoring, and management burden. | Observed site data or predicted site count. | Startup and monitoring cost per site. |
| Number of countries | Country count drives regulatory, startup, translation, and vendor complexity. | Observed country data or predicted country count. | Startup cost per country and geography multiplier. |
| Trial duration | Duration drives monthly management and vendor costs. | Planned duration or predicted final duration. | Monthly trial management cost. |
| Randomization | Randomized trials may increase operational and statistical complexity. | Design-stage allocation field. | Randomization multiplier. |
| Masking | Blinding can increase drug supply, monitoring, and operational burden. | Masking field. | Masking multiplier. |
| Intervention model | Parallel, crossover, factorial, or single-group designs have different operational profiles. | Design-stage intervention model. | Intervention-model multiplier. |
| Endpoint complexity | Complex endpoints may increase assessment and data management cost. | Outcome count, endpoint duration, endpoint proxies. | Endpoint complexity multiplier. |
| Eligibility complexity | Restrictive criteria can slow recruitment and increase screening burden. | Eligibility text-derived proxies, age and population flags. | Screening and recruitment multiplier. |
| Intervention type | Drug, biologic, device, procedure, or behavioral interventions may differ in cost. | Intervention type mapping. | Intervention-type multiplier. |
| Recruitment burden | Captures expected difficulty of finding and retaining patients. | Recruitment complexity model or proxy. | Recruitment burden multiplier. |
| Geography complexity | Multicountry and high-site-count trials increase coordination cost. | Country count, site count, region mix if available. | Geography complexity multiplier. |

### 10.1 Optional cost outputs

Cost outputs are downstream and optional for this branch. If later enabled, the first phase should produce a small set of estimate-derived cost outputs:

| Output | Meaning | UI use |
| ------ | ------- | ------ |
| Estimated total current-trial cost | Full expected cost if the current trial is completed. | Asset detail and cost explanation. |
| Cost incurred to decision date | Estimated spend already consumed. | Sunk-cost context, not a reason to continue by itself. |
| Remaining committed cost | Estimated additional cost implied by remaining trial activity. | Later planning input. |
| Saveable cost this year | Estimated cost that could be avoided in the current calendar year. | Budget-reduction exercise. |
| Saveable cost next year | Estimated cost that could be avoided in the next calendar year. | Medium-term budget planning. |
| Saveable cost in following years | Estimated later cost that could be avoided. | Long-horizon planning view. |
| Future development commitment | Estimated cost of invented future phases needed to reach market. | Strategic continuation burden. |
| Avoided future commitment | Future commitment avoided in a downstream scenario. | Later planning input. |

Any future UI should show a simplified subset: source operational estimates, estimated total cost, estimate confidence, and major cost drivers.

## 11. Calendar-year spend model

Optional downstream planning may need costs over time, not only total cost. Analysts should understand both the total obligation and the near-term budget impact implied by the operational estimates.

Key concepts:

- Estimation date.
- Trial start date.
- Expected end date.
- Elapsed duration.
- Spent-to-date.
- Remaining spend.
- Cost by calendar year.
- Cost avoided if terminated.

For ongoing trials, the calendar model should anchor on the estimation date. Costs before that date are incurred. Costs after that date are forecast commitments derived from the operational estimate.

Simple planned cost-curve structure:

```text
0-15% of duration: startup-heavy cost
15-70% of duration: recruitment and treatment-heavy cost
70-90% of duration: follow-up and data-cleaning cost
90-100% of duration: closeout and reporting cost
```

The first version can use a deterministic curve. Later versions may use phase-specific curves, therapeutic-area-specific curves, or empirical calibration if suitable data becomes available.

The MVP does not need a complex cash-flow model. A deterministic curve is sufficient if assumptions are visible and if the UI presents only decision-relevant totals.

This spend curve should be used to calculate:

- Cost incurred before the missing-value review date.
- Remaining committed cost after the missing-value review date.
- Saveable cost in the current calendar year.
- Saveable cost in the next calendar year.
- Saveable cost in following years.

Suggested deterministic MVP spend weights:

| Trial lifecycle segment | Share of duration | Suggested share of total cost | Rationale |
| ----------------------- | ----------------- | ----------------------------- | --------- |
| Startup | 0-15% | 20% | Country, site, vendor, protocol, and activation cost can be front-loaded. |
| Recruitment and treatment | 15-70% | 55% | Patient cost, monitoring, drug supply, and operations dominate this period. |
| Follow-up and data cleaning | 70-90% | 15% | Lower than recruitment, but still operationally meaningful. |
| Closeout and reporting | 90-100% | 10% | Database lock, analysis, reporting, and closeout. |

These weights are assumptions, not observed sponsor accounting data. They should be configurable in the notebook and later in facilitator settings if needed.

## 12. Downstream development commitment

Continuing an asset may imply large future commitments beyond the currently visible trial. The estimation workflow should make that future obligation visible without presenting it as observed truth.

Examples:

- Continuing a Phase II asset may imply a future Phase III program.
- Continuing a Phase I/II asset may imply Phase II and Phase III development.
- Terminating early may avoid large downstream investment.
- Keeping a risky but high-potential asset may still be strategically rational.

| Current asset stage | Current cost view | Future commitment view |
| ------------------- | ----------------- | ---------------------- |
| Phase I/II | Current early-stage study cost, remaining current-trial spend, and completion risk. | Potential Phase II expansion, future Phase III program, and larger evidence-generation commitment. |
| Phase II | Current proof-of-concept or dose-finding trial cost. | Potential pivotal Phase III commitment, manufacturing scale-up, and regulatory preparation cost. |
| Phase II/III | Current combined-stage or adaptive development cost. | Remaining pivotal-stage cost and possible post-approval or launch-readiness investment. |
| Phase III | Current pivotal-trial remaining cost. | Regulatory submission, launch-readiness, post-marketing, or lifecycle-management cost if continued. |

### 12.1 Invented future phases

Many assets in the source data will not have an explicit next-phase trial in the dataset. The estimation should therefore invent future phases as scenario estimates. These invented phases are not observed facts.

Recommended first approach:

- Define a next-phase template from the current asset stage.
- Find similar completed historical trials in the target next phase.
- Estimate typical duration, enrollment, site count, and country count using median or model-based values.
- Convert those operational estimates into cost using the same transparent cost engine.
- Apply simple phase-level and completion-risk assumptions to show likelihood-adjusted future commitment.

Suggested future-path templates:

| Current asset stage | Synthetic path to market |
| ------------------- | ------------------------ |
| Phase I/II | Phase II estimate plus Phase III estimate. |
| Phase II | Phase III estimate. |
| Phase II/III | Remaining pivotal estimate plus regulatory / launch-readiness placeholder. |
| Phase III | Remaining pivotal cost plus regulatory / launch-readiness placeholder. |

The future commitment engine should be framed as scenario logic, not as a claim that the exact future trial exists.

The output should be "future-phase cost to reach market", not only "next-phase cost". For a Phase II asset, this may mean the estimated Phase III commitment. For a Phase I/II asset, this may include a Phase II continuation and a later Phase III commitment. This is later planning logic and should remain separate from the immediate missing-value estimation objective.

Suggested outputs:

| Output | Meaning |
| ------ | ------- |
| Next required phase | Synthetic next development step implied by current stage. |
| Future phases to market | Full estimated path from current stage to market. |
| Future-phase cost to market | Estimated future development cost beyond the current trial. |
| Remaining research duration | Estimated time needed to complete current and future development. |
| Estimated time to market | Approximate years from committee date to potential launch / marketable asset. |
| Probability-adjusted future commitment | Future spend adjusted by simple risk assumptions where useful. |

### 12.2 Risk to market

The existing XGBoost score estimates trial completion or early-termination risk. It is not a full probability of regulatory approval or market launch. For the estimation MVP, this score can be used as one risk signal and combined with simple phase-level assumptions for future development risk.

Example MVP framing:

```text
near_term_trial_risk = current XGBoost completion / termination score
future_development_risk = phase-level scenario assumption
planning_risk_signal = combination of near-term risk and future-stage assumption
```

This keeps the first version honest: the existing model informs operational continuation risk, while future market-reaching probability remains a transparent scenario assumption until a dedicated model exists.

The current dataset does not contain true regulatory approval or launch outcomes. Therefore, a real probability of reaching market cannot be learned directly from `data_clinpred.csv` alone. For the MVP, risk-to-market should be a scenario estimate based on:

- The current XGBoost trial completion / termination risk.
- Simple phase-level future risk assumptions.
- Optional therapeutic-area or rare-disease adjustment if agreed later.

If a later version needs validated market-reaching probabilities, it will require external phase-transition or likelihood-of-approval benchmark data.

## 13. Notebook implementation plan

The first implementation should be developed in `notebooks/estimation.ipynb` as the analytical build notebook for the `estimation` module. It should follow the style of `notebooks/validation_clinpred.ipynb` and `notebooks/production_01.ipynb`: markdown explanation first, then focused code cells, with explicit audit checkpoints and reproducible outputs.

Recommended notebook structure:

| Notebook block | Purpose | Output |
| -------------- | ------- | ------ |
| `<REF:ENV_CONFIG>` | Configure autoreload, warnings, display settings, and reproducibility seeds. | Stable notebook environment. |
| `<REF:PATH_RESOLUTION>` | Resolve project root and data paths. | Portable paths to `data/`, `models/`, and future outputs. |
| `<REF:LIB_INIT>` | Import pandas, numpy, sklearn, xgboost if available, plotting, and metrics. | Shared notebook imports. |
| `<REF:DATA_LOAD>` | Load `data/data_clinpred.csv` and supporting `data/countries.txt`. | Base dataframe and country-count table. |
| `<REF:DATA_AUDIT>` | Audit schema, missingness, status counts, phase counts, and key cost-driver distributions. | Data-quality summary. |
| `<REF:FIELD_CONTRACT>` | Classify fields as direct, lower-bound, final target, feature, synthetic, or excluded. | Field contract table. |
| `<REF:TARGET_READINESS_FLAGS>` | Create explicit flags for completed training targets, ongoing lower bounds, and planned estimates. | Auditable target/lower-bound flags. |
| `<REF:TARGET_BUILD>` | Build completed-trial targets for duration, enrollment, sites, and countries. | Modelling dataset with target definitions. |
| `<REF:FEATURE_BUILD>` | Build leakage-safe features for operational models. | Feature matrix and preprocessing plan. |
| `<REF:MODEL_BENCHMARK>` | Compare baselines and candidate models per target. | Champion model selection per indicator. |
| `<REF:PREDICT_OPERATIONAL>` | Generate final-if-continued operational estimates for all candidate assets. | Predicted duration, enrollment, sites, countries. |
| `<REF:COST_ASSUMPTIONS>` | Define transparent cost assumptions and lifecycle spend weights. | Cost assumption table. |
| `<REF:COST_ENGINE>` | Calculate current-trial total, incurred, remaining, and saveable cost. | Current-trial cost table. |
| `<REF:FUTURE_PHASES>` | Create synthetic future path to market and estimate future-phase cost and time. | Future commitment table. |
| `<REF:RISK_TO_MARKET>` | Combine XGBoost trial risk with phase-level scenario assumptions. | Risk-to-market scenario fields. |
| `<REF:PORTFOLIO_EXPORT>` | Produce reusable asset-level table for estimation scenarios. | Portfolio-ready dataset. |
| `<REF:VALIDATION_AUDIT>` | Validate ranges, subgroup stability, and cost plausibility. | Audit summary and known limitations. |

The notebook should be written as an analytical build notebook, not as a UI implementation. It should include markdown between code blocks explaining each decision, especially target definitions, model choices, assumptions, and limitations.

Until the notebook outputs are validated, future work should not wire these estimates into Streamlit. The notebook should first prove the data contract, target construction, model selection, cost assumptions, and output table.

## 14. Reproducible output contract

The first estimation dataset should produce one row per trial or asset with a small set of stable columns. This table should be reusable by future Streamlit work without requiring the UI to rerun model training.

Recommended output groups:

- Identity: `nct_id`, title, sponsor, phase, therapeutic area, indication.
- Risk: existing XGBoost completion-risk score and simplified risk category.
- Operational estimates: predicted final duration, enrollment, site count, country count.
- Current-trial cost: estimated total cost, incurred cost, remaining cost, saveable cost by year.
- Future path: next required phase, future phases to market, future-phase cost to market, time to market.
- Market potential: initial category and later GBD / rare-disease value signals.
- Assumption metadata: cost-assumption version, model version, missing-value review date if date-specific values are materialized.

The UI should consume this as an estimation table. It should not expose training diagnostics, detailed model errors, or all raw cost-driver variables.

## 15. Market potential module

The market potential layer should estimate or categorize commercial and strategic value. It must remain separate from trial cost so users can distinguish operational affordability from potential value.

MVP fields may include:

- Market potential category.
- Competitive intensity.
- Strategic priority.
- Unmet need proxy.
- Indication size proxy.
- Expected commercial value category.

Initial market potential can be synthetic or categorical:

- Low.
- Medium.
- High.
- Transformational.

Later versions may evolve into:

- Peak sales estimate.
- Risk-adjusted NPV.
- Expected commercial value.
- Probability-adjusted planning contribution.

Any future valuation model should explicitly document assumptions and should not be mixed into the completion-risk model without a clear reason.

## 16. Estimation output engine

Each trial should receive explicit asset-level estimation outputs. These outputs should be stable enough for later UI, cost, planning, or forecasting layers without requiring the consumer to rerun model training.

Recommended output groups:

- Source identifiers: `nct_id`, title, sponsor, phase, therapeutic area, indication.
- Target availability flags: final target, observed lower bound, planned estimate, missing value.
- Point estimates: duration, enrollment, site count, and country count.
- Uncertainty ranges: low, expected, and high values where supported.
- Reconciliation flags: countries greater than sites, implausible patients per site, implausible sites per country.
- Model metadata: model family, version, feature mode, training cohort, validation metrics.
- Source metadata: source fields used, target definition, estimation date if relevant.

The output table should make missingness and estimate provenance visible. A downstream user should be able to distinguish an observed final value, an observed lower bound, a planned estimate, and a modelled value without reading notebook code.

## 17. Estimation validation engine

The validation engine should evaluate estimate quality across multiple dimensions. It should not only report global mean error; heavily skewed operational targets require robust metrics, subgroup checks, and plausibility audits.

| Validation dimension | What it measures | Possible calculation | MVP or later |
| --------------- | ---------------- | -------------------- | ------------ |
| Point error | Absolute prediction error on original scale. | MAE, RMSE, median absolute error, p75 error. | MVP |
| Log-scale error | Stability on skewed positive targets. | MAE and R2 on `log1p(target)`. | MVP |
| Order-of-magnitude quality | Whether estimates land in the right size band. | Enrollment band accuracy, within-2x accuracy. | MVP |
| Range calibration | Whether uncertainty intervals cover plausible truth. | Prediction interval coverage and width. | MVP |
| Subgroup stability | Whether errors concentrate in specific trial types. | Metrics by phase, therapeutic area, rare disease flag, sponsor tier. | MVP |
| Reconciliation quality | Whether estimated operational bundles are plausible. | Patients per site, sites per country, countries <= sites checks. | MVP |

## 18. Workflow principles

Future estimation UI work should follow these principles:

- The experience should feel like a missing-value estimation review.
- Avoid overwhelming the analyst with too many variables at once.
- Show observed values, lower bounds, estimates, and uncertainty together.
- Keep assumptions explainable.
- Allow drill-down from cohort view to trial detail.
- Maintain the clean professional CTPredict visual identity.
- Preserve current app stability.
- Avoid major redesign during early planning.
- Add estimation mode progressively.

The early product should extend the existing architecture rather than replace it.

## 19. Future screens

The following screens are possible future surfaces. They are not implemented yet.

| Screen | Purpose | Key information shown | MVP priority |
| ------ | ------- | --------------------- | ------------ |
| Estimation Cohort Setup | Select the trial cohort and target quantities. | Filters, target selection, feature mode, estimation date if used. | High |
| Missingness Dashboard | Show which operational values are missing, lower-bound only, or final. | Counts by target, status, phase, therapeutic area, data quality flags. | High |
| Trial Estimate Detail | Support drill-down on a single trial estimate. | Source fields, lower bounds, model estimate, uncertainty range, reconciliation flags. | High |
| Model Diagnostics | Explain validation performance and limitations. | Metrics by target, model family, phase, therapeutic area, and outlier group. | High |
| Reconciliation View | Audit implausible operational bundles. | Enrollment/site/country ratios, flags, worst examples. | High |
| Export Preview | Review the reusable estimation table before saving. | Output columns, metadata, row counts, version labels. | Medium |
| Assumption Settings | Allow approved maintainers to adjust estimation assumptions. | Target cleaning thresholds, model choices, range policy, save paths. | Later |

## 20. MVP definition

The first achievable version should be small, transparent, and estimation-driven.

MVP should include:

- A reproducible cohort of selected trial records.
- Existing completion-risk score.
- Predicted final duration.
- Predicted final enrollment.
- Predicted final sites and countries.
- Lower-bound handling for ongoing observed patients, sites, countries, and elapsed duration.
- Model comparison against grouped-median baselines.
- Dependency-pruned and sequenced benchmark modes.
- Reconciliation checks for enrollment, sites, and countries.
- Optional export of target datasets and estimates after validation.
- Clear model and target metadata.

Explicitly excluded from MVP:

- Full rNPV model.
- Real-world budget validation.
- Complex commercial forecasting.
- Complex multi-round scenario logic.
- Advanced facilitator controls.
- Automatic optimization engine.
- Real-time multiplayer features.
- Portfolio decision UI.

## 21. Future implementation phases

### Phase 0 - Architecture and planning only

No code. Build this document and refine assumptions.

### Phase 1 - Data audit

Identify available design-stage, observed-to-date, and final-observed variables. Confirm which variables are safe to use for each estimation mode. Reconstruct country count from `data/countries.txt` if needed. Produce the estimation data contract before cost or UI implementation.

### Phase 2 - Operational estimation datasets

Create modelling datasets for duration, enrollment, sites, and countries. Train-target construction should prefer completed trials for final scale. Ongoing values should be retained as lower bounds. Terminated and withdrawn trials should be flagged as partial-spend evidence rather than clean final-scale targets.

### Phase 3 - First operational models

Train first models or rule-based estimators for missing operational quantities. Keep outputs interpretable enough for optional cost translation and enforce lower-bound constraints for ongoing trials.

### Phase 4 - First cost engine

Create transparent assumption-driven cost calculations. Make assumptions configurable and clearly labelled as synthetic. Do not train a direct black-box cost model in the first phase.

### Phase 5 - Calendar spend engine

Convert total cost into incurred cost, remaining committed cost, saveable cost by calendar year, and annual spend. Use a simple non-linear lifecycle spend curve rather than assuming linear spend.

### Phase 6 - Estimation output MVP

Create a reusable estimation output table with source fields, estimated values, lower bounds, uncertainty ranges, reconciliation flags, and model metadata.

### Phase 7 - Estimation validation and review

Add validation summaries, estimate-quality notes, and analyst-friendly outputs.

### Phase 8 - Market potential and downstream development

Add future value and downstream cost logic. Keep market potential separate from cost and completion risk. Treat next phases as synthetic scenario estimates unless explicitly linked to observed successor trials.

### Phase 9 - Refinement and validation

Validate assumptions with expert review and scenario testing. Refine scoring and user experience based on observed use.

## 22. Open questions

- What is the first target user: data science, clinical operations, forecasting, or finance analytics?
- Should the estimation workflow use all eligible historical trials or a curated modelling cohort?
- Should trial names and sponsor names be anonymized in exported estimation tables?
- Is optional financial translation needed in this branch, or should it remain out of scope?
- Which therapeutic areas should be included first?
- Should the first MVP focus only on Phase II / Phase III assets?
- How should market potential be estimated initially?
- How should downstream Phase III commitment be estimated?
- Should next-phase cost use medians from similar completed trials, model-based estimates, or a hybrid?
- Should ongoing actual enrollment, site count, and country count be used only as current-extract lower bounds, or should future versions attempt historical timestamp reconstruction?
- Should terminated trials be used only for incurred-cost calibration, or also for partial-spend curve validation?
- Should historical estimation use current-extract approximations or later reconstructed AACT snapshots?
- What tolerance is acceptable if exact observed-to-date patient/site/country values are unavailable for the missing-value review date?
- What default uncertainty range policy should be used for the first version?
- Should estimates be compared with a model recommendation, grouped median, or both?
- Should the estimation workflow include maintainer-adjustable assumptions?

## 23. Non-goals for now

The following should not be done yet:

- No implementation.
- No UI redesign.
- No new model training yet.
- No database schema changes yet.
- No cost claims presented as real observed sponsor costs.
- No automatic recommendation engine yet.
- No complex commercial valuation yet.
- No multiplayer logic.
- No authentication or user management planning unless separately requested.
- No direct black-box prediction of total sponsor cost in the MVP.
- No use of ongoing actual patient, site, or country counts as if they were final totals.
- No user-facing use of final observed outcomes in a historical scenario unless they were known at the estimation date.

## 24. Instructions for future coding assistants

Gemini CLI, Codex CLI, or another assistant should use this document as the backbone for future estimation development.

Rules for future work:

- Treat this document as the backbone for future estimation development.
- Before implementing any estimation feature, update or reference the relevant section.
- Do not mix design-stage features with future-observed outcomes.
- Anchor estimation datasets to an estimation date when historical feature visibility matters.
- Classify every user-facing field as visible at estimation date, observed-to-date lower bound, final observed only, prediction feature, synthetic estimate, or excluded for leakage.
- Precompute reusable asset-level estimates where possible, then calculate date-dependent values only when required.
- Treat completed trials as the preferred source for final operational scale.
- Treat ongoing patient, site, country, and elapsed-duration values as lower bounds when estimating final scale.
- Treat terminated and withdrawn trials as partial-spend evidence, not clean examples of completed-trial scale.
- Treat future phases to market and time to market as synthetic scenario estimates unless an explicit observed successor path is being used.
- Use a simple non-linear lifecycle spend curve for incurred and saveable cost; do not assume linear spend by default.
- Keep cost assumptions transparent and configurable.
- Prefer incremental implementation.
- Preserve the existing CTPredict app architecture unless explicitly asked to refactor.
- Do not redesign the visual identity unless explicitly requested.
- When proposing code changes later, provide small, copy-paste-safe patches with filenames and exact locations.
- Maintain a distinction between existing functionality, planned MVP functionality, and future extensions.
