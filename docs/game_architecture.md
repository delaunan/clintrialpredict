# CTPredict Serious-Game Architecture

## 1. Purpose of this document

This document is the planning backbone for the future CTPredict serious-game mode. It defines the strategic and technical architecture for turning the current trial prediction product into a portfolio decision simulation.

This is not an implementation specification yet. It does not prescribe immediate code edits, model training, database changes, or UI changes. It should guide future discussions with Gemini CLI, Codex CLI, or another coding assistant before implementation begins.

## 2. Product vision

The serious-game ambition is to simulate pharma portfolio decision-making under capital constraints. Participants should review a set of clinical development assets, compare risk, cost, operational burden, sunk cost, future commitments, and market potential, then decide which assets deserve continued investment.

The experience should feel like a portfolio committee review, not a prediction dashboard. The core question is not only "Which trial is risky?" but "Given limited capital, which assets should receive investment, and which should be stopped?"

## 3. Current foundation

CTPredict currently provides:

- An AACT / ClinicalTrials.gov-based trial database.
- A focus on industry-led Phase II / Phase III clinical trials.
- An XGBoost completion / early-termination prediction engine.
- A Streamlit application for trial search, trial detail review, and completion-risk scoring.
- A scoring architecture aligned with the existing API, registry, and parity audit workflow.
- A future product direction toward serious-game and strategic forecasting modes.

## 4. Target serious-game scenario

The future scenario should place participants in a pharma portfolio review. They receive a portfolio of molecules or clinical assets at different development stages. They cannot continue funding everything, so they must allocate scarce capital across competing programs.

Example participant roles:

- Portfolio committee member.
- Business development analyst.
- Clinical operations lead.
- Finance partner.
- Strategy lead.

Example decisions:

- Continue.
- Terminate.
- Pause.
- Accelerate.
- Partner / out-license.
- Defer until more evidence is available.

Example constraints:

- Limited annual budget.
- Required cost reduction.
- Maximum acceptable risk exposure.
- Need to preserve future commercial value.
- Strategic therapeutic-area priorities.

## 5. Main architecture overview

| Module | Purpose | Input | Output | Implementation stage |
| ------ | ------- | ----- | ------ | -------------------- |
| Trial data layer | Provide structured trial records from AACT / ClinicalTrials.gov and existing registry artifacts. | Raw and processed trial data, sponsor data, design data, eligibility data, intervention data. | Trial-level records for search, scoring, and future simulation. | Existing |
| Design-stage feature layer | Represent information available at or near trial start. | Trial design, phase, sponsor, therapeutic area, condition, intervention, eligibility, planned dates. | Leakage-safe feature set for prediction and simulation. | Existing / Planned MVP |
| Completion-risk engine | Estimate completion or early-termination risk using the current model. | Design-stage model features from the existing pipeline. | Completion-risk score, probability, and explanation components. | Existing |
| Operational scale estimation engine | Estimate operational quantities needed by finance and portfolio simulation. | Design-stage features, historical completed trials, active-trial censoring logic. | Predicted duration, enrollment, site count, country count, recruitment burden. | Planned MVP |
| Cost simulation engine | Convert operational estimates into transparent cost estimates. | Operational quantities, phase, therapeutic area, complexity assumptions, cost parameters. | Total trial cost, remaining cost, avoided cost, cost drivers. | Planned MVP |
| Calendar-year spend engine | Spread total cost over time for budget-constrained decision-making. | Trial start date, expected end date, simulation date, cost curve assumptions. | Spent-to-date, remaining spend, annual spend by calendar year. | Planned MVP |
| Downstream development engine | Estimate future investment implied by continuing an asset. | Current phase, asset stage, expected next phase, phase-specific cost assumptions. | Downstream phase commitment and probability-weighted future investment. | Planned later |
| Market potential engine | Estimate or categorize commercial and strategic value. | Therapeutic area, indication, unmet need proxy, market category, strategic priority. | Market potential category and future value signal. | Planned MVP / Planned later |
| Portfolio decision engine | Represent participant choices and enforce scenario constraints. | Asset list, budgets, decisions, risk and value measures. | Portfolio outcome, budget usage, asset-level decisions. | Planned MVP |
| Serious-game scoring engine | Evaluate participant decisions against scenario goals. | Decisions, budget constraints, risk, cost, value, scenario rules. | Scores for budget discipline, value preservation, risk balance, and rationale quality. | Planned later |
| Debrief and explanation layer | Explain outcomes and compare decisions with model-based or scenario-based guidance. | Portfolio outcomes, model outputs, assumptions, participant decisions. | Final debrief, decision summary, facilitator outputs. | Planned later |

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

Synthetic financial data is created by the future cost engine. It is a simulation layer, not observed sponsor truth.

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
| Duration | `start_date`, `completion_date`, `primary_completion_date`, `primary_duration_months`, `is_duration_unknown` | Completed trials can define final duration; ongoing trials may contain planned dates but require uncertainty handling. |
| Sites | `number_of_facilities` | Final-ish for completed trials; observed-to-date lower bound for ongoing trials. |
| Countries | Reconstruct from `data/countries.txt` by `nct_id`; `includes_us` is already in `data_clinpred.csv` | Country count is not directly in `data_clinpred.csv`; ongoing country lists are observed-to-date lower bounds. |
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

This separation is critical because the serious-game mode will ask users to make decisions at a specific point in time. Inputs available after that point must not be treated as if they were available to the decision-maker.

Rule: Any future model must document whether each input feature is available at design stage, observed after trial execution, or synthetically estimated.

This rule applies to completion-risk modelling, operational scale estimation, cost simulation, market potential estimation, downstream commitment estimation, and serious-game scoring.

## 8. Simulation decision-date data contract

The serious-game mode should be anchored to a simulation decision date. This is the date at which participants are assumed to review the portfolio. A trial may later be completed, terminated, withdrawn, or still ongoing in the real dataset, but the participant-facing simulation should treat the selected assets as ongoing and actionable at the committee date.

For the first serious-game version, the objective is not to reconstruct a perfect historical AACT snapshot. The objective is to create a roughly realistic portfolio discussion. Final completion or termination outcomes should not be shown to the participant, because the scenario assumes the assets are still under review. The existing XGBoost risk module is already based on design-stage features and can provide the near-term completion / termination risk signal without relying on enrollment shortcuts.

### 8.1 Scenario date concepts

| Concept | Meaning | Use |
| ------- | ------- | --- |
| Simulation decision date | Date at which the portfolio committee makes decisions. | Defines what is visible, incurred, remaining, and saveable. |
| Trial start date | Date trial activity begins. | Determines eligibility and elapsed duration. |
| Known or planned completion date as of decision date | Completion expectation visible at that decision point if available. | Input to duration estimate, subject to uncertainty. |
| Final observed completion date | Completion date visible only after trial execution. | Training target or validation data, not participant-facing for historical scenarios. |
| Current extracted status | Trial status in the current source extract. | Useful for data audit, but not necessarily the scenario status at a historical decision date. |
| Scenario status | Scenario assumption that the asset is ongoing and actionable at the decision date. | Keeps the game focused on portfolio decisions rather than retrospective outcome knowledge. |
| Expected completion date | Estimated or planned end date used for spend phasing. | Determines remaining current-trial duration and saveable cost. |
| Expected time to market | Estimated remaining research and development time after the decision date. | Supports strategic value, opportunity cost, and sales-timing discussion. |

### 8.2 Trial eligibility for a portfolio scenario

For a trial to appear in a portfolio scenario, it should normally satisfy:

- The trial start date is on or before the simulation decision date.
- The selected portfolio committee date falls within a plausible active development window for the asset.
- The asset is treated as strategically actionable at the decision date.
- The phase and therapeutic context are available.
- The expected completion date is after the portfolio committee date, or can be estimated as after that date.

The number of selected trials does not need to be fixed at architecture time. A facilitator may hand-pick a small or larger portfolio later. The same data contract should work whether the portfolio contains 10, 15, 30, or another number of assets.

Future implementation may support fictionalized assets or manually curated portfolios. In that case, the same data contract still applies: every participant-facing field should be either known for the scenario or explicitly marked as an estimate.

### 8.3 Field-use contract

Each candidate field should be classified before use.

| Field class | Description | Example | Allowed participant-facing use |
| ----------- | ----------- | ------- | ------------------------------ |
| Directly visible at decision date | Known to the decision-maker at the scenario date. | Phase, sponsor, therapeutic area, trial title, planned enrollment if available. | Yes. |
| Observed-to-date lower bound | Current operational value visible by the scenario date but not necessarily final. | Patients enrolled to date, active/listed sites, listed countries, elapsed duration. | Yes, as lower bound and incurred-cost input. |
| Final observed only | Known only after trial execution. | Final actual enrollment, final duration, final site count, final country count, final outcome. | No, except for training, validation, and retrospective audit. |
| Prediction feature | Field allowed as input to an estimator. | Phase, TA, rare disease flag, modality, planned enrollment, number of arms. | Yes if available at decision date. |
| Synthetic estimate | Modelled or rule-based output. | Predicted final enrollment if continued, future Phase III cost, path-to-market commitment. | Yes, if labelled as estimate. |
| Excluded for leakage | Field that reveals future knowledge relative to the decision date. | Final completion outcome for an active historical trial. | No. |

### 8.4 Precomputation versus runtime calculation

Anchoring to a simulation decision date does not mean every value must be calculated interactively. Most asset-level estimates can be precomputed in advance. The date-dependent values are lightweight and can be calculated when a portfolio committee date is selected.

Precomputed asset-level estimates:

- Predicted final duration if continued.
- Predicted final enrollment if continued.
- Predicted final site count if continued.
- Predicted final country count if continued.
- Estimated total current-trial cost.
- Estimated future-phase cost to reach market.
- Estimated remaining time to market.
- Completion-risk score from the existing XGBoost engine.
- Market potential category.

Runtime date-dependent calculations:

- Elapsed duration at the portfolio committee date.
- Cost incurred by the portfolio committee date.
- Remaining current-trial cost after the date.
- Saveable cost this year.
- Saveable cost next year.
- Saveable cost in following years.
- Whether the selected date is plausible for the selected asset.

For a portfolio of hand-picked trials, these runtime calculations should be computationally light. They are mainly date arithmetic plus a deterministic spend curve applied to precomputed trial-cost and duration estimates.

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
9. At scenario time, apply the selected portfolio committee date to calculate incurred, remaining, and saveable cost.

This sequence should be completed before any serious-game UI work beyond a minimal prototype.

## 9. Operational scale estimation plan

Future models should estimate the operational quantities required by the cost engine. These estimates should be separated from the current completion-risk model so that cost assumptions remain interpretable and modular.

Completed trials should be the preferred training base for final operational scale. Active, recruiting, not-yet-recruiting, enrolling-by-invitation, and active-not-recruiting trials require careful treatment because duration, enrollment, site count, and country count may be censored, planned, revised, or incomplete.

Terminated and withdrawn trials are useful for modelling partial spend and termination risk, but they should not be treated as clean examples of the final scale a trial would have reached if completed.

### 9.1 Operational value types

Future datasets should distinguish three operational values per trial and per quantity:

| Value type | Meaning | Main use |
| ---------- | ------- | -------- |
| Observed-to-date | Value currently visible in AACT / ClinicalTrials.gov at the simulation decision date. | Lower bound for ongoing trials; partial-spend input. |
| Predicted final if continued | Estimated final operational scale if the asset continues. | Main driver of total cost, remaining cost, and future commitments. |
| Realized if stopped | Partial scale and spend at termination or scenario decision date. | Incurred cost and avoided-cost calculation. |

For ongoing trials, observed patient, site, and country values should constrain the prediction but should not be assumed to be the final total. If an ongoing trial has already reported 120 patients, 20 sites, or 4 countries, the predicted final value should not be lower than those observed-to-date values.

```text
predicted_final_if_continued = max(model_prediction, observed_to_date)
```

The same principle applies to duration:

```text
predicted_final_duration = max(model_prediction, elapsed_duration_to_decision_date)
```

For the first version, exact historical observed-to-date patient, site, and country values may not be available for a past portfolio committee date. In that case, use the current extract as a pragmatic lower-bound approximation where appropriate, clearly label it as an approximation, and keep the participant-facing scenario focused on discussion-quality decisions rather than forensic reconstruction.

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
| Final duration model | Final duration in months for completed trials. | Drives monthly management cost, calendar spend, and time-to-next-decision. | Phase, therapeutic area, intervention type, planned dates or duration when available, planned enrollment, number of arms, masking, allocation, endpoint proxies. | High |
| Actual enrollment model | Final actual enrolled patients for completed trials. | Drives patient-level cost and recruitment burden. | Planned or observed-to-date enrollment, phase, condition, sponsor type, therapeutic area, rare disease proxy, eligibility complexity. | High |
| Site count model | Final site count for completed trials. | Drives site startup, monitoring, and geography complexity. | Planned or observed-to-date sites, planned enrollment, phase, therapeutic area, sponsor tier, intervention model. | Medium |
| Country count model | Final country count for completed trials. | Drives country startup cost and operational complexity. | Reconstructed country count, includes-US flag, sponsor tier, trial size, therapeutic area, phase, planned enrollment. | Medium |
| Recruitment complexity proxy | Relative difficulty of recruiting and retaining patients. | Modifies patient cost, duration risk, and operational burden. | Rare disease proxy, eligibility restrictions, age eligibility, condition severity, intervention type, trial duration. | High |
| Downstream phase size estimate | Expected size of invented next phase. | Supports future commitment estimates beyond the current trial. | Current phase, next-phase template, therapeutic area, condition, historical similar completed trials, market category. | Later |

### 9.3 Model selection protocol

Each operational indicator should use the simplest model that is accurate enough and stable enough for cost simulation. The first notebook should compare a small set of candidates, then choose a champion per target.

| Target | Candidate models | Recommended first champion | Validation focus |
| ------ | ---------------- | -------------------------- | ---------------- |
| Final duration months | Phase/TA median baseline, `HistGradientBoostingRegressor`, `XGBRegressor` on `log1p(target)`. | `XGBRegressor` if it clearly improves over baselines; otherwise gradient boosting or median fallback. | MAE and RMSE on months after inverse transform; calibration by phase and TA; minimum duration sanity checks. |
| Final enrollment | Phase/TA median baseline, `HistGradientBoostingRegressor`, `XGBRegressor` on `log1p(target)`, optional quantile model later. | `XGBRegressor` on `log1p(enrollment)` for MVP if validation is stable. | Error on log scale and original scale; high-enrollment outlier handling; phase-level calibration. |
| Final site count | Phase/TA median baseline, `HistGradientBoostingRegressor`, `XGBRegressor` on `log1p(target)`. | Best validated gradient-boosting model, with median fallback for sparse groups. | Count plausibility, lower-bound enforcement, calibration by phase and multinational proxy. |
| Final country count | Phase/TA median baseline, `HistGradientBoostingRegressor`, `XGBRegressor` on `log1p(target)`. | Best validated gradient-boosting model after reconstructing country count from `data/countries.txt`. | Count plausibility, includes-US sanity check, calibration by phase and sponsor tier. |
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

### 9.4 Required and optional data sources

The primary source for the first notebook should be `data/data_clinpred.csv`.

Recommended supporting source already present:

- `data/countries.txt`: reconstruct country count by `nct_id`.

Potentially useful existing sources if the notebook needs deeper audits:

- `data/design_outcomes.txt`: outcome count or endpoint timing audit if existing derived fields are insufficient.
- `data/interventions.txt`: intervention-type audit if `therapeutic_modality` requires validation.
- `data/conditions.txt`: condition text audit if indication grouping needs review.

Files not currently present but useful only for later versions:

- Historical AACT snapshots by date, if exact observed-to-date reconstruction becomes important.
- External phase-transition / likelihood-of-approval benchmarks, if the game needs a true probability of reaching market rather than a scenario assumption.
- External clinical trial cost benchmarks, if assumptions need validation against published or licensed cost data.

The MVP does not require these external files. It can proceed with `data_clinpred.csv`, reconstructed country counts, transparent cost assumptions, and scenario-level future risk assumptions.

## 10. Cost simulation engine

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

The cost engine should use `predicted_final_if_continued` values for total cost and remaining commitment. It should use `observed-to-date` values and elapsed duration for incurred cost. This allows the game to estimate how much has already been spent, how much is committed if the trial continues, and how much can be saved by stopping or pausing.

The MVP should not assume linear spend. Current-trial cost should be allocated over the expected trial lifecycle using a simple spend curve. This is sufficient for discussion-quality portfolio simulation and avoids overcomplicating the UI.

Cost calculation should be deterministic once the operational estimates and assumptions are fixed. This makes the output reproducible in a notebook and explainable in a workshop.

Recommended first calculation order:

1. Build final-if-continued operational estimates.
2. Calculate estimated total current-trial cost.
3. Apply the non-linear lifecycle spend curve to split cost before and after the portfolio committee date.
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

### 10.1 Cost outputs for portfolio decisions

The first phase should produce a small set of decision-grade cost outputs:

| Output | Meaning | UI use |
| ------ | ------- | ------ |
| Estimated total current-trial cost | Full expected cost if the current trial is completed. | Asset detail and cost explanation. |
| Cost incurred to decision date | Estimated spend already consumed. | Sunk-cost context, not a reason to continue by itself. |
| Remaining committed cost | Estimated additional cost if the current trial continues. | Main portfolio budget pressure. |
| Saveable cost this year | Estimated cost that could be avoided in the current calendar year. | Budget-reduction exercise. |
| Saveable cost next year | Estimated cost that could be avoided in the next calendar year. | Medium-term budget planning. |
| Saveable cost in following years | Estimated later cost that could be avoided. | Long-horizon portfolio view. |
| Future development commitment | Estimated cost of invented future phases needed to reach market. | Strategic continuation burden. |
| Avoided future commitment | Future commitment avoided if the asset is terminated or out-licensed. | Portfolio-reduction debrief. |

The portfolio UI should show a simplified subset: cost incurred, remaining committed cost, saveable cost by year, future development commitment, completion risk, and market potential category.

## 11. Calendar-year spend model

The serious game needs costs over time, not only total cost. Participants should understand both the total obligation and the near-term budget impact of continuing an asset.

Key concepts:

- Simulation decision date.
- Trial start date.
- Expected end date.
- Elapsed duration.
- Spent-to-date.
- Remaining spend.
- Cost by calendar year.
- Cost avoided if terminated.

For ongoing trials, the calendar model should anchor on the simulation decision date. Costs before that date are incurred. Costs after that date are committed only if the participant continues, accelerates, or otherwise preserves the asset.

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

- Cost incurred before the portfolio committee date.
- Remaining committed cost after the portfolio committee date.
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

Continuing an asset may imply large future commitments beyond the currently visible trial. The game should make that future obligation visible without presenting it as observed truth.

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

Many assets in the source data will not have an explicit next-phase trial in the dataset. The serious game should therefore invent future phases as scenario estimates. These invented phases are not observed facts.

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

The output should be "future-phase cost to reach market", not only "next-phase cost". For a Phase II asset, this may mean the estimated Phase III commitment. For a Phase I/II asset, this may include a Phase II continuation and a later Phase III commitment. The architecture should also estimate remaining time to market so that participants understand not only how much investment is required, but how long capital remains tied up before a possible launch or sale.

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

The existing XGBoost score estimates trial completion or early-termination risk. It is not a full probability of regulatory approval or market launch. For the serious-game MVP, this score can be used as one risk signal and combined with simple phase-level assumptions for future development risk.

Example MVP framing:

```text
near_term_trial_risk = current XGBoost completion / termination score
future_development_risk = phase-level scenario assumption
portfolio_risk_signal = combination of near-term risk and future-stage assumption
```

This keeps the first version honest: the existing model informs operational continuation risk, while future market-reaching probability remains a transparent scenario assumption until a dedicated model exists.

The current dataset does not contain true regulatory approval or launch outcomes. Therefore, a real probability of reaching market cannot be learned directly from `data_clinpred.csv` alone. For the MVP, risk-to-market should be a scenario estimate based on:

- The current XGBoost trial completion / termination risk.
- Simple phase-level future risk assumptions.
- Optional therapeutic-area or rare-disease adjustment if agreed later.

If a later version needs validated market-reaching probabilities, it will require external phase-transition or likelihood-of-approval benchmark data.

## 13. Notebook implementation plan

The first implementation should be developed in `notebooks/serious_game.ipynb` as the analytical build notebook for the `simulation` module. It should follow the style of `notebooks/validation_clinpred.ipynb` and `notebooks/production_01.ipynb`: markdown explanation first, then focused code cells, with explicit audit checkpoints and reproducible outputs.

Recommended notebook structure:

| Notebook block | Purpose | Output |
| -------------- | ------- | ------ |
| `<REF:ENV_CONFIG>` | Configure autoreload, warnings, display settings, and reproducibility seeds. | Stable notebook environment. |
| `<REF:PATH_RESOLUTION>` | Resolve project root and data paths. | Portable paths to `data/`, `models/`, and future outputs. |
| `<REF:LIB_INIT>` | Import pandas, numpy, sklearn, xgboost if available, plotting, and metrics. | Shared notebook imports. |
| `<REF:DATA_LOAD>` | Load `data/data_clinpred.csv` and supporting `data/countries.txt`. | Base dataframe and country-count table. |
| `<REF:DATA_AUDIT>` | Audit schema, missingness, status counts, phase counts, and key cost-driver distributions. | Data-quality summary. |
| `<REF:FIELD_CONTRACT>` | Classify fields as direct, lower-bound, final target, feature, synthetic, or excluded. | Field contract table. |
| `<REF:TARGET_BUILD>` | Build completed-trial targets for duration, enrollment, sites, and countries. | Modelling dataset with target definitions. |
| `<REF:FEATURE_BUILD>` | Build leakage-safe features for operational models. | Feature matrix and preprocessing plan. |
| `<REF:MODEL_BENCHMARK>` | Compare baselines and candidate models per target. | Champion model selection per indicator. |
| `<REF:PREDICT_OPERATIONAL>` | Generate final-if-continued operational estimates for all candidate assets. | Predicted duration, enrollment, sites, countries. |
| `<REF:COST_ASSUMPTIONS>` | Define transparent cost assumptions and lifecycle spend weights. | Cost assumption table. |
| `<REF:COST_ENGINE>` | Calculate current-trial total, incurred, remaining, and saveable cost. | Current-trial cost table. |
| `<REF:FUTURE_PHASES>` | Create synthetic future path to market and estimate future-phase cost and time. | Future commitment table. |
| `<REF:RISK_TO_MARKET>` | Combine XGBoost trial risk with phase-level scenario assumptions. | Risk-to-market scenario fields. |
| `<REF:PORTFOLIO_EXPORT>` | Produce reusable asset-level table for serious-game scenarios. | Portfolio-ready dataset. |
| `<REF:VALIDATION_AUDIT>` | Validate ranges, subgroup stability, and cost plausibility. | Audit summary and known limitations. |

The notebook should be written as an analytical build notebook, not as a UI implementation. It should include markdown between code blocks explaining each decision, especially target definitions, model choices, assumptions, and limitations.

Until the notebook outputs are validated, future work should not wire these estimates into Streamlit. The notebook should first prove the data contract, target construction, model selection, cost assumptions, and output table.

## 14. Reproducible output contract

The first serious-game dataset should produce one row per trial or asset with a small set of stable columns. This table should be reusable by future Streamlit work without requiring the UI to rerun model training.

Recommended output groups:

- Identity: `nct_id`, title, sponsor, phase, therapeutic area, indication.
- Risk: existing XGBoost completion-risk score and simplified risk category.
- Operational estimates: predicted final duration, enrollment, site count, country count.
- Current-trial cost: estimated total cost, incurred cost, remaining cost, saveable cost by year.
- Future path: next required phase, future phases to market, future-phase cost to market, time to market.
- Market potential: initial category and later GBD / rare-disease value signals.
- Assumption metadata: cost-assumption version, model version, portfolio committee date if date-specific values are materialized.

The UI should consume this as a decision table. It should not expose training diagnostics, detailed model errors, or all raw cost-driver variables.

## 15. Market potential module

The market potential layer should estimate or categorize commercial and strategic value. It must remain separate from trial cost so users can distinguish operational affordability from potential portfolio value.

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
- Probability-adjusted portfolio contribution.

Any future valuation model should explicitly document assumptions and should not be mixed into the completion-risk model without a clear reason.

## 16. Portfolio decision engine

Participant decisions should be represented as explicit asset-level actions. Each decision should have budget, risk, value, and timing effects that are visible in the portfolio simulation.

Decision options:

- Continue.
- Terminate.
- Pause.
- Accelerate.
- Partner / out-license.

| Decision | Budget effect | Risk effect | Value effect | Timing effect |
| -------- | ------------- | ----------- | ------------ | ------------- |
| Continue | Commits remaining current-trial spend and may preserve downstream commitment. | Maintains exposure to completion and development risk. | Preserves full upside if the asset succeeds. | Keeps current timeline. |
| Terminate | Avoids remaining current-trial spend and possible downstream spend. | Removes future development risk for the asset. | Gives up potential future value. | Ends program immediately in the scenario. |
| Pause | Reduces or delays near-term spend. | May increase operational, competitive, or evidence-generation risk. | Preserves option value but may reduce strategic momentum. | Delays key milestones. |
| Accelerate | Increases near-term spend. | May reduce timing risk but can increase execution pressure. | Pulls forward potential value if successful. | Shortens expected timeline in the scenario. |
| Partner / out-license | Reduces internal spend and may share future costs. | Transfers or shares execution and commercial risk. | Preserves partial economics instead of full value. | May delay or stabilize development depending on partner assumptions. |

## 17. Serious-game scoring engine

The scoring engine should evaluate participant decisions across multiple dimensions. It should not simply reward selecting the lowest-risk trials. A good portfolio decision may preserve uncertain but strategically important assets while cutting high-cost, low-value programs.

| Score dimension | What it measures | Possible calculation | MVP or later |
| --------------- | ---------------- | -------------------- | ------------ |
| Budget discipline | Whether the participant meets the budget or cost-reduction target. | Remaining annual budget, total spend reduction, avoided cost. | MVP |
| Value preservation | Whether high-potential assets remain funded or strategically protected. | Retained market potential category, retained expected value, avoided cuts to transformational assets. | MVP |
| Risk balance | Whether the portfolio avoids excessive concentration in high-risk assets. | Weighted completion-risk exposure across continued assets. | MVP |
| Strategic coherence | Whether decisions align with stated therapeutic-area or business priorities. | Match between decisions and scenario priority weights. | Later |
| Quality of rationale | Whether participant explanations are consistent with risk, cost, and value evidence. | Facilitator assessment or structured rationale checklist. | Later |
| High-cost / low-value identification | Whether assets with weak value relative to cost are paused, partnered, or terminated. | Cost-value ratio, risk-adjusted value threshold, scenario rule comparison. | MVP |
| High-potential preservation under uncertainty | Whether the participant keeps strategically valuable assets despite moderate uncertainty. | Continued assets with high market potential and acceptable risk-cost profile. | Later |

## 18. User experience principles

Future serious-game UI work should follow these principles:

- The experience should feel like a portfolio review.
- Avoid overwhelming the participant with too many variables at once.
- Show cost, risk, and value together.
- Keep assumptions explainable.
- Allow drill-down from portfolio view to asset detail.
- Maintain the clean professional CTPredict visual identity.
- Preserve current app stability.
- Avoid major redesign during early planning.
- Add serious-game mode progressively.

The early product should extend the existing architecture rather than replace it.

## 19. Future screens

The following screens are possible future surfaces. They are not implemented yet.

| Screen | Purpose | Key information shown | MVP priority |
| ------ | ------- | --------------------- | ------------ |
| Serious Game Landing / Scenario Setup | Introduce the scenario and choose constraints. | Scenario name, participant role, budget target, asset set, time horizon. | High |
| Portfolio Dashboard | Provide the main portfolio review surface. | Assets, stage, risk, cost, market potential, time to market, decision status, budget impact. | High |
| Asset Detail | Support drill-down on a single asset. | Trial details, completion-risk score, operational estimates, cost drivers, market potential. | High |
| Cost Breakdown | Explain total, spent, remaining, and avoided cost. | Cost driver table, assumptions, calendar-year spend. | High |
| Decision Panel | Capture participant choices. | Continue, terminate, pause, accelerate, partner / out-license, rationale field. | High |
| Budget Tracker | Show whether the participant is meeting constraints. | Annual budget, spend committed, savings achieved, remaining gap. | High |
| Portfolio Impact View | Summarize the consequences of decisions. | Retained value, avoided cost, risk exposure, therapeutic-area mix. | Medium |
| Final Debrief | Explain outcomes after decisions are locked. | Score summary, asset decisions, model-based or scenario-based comparison, learning points. | Medium |
| Facilitator / Assumption Settings | Allow scenario owners to adjust assumptions. | Cost multipliers, market categories, budget targets, scoring weights. | Later |

## 20. MVP definition

The first achievable version should be small, transparent, and scenario-driven.

MVP should include:

- A small portfolio of selected assets.
- Existing completion-risk score.
- Predicted final duration if continued.
- Predicted final enrollment if continued.
- Predicted final sites and countries if continued.
- Lower-bound handling for ongoing observed patients, sites, countries, and elapsed duration.
- Synthetic total cost.
- Spent-to-date.
- Remaining cost.
- Saveable cost this year, next year, and following years.
- Simple annual spend.
- Synthetic future-phase commitment to market.
- Estimated remaining research duration and time to market.
- Simple market potential category.
- Continue / terminate decision.
- Budget reduction target.
- Final scoring summary.

Explicitly excluded from MVP:

- Full rNPV model.
- Real-world budget validation.
- Complex commercial forecasting.
- Complex multi-round game logic.
- Advanced facilitator controls.
- Automatic optimization engine.
- Real-time multiplayer features.

## 21. Future implementation phases

### Phase 0 - Architecture and planning only

No code. Build this document and refine assumptions.

### Phase 1 - Data audit

Identify available design-stage, observed-to-date, and final-observed variables. Confirm which variables are safe to use at each simulated decision point. Reconstruct country count from `data/countries.txt` if needed. Produce the simulation decision-date data contract before cost or UI implementation.

### Phase 2 - Operational estimation datasets

Create modelling datasets for duration, enrollment, sites, and countries. Train-target construction should prefer completed trials for final scale. Ongoing values should be retained as lower bounds. Terminated and withdrawn trials should be flagged as partial-spend evidence rather than clean final-scale targets.

### Phase 3 - First operational models

Train first models or rule-based estimators for missing operational quantities. Keep outputs interpretable enough for cost simulation and enforce lower-bound constraints for ongoing trials.

### Phase 4 - First cost engine

Create transparent assumption-driven cost calculations. Make assumptions configurable and clearly labelled as synthetic. Do not train a direct black-box cost model in the first phase.

### Phase 5 - Calendar spend engine

Convert total cost into incurred cost, remaining committed cost, saveable cost by calendar year, and annual spend. Use a simple non-linear lifecycle spend curve rather than assuming linear spend.

### Phase 6 - Portfolio simulation MVP

Create a simple portfolio decision interface with a facilitator-selected asset set, budget target, portfolio committee date, and continue / terminate actions. Keep the visible cost language focused on incurred, committed, saveable, future commitment, and time to market.

### Phase 7 - Serious-game scoring and debrief

Add scoring, decision summary, and facilitator-friendly outputs.

### Phase 8 - Market potential and downstream development

Add future value and downstream cost logic. Keep market potential separate from cost and completion risk. Treat next phases as synthetic scenario estimates unless explicitly linked to observed successor trials.

### Phase 9 - Refinement and validation

Validate assumptions with expert review and scenario testing. Refine scoring and user experience based on observed use.

## 22. Open questions

- What is the first target user: recruitment assessment center, pharma finance training, clinical operations training, or portfolio strategy workshop?
- Should the game use real historical trials, fictionalized assets, or a mixture?
- Should trial names and sponsor names be anonymized?
- What level of financial realism is required for the MVP?
- Which therapeutic areas should be included first?
- Should the first MVP focus only on Phase II / Phase III assets?
- How should market potential be estimated initially?
- How should downstream Phase III commitment be estimated?
- Should next-phase cost use medians from similar completed trials, model-based estimates, or a hybrid?
- Should ongoing actual enrollment, site count, and country count be used only as current-extract lower bounds, or should future versions attempt historical timestamp reconstruction?
- Should terminated trials be used only for incurred-cost calibration, or also for partial-spend curve validation?
- Should historical scenarios use current-extract approximations, manually curated scenario portfolios, or later reconstructed AACT snapshots?
- What tolerance is acceptable if exact observed-to-date patient/site/country values are unavailable for the portfolio committee date?
- What default non-linear spend curve should be used for the first version?
- Should participant decisions be compared with a model recommendation or only with scenario rules?
- Should the game include facilitator-adjustable assumptions?

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
- No participant-facing use of final observed outcomes in a historical scenario unless they were known at the simulation decision date.

## 24. Instructions for future coding assistants

Gemini CLI, Codex CLI, or another assistant should use this document as the backbone for future serious-game development.

Rules for future work:

- Treat this document as the backbone for future serious-game development.
- Before implementing any serious-game feature, update or reference the relevant section.
- Do not mix design-stage features with future-observed outcomes.
- Anchor serious-game datasets to a portfolio committee date before estimating incurred, committed, and saveable costs.
- Classify every participant-facing field as visible at decision date, observed-to-date lower bound, final observed only, prediction feature, synthetic estimate, or excluded for leakage.
- Precompute reusable asset-level estimates where possible, then calculate date-dependent spend at scenario time.
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
