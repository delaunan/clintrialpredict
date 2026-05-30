# Edit Mode Live Prediction Architecture

This document started as the planning reference for turning the current read-only/audit demo into a simple, robust live simulation workflow. It now also records the implementation state of the isolated edit-mode variant so future sessions can resume without rediscovering the same decisions.

## Implementation Status - 2026-05-30

The first edit-mode implementation is in place on branch `edit_mode`.

Implemented:

- `frontend/app.py` routes `APP_VARIANT=edit_trial` to a new isolated view, `frontend/views/edit_trial.py`.
- `frontend/views/trial_audit.py` remains the stable deployed audit view and has not been modified for edit-mode UI work.
- `frontend/views/edit_trial.py` contains the simulation UI, including `Trial Features`, shared edited field state, live prediction workflow, mode-off reset handling, and simulation-only gauge/bar deltas.
- `api/main.py` keeps the precomputed SHAP audit path for normal mode and adds a `simulation_mode` live path using the production pipeline, `predict_proba`, and native XGBoost `pred_contribs=True`.
- The live simulation path normalizes submitted taxonomy labels/codes into model values, recomputes the TA calibration offset from edited `therapeutic_area_ml` / `therapeutic_area_ui`, and returns the existing chart response shape.
- `frontend/utils/plot.py` supports simulation-only pillar delta annotations in the impact bar.

Important fixes already made:

- Edited therapeutic area now takes priority over original row `therapeutic_area` in simulation scoring. This fixed the bug where the impact chart could move while the gauge score stayed unchanged after a TA edit.
- Normal/audit mode prediction now uses the original selected row instead of edited session state, so toggling simulation mode off should return to prerecorded audit values.
- Simulation-only KPIs disappear when edit mode is off.
- Trial Features widget state is reset on mode-off across all simulation fields, not only the original Trial Information fields.
- Raw boolean-like values such as `TRUE`, `FALSE`, `YES`, and `NO` are normalized back to intended UI labels before dropdown rendering.

Current Trial Features layout:

- Top left: `Therapeutic Context`
- Top right: `Patient Profile`
- Bottom left: `Scientific Challenge`
- Bottom right: `Execution Framework`
- The layout is visually finalized as of this session: four white rounded pillar cards, two fields per row by default, single-field rows left-aligned at half-card width, equal top-row card heights, equal bottom-row card heights, compact row-to-row spacing, and enlarged pillar icon/title headers with extra separation before the first field row.
- `number_of_arms_ml` remains an integer input. `primary_duration_months_ml` is displayed and edited with one decimal place and `0.1` increments.

Current Trial Features row structure:

- Therapeutic Context:
  - `therapeutic_area_ml`
  - `gbd_cause_id_3_ml`, `is_rare_disease_ml`
  - `phase_ml`, `strategic_ambition_ml`
- Patient Profile:
  - `healthy_volunteers_ml`, `gender_ml`
  - `adult_ml`, `child_ml`, `older_adult_ml`
  - `patient_severity_ml`, `line_of_therapy_ml`
- Scientific Challenge:
  - `target_precedent_ml`, `target_pathway_class_ml`
  - `therapeutic_modality_ml`, `innovation_tier_ml`
  - `primary_purpose_ml`, `intervention_model_ml`
  - `adaptive_design_ml`, `biomarker_stratification_ml`
  - `endpoint_rigor_ml`, `endpoint_structure_ml`
- Execution Framework:
  - `allocation_ml`, `masking_ml`
  - `comparator_benchmark_ml`, `has_placebo_ml`
  - `administration_complexity_ml`
  - `number_of_arms_ml`, `primary_duration_months_ml`
  - `has_dmc_ml`, `sponsor_tier_ml`

Current Trial Features label overrides:

- `primary_duration_months_ml`: `Max Primary Endpoint Duration (in months)`, with `(in months)` visually on the second label line.
- `has_dmc_ml`: `Data Monitoring Comittee`.

Verification completed:

- `python -m py_compile api/main.py frontend/views/edit_trial.py frontend/utils/plot.py`
- During the latest Trial Features layout refinement, `python -m py_compile frontend/views/edit_trial.py` was run repeatedly after each visual/control change.
- Earlier parity run: `python refresh_registry.py` and `python audit_parity.py`, with `4,423/4,423` audit parity.
- Sample unchanged simulation rows matched registry scores in earlier checks.
- Local API probes confirmed frontend-style TA edit changed score from `39.4` to `45.6` after the TA canonicalization fix.
- Local API and Streamlit health checks returned `200` on test ports.

Remaining work before deployment:

- Manual browser smoke test of behavior when users change Trial Features values, including queued/recomputed simulation prediction behavior and score/chart consistency.
- Confirm visually that normal/audit mode never carries edited values into gauge, treemap, or bar chart after toggling edit mode off.
- Confirm simulation-only gauge and bar deltas disappear in edit mode off.
- Re-run `python refresh_registry.py` and `python audit_parity.py` before deployment if any additional scoring/parity-sensitive edits are made.

## Current State

- `frontend/views/trial_audit.py` remains the audit/demo view. It should continue to be treated as the stable deployed variant.
- `frontend/views/edit_trial.py` is the current simulation/edit view. It owns the active edit-mode UI work.
- Clicking `Predict Trial Completion` works in simulation mode in the edit view and calls the live simulation path.
- The `/predict` API in `api/main.py` behaves as an audit endpoint by default and as a live simulation endpoint when payloads include `simulation_mode: true`.
- The production model is available in `models/model_prod_01.joblib` as a pipeline with `prep` plus XGBoost `clf`.
- In this context, `clf` means classifier: the trained XGBoost classifier step inside the saved pipeline.
- The model feature set is small and tabular: 27 transformed model inputs from `prep.get_feature_names_out()`, sourced from 31 UI-taxonomy fields once disabled/internal fields are considered.
- `models/taxonomy_01.json` is the correct UI source of truth for labels, pillars, priorities, options, encodings, and mappings.
- `frontend/data/search_registry.csv` is a precalculated display/search registry. It contains model feature columns, UI columns, precalculated `Clinical_Score`, and pillar scores.
- `data/data_clinpred.csv` contains the richer modelling dataset with 34,066 rows, GBD IDs/names, and model features.

## Goal

When simulation mode is enabled, the user should be able to change model-driving trial features, request a new live prediction, and compare the simulated score with the initial score while keeping the UI close to the existing visual language.

The intended user flow is:

1. Open a trial.
2. Toggle `Simulation Mode` on.
3. The existing `Completion Score` tab disappears until a new prediction is requested.
4. A new `Trial Features` tab appears.
5. `Trial Features` contains four white rounded boxes, one per pillar:
   - Therapeutic Context
   - Scientific Challenge
   - Patient Profile
   - Execution Framework
6. Each box contains editable controls for the model features in that pillar, using UI labels from `models/taxonomy_01.json`.
7. Changing a field updates the matching value anywhere else it appears in the other tabs.
8. Clicking `Predict Trial Completion` calls live prediction.
9. After prediction, `Completion Score` appears after `Trial Features`.
10. `Trial Features` remains visible after prediction.
11. When simulation mode is toggled off, `Trial Features` disappears and the app returns to the normal audit view.
12. After a simulated prediction, the gauge shows a small top-right score-change indicator such as `+4.0%` or `-3.2%`, computed as `(new_score / initial_score - 1) * 100`.

## Recommended Architecture

Protect the deployed demo by implementing simulation as a separate app variant, while sharing only the backend artifacts and API service.

`frontend/app.py` already routes by `APP_VARIANT`. The current demo should remain `APP_VARIANT=trial_audit`. The simulation version should be implemented as a new variant, tentatively `APP_VARIANT=edit_trial`, backed by a new view module:

- Keep `frontend/views/trial_audit.py` as the stable deployed demo view.
- Create `frontend/views/edit_trial.py` from the current audit view as the starting point for simulation-specific UI work.
- Add an `elif variant == "edit_trial"` branch in `frontend/app.py`.
- Prefer deploying a second Cloud Run UI service for the simulation link, using `APP_VARIANT=edit_trial`.
- Keep the current demo Cloud Run UI service on `APP_VARIANT=trial_audit`.

This gives the demo a clean safety boundary. Bugs in the simulation view should not affect the published audit demo.

Inside the simulation view, use one shared edited-row state and one live API path.

The frontend should not maintain separate values for the same feature in multiple tabs. Every editable widget should write to the existing `input_{nct_id}_{field_id}` convention, and every display panel should read through the same state-aware helpers. This keeps synchronization simple: when `phase_ml` changes in `Trial Features`, the phase field in `Trial Information` reads the same session-state value on rerun.

The backend should not depend on precomputed SHAP values when any feature can be changed. The `/predict` endpoint should support two modes:

- `audit`: existing behavior for unchanged known trials, using precomputed SHAP values if desired.
- `simulation`: live behavior for edited rows, using the model pipeline to recompute score and model contributions from the submitted feature values.

The simplest robust version is to always use the live path when the frontend sends `simulation_mode: true`.

Implementation decision: keep the existing precomputed/audit path in normal mode and use live scoring only for simulation mode. After unchanged-row live scores match `Clinical_Score` and latency is acceptable, the project can reconsider whether to unify all prediction requests behind the live path.

## Production Scoring Source

The production notebook, `notebooks/production_01.ipynb`, confirms that the existing explanation and scoring layer is reproducible for live simulation.

The relevant notebook flow is:

1. Save the trained model pipeline to `models/model_prod_01.joblib`.
2. Transform the full modelling universe with `model.named_steps["prep"].transform(...)`.
3. Compute SHAP values with `shap.TreeExplainer(model.named_steps["clf"])`.
4. Save per-trial SHAP vectors to `models/shap_values_01.joblib`.
5. Save TA-specific thresholds, gain factor, and model base value to `models/thresholds_01.json`.
6. Export UI/model metadata to `models/taxonomy_01.json`.
7. Convert SHAP values into UI scores and pillar/subcategory impacts.

The current demo API uses step 4: it looks up a stored SHAP vector by `nct_id`. This is valid for audit mode because the trial values are unchanged.

Live simulation cannot use `models/shap_values_01.joblib` after edits, because the edited row has no stored SHAP vector. Instead, it should recompute contributions from the trained XGBoost booster inside `models/model_prod_01.joblib`.

Validation finding: XGBoost native `pred_contribs=True` reproduced the saved notebook SHAP feature vectors exactly for sampled unchanged rows (`max_abs_diff_features = 0.0`). Therefore live simulation can use native XGBoost contribution output without adding or persisting a separate SHAP explainer artifact.

No model retraining is required. No model artifact should be modified for the first implementation.

Important terminology:

- `models/model_prod_01.joblib` contains the fitted preprocessor and trained XGBoost classifier. It does not contain saved per-trial SHAP vectors.
- `models/shap_values_01.joblib` contains saved per-trial SHAP vectors for existing unedited trials only.
- There are no separate "SHAP weights" to save or load. SHAP values are derived from the trained XGBoost trees.
- Audit mode uses saved answers from `shap_values_01.joblib`.
- Simulation mode asks the trained XGBoost model to calculate new TreeSHAP contributions for the edited row.

## Feature List

The `Trial Features` tab should be generated from `models/taxonomy_01.json`, not hard-coded field by field except for special control types.

Include all 31 taxonomy model-facing fields with a non-`Metadata` UI pillar. This intentionally includes fields that are not transformed into the current 27 XGBoost inputs, because they are still useful for scenario completeness, calibration, UI consistency, or future model expansion.

### Therapeutic Context

- `therapeutic_area_ml` - Therapeutic Area
- `gbd_cause_id_3_ml` - Indication
- `is_rare_disease_ml` - Rare Condition Status
- `phase_ml` - Clinical Phase
- `strategic_ambition_ml` - Regulatory Intent

### Scientific Challenge

- `target_precedent_ml` - Target Precedent Status
- `target_pathway_class_ml` - Pathway Profile
- `therapeutic_modality_ml` - Therapeutic Modality
- `innovation_tier_ml` - Innovation Rank
- `intervention_model_ml` - Intervention Model
- `primary_purpose_ml` - Primary Purpose
- `adaptive_design_ml` - Design Flexibility Level
- `endpoint_rigor_ml` - Primary Endpoint Type
- `endpoint_structure_ml` - Primary Endpoints Number
- `biomarker_stratification_ml` - Biomarker Patient Selection

### Patient Profile

- `patient_severity_ml` - Patient Severity
- `line_of_therapy_ml` - Line of Therapy
- `gender_ml` - Patient Gender Eligibility Status
- `healthy_volunteers_ml` - Population Type
- `adult_ml` - Adult Profile Eligibility Status
- `child_ml` - Pediatric Profile Eligibility Status
- `older_adult_ml` - Geriatric Profile Eligibility Status

### Execution Framework

- `masking_ml` - Bias Control
- `allocation_ml` - Allocation Method
- `has_dmc_ml` - Data Monitoring Comittee
- `has_placebo_ml` - Placebo Control
- `comparator_benchmark_ml` - Benchmark Comparator
- `administration_complexity_ml` - Delivery Profile
- `number_of_arms_ml` - Number of Arms
- `sponsor_tier_ml` - Sponsor Type
- `primary_duration_months_ml` - Max Primary Endpoint Duration (in months)

## GBD Indication Selection

`gbd_cause_id_3_ml` has no static options in `taxonomy_01.json`, so it needs a dynamic option source.

Recommended source for the first implementation:

- Build options from `data/data_clinpred.csv` or from a small generated lookup derived from it.
- Use distinct observed `(therapeutic_area, gbd_cause_id_3, gbd_indication_name_3)` combinations.
- Display as `"{gbd_indication_name_3} ({gbd_cause_id_3})"` and store `gbd_cause_id_3_ml`.
- Filter options by the currently selected `therapeutic_area_ml`.
- Always include the current trial's indication even if it is outside the selected TA, to avoid losing the existing value.
- Always include an `Other / Unclassified` option mapped to `0` if present, or as a local fallback.

Important data finding: `data/data_clinpred.csv` has 155 distinct L3 indications but 325 observed TA+L3 pairs. Sixty-seven L3 IDs appear under more than one therapeutic area. Therefore the UI menu should be TA-filtered using the observed pair, not assume a globally unique TA for each L3 ID.

When a user changes therapeutic area:

- The indication menu should rerender with the new TA's observed indications.
- If the current indication is not valid for the new TA, keep it as a temporary first option.
- Do not automatically reset indication to `Other / Unclassified`; the user should choose the new indication explicitly.

When a user changes indication:

- Set `gbd_cause_id_3_ml`.
- Set `gbd_indication_name_3` for display and API explanation labels.
- Do not automatically change therapeutic area in the first implementation, because the same L3 can appear in multiple TAs and the user has already selected the TA filter.

## Backend Live Prediction

The live prediction path should:

1. Receive the edited row payload.
2. Extract the exact model input columns expected by the pipeline.
3. Normalize UI labels/codes into ML values using `taxonomy_01.json`.
4. Run `model.predict_proba(input_df)`.
5. Convert probability/logit to the existing calibrated 1-99 score scale using `models/thresholds_01.json`, preserving the current parity formula.
6. Compute live feature contributions.
7. Aggregate contributions to the same pillar/subcategory structures currently returned by `/predict`.
8. Return the same response shape used by `render_completion_prediction_tab`.

For live feature contributions, use XGBoost native contribution output:

- Transform the row with `model.named_steps["prep"].transform(input_df)`.
- Use `model.named_steps["clf"].get_booster().predict(..., pred_contribs=True)`.
- Treat all columns except the final contribution column as feature SHAP values.
- Treat the final contribution column as the live bias/base term.
- Map transformed feature names from `prep.get_feature_names_out()` back through `taxonomy_01.json`.
- Apply the same gain factor and calibration offset as the existing API.

This keeps the implementation small, avoids adding a new dependency, and makes explanations reflect the edited values instead of stale precomputed SHAP vectors.

The scoring math should mirror `notebooks/production_01.ipynb`, `refresh_registry.py`, and the current API:

```text
feature_impact_points = -shap_value * gain_factor
calibration_offset_points = (ta_threshold_logit - base_value) * gain_factor
Clinical_Score = clip(round(50 + sum(rounded pillar impacts), 1), 1, 99)
```

Then absorb any clipping/rounding residual into `Therapeutic Context` / `Therapeutic Area Profile` so that:

```text
Clinical_Score == 50.0 + sum(pillar_impacts)
```

This parity rule is what keeps the gauge, impact bar, and treemap mathematically aligned.

## Therapeutic Area Adjustment

The therapeutic-area adjustment is not learned inside the XGBoost model. It is a post-model calibration layer applied during score construction.

The adjustment source is `models/thresholds_01.json`:

- `ta_threshold_logits`: TA-specific decision thresholds in logit space.
- `global_threshold_logit`: fallback threshold for sparse or missing TAs.
- `gain_factor`: currently `25.0`.
- `base_value`: the model SHAP expected value / intercept saved from the production notebook.

Audit mode currently applies the adjustment in `api/main.py`:

```text
ta = original therapeutic_area
threshold_logit = ta_threshold_logits.get(ta, global_threshold_logit)
calibration_offset_points = (threshold_logit - base_value) * gain_factor
```

That offset is added to `Therapeutic Context` / `Therapeutic Area Profile`, then flows into the subcategory impacts, pillar impacts, and final `Clinical_Score`.

Simulation mode must apply the same formula, but using the edited therapeutic area:

```text
edited therapeutic_area_ml -> canonical TA code
canonical TA code -> ta_threshold_logits
calibration_offset_points = (edited_ta_threshold_logit - base_value) * gain_factor
```

Therefore changing `Therapeutic Area` in `Trial Features` can change the score through two mechanisms:

- direct model/preprocessor inputs where applicable,
- the post-model TA-specific calibration offset.

The first implementation should keep the calibration offset anchored in `Therapeutic Context` / `Therapeutic Area Profile`, exactly as audit mode does.

## Live Encoding Contract

Live simulation has two encoding layers.

Frontend/UI normalization:

- Widgets display user-friendly labels from `models/taxonomy_01.json`.
- Edited values should be stored with stable field IDs such as `phase_ml`, `gbd_cause_id_3_ml`, and `sponsor_tier_ml`.
- For fields with taxonomy `ui.options`, the frontend should know both the option key and label.
- For GBD indication, the displayed label should be `"{gbd_indication_name_3} ({gbd_cause_id_3})"` while the stored model value remains the numeric `gbd_cause_id_3_ml`.

Backend normalization:

- The simulation API path must defensively normalize submitted values before prediction.
- `therapeutic_area_ml` must also produce the canonical TA code used by `thresholds_01.json`, such as `ONCOLOGY`, `CARDIOVASCULAR`, or `UNCLASSIFIED`.
- If the frontend sends a UI label, map it to the taxonomy option key.
- If the frontend sends an option key, map it through the taxonomy `mapping`.
- If the frontend sends an already encoded numeric value, keep it if valid.
- Numeric fields should be coerced with `pd.to_numeric`.
- Target fields such as `gbd_cause_id_3_ml` should remain numeric and are then target-encoded by the fitted preprocessor.

Model preprocessing:

- The fitted preprocessor in `models/model_prod_01.joblib` handles the final model transformations.
- Ordinal fields pass through after registry imputation.
- `gbd_cause_id_3_ml` is target-encoded by the fitted `TargetEncoder`.
- `number_of_arms_ml` and `primary_duration_months_ml` are imputed/scaled.
- Dropped/calibration fields remain useful in the UI, but only transformed features returned by `prep.get_feature_names_out()` receive direct XGBoost contribution values.
- `therapeutic_area_ml` is one of the important dropped/calibration fields: it may not receive a direct XGBoost feature contribution, but it drives the TA threshold offset.

This means all UI names must be tied back to their taxonomy field key and encoding method before live prediction.

## Frontend State Model

Recommended session-state additions:

- `simulation_prediction_result`: latest live prediction result for selected trial.
- `simulation_prediction_nct_id`: selected trial associated with the result.
- `simulation_initial_score`: original `Clinical_Score` for selected trial.
- `simulation_last_score`: latest predicted score.
- `simulation_has_edits`: whether any editable feature differs from its initial value.
- `detail_completion_tab_visible`: keep existing flag, but make it true after successful simulation prediction.

Existing state to keep:

- `global_edit_mode`
- `trigger_prediction`
- `analysis_result`
- `analysis_nct_id`
- `completion_score_tab_jump_nonce`
- `input_{nct_id}_{field_id}` widget keys

Resolved behavior decisions:

- When therapeutic area changes, keep the current indication until the user chooses another indication.
- `Trial Features` should include all 31 taxonomy model-facing fields.
- Sponsor name should remain editable only in `Trial Information`.
- `Trial Features` should expose `sponsor_tier_ml` / Sponsor Type, not sponsor name.
- Large text fields can remain editable for now, but the current simulation workflow should guide users to change model inputs from `Trial Features`.
- LLM-derived dropdown values should not be recomputed from edited text in this phase. That can be a later capability after direct dropdown editing and live reprediction are stable.
- Normal mode keeps the existing precomputed/audit prediction path.
- Simulation mode uses live scoring.

Recommended reset behavior:

- On selected trial change: clear prediction result and reinitialize editor values.
- On simulation mode on: clear completion-score visibility and prediction result, show `Trial Features`.
- On simulation mode off: clear simulation-only state, hide `Trial Features`, restore normal tab layout.
- On first feature edit: keep `Trial Features` visible and mark `simulation_has_edits`.
- On successful prediction: show both `Trial Features` and `Completion Score`.

## Tab Behavior

Normal mode:

- `Trial Information`
- `Population Details`
- `Completion Score` only if already visible under the existing audit workflow

Simulation mode before prediction:

- `Trial Information`
- `Population Details`
- `Trial Features`

Simulation mode after prediction:

- `Trial Information`
- `Population Details`
- `Trial Features`
- `Completion Score`

The `Predict Trial Completion` button should no longer be blocked by simulation mode. Instead, when simulation mode is on, it should submit the edited row and set completion-score visibility after success.

## Gauge Delta Indicator

Compute only after a simulated prediction exists:

```text
delta_pct = (new_score / initial_score - 1) * 100
```

Display:

- one decimal place
- explicit sign
- blue font for positive
- red font for negative
- no badge or heavy visual treatment
- place at the top-right of the gauge card

Edge case:

- If `initial_score` is missing, zero, or invalid, hide the delta.

## Implementation Phases And Status

### Phase 0 - Variant isolation - Implemented

- Duplicate the current audit view into `frontend/views/edit_trial.py`.
- Add `APP_VARIANT=edit_trial` routing in `frontend/app.py`.
- Keep `frontend/views/trial_audit.py` unchanged for the deployed demo.
- Verify `APP_VARIANT=trial_audit` still renders the current demo. Manual visual verification still recommended after latest edits.
- Verify `APP_VARIANT=edit_trial` initially renders the copied view before simulation edits begin. Superseded by the current edit-mode implementation.

### Phase 1 - Frontend contract and tab skeleton - Implemented

- Generate editable feature groups from `taxonomy_01.json`.
- Add `Trial Features` tab only in simulation mode.
- Use the same white rounded box style as existing detail panels. Current implementation uses four rounded white panels with compact, left-aligned controls.
- Render static taxonomy options and numeric inputs.
- Keep all widgets on the existing shared `input_{nct_id}_{field_id}` key pattern.

### Phase 2 - GBD indication dropdown - Implemented

- Build a cached TA-filtered indication lookup.
- Add the indication dropdown under `Therapeutic Context`.
- Synchronize selected indication ID and display name into the edited row.
- Preserve current indication as a safe option.

### Phase 3 - Live backend scoring - Implemented

- Add simulation-mode prediction path in `api/main.py`.
- Keep the existing precomputed SHAP lookup path for normal/audit mode.
- Recompute model probability and calibrated score from the submitted row.
- Recompute live contributions with XGBoost native contribution output.
- Recompute the TA-specific calibration offset from the edited therapeutic area. Current implementation prioritizes edited `therapeutic_area_ml` / `therapeutic_area_ui` over original `therapeutic_area`.
- Keep response shape compatible with the current frontend charts.

### Phase 4 - Prediction workflow - Implemented

- Let `Predict Trial Completion` work while simulation mode is on.
- After success, show `Completion Score` after `Trial Features`.
- Keep `Trial Features` visible.
- Add score delta on the gauge.
- Add simulation-only pillar delta annotations in the impact bar.
- If Completion Score is already visible, changing Trial Features queues a fresh simulation prediction for the next Completion Score render.

### Phase 5 - Verification - Partially complete

- Verify current demo behavior under `APP_VARIANT=trial_audit`.
- Verify simulation behavior under `APP_VARIANT=edit_trial`.
- For an unchanged row, compare live API score against `Clinical_Score` from `frontend/data/search_registry.csv`.
- For sampled unchanged rows, compare native live contribution vectors against `models/shap_values_01.joblib` if needed.
- Verify that changing only therapeutic area updates the TA calibration offset and that the offset appears under `Therapeutic Context` / `Therapeutic Area Profile`.
- Confirm edited values are reflected in all tabs after rerun.
- Confirm TA-filtered GBD options include expected values and `Other / Unclassified`.
- Confirm simulation-mode off resets the tab layout.
- Run `python refresh_registry.py` and `python audit_parity.py` before deployment if scoring code or parity-sensitive paths are changed.

Completed verification includes py_compile, local API probes, local health checks, and an earlier full audit parity run. Remaining verification is primarily manual browser smoke testing of the latest visual layout and mode-off reset behavior.

## Open Questions

No open architecture questions remain for the first implementation pass.
