# Estimation V1 Completion Note

## Status

Estimation v1 is complete for the deterministic operational-assumption layer.

Active operational assumptions:

- `planned_enrollment`
- `planned_sites`
- `planned_duration_months`

These assumptions are active in Simulation Mode and remain outside `/predict`, XGBoost, SHAP, Completion Score, pillar impacts, impact bar, treemap, therapeutic-area calibration, API payloads, model artifacts, and taxonomy artifacts.

## Source Of Truth

- `docs/architecture_estimation.md`: final estimation architecture and implementation contract.
- `docs/architecture_narratives.md`: future narrative / Coherence consumption contract.
- `notebooks/estimation.ipynb`: explanatory notebook for operational benchmark construction and checks.
- `notebooks/validation_clinpred.ipynb`: validation workflow aligned to the one-decimal endpoint-duration rule.
- `notebooks/production_01.ipynb`: production workflow aligned to the one-decimal endpoint-duration rule and final exported-registry parity trigger.

## Implemented Runtime

- `scripts/build_operational_benchmarks.py`: benchmark artifact builder.
- `scripts/check_operational_benchmarks.py`: benchmark checker.
- `src/operational_benchmarks.py`: runtime lookup/defaulting/classification utility.
- `frontend/data/operational_benchmarks_v1.csv`: compact runtime artifact.
- `frontend/data/operational_benchmarks_v1_report.json`: build report.
- `frontend/data/operational_benchmarks_v1.xlsx`: analyst inspection export.

## Endpoint-Duration Precision

The existing model-facing maximum primary endpoint duration feature, `primary_duration_months_ml`, is rounded to one decimal before model training/scoring inputs are built.

Implemented in:

- `src/prep/data_loader_clinpred.py`: rounds `primary_duration_months_ml` after numeric coercion when `data_clinpred.csv` is rebuilt.
- `src/prep/pipeline.py`: rounds only `NUM_DURATION_COL = ['primary_duration_months_ml']` inside the `num_duration` preprocessing pipeline before imputation/scaling.

This rule is separate from operational `planned_duration_months`.

## Final Validation Facts

- Rounded validation smoke result: `test_auc = 0.7796800751`, `test_acc_0_5 = 0.6999132697`.
- Final production notebook exported-registry parity check reported `5,890 / 5,890` rows with perfect gauge-to-pillar parity.
- Production parity rule: `Clinical_Score == 50.0 + Therapeutic Context + Scientific Challenge + Execution Framework + Patient Profile`.

## Cross-Branch Propagation

The following files were copied to `edit-trial`, `master`, `narratives`, and `trial-audit`:

- `notebooks/production_01.ipynb`
- `notebooks/validation_clinpred.ipynb`
- `src/prep/data_loader_clinpred.py`
- `src/prep/pipeline.py`

Commits:

- `edit-trial`: `6ea023c Round endpoint duration before training`
- `master`: `b6ed571 Round endpoint duration before training`
- `narratives`: `486336d Round endpoint duration before training`
- `trial-audit`: `5f40a13 Round endpoint duration before training`

## Future Work Not In V1

- LLM narrative generation.
- Coherence Score.
- Adjusted Trial Value Score.
- Planned Country Count.
- Cost, market potential, spend curve, and downstream commitment logic.
- Full operational-scale estimation.
