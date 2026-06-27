#!/usr/bin/env bash
set -euo pipefail

python scripts/check_trial_score_contract.py
python scripts/check_trial_score_no_obsolete_active_fields.py
python scripts/check_narrative_prompt_builder.py
python scripts/check_narrative_provider.py
python scripts/check_narrative_mock_reviewer.py
python scripts/check_narrative_review_store.py
python scripts/check_narrative_packet_builder.py
python scripts/check_completion_decomposition.py
python scripts/check_narrative_live_snapshot_flow.py
python scripts/check_scenario_review_failure_formatting.py
python scripts/check_scenario_review_diagnostics.py
python scripts/check_trial_score_visual_data.py
python -m py_compile \
  src/narratives/trial_score_contract.py \
  src/narratives/scoring.py \
  src/narratives/prompt_builder.py \
  src/narratives/provider.py \
  src/narratives/mock_reviewer.py \
  src/narratives/review_store.py \
  src/narratives/packet_builder.py \
  src/narratives/storyline.py \
  src/narratives/review_controls.py \
  src/scoring/decomposition.py \
  frontend/utils/scenario_review_diagnostics.py \
  frontend/utils/scenario_review_failure.py \
  frontend/utils/scenario_review_plot_data.py \
  frontend/views/trial_simulator.py \
  scripts/check_trial_score_contract.py \
  scripts/check_completion_decomposition.py \
  scripts/check_trial_score_no_obsolete_active_fields.py \
  scripts/check_scenario_review_diagnostics.py \
  scripts/check_trial_score_visual_data.py
git diff --check

echo "Validated Trial Score V1 migration checks."
