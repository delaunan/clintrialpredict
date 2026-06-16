# Implementation Plan: Narrative Design Confidence Migration

> Superseded on 2026-06-16 by `docs/strategic_review_phase1.md`.
> This file is historical implementation provenance for the old
> `Design Confidence` / `Total Scenario Score` direction. Do not use it as the
> active implementation plan. The active score stack is
> `Completion Outlook + Strategic Review = Trial Score`, with provider output
> limited to Strategic Review classification/rationale and application-owned
> numeric scoring.

## Scope

Architecture scope: `architecture_narratives`

Primary architecture source: `docs/architecture_narratives.md`

Current branch: `trial-simulator`

Goal: finish migrating the serious-game narrative layer from the old `Quality Review` / `Quality Adjustment` / `Final Candidate Score` implementation to the active `Scenario Review` / `Design Confidence` / `Total Scenario Score` model.

This document intentionally resets the earlier long planning notebook. The useful design decisions have been consolidated into `docs/architecture_narratives.md`; this plan is the working execution tracker.

## Active Product Model

The simulator keeps the existing XGBoost `Completion Score` unchanged.

The narrative layer adds:

- `Completion Outlook`: explanation of model-derived Completion Score movement.
- `Scenario Review`: structured participant-facing review of trade-offs.
- `Design Confidence`: app-owned point adjustment derived from validated evidence-backed design subcategories.
- `Total Scenario Score`: optional combined score, calculated as `Completion Score + Design Confidence`, clamped to `0..100`.

The four Design Confidence subcategories are aligned to the existing four participant-facing Completion Outlook pillars:

| Pillar | Existing model-derived view | New design subcategory |
| --- | --- | --- |
| Therapeutic Context | Completion Outlook evidence | Phase & Intent Alignment |
| Scientific Challenge | Completion Outlook evidence | Endpoint & Evidence Strength |
| Patient Profile | Completion Outlook evidence | Target Population Alignment |
| Execution Framework | Completion Outlook evidence | Operational Burden Balance |

Scoring guardrails:

- XGBoost, SHAP, therapeutic-area calibration, audit parity, and `/predict` scoring stay unchanged.
- The provider must not return final app-owned scores.
- The app validates review JSON and calculates Design Confidence deterministically.
- Non-zero Design Confidence requires supported packet evidence.
- Each design subcategory now separates current design state from movement. `current_state` describes the current full scenario; `movement_direction`, `movement_materiality`, and `effect_role` drive app-owned points.
- Movement scoring uses a small bottom-up scale: `minor = 0.5`, `moderate = 1`, `major = 2`; unchanged movement scores `0`.
- `effect_role=confirming` halves the point effect to reduce double counting with Completion Outlook; `counterweight` and `independent` keep full movement weight; `unchanged` scores `0`.
- The app applies a proportional net Design Confidence cap from Completion Score movement and changed-field materiality. This preserves subcategory trade-offs while preventing a second-score jump such as `+12` on a flat Completion Outlook.
- Total Scenario Score is clamped to `0..100`.
- Operational assumptions remain outside XGBoost and feed only Scenario Review / Design Confidence.

## What Is Done

### Done: Architecture Target

Commits:

- `4dbcf4f` aligned `docs/architecture_narratives.md` and the previous plan with the active four-pillar Design Confidence model.

Completed decisions:

- Replaced the earlier three-pillar Quality Assessment direction with Design Confidence.
- Kept Completion Score as the sole XGBoost score.
- Defined Scenario Review as the narrative review layer.
- Defined app-owned Design Confidence and optional Total Scenario Score.
- Documented boundaries, non-goals, evidence gates, baseline rules, provider responsibilities, and UI principles.

Verification already run:

- `git diff --check`

### Done: V2 Contract Fixtures

Commits:

- `35b9732` migrated `src/narratives/contract_fixtures.py` to `narratives_v2` / `design_confidence_v1`.

Completed implementation:

- Added the required Design Confidence subcategories.
- Added fixture scenarios for baseline, score/design divergence, operational-only edits, text contradictions, population mismatch, phase/intent mismatch, modality/governance mismatch, no-op/minor text, and no-adjustment despite large Completion Outlook movement.
- Added expected Design Confidence and Total Scenario Score behavior to fixtures.
- Kept a temporary compatibility alias for old callers during migration.

Verification already run:

- `python scripts/check_narrative_contract_fixtures.py`
- `python -m py_compile src/narratives/contract_fixtures.py scripts/check_narrative_contract_fixtures.py`
- `git diff --check`

### Already Exists But Still Uses Old Schema

These components exist from earlier work, but still need migration from old `Quality Review` terms and old `quality_review_domains` schema:

- `src/narratives/packet_builder.py`
- `src/narratives/scoring.py`
- `src/narratives/mock_reviewer.py`
- `src/narratives/provider.py`
- `src/narratives/provider_config.py`
- `src/narratives/prompt_builder.py`
- `src/narratives/review_store.py`
- `frontend/views/trial_simulator.py`
- `scripts/check_narrative_packet_builder.py`
- `scripts/check_narrative_scoring.py`
- `scripts/check_narrative_mock_reviewer.py`
- `scripts/check_narrative_provider.py`
- `scripts/check_narrative_prompt_builder.py`
- `scripts/check_narrative_review_store.py`

Current symptom:

- Code and checks still reference `Quality Review`, `Quality Adjustment`, `Final Candidate Score`, and `quality_review_domains`.
- Fixtures now expose `design_confidence_subcategories`, so downstream code is behind the contract.

## Next Work

### Historical Phase 3: Migrate Scoring First

Status: done

Note: this is the older scoring-migration phase. Current references to "Phase 3" in the active prompt work mean point 3 of `prompt_enhancement_plan.md` (`Prompt simplification`), not this completed scoring phase.

Why first: scoring defines the contract that prompt, provider, mock reviewer, store, and UI must all obey.

Files to update:

- `src/narratives/scoring.py`
- `scripts/check_narrative_scoring.py`
- possibly `src/narratives/contract_fixtures.py` only if fixture edge cases are missing

Required behavior:

- Validate `design_confidence_subcategories`.
- Reject or ignore provider-returned app-owned score fields.
- Enforce supported packet evidence before any point effect.
- Calculate per-subcategory Design Confidence contributions from validated `rating + score_materiality + context guardrails`.
- Sum subcategories into total Design Confidence.
- Calculate Total Scenario Score as `completion_score + design_confidence`, clamped to `0..100`.
- Preserve half-point values.
- Preserve unsupported evidence references for trace/debug, but give them zero scoring effect.
- Return no Design Confidence or Total Scenario Score when required review fields are malformed or incomplete.

Verification:

- `python scripts/check_narrative_contract_fixtures.py` passed.
- `python scripts/check_narrative_scoring.py` passed.
- `python -m py_compile src/narratives/scoring.py scripts/check_narrative_scoring.py` passed.
- `git diff --check -- src/narratives/scoring.py scripts/check_narrative_scoring.py implementation_plan.md` passed.

Notes:

- `src/narratives/scoring.py` now returns V2 fields: `design_confidence`, `total_scenario_score`, and `design_confidence_assessment`.
- Legacy Quality Review output aliases are intentionally not preserved in the scoring result; downstream modules are expected to migrate in Phase 4 and Phase 5.
- The deterministic rating-to-points mapping uses rating direction plus packet context, so the app can distinguish small supportive signals from stronger evidence-backed trade-offs without accepting provider-owned final score values.

Completed scoring-scale migration:

- Added qualitative `score_materiality` to each Design Confidence subcategory in the prompt/schema.
- Provider-owned numeric Design Confidence points remain forbidden.
- `rating + score_materiality` maps to app-owned points across `-5.0..+5.0` in `0.5` increments.
- Supported-evidence gating is preserved: unsupported non-neutral ratings still produce `0.0` point effect.
- Added context guardrails so already-positive Completion Outlook pillars are not over-rewarded unless the packet shows a resolved weakness or new design-quality evidence.
- Operational-only assumptions remain neutral or negative unless they are matched by evidence, patient-relevance, governance, or proportionality gains.
- Updated fixtures, mock/provider paths, prompt checker, scoring checker, review-store traces, eval harness summaries, and UI trace details together.

### Phase 4: Migrate Prompt, Schema, Provider, And Mock

Status: done

Files to update:

- `src/narratives/prompt_builder.py`
- `src/narratives/provider.py`
- `src/narratives/mock_reviewer.py`
- `scripts/check_narrative_prompt_builder.py`
- `scripts/check_narrative_provider.py`
- `scripts/check_narrative_mock_reviewer.py`

Required behavior:

- Rename provider-facing contract from Quality Review to Scenario Review.
- Replace `quality_review_domains` with `design_confidence_subcategories`.
- Enforce evidence-first provider reasoning for each Design Confidence subcategory:
  1. choose packet-supported `evidence_fields`;
  2. write the `rationale` from those fields;
  3. assign the `rating` from the evidence and rationale;
  4. assign qualitative `score_materiality` from supported-evidence strength and context guardrails.
- Make prompt examples and response schema present subcategory fields in evidence/rationale/rating/score-materiality order where possible, while keeping JSON parsing order-independent.
- Store enough trace data to audit the score rationale later: original provider JSON, normalized validated review, validation errors, supported/unsupported evidence fields, app-calculated subcategory points, Design Confidence, Total Scenario Score, prompt/rubric versions, provider/model namespace, and input hash.
- Make provider schema strict enough to require all top-level Scenario Review sections and all four Design Confidence subcategories on every review.
- Add one bounded provider-wrapper retry/repair attempt for malformed JSON, missing required fields, or incomplete required subcategories; record the retry reason and use the same packet.
- Do not fall back to a different provider for malformed or incomplete review JSON unless a later explicit product decision changes this; fallback is for provider/network/rate-limit/unavailable failure, not for provider-shopping around validation errors.
- Ensure mock, prompt, and provider checkers assert all four subcategories are present, valid, evidence-first, and do not include app-owned score fields.
- Ensure the participant UI can show Completion Score only with Scenario Review unavailable/retry when validation fails after retry, rather than showing a partial Design Confidence score.
- Defer any automatic response-repair step unless it is deterministic, auditable, uses the same provider output and packet evidence, and cannot invent missing clinical reasoning.
- Keep baseline-review prompts qualitative-only and avoid hidden numeric leakage.
- Ensure provider schema does not ask the LLM to calculate Design Confidence or Total Scenario Score.
- Ensure mock reviewer emits fixture-backed V2 Scenario Review JSON.
- Keep live providers opt-in through existing environment controls.

Verification:

- `python scripts/check_narrative_prompt_builder.py` passed.
- `python scripts/check_narrative_provider.py` passed.
- `python scripts/check_narrative_mock_reviewer.py` passed.
- `python scripts/check_narrative_scoring.py` passed.
- `python scripts/check_narrative_contract_fixtures.py` passed.
- `python -m py_compile src/narratives/prompt_builder.py src/narratives/mock_reviewer.py src/narratives/provider.py scripts/check_narrative_prompt_builder.py scripts/check_narrative_mock_reviewer.py scripts/check_narrative_provider.py` passed.
- `git diff --check -- src/narratives/prompt_builder.py src/narratives/mock_reviewer.py src/narratives/provider.py scripts/check_narrative_prompt_builder.py scripts/check_narrative_mock_reviewer.py scripts/check_narrative_provider.py implementation_plan.md docs/architecture_narratives.md` passed.

Notes:

- `src/narratives/prompt_builder.py` now emits a V2 Scenario Review contract with `completion_outlook_review`, `design_confidence_subcategories`, `pillar_reviews`, `tradeoff_review`, `participant_review`, `continuity`, and `trace`.
- Provider prompts require evidence-first reasoning for each Design Confidence subcategory and explicitly forbid provider-owned score fields.
- `src/narratives/mock_reviewer.py` now returns V2 Design Confidence / Total Scenario Score scoring fields.
- `src/narratives/provider.py` now treats valid V2 scoring as `design_confidence != None`, suppresses scores for malformed/incomplete V2 reviews, and records one bounded same-packet validation retry for malformed or incomplete provider JSON.
- Live provider smoke tests were not run; they remain opt-in because they use network/API spend.

### Phase 4 Prompt Quality Refinement

Status: complete

Goal: make Scenario Review output read like expert clinical-development analysis while staying auditable, bounded, and non-prescriptive before Phase 5 storage/UI wiring.

Files updated:

- `docs/architecture_narratives.md`
- `src/narratives/prompt_builder.py`
- `scripts/check_narrative_prompt_builder.py`

Implemented behavior:

- Added senior clinical-development / medical-strategy reviewer role language.
- Added `because / however / therefore` reasoning guidance so participant-facing comments identify the evidence signal, limitation or trade-off, and discussion implication.
- Added expert lenses for evidence interpretability, development intent fit, target-population relevance, operational proportionality, shortcut risk, governance adequacy, and cross-pillar Completion Outlook / Design Confidence tension.
- Added explicit overclaim limits: do not treat model score as clinical truth, infer efficacy/safety/regulatory acceptability beyond packet evidence, equate higher Completion Score with better design, or prescribe exact next edits.
- Added compact good/weak participant-output examples to the provider contract.
- Added richer scenario examples for three high-value cases: Completion Outlook improves while evidence weakens, Completion Outlook declines while design confidence improves, and operational burden increases without matching evidence gain.
- Added packet-level `structured_feature_meanings` and `text_context_field_meanings` sourced from taxonomy metadata so the provider sees clinical field meaning alongside canonical values and display labels.
- Added required `tradeoff_review.central_tension` to capture the single most important Completion Outlook versus Design Confidence trade-off in one auditable sentence.
- Strengthened taxonomy field meanings so they explain why fields matter for evidence interpretability, population relevance, operational proportionality, governance, and strategic defensibility rather than only defining the field.
- Activated curated reference-pack summaries in review packets. Default packet context now includes core clinical development, 2026 strategic context, and ICH E8 quality-by-design summaries, with ICH E6/E9 specialist packs selected when scenario evidence supports them.
- Added `strategic_context_2026_v1` for current practice themes such as access, representativeness, decentralised/digital elements, estimand clarity, data reliability, and governance proportionality.
- Strengthened final debate-question rules so the questions must be open-ended, strategic, non-prescriptive, grounded in the current narrative/reference packs, and not answerable with yes/no.
- Kept the V2 response shape stable except for the narrow `tradeoff_review.central_tension` addition, which Phase 5 should store with the rest of `tradeoff_review`.

Verification:

- `python scripts/check_narrative_prompt_builder.py` passed.
- `python scripts/check_narrative_provider.py` passed.
- `python scripts/check_narrative_mock_reviewer.py` passed.
- `python scripts/check_narrative_scoring.py` passed.
- `python scripts/check_narrative_contract_fixtures.py` passed.
- `python scripts/check_narrative_packet_builder.py` passed.
- `python scripts/check_narrative_live_snapshot_flow.py` passed.
- `python scripts/check_narrative_taxonomy.py` passed.
- `python scripts/check_narrative_reference_packs.py` passed.
- `python -m py_compile src/narratives/packet_builder.py src/narratives/prompt_builder.py src/narratives/contract_fixtures.py scripts/check_narrative_packet_builder.py scripts/check_narrative_prompt_builder.py` passed.
- `git diff --check -- docs/architecture_narratives.md src/narratives/packet_builder.py src/narratives/prompt_builder.py src/narratives/contract_fixtures.py scripts/check_narrative_packet_builder.py scripts/check_narrative_prompt_builder.py` passed.

### Phase 5: Migrate Packet Builder And Review Store

Status: complete

Files to update:

- `src/narratives/packet_builder.py`
- `src/narratives/review_store.py`
- `scripts/check_narrative_packet_builder.py`
- `scripts/check_narrative_review_store.py`
- `scripts/check_narrative_live_snapshot_flow.py`

Required behavior:

- Keep existing deterministic packet content: baseline/current/previous snapshots, field changes, `text_context` Trial description fields, operational assumptions, Completion Outlook score movement, impact deltas, and storyline memory.
- Update review-context summaries from Quality Review to Scenario Review.
- Store Design Confidence, Total Scenario Score, design subcategory contributions, validation status, provider metadata, and compact storyline memory.
- Cache by input hash plus provider/model/prompt/rubric namespace.
- Do not reuse mock cache entries as live-provider reviews.

Verification:

- `python scripts/check_narrative_packet_builder.py` passed.
- `python scripts/check_narrative_review_store.py` passed.
- `python scripts/check_narrative_live_snapshot_flow.py` passed.
- `python scripts/check_narrative_contract_fixtures.py` passed.
- `python scripts/check_narrative_scoring.py` passed.
- `python scripts/check_narrative_mock_reviewer.py` passed.
- `python scripts/check_narrative_provider.py` passed.
- `python scripts/check_narrative_prompt_builder.py` passed.
- `python scripts/check_narrative_taxonomy.py` passed.
- `python scripts/check_narrative_reference_packs.py` passed.
- `python -m py_compile src/narratives/packet_builder.py src/narratives/review_store.py` passed.

Notes:

- `src/narratives/review_store.py` now stores `design_confidence`, `total_scenario_score`, `design_confidence_assessment`, design subcategory ratings/contributions, `central_tension`, available/used reference-pack IDs, validation status/errors, provider metadata, and compact storyline memory.
- Temporary `quality_adjustment`, `final_candidate_score`, and `quality_assessment` aliases remain only to keep the current Phase 5/6 boundary from breaking the existing simulator panel before UI migration.
- `src/narratives/packet_builder.py` now passes compact V2 review context forward: Completion Outlook summary, Design Confidence subcategory ratings/contributions, participant-review V2 fields, central tension, continuity, and hidden-baseline qualitative-only policy.
- The reference-pack manifest default now matches code/docs: `core_clinical_development_v1`, `strategic_context_2026_v1`, and `ich_e8_quality_by_design_v1`.

### Phase 6: Migrate Simulator UI

Status: complete, with uncommitted polish pending final browser verification

Files to update:

- `frontend/views/trial_simulator.py`
- `frontend/utils/plot.py`

Required behavior:

- Rename participant UI from `Quality Review` to `Scenario Review`.
- Replace `Quality Adjustment` / `Final Candidate Score` display with `Design Confidence` / optional `Total Scenario Score`.
- Keep Completion Score visible and clearly separate from Design Confidence.
- Rename the score tab to `Trial Score`.
- Rename the prediction action button to `Review Scenario`.
- Add a `Score View` radio control in the gauge card with:
  - `Completion Outlook`: gauge/bar/treemap show XGBoost Completion Score and model drivers only.
  - `Design Confidence`: gauge/bar/treemap show Design Confidence adjustment and design subcategories only. The gauge is centered at zero and displays signed points, not `50 +/- adjustment`.
  - `Total Scenario Score`: gauge/bar/treemap show the combined Completion Outlook plus Design Confidence view.
- Keep the Scenario Review narrative report below the chart cards and keep the narrative content identical across all radio views. The radio changes the visual score lens only.
- Present the four familiar pillars and one Design Confidence subcategory under each in the Design/Total contribution visuals.
- Keep timing/diagnostics and raw validation details behind expanders.
- Preserve hidden baseline behavior and no-op review reuse.
- Reserve operational-assumption details for a future facilitator view. The plan should preserve the information and audit trail, but the participant Scenario Review should not become an operational-assumption debug panel.

Verification:

- `python -m py_compile frontend/views/trial_simulator.py frontend/utils/plot.py` passed.
- `python scripts/check_narrative_live_snapshot_flow.py` passed.
- `python scripts/check_narrative_scoring.py` passed.
- `python scripts/check_narrative_mock_reviewer.py` passed.
- `python scripts/check_narrative_review_store.py` passed.
- `python scripts/check_narrative_packet_builder.py` passed.
- `git diff --check -- frontend/views/trial_simulator.py frontend/utils/plot.py implementation_plan.md` passed.
- Streamlit simulator smoke passed on `APP_VARIANT=trial_simulator`, port `8503`.
- Browser smoke confirmed `Review Scenario`, `Trial Score`, `Score View`, `Completion Outlook`, `Design Confidence`, and `Total Scenario Score` render.
- Browser radio smoke confirmed `Design Confidence` view switches without crashing; no-edit baseline correctly keeps participant-visible Scenario Review unavailable under the hidden-baseline rule.
- Live-provider review smoke was not run; it remains opt-in because it uses network/API spend.

Current uncommitted status:

- Trial Score radio behavior has been refined so Simulation Mode opens on `Completion Outlook`, while `Review Scenario` switches the view to `Total Scenario Score`.
- Design Confidence now uses the same participant tier labels as the other score views: `Low Risk`, `Favorable`, `Watchlist`, and `High Risk`.
- Design Confidence uses a signed `-50..+50` gauge mapped across the full `0..100` visual span: `-50 -> 0`, `-25 -> 25`, `0 -> 50`, `+25 -> 75`, `+50 -> 100`.
- Design Confidence point deltas are displayed as a one-line tag; Completion Outlook and Total Scenario Score keep the two-line points-plus-percent tag.
- The participant Scenario Review no longer shows the Design Confidence contribution/debug section.
- Baseline Scenario Review is temporarily visible for testing and should be revisited before final participant-live behavior.
- `Preparing Simulation Mode...` overlay has been changed to persist until Trial Features render, using a Trial Features readiness marker.
- Final browser verification of the persistent Simulation Mode overlay is still pending because the last smoke attempt was interrupted.

### Pre-Automation Checkpoint - 2026-06-13

Status: first-wave automated narrative eval harness implemented; ready for controlled live Gemini run after user approval.

Completed from the manual four-iteration live-review feedback:

- Pending edits preserve the latest visible `Scenario Review`, `Design Confidence`, and `Total Scenario Score` instead of blanking those views.
- `Completion Outlook` now shows `Score update pending`; Design/Total score views and the Scenario Review card show `Review update pending`.
- Hidden baseline still suppresses participant-visible Design Confidence and Total Scenario Score.
- First visible `Total Scenario Score` delta compares against baseline `Completion Outlook`; first visible Design Confidence has no previous-value delta; later Design/Total deltas compare against the previous visible review.
- Design Confidence treemap leaves include short rationale text when available.
- Design Confidence treemap data preparation now lives in `frontend/utils/scenario_review_plot_data.py`, so narrative checkers can verify treemap rationale details without importing the Streamlit view or emitting Streamlit runtime warnings.
- Prompt instructions now explicitly state that `structured_features` values are the source of truth when they conflict with Trial description fields, while a contradictory Trial description field can only create a scenario-readiness warning.
- Prompt instructions now require materially fresh key questions across visible iterations unless the same dilemma is genuinely reopened.
- Prompt instructions now soften regulatory/evidence language and discourage unsupported categorical phrasing such as `required for registration` or `can provide the necessary evidence`.
- Prompt instructions now discourage repeating the same Design Confidence concern across the summary, subcategory rationales, central tension, and questions.

Verification completed:

- `python -m py_compile frontend/views/trial_simulator.py`
- `python -m py_compile src/narratives/prompt_builder.py scripts/check_narrative_prompt_builder.py`
- `python scripts/check_narrative_prompt_builder.py`
- `python scripts/check_narrative_provider.py`
- `python scripts/check_narrative_mock_reviewer.py`
- `python scripts/check_narrative_scoring.py`
- `python scripts/check_narrative_contract_fixtures.py`
- `python scripts/check_narrative_packet_builder.py`
- `python scripts/check_narrative_review_store.py`
- `python scripts/check_narrative_live_snapshot_flow.py`
- `git diff --check`
- Local Streamlit health smoke on port `8504` returned `ok`.
- `python -m py_compile frontend/utils/scenario_review_plot_data.py frontend/views/trial_simulator.py scripts/check_narrative_live_snapshot_flow.py` passed after extracting the Streamlit-free plot-data helper.

Automated eval harness:

- Added `scripts/run_narrative_eval_suite.py`.
- The harness selects real registry trials across target Therapeutic Areas and starting Completion Score bands.
- It applies four scenario iterations without Streamlit UI interaction: Completion Outlook-favorable evidence shortcut, harder but more clinically focused design, planning-assumption-only operational burden change, and `structured_features` / `text_context` contradiction.
- It builds the same narrative review packets used by the simulator, calls `mock`, explicit `gemini`/`openai`, or configured provider chain, stores review traces, grades deterministic quality checks, and writes Markdown/JSON reports under ignored `reports/narrative_evals/`.
- Reports include trial identity, exact UI-label changes, expectations, narratives, Design Confidence ratings/points, Total Scenario Score, key questions, deterministic gap analysis, and Codex comments.
- JSON reports archive the full input packet, provider prompt, raw provider JSON, and validated review whenever a provider returns a reviewed trace. Markdown stays concise for user review.
- Expectation flags are separated into deterministic checks and human-review focus items so subjective narrative-quality expectations are not treated as fully automated pass/fail signals.
- Added `--success-smoke` to validate the full reviewed-trace path and the `review_controls` operational-boundary override path locally with a fixture-backed mock review before any external provider run.

Residual risk:

- A one-trial live Gemini rerun after the latest prompt wording change returned 4/4 reviewed iterations and identified targeted calibration items now implemented: operational-only checks are subcategory-level on `operational_burden_balance`, verbatim repeated questions fail deterministic freshness, and `structured_features` / `text_context` mismatch remains a scenario-readiness warning rather than overriding structured-feature evidence or becoming the main Design Confidence score driver.
- The latest post-audit one-trial live Gemini run returned 4/4 reviewed iterations with `3` failed checks and `2` warnings. Implemented follow-up changes: remove participant-facing internal model vocabulary, narrow the Completion Outlook operational boundary to the three planning-assumption fields only, fail extra planning-assumption detail after the agreed zero-delta sentence, fail stale planning-assumption carryover in later non-operational iterations, and strengthen question freshness around operational-only and `structured_features` / `text_context` contradiction changes.
- After planning review, added eval-packet `review_controls` / `question_controls` so the prompt receives explicit product controls for the latest-change focus and Completion Outlook mode. For the hard operational planning-assumption zero-delta case, the eval harness now applies a narrow deterministic override to the Completion Outlook summary before storing the trace; Design Confidence and scoring remain provider/app-derived as before.
- Follow-up audit fixes implemented: raw pre-control provider review/scoring are preserved in report metadata before the operational-boundary override; remaining active prompt wording was changed from internal model vocabulary to participant-friendlier Completion Outlook score-input language; and `--success-smoke` now exercises the `review_controls` override path directly.
- The first control-layer rerun returned 4/4 reviewed iterations with `1` failed deterministic check and `0` warnings. The fail was an overly narrow operational-question vocabulary check, while the generated questions did focus on enrollment, duration, expanded network, oversight, and data quality. Implemented follow-up changes keep prompt additions minimal and move detailed enforcement into the eval checker: broader operational-question vocabulary, participant-facing internal-language failures, and `structured_features` / `text_context` scenario-readiness dominance warnings.
- Simplified question-freshness prompt wording to avoid rule saturation: later visible iterations now use a two-question-pair rule, with the medical/development question anchored to the newest material medical/evidence implication and the clinical-operations question raising a broader operational-development debate rooted in the trial or latest change. The old `unless the latest change genuinely reopens the same dilemma` escape hatch was removed; persistent dilemmas should be reframed through the newest change.
- Added a compact Operational Burden Balance resource/budget principle: qualitative resource, staffing, and budget implications may be discussed when packet fields imply added burden, but monetary cost, affordability, or financial feasibility must not be estimated without explicit financial evidence. Resource intensity affects Design Confidence through proportionality rather than automatically lowering the subcategory score.
- Refined structured_features/text_context conflict handling without adding a new rule block: `text_context` Trial description fields may support Completion Outlook narrative only when aligned with selected Completion Outlook score inputs; this applies across all Trial description fields and all relevant `structured_features`, not only intervention descriptions. Only the conflicting Trial description field detail is stale/superseded, while aligned or non-conflicting Trial description field content and latest useful `text_context` changes remain supporting context. Structured_features/text_context conflict is framed as a scenario-readiness warning rather than evidence that the selected structured design has the contradicted feature.
- Added the final boundary cleanup after `first_wave_question_general_prompt_1`: the fixed planning-assumption Completion Outlook sentence is exclusive to planning-assumption-only changes; non-planning-only reuse now fails deterministic eval; planning-assumption-only questions get a medical/development bridge to evidence ambition versus added burden; and `structured_features` / `text_context` mismatch dominance now warns when it drives multiple strong negative subcategory ratings without independent non-conflicting structured support.
- Added a compact field-source/output glossary to the provider prompt and architecture docs using exact packet groups: `model_interpretation.completion_score`, `structured_features`, `structured_feature_display_values`, `structured_feature_meanings`, `text_context`, `text_context_field_meanings`, `operational_assumptions`, `review_controls`, `completion_outlook_analysis`, `design_confidence_analysis`, and `design_confidence_subcategories`. Active wording now prefers `Completion Outlook score`, `Completion Outlook narrative`, `Design Confidence narrative`, and `Design Confidence subcategory ratings` over generic prediction/text terms.
- Tightened participant-facing style without adding a heavy prompt block: narratives should state unresolved concerns rather than prescribe exact redesign paths, and the eval harness now fails common prescriptive redesign phrases such as telling the scenario to transition, switch, add blinding, or add a comparator.
- After the three-trial live run, tightened question quality with two small controls: avoid repeating the same opening stem across consecutive visible questions, and ensure `structured_features` / `text_context` contradiction questions ask how to resolve or reconcile the scenario rather than how to operationalize stale contradictory Trial description detail.
- Added a shortcut-simplification calibration after the hematology fail, then refined it after the final-settings quality wave: operational simplification can legitimately improve Operational Burden Balance, but strong positive credit (`+3` to `+5`) in shortcut scenarios needs bounded rationale, independent operational value, or safety-extension/proportionality context. The eval now checks justification for strong shortcut credit instead of failing all positive shortcut credit. The eval also fails key questions that address `the team` and accepts hyphenated `small-molecule` as Small Molecule acknowledgment.
- Final-settings qualitative follow-up kept prompt changes deliberately compact: Design Confidence should open with the main cross-functional decision tension; participant-facing language now also avoids `model signal`, `model-score inputs`, and `the model reflects`; and question checks now catch direct address to a team, sponsor, investigator, or `you` rather than only `the team`.
- After `first_wave_broader_trials_5_1`, added three lightweight eval/prompt-boundary refinements: replace `model signals` with score-pattern wording, make the Target Population Alignment expectation conditional when the synthetic population edit conflicts with prevention/vaccine-style trial context, and require operational-only medical questions to reference planning burden, scale, or proportionality.
- After `first_wave_broader_trials_5_2`, the residual failures were mostly question-generation issues. The active response contract now adds `key_questions.strategic_field_question` as a third participant question for broader Therapeutic Area or field-level development-design tensions, while keeping the medical/development and clinical-operations questions focused on scenario-specific implications.
- Audit follow-up fixed the material coverage gap by including `strategic_field_question` in participant-language/prescriptive-redesign eval checks, and added a focused scoring smoke check that legacy `participant_review` two-question payloads still validate with an empty strategic-field question. No additional simplification was needed beyond this narrow regression test.
- After `first_wave_three_question_contract_5_1`, kept tuning deliberately narrow: strengthen participant-facing replacement language for `in the model`, broaden the prevention/vaccine Target Population Alignment expectation skip using Trial description context, vary strategic/field question lenses across evidence standard, access, governance, data reliability, representativeness, feasibility, and interpretability, and sanitize provider failure messages in the simulator so participants see `Scenario Review could not be generated (ErrorType)` without provider names.
- Current constraint: do not implement post-narrative deterministic cleanup for participant wording or question rewriting. For now, the system should only prompt Gemini to avoid those issues and let the eval harness flag remaining failures; the existing planning-assumption boundary sentence and unavailable-review error formatting remain the only deterministic participant-facing controls in scope.
- Temperature evaluation setup: the eval harness now accepts `--temperature` with numeric values or `omit/default/none/unset`. Numeric values are passed to Gemini; `omit` leaves the provider temperature field unset while preserving the setting as `None` in report metadata. Direct provider runs now carry the selected provider in config metadata and use the generation-control cache namespace. `scripts/compare_narrative_temperature_reports.py` stores cross-report summaries and duplicate-run reproducibility checks, including Design Confidence and subcategory drift deltas.
- Thinking-level evaluation setup: the eval harness now accepts `--gemini-thinking-level low|medium|high`. Unset keeps the current provider default, now omitted/default temperature plus primary Gemini `high` thinking; explicit values are stored in report metadata, applied to Gemini primary calls, and included in generation-control cache namespaces.
- Final-settings quality/reproducibility plan: use `scripts/run_final_narrative_quality_plan.py` to run or print the recommended wave. The default plan runs 10 Gemini trials for quality/adherence pattern detection, then runs the first 3 trials twice for duplicate reproducibility/drift inspection, then writes `final_settings_quality_repro_comparison`. The helper passes `--temperature omit` and `--gemini-thinking-level high` explicitly so report metadata shows the final settings. Run without `--execute` to print the exact commands; run with `--execute` to launch them.
- Final narrative validation plan: use `scripts/run_final_narrative_validation_plan.py` to run or print the final behavior/storyline/reproducibility wave. The helper keeps the recommended generation settings, then runs non-cumulative `--scenario-plan boundary` for isolated unusual latest-change behavior, cumulative `--scenario-plan storyline` for 12 credible candidate one-shot examples, and two duplicate `storyline` runs for reproducibility/drift inspection. The boundary matrix covers structured-only, Trial description-only, planning-assumption-only, structured + Trial description, Trial description + planning assumptions, structured + planning assumptions, all three input types together, `structured_features` / `text_context` contradiction, aligned non-conflict structured/text, and shortcut simplification. The reports should be reviewed separately: boundary reports for rule adherence, storyline reports for presentation-ready example selection, and comparison reports for duplicate score/narrative drift.
- Implemented structured-field red flags: UI-only red highlighting for impossible structured Trial Feature combinations, separate from Design Confidence scoring calibration. It mirrors the existing placebo consistency behavior: runs on field change, red-highlights involved controls until the incompatibility is resolved, has no amber state, no compact warning card, no auto-correction except existing placebo sync, and does not disable `Review Scenario` or block Scenario Review generation. The validated hard flags are: parallel with one arm; randomized single-group; randomized with one arm; placebo comparator with no placebo control; placebo control with no control group; placebo control with one arm; single-group with placebo control; active comparator with single-group; active comparator with one arm; double/triple/quadruple blind with single-group/no comparator/no placebo; double/triple/quadruple blind with one arm/no comparator/no placebo; factorial with one arm; crossover with one arm; sequential with one arm. Implementation artifacts: `frontend/utils/structured_incompatibility.py`, `frontend/views/trial_simulator.py`, and `scripts/check_structured_incompatibility.py`.
- Implemented Design Confidence scoring calibration: the LLM contract is unchanged and numeric scoring remains app-owned. Each subcategory now stores `raw_points` from `rating + score_materiality`, calibrated `points`, and `calibration_notes` when a deterministic cap changes the score. Same-direction double-counting is triggered only by strong matching Completion Outlook pillar movement at `>= +3.0` or `<= -3.0` points. If the matching pillar moved strongly positive and is now still negative, strongly positive mapped Design Confidence is capped at `+2.5`; if it moved strongly positive and is now neutral or positive, it is capped at `+1.5`. If the matching pillar moved strongly negative and is now still positive, strongly negative mapped Design Confidence is softened to `-2.5`; if it moved strongly negative and is now neutral or negative, it is softened to `-1.5`. Total Completion Outlook score movement and previous-score thresholds are not calibration triggers because total score movement is an aggregate and the total is where subcategory trade-offs reconcile. Opposite-direction counterweight behavior is preserved, and Design Confidence can still move when Completion Outlook is flat or barely changed. Operational assumptions follow the same Design Confidence scoring and calibration rules as other supported packet evidence: they may improve or worsen Operational Burden Balance, and may create compensating effects in other Design Confidence subcategories when supported by rationale. Their only special boundary is Completion Outlook: planned enrollment, planned site count, and planned total duration must not explain Completion Outlook movement because they do not feed that score. There is no field-family-specific cap for Operational Burden Balance; subcategory meaning is preserved through the LLM rationale plus global same-pillar double-counting controls. Participant-facing Design Confidence treemap now shows signed subcategory points plus `short_rationale`, not `rating` or `score_materiality` labels.
- The harness currently uses planned/synthetic Completion Outlook score movements rather than calling live `/predict`; this is intentional for fast prompt-quality testing and can be replaced by API scoring later if needed.

Next step:

- Run the final narrative validation plan, then inspect boundary behavior, candidate one-shot storylines, duplicate-run drift, failed checks, warning clusters, Design Confidence calibration, question freshness, participant-language leaks, and whether issues repeat across trials enough to justify another prompt change or one-shot example A/B test.

### Phase 7: Full Narrative Regression Pass

Status: superseded as the next manual-only step by the automated first-wave eval harness described above. The named live-review trials remain useful seeds for that harness.

Named live-review target:

- Review `NCT03386721` - Simlukafusp alfa (ROCHE) | Oncology (2018), with special attention to whether Execution Framework / Operational Burden Balance narratives are specific, expert, auditable, and strategically useful without prescribing the solution.
- Review `NCT03896581` - `[BE COMPLETE]` Bimekizumab (UCB) | Musculoskeletal (2019), with special attention to the narrative when Pathway Profile changes from `Interleukin Cytokine` to `Kinase Inhibitor`. Use this case to check whether the Scenario Review explains the clinical-development meaning of a pathway-class change without overclaiming mechanism, efficacy, or regulatory implications.

Verification:

- all narrative checkers:
  - `python scripts/check_narrative_contract_fixtures.py`
  - `python scripts/check_narrative_packet_builder.py`
  - `python scripts/check_narrative_scoring.py`
  - `python scripts/check_narrative_mock_reviewer.py`
  - `python scripts/check_narrative_review_store.py`
  - `python scripts/check_narrative_provider_config.py`
  - `python scripts/check_narrative_prompt_builder.py`
  - `python scripts/check_narrative_provider.py`
  - `python scripts/check_narrative_live_snapshot_flow.py`
- `python -m py_compile` for changed modules
- `git diff --check`
- Streamlit simulator smoke

Parity note:

- These phases should not touch XGBoost scoring, preprocessing, model artifacts, taxonomy encoding, SHAP, or audit/demo prediction behavior.
- If any future change touches scoring/prep/model artifacts, run `python refresh_registry.py` and `python audit_parity.py` before deployment.

### Phase 8: Prompt Engineering Brief And Knowledge Substrate Review

Status: manual-feedback prompt/UI corrections complete; automated first-wave eval harness implemented and awaiting live provider run.

Purpose:

- Make the Scenario Review prompt system understandable, reproducible, and reusable before deeper prompt editing.
- Separate prompt/content engineering from provider-settings tuning.
- Document what the LLM currently receives as if the prompt were manually submitted to ChatGPT/Gemini with attached context.

Current artifact:

- `docs/narrative_prompt_engineering_brief.md`
- `prompt_enhancement_plan.md`

The brief captures:

- current prompt anatomy
- current context/files feeding the LLM
- current LLM flow architecture
- reference-pack selection
- provider settings and retry behavior
- test cases used for regression
- before/after observation template
- first recommendations for prompt, reference-pack, taxonomy, and settings changes

Next work:

- Use the prompt reconstruction helper to inspect one representative prompt, packet, selected reference packs, response contract, and sanitized provider settings:
  - `python scripts/export_narrative_prompt_brief.py --fixture operational_only_ambitious_enrollment_v2 --out /tmp/narrative_prompt_export`
- Use `prompt_enhancement_plan.md` as the working implementation plan for the next narrative prompt/schema migration.
- Before code changes, promote accepted durable decisions from `prompt_enhancement_plan.md` into `docs/architecture_narratives.md`.
- Then implement in staged order: contract fixtures, packet builder, prompt/schema, mock/provider normalization, scoring/storage, prompt export review, UI integration, regression/live review, then provider settings.

## Deferred Work

These are real needs, but not blockers for the immediate schema migration:

- Durable database-backed hidden baseline review store for cross-team reuse.
- Calibrated 10-20 scenario playtest set.
- Final product decision on whether Total Scenario Score is participant-primary, participant-secondary, or facilitator-only.
- Reference-pack and local context-stat expansion beyond the existing checked packs.
- Facilitator-specific debug/explanation mode.
- Future facilitator view for operational assumptions, including benchmark source, defaulting logic, user overrides, stale-state behavior, and how operational evidence influenced Design Confidence.
- Two-branch adjusted treemap or richer visual decomposition after the simpler four-pillar view is stable.

## Immediate Next Step

Final validation wave completed successfully and should now be reviewed qualitatively before any one-shot example is selected or embedded. Generated artifacts:

- `reports/narrative_evals/final_validation_boundary_10_1.json` / `.md`: 110/110 reviewed visible iterations, 10 failed checks, 12 warnings. Boundary behavior stayed stable; remaining failures are mainly accepted participant-language/question-style residuals, plus one operational-only question focus finding.
- `reports/narrative_evals/final_validation_storyline_candidates_12_1.json` / `.md`: 48/48 reviewed visible iterations, 5 failed checks, 19 warnings. This is the primary source for one-shot candidate review.
- `reports/narrative_evals/final_validation_repro_3_a.json`, `final_validation_repro_3_b.json`, and `final_validation_repro_3_comparison.json`: duplicate storyline reproducibility was 12/12 exact and normalized iteration matches, with 12/12 score matches.

Next work is a human-readable quality review, not another prompt-control edit: inspect full storylines, what changed at each iteration, Completion Outlook and Design Confidence narratives, subcategory rationales, score movements, and questions. Early storyline candidates to inspect first are `NCT04393298` UCB Oncology, `NCT05590793` Oncology, `NCT06315322` UCB Neurology, `NCT04009499` UCB Musculoskeletal, and `NCT04643457` UCB Dermatology. Classify candidates later as `presentation_ready`, `good_after_light_edit`, `useful_for_stress_test_only`, or `discard`.
