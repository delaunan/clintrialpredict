# Implementation Plan: Narrative Design Confidence Migration

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
- Each design subcategory uses `-5.0..+5.0` in `0.5` increments, derived by the app from qualitative `rating + score_materiality + context guardrails`, not from provider-owned numeric point fields.
- There is no hidden Design Confidence total cap; only Total Scenario Score is clamped to `0..100`.
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

### Phase 3: Migrate Scoring First

Status: done

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

- Keep existing deterministic packet content: baseline/current/previous snapshots, field changes, text context, operational assumptions, XGBoost score movement, impact deltas, and storyline memory.
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
- Prompt instructions now explicitly state that selected structured/categorical fields are the source of truth when free text conflicts, while contradictory free text can only create a scenario-coherence concern.
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
- It applies four scenario iterations without Streamlit UI interaction: model-favorable evidence shortcut, harder but more clinically focused design, operational-burden-only change, and structured/free-text contradiction.
- It builds the same narrative review packets used by the simulator, calls `mock`, explicit `gemini`/`openai`, or configured provider chain, stores review traces, grades deterministic quality checks, and writes Markdown/JSON reports under ignored `reports/narrative_evals/`.
- Reports include trial identity, exact UI-label changes, expectations, narratives, Design Confidence ratings/points, Total Scenario Score, key questions, deterministic gap analysis, and Codex comments.
- JSON reports archive the full input packet, provider prompt, raw provider JSON, and validated review whenever a provider returns a reviewed trace. Markdown stays concise for user review.
- Expectation flags are separated into deterministic checks and human-review focus items so subjective narrative-quality expectations are not treated as fully automated pass/fail signals.
- Added `--success-smoke` to validate the full reviewed-trace path and the `review_controls` operational-boundary override path locally with a fixture-backed mock review before any external provider run.

Residual risk:

- A one-trial live Gemini rerun after the latest prompt wording change returned 4/4 reviewed iterations and identified targeted calibration items now implemented: operational-only checks are subcategory-level on `operational_burden_balance`, verbatim repeated questions fail deterministic freshness, and structured/free-text mismatch remains bounded narrative context rather than overriding selected-field evidence or becoming the main Design Confidence score driver.
- The latest post-audit one-trial live Gemini run returned 4/4 reviewed iterations with `3` failed checks and `2` warnings. Implemented follow-up changes: remove participant-facing `model-facing` wording, narrow the Completion Outlook operational boundary to the three planning-assumption fields only, fail extra planning-assumption detail after the agreed zero-delta sentence, fail stale planning-assumption carryover in later non-operational iterations, and strengthen question freshness around operational-only and structured/free-text contradiction changes.
- After planning review, added eval-packet `review_controls` / `question_controls` so the prompt receives explicit product controls for the latest-change focus and Completion Outlook mode. For the hard operational planning-assumption zero-delta case, the eval harness now applies a narrow deterministic override to the Completion Outlook summary before storing the trace; Design Confidence and scoring remain provider/app-derived as before.
- Follow-up audit fixes implemented: raw pre-control provider review/scoring are preserved in report metadata before the operational-boundary override; remaining active prompt wording was changed from `model-supported` / `model evidence` to participant-friendlier score-input language; and `--success-smoke` now exercises the `review_controls` override path directly.
- The first control-layer rerun returned 4/4 reviewed iterations with `1` failed deterministic check and `0` warnings. The fail was an overly narrow operational-question vocabulary check, while the generated questions did focus on enrollment, duration, expanded network, oversight, and data quality. Implemented follow-up changes keep prompt additions minimal and move detailed enforcement into the eval checker: broader operational-question vocabulary, participant-facing internal-model-language failures, and structured/free-text bounded-impact warnings.
- Simplified question-freshness prompt wording to avoid rule saturation: later visible iterations now use a two-question-pair rule, with one question anchored to the newest material change and the other raising a broader strategic development-design tension. The old `unless the latest change genuinely reopens the same dilemma` escape hatch was removed; persistent dilemmas should be reframed through the newest change.
- Added a compact Operational Burden Balance resource/budget principle: qualitative resource, staffing, and budget implications may be discussed when packet fields imply added burden, but monetary cost, affordability, or financial feasibility must not be estimated without explicit financial evidence. Resource intensity affects Design Confidence through proportionality rather than automatically lowering the subcategory score.
- The harness currently uses planned/synthetic Completion Outlook score movements rather than calling live `/predict`; this is intentional for fast prompt-quality testing and can be replaced by API scoring later if needed.

Next step:

- Rerun the same controlled one-trial Gemini eval with a new run id to verify the control-layer prompt/checker adaptations before expanding to `--max-trials 2`.

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

Run the first-wave automated Scenario Review quality-eval harness before further manual prompt tuning. Start small, inspect `reports/narrative_evals/*.md`, summarize recurring quality gaps into `prompt_enhancement_plan.md`, then refine the prompt only from repeated evidence across trials/iterations.
