# Implementation Plan: Narrative Dimension Redesign

## Scope

Architecture scope: `architecture_narratives`, with secondary read-only inputs from `architecture_edit`, `architecture_estimation`, and `core_scoring`.

Goal: redesign the serious-game Quality Review so it gives medical-director-grade interpretation of trial scenario changes. The review should explain Completion Score movement from XGBoost evidence, then evaluate complementary trial-value dimensions beyond completion likelihood.

This plan replaces the prior narrative next-step plan about the lightweight consistency check. The consistency check may still be useful, but it becomes one subtask inside the broader dimension redesign.

## Product Framing

The simulator should answer two different questions:

1. `Will this trial complete?`
   - Owned by the existing XGBoost Completion Score.
   - Explained through feature, subcategory, pillar, and score movement.
   - No LLM scoring should alter this value.

2. `Is this a strategically valuable and defensible trial design?`
   - Owned by a redesigned `Trial Value Review`.
   - Uses XGBoost fields, non-XGBoost operational assumptions, text changes, baseline context, previous iteration context, and curated clinical-development references.
   - Can highlight that a scenario improved completion likelihood by reducing scientific, regulatory, population, operational, or evidence value.

Recommended user-facing naming:

- Keep `Completion Score`.
- Rename broad review from `Quality Review` to `Trial Value Review` if playtesting confirms it feels clearer.
- Rename `Quality Adjustment` to `Design Value Adjustment` or `Value Adjustment`.
- Rename `Final Candidate Score` to `Strategic Candidate Score` only if a single adjusted score remains necessary.

Current recommendation: simplify the visible product language further before implementation. Keep the numeric adjustment small and secondary until UX tests show that a combined score is genuinely clearer. The primary participant learning surface should be the bar chart, state labels, trade-off map, and debate question.

Possible naming families:

1. Conservative / clinical-operations language:
   - `Completion Outlook`
   - `Scenario Review`
   - `Scenario Adjustment`
   - `Total Scenario Score`

2. Medical-director / portfolio language:
   - `Development Outlook`
   - `Strategic Design Review`
   - `Design Strength Adjustment`
   - `Development Confidence Score`

3. Workshop / game language:
   - `Scenario Outlook`
   - `Design Trade-off Review`
   - `Scenario Modifier`
   - `Candidate Score`

Recommendation for now:

- use `Completion Outlook` for the model score view
- use `Scenario Review` for the narrative panel
- use `Total Scenario Score` only if the combined view is activated
- avoid `Quality Score` and probably avoid `Value Score`, because both sound more absolute than the evidence supports

## Design Principle

Do not merge the new dimensions into the four XGBoost Completion Score pillars by default.

Reason: the XGBoost pillars explain modeled early-termination/completion likelihood. The new dimensions evaluate a different construct: decision usefulness and strategic defensibility. If they are nested inside the Completion Score pillars, users may think the model itself has learned cost, regulatory strength, scientific rigor, or patient relevance. It has not.

Detailed internal reasoning structure from the first planning pass:

```text
Scenario Review
├── Completion Movement Analysis
│   ├── Score delta
│   ├── Feature impact movement
│   ├── Subcategory movement
│   ├── Pillar movement
│   └── cross-feature dependency hypotheses
└── Trial Value Review
    ├── Evidence Decisiveness
    ├── Population & Patient Relevance
    ├── Scientific & Mechanistic Rationale
    ├── Regulatory & Development Strategy Fit
    └── Operational Deliverability & Risk Governance
```

This five-domain version is probably too granular for the participant UI. It is useful as an internal reasoning map, but the visible simulator needs fewer, more familiar clusters.

Recommended simplified UI structure:

```text
Total Scenario Score
├── Completion Outlook
│   ├── Patient Profile
│   ├── Scientific Challenge
│   ├── Execution Framework
│   └── Operational Complexity
├── Evidence Strength
│   ├── Endpoint & Comparator
│   └── Interpretability
├── Strategic Fit
│   ├── Development Intent
│   └── Patient Relevance
└── Delivery Confidence
    ├── Operational Assumptions
    └── Risk Governance
```

In this simplified model, `Completion Outlook` is the current XGBoost Completion Score contribution layer. The three additional pillars explain what the scenario adds or sacrifices beyond pure completion likelihood. They can be plotted next to the existing completion pillars in a combined bar chart, while still allowing a radio toggle between:

- `Completion only`: current XGBoost Completion Score and existing pillars.
- `Full scenario`: Completion Outlook plus Evidence Strength, Strategic Fit, and Delivery Confidence.

The key UX rule is that the same field should not appear as two unrelated concepts with confusing names. For example, current `Patient Profile` should not sit next to a separate visible pillar called `Population & Patient Relevance`. Instead, keep the familiar `Patient Profile` naming under Completion Outlook, and use `Strategic Fit -> Patient Relevance` only as the extra-value interpretation layer.

Alternative even simpler structure:

```text
Total Scenario Score
├── Completion Outlook
├── Evidence Strength
├── Strategic Fit
└── Delivery Confidence
```

Each visible pillar can expose only two subpillars. Facilitator/debug mode can show the deeper five-domain analysis.

Feature-density review after inspecting the current 31 model-facing fields suggests an even stronger simplification:

```text
Total Scenario Score
├── Completion Outlook
│   ├── Therapeutic Context
│   ├── Scientific Challenge
│   ├── Patient Profile
│   └── Execution Framework
├── Evidence & Strategy Strength
│   ├── Endpoint / Comparator / Bias Control
│   └── Development Fit / Patient Relevance
└── Delivery & Investment Burden
    ├── Operational Feasibility
    └── Cost / Complexity Pressure
```

This may be the best V1 participant-facing version because it respects the actual evidence density:

- `Completion Outlook` keeps all four existing XGBoost pillars, so it is not visually underrepresented.
- `Evidence & Strategy Strength` merges endpoint, comparator, interpretability, regulatory intent, and patient relevance into one clinically meaningful pillar. These concepts are deeply linked in practice and share many of the same fields.
- `Delivery & Investment Burden` merges operational assumptions, risk governance, and cost/complexity pressure. Current fields do not support true budget calculation, but they do support a relative burden estimate.

Cost should be represented as `Cost / Complexity Pressure`, not as exact cost. Current evidence can support relative cost pressure from:

- planned enrollment
- planned sites
- planned duration
- number of arms
- administration complexity
- therapeutic modality
- masking
- placebo/control structure
- DMC / oversight
- phase
- sponsor type
- endpoint duration

The narrative should compare whether additional burden appears justified by additional evidence or strategic strength. Example: a larger, longer, more controlled design may increase cost pressure but also improve evidence strength; a simplification may reduce cost and raise completion likelihood while weakening interpretability.

Recommended V1 radio options:

- `Completion outlook`: current Completion Score and the four XGBoost pillars.
- `Full scenario`: Completion Outlook plus Evidence & Strategy Strength plus Delivery & Investment Burden.

This gives a simple combined bar chart without asking participants to parse too many new categories.

Important scoring guardrail:

`Operational Feasibility` must not simply duplicate the Completion Score. It should not reward easy execution by default. It should evaluate whether operational assumptions are proportionate to the evidence ambition and patient context.

Examples:

- Lower enrollment, fewer sites, and shorter duration may improve completion outlook but weaken evidence precision, representativeness, endpoint maturity, or strategic credibility.
- More sites, longer duration, and stronger oversight may reduce completion outlook or increase cost pressure but be justified if they protect endpoint interpretability, rare-disease access, safety governance, or regulatory confidence.
- Benchmark-typical operational assumptions are neutral unless they are coherent with the current scenario and support a meaningful design choice.

The review should therefore seek contradictory stakes:

```text
Completion improved, but evidence weakened.
Completion improved, but cost/burden increased without clear evidence gain.
Completion worsened, but evidence or strategic confidence improved.
Operational feasibility improved, but the trial became less decision-useful.
Operational burden increased, but the added burden appears justified.
```

## Score Architecture Alternatives

### Option A: Two-Level Score Stack

This is the cleanest conceptual model.

```text
Total Scenario Score
├── Completion Outlook
│   ├── Therapeutic Context
│   ├── Scientific Challenge
│   ├── Patient Profile
│   └── Execution Framework
└── Design Confidence
    ├── Evidence & Decision Strength
    └── Feasibility & Resource Balance
```

Participant reading:

- `Completion Outlook`: how completion-like the scenario appears.
- `Design Confidence`: whether the trial remains worth doing and defensible.
- `Total Scenario Score`: combined view.

This avoids too many same-level bars. It also makes it clear that the non-XGBoost layer is an adjustment layer.

### Option B: Two-Level Score With Four Design Pillars

This is more explicit but still readable.

```text
Total Scenario Score
├── Completion Outlook
│   ├── Therapeutic Context
│   ├── Scientific Challenge
│   ├── Patient Profile
│   └── Execution Framework
└── Design Confidence
    ├── Endpoint Strength
    ├── Development Fit
    ├── Operational Feasibility
    └── Cost / Complexity Pressure
```

Participant reading:

- the current model remains intact
- the added layer explains what the completion score does not judge directly
- the four design pillars are concrete enough for discussion

Risk:

- four extra pillars may be too many for a compact chart unless the UI groups them visually under `Design Confidence`.

### Option C: Same Four Pillars, Full-Scenario Overlay

This is the most visually consistent with the current product.

```text
Completion Outlook View
├── Therapeutic Context
├── Scientific Challenge
├── Patient Profile
└── Execution Framework

Full Scenario View
├── Therapeutic Context
│   └── development fit / disease-context adjustment
├── Scientific Challenge
│   └── endpoint, comparator, evidence-strength adjustment
├── Patient Profile
│   └── patient relevance / representativeness adjustment
└── Execution Framework
    └── feasibility, risk governance, cost/complexity adjustment
```

Participant reading:

- same four pillars in both radio-button views
- `Completion Outlook` shows XGBoost-only contribution
- `Full Scenario` shows XGBoost contribution plus design-confidence adjustment inside each familiar pillar

This may be the best UX if the bar chart is the main interface. It minimizes new vocabulary and lets participants see how broader considerations modify the current model view.

Risk:

- it may blur the boundary between XGBoost evidence and LLM/app-owned design adjustment.
- the UI must clearly label each bar as `Completion component` plus `Design adjustment`.

Recommended implementation sequence:

1. Prototype Option A in the prompt/schema because it keeps the conceptual boundary clean.
2. In UI mockups, test Option C because it may be easiest for participants to read.
3. Keep Option B as facilitator/debug detail if users want the four design levers exposed.

## Review Input Evidence Model

The Scenario Review must be fully tied to the evidence the application can provide. The LLM should not infer a rich protocol that is not present in the packet.

Available current evidence sources:

- `models/taxonomy_01.json` and `src/prep/pipeline.py`: field labels, options, current four pillars, model subgroups, and narrative meanings.
- `frontend/data/search_registry.csv`: selected-trial baseline values, prerecorded Completion Score, existing pillar scores, text fields, metadata, and display fields.
- `src/scoring/decomposition.py`: local prerecorded baseline decomposition for audit/search-registry trials.
- `/predict` via `api/main.py`: live simulation Completion Score, pillar impacts, subcategory impacts, and feature drivers for model-facing edits.
- `src/narratives/packet_builder.py`: deterministic baseline/previous/current review packet with structured features, display values, text context, operational assumptions, score deltas, field changes, and `xgboost_impact_changes`.
- `frontend/data/operational_benchmarks_v1.csv` and `src/operational_benchmarks.py`: planned enrollment, planned sites, planned duration, benchmark status, source, cohort, percentile, and confidence metadata.
- `frontend/data/gbd_l3_indication_lookup.csv`: indication/TA context and observed TA support for the indication selector.
- Future `frontend/data/narrative_context_stats_v1.json`: mature-cohort design-pattern and outcome-context summaries derived from `data/data_clinpred.csv`.

Current packet facts:

- `STRUCTURED_FEATURE_KEYS` contains the 31 active structured Trial Features used for scenario review.
- `ACTIVE_OPERATIONAL_ASSUMPTION_KEYS` contains `planned_enrollment`, `planned_sites`, and `planned_duration_months`.
- `TEXT_CONTEXT_KEYS` contains `title`, `summary_ui`, `conditions_ui`, `primary_outcomes_ui`, and `interventions_ui`.
- `model_interpretation.xgboost_impact_changes` already captures baseline, previous, current, and delta values for pillar and subcategory movement.
- `iteration_context.field_changes` already captures baseline, previous, and current values/labels for structured, text, and operational edits.

The plan should leverage these existing packet fields before adding new artifacts.

### Required Packet Context

Every visible iteration review should receive:

```text
1. Trial baseline
   - original structured Trial Features
   - original text context
   - original operational assumptions and source metadata
   - original prerecorded XGBoost Completion Score
   - original pillar/subcategory/feature decomposition when available

2. Previous iteration
   - previous submitted structured values
   - previous operational assumptions
   - previous text context
   - previous Completion Score
   - previous pillar/subcategory/feature decomposition
   - previous Design Confidence review summary

3. Current iteration
   - current submitted structured values
   - current operational assumptions
   - current text context
   - current Completion Score
   - current pillar/subcategory/feature decomposition
   - changed fields since previous iteration
   - changed fields from original baseline

4. XGBoost movement evidence
   - score delta from previous
   - score delta from baseline
   - feature impact deltas
   - subcategory impact deltas
   - pillar impact deltas
   - top positive and negative feature movement

5. Design Confidence evidence
   - operational benchmark metadata
   - text/structured consistency signals
   - clarification context, if any
   - selected document-pack summaries
   - selected local database context stats
```

### Baseline Role

The baseline review should do two things:

1. Establish why the original trial had its initial Completion Outlook.
2. Establish the original design context so later participant edits can be interpreted as genuine improvement, acceptable simplification, or shortcut behavior.

The baseline should not be a hidden answer key. It should create a compact memory of:

- baseline completion drivers
- baseline strengths
- baseline watchlist issues
- original patient/indication logic
- original endpoint/evidence logic
- original operational burden logic

### Iteration Role

Each iteration should answer:

```text
What changed?
What moved the Completion Outlook?
Which pillar moved most?
Which feature or subcategory movements explain that pillar movement?
Which design adjustment topics were affected?
What collateral impacts appeared in other pillars?
Did the scenario become easier, stronger, narrower, riskier, more expensive, or more defensible?
What should the team debate next?
```

## Completion Outlook Analysis Contract

The Completion Outlook analysis is not a generic LLM opinion. It is a structured explanation of the XGBoost score movement.

### Required Analysis Levels

For each visible iteration, the review should produce:

```text
Completion Outlook
├── Overall score movement
├── Pillar movement summary
├── Top affected pillar
├── Feature and subcategory drivers
├── Cross-pillar interaction hypotheses
└── Model limitation note
```

### Per-Pillar Completion Review

For every current pillar, the review should identify:

- baseline pillar contribution
- previous pillar contribution
- current pillar contribution
- delta from previous
- delta from baseline
- changed fields directly mapped to the pillar
- top feature/subcategory movements
- whether the movement is model-supported, design-plausible, or uncertain

Pillar-level explanation template:

```text
Pillar: Scientific Challenge
Movement: +3.0 from previous, +1.5 from baseline
Likely model-supported drivers:
- endpoint_rigor_ml changed from surrogate to hard clinical endpoint
- biomarker_stratification_ml changed from no to yes
Clinical interpretation:
- The model may be responding to stronger evidence architecture, but this also increases operational and recruitment pressure.
Collateral impacts:
- Execution Framework may become more pressured because the stronger endpoint may require longer duration or more complex operations.
```

### Cross-Pillar Interaction Library

The prompt should explicitly ask the reviewer to look for these interactions:

```text
Therapeutic Context x Scientific Challenge
- phase/regulatory intent vs endpoint rigor
- rare disease vs biomarker or endpoint expectations
- indication/disease course vs endpoint duration

Therapeutic Context x Patient Profile
- indication/rareness vs age/severity/line-of-therapy scope
- phase/intent vs population breadth

Therapeutic Context x Execution Framework
- phase/intent vs enrollment/sites/duration
- rare disease vs feasible recruitment footprint
- sponsor type vs operational scale

Scientific Challenge x Patient Profile
- biomarker strategy vs target population
- endpoint relevance vs severity/line of therapy
- modality/target novelty vs patient-risk tolerance

Scientific Challenge x Execution Framework
- endpoint rigor vs duration/sites/enrollment
- comparator/masking/allocation vs number of arms and complexity
- modality/administration complexity vs delivery feasibility

Patient Profile x Execution Framework
- population restrictions vs enrollment feasibility
- pediatric/older/severe population vs oversight and site burden
- rare/severe population vs duration and retention pressure
```

Cross-pillar interactions should be framed as hypotheses, not model-proven causality.

## Design Confidence Analysis Contract

Design Confidence is a bounded review-derived adjustment layer. It should be analyzed at the same four-pillar level as Completion Outlook, but as app/LLM-owned design reasoning rather than XGBoost contribution.

Current preferred full-scenario hierarchy:

```text
Therapeutic Context
├── Therapeutic Area Profile
├── Development Phase and Goal
└── Phase & Intent Alignment

Scientific Challenge
├── Biological Profile
├── Protocol Architecture
└── Endpoint & Evidence Strength

Patient Profile
├── Clinical Severity
├── Population Scope
└── Target Population Alignment

Execution Framework
├── Methodological Setup
├── Trial Complexity Footprint
└── Operational Burden Balance
```

### Per-Pillar Design Adjustment Review

For each design-adjustment subcategory, the review should produce:

- rating/state: `Favorable`, `Watchlist`, `High risk`, or `Trade-off`
- point adjustment, if numeric scoring is enabled
- supporting evidence fields
- baseline comparison
- previous-iteration comparison
- collateral impacts on other pillars
- one concise rationale

Example:

```text
Patient Profile -> Target Population Alignment
State: Watchlist
Adjustment: -1.0
Evidence:
- child_ml changed from included to excluded
- patient_severity_ml changed from severe to moderate
- is_rare_disease_ml remains yes
- strategic_ambition_ml remains confirmatory
Rationale:
- The change may improve completion by simplifying recruitment, but it appears to move away from the high-need population implied by the rare-disease baseline.
Collateral impact:
- Operational Burden Balance may improve, but Phase & Intent Alignment may weaken if confirmatory intent is retained.
```

## End-to-End Review Output Shape

The participant-facing review should be rich but structured.

Important participant UI assumption:

Participants may see only the four familiar pillars in the main chart. They may not see Completion Outlook impacts and Design Confidence adjustments as two separate chart systems. Therefore, the narrative must be able to explain a single four-pillar `Full Scenario` view while keeping provenance clear in the trace.

Recommended display rule:

```text
Participant view:
Four pillars only, with model subcategories and design-adjustment subcategory integrated.

Trace / facilitator view:
Separates XGBoost completion component, Design Confidence adjustment, evidence fields, and validation details.
```

Participant-facing wording should avoid over-technical labels such as XGBoost, SHAP, feature deltas, or validation domains unless the facilitator/debug view is open. It should instead say:

```text
The completion model moved mainly because...
The scenario review adjusts this pillar because...
The main trade-off is...
```

The narrative must be backed by packet evidence. If the packet does not support a claim, the output should say the issue is uncertain or not assessable.

Recommended output sections:

```text
1. Scenario movement snapshot
   - Completion Score changed from X to Y
   - top moved Completion Outlook pillar
   - top design-confidence counterpressure

2. What moved the Completion Outlook
   - top model-supported drivers
   - top pillar/subcategory movements
   - cross-pillar interaction hypotheses

3. Full Scenario pillar review
   - Therapeutic Context: completion movement + Phase & Intent Alignment
   - Scientific Challenge: completion movement + Endpoint & Evidence Strength
   - Patient Profile: completion movement + Target Population Alignment
   - Execution Framework: completion movement + Operational Burden Balance

4. Trade-off summary
   - what improved
   - what weakened
   - what became easier
   - what became more burdensome
   - what remains uncertain

5. Two expert questions
   - one strategic/medical question
   - one operational/clinops question
```

### Two Expert Questions

Every visible iteration should end with exactly two questions:

1. Medical / development question:
   - focuses on evidence, patient relevance, phase/intent, endpoint interpretation, or strategic defensibility.
2. Clinops / execution question:
   - focuses on recruitment, sites, duration, oversight, cost/complexity pressure, or feasibility versus evidence ambition.

Examples:

```text
Medical / development:
If this scenario is easier to complete, what evidence would convince you that the revised endpoint and population still answer the intended development question?

Clinops / execution:
Which operational assumption is the tightest constraint in this scenario: enrollment, site footprint, duration, or delivery complexity, and is that constraint justified by the evidence gain?
```

The questions should be open-ended and should not recommend a specific field edit.

## Document and Data Context Strategy

The narrative should use three context layers:

```text
1. Packet evidence
2. Fixed free-access document summaries
3. Local database statistics
```

The ordering matters. Packet evidence wins. Documents and database stats enrich interpretation; they do not override the scenario.

### Fixed Free-Access Document Summaries

Always-on core:

- ICH E8(R1): quality by design, critical-to-quality factors, study objective/design alignment.
- ICH E6(R3): risk-proportionate GCP, participant protection, data integrity, fit-for-purpose conduct, oversight.
- ICH E9(R1): estimands, endpoint/population/treatment-condition/interpretation alignment.
- FDA clinical-trials guidance index: source index for optional packs, not full always-on content.

Conditional:

- FDA PFDD / clinical outcome assessment guidance when endpoints or patient experience fields matter.
- FDA eligibility/diversity guidance when population scope materially changes.
- FDA adaptive design guidance or ICH E20 when adaptive design becomes central.
- UCB public disease-area/pipeline context only for workshop personalization, not scoring.

Implementation rule:

- Use short authored summaries with source URLs and applicability tags.
- Do not paste full guidance text into runtime prompts.
- Do not let the LLM browse during gameplay.

### Local Database Statistics

Create a versioned local context artifact from `data/data_clinpred.csv`.

Recommended artifact:

```text
frontend/data/narrative_context_stats_v1.json
```

Stats to include:

- mature-cohort completion/termination rates by phase, TA, indication group, rare status, modality, endpoint profile, comparator profile, biomarker strategy, and population scope
- distribution of enrollment, site count, and duration by relevant cohort
- frequency of design patterns over time:
  - biomarker stratification
  - adaptive design
  - endpoint rigor
  - endpoint structure
  - comparator type
  - DMC use
  - masking/randomization
  - modality
- terminated/withdrawn pattern summaries when mature and reliable
- benchmark percentile context already available for operational assumptions

Censoring guardrail:

- Recent-start trials are immature. Do not report naive recent completion rates as failure trends.
- Prefer mature cohorts, status-aware summaries, or design-pattern frequencies that do not require final trial outcome.

Prompt use:

- The LLM can cite database context only as broad local context, for example "similar completed trials in this cohort often require longer duration."
- It must not claim causal proof from database summaries.
- It must not use database stats to override current packet evidence.

## Rich Iteration Analysis Algorithm

Conceptual algorithm for each prediction:

```text
1. Build baseline/current/previous packet.
2. Compute XGBoost score movement:
   - score delta from previous
   - score delta from baseline
   - pillar deltas
   - subcategory deltas
   - feature deltas
3. Identify directly changed fields.
4. Identify cross-pillar collateral effects:
   - changed field in one pillar
   - movement in another pillar or design adjustment topic
5. Pull applicable document summaries:
   - always-on core pack
   - conditional packs triggered by field changes
6. Pull applicable local database context stats:
   - trial cohort
   - operational benchmarks
   - mature design-pattern summaries
7. Generate structured review:
   - Completion Outlook movement
   - per-pillar design adjustment
   - collateral impacts
   - trade-off summary
   - two expert questions
8. Validate:
   - every non-neutral adjustment has supported evidence fields
   - no unsupported guidance claims
   - no stale score reuse
   - Design Confidence remains bounded
```

## Generic Naming Options

Avoid names that sound regulatory-only or financially exact.

Recommended top-level non-completion score name:

- `Design Confidence`

Why:

- broad enough to include evidence, strategy, feasibility, governance, and resource burden
- does not imply pure regulatory strength
- does not imply true budget or ROI calculation
- easier to explain than `Quality`, `Value`, or `Strategic Defensibility`

Alternative names:

- `Design Strength`
- `Scenario Strength`
- `Development Confidence`
- `Trial Confidence`
- `Design Balance`

Current recommendation:

```text
Total Scenario Score
├── Completion Outlook
└── Design Confidence
```

Design Confidence can then be decomposed into:

```text
Design Confidence
├── Evidence & Decision Strength
└── Feasibility & Resource Balance
```

or, in a more detailed view:

```text
Design Confidence
├── Endpoint Strength
├── Development Fit
├── Operational Feasibility
└── Cost / Complexity Pressure
```

Generic label definitions:

- `Evidence & Decision Strength`: whether the scenario can produce interpretable evidence that supports a meaningful decision.
- `Feasibility & Resource Balance`: whether the scenario can be delivered credibly and whether the operational/cost pressure appears justified by the intended evidence gain.

These are intentionally broader than `regulatory strength` or `investment burden`, while still allowing regulatory and cost considerations to appear when supported by fields.

## Integration With Current Pipeline Hierarchy

Source of truth: `src/prep/pipeline.py` / `PIPELINE_REGISTRY`.

Current model-facing hierarchy:

```text
Therapeutic Context
├── Therapeutic Area Profile
└── Development Phase and Goal

Scientific Challenge
├── Biological Profile
└── Protocol Architecture

Execution Framework
├── Methodological Setup
└── Trial Complexity Footprint

Patient Profile
├── Clinical Severity
└── Population Scope
```

The Design Confidence layer should not replace this hierarchy. It should reuse it.

### Recommended Overlay Mapping

```text
Therapeutic Context
├── Therapeutic Area Profile
├── Development Phase and Goal
└── Phase & Intent Alignment

Scientific Challenge
├── Biological Profile
├── Protocol Architecture
└── Endpoint & Evidence Strength

Patient Profile
├── Clinical Severity
├── Population Scope
└── Target Population Alignment

Execution Framework
├── Methodological Setup
├── Trial Complexity Footprint
└── Operational Burden Balance
```

This preserves the current four-pillar reading while adding one design-adjustment subcategory directly below each pillar. The UI can still distinguish model-owned subcategories from app/LLM-owned design-adjustment subcategories through styling or tooltip labels, but the hierarchy stays flat and readable.

### Feature-Level Mapping

#### Therapeutic Context -> Phase & Intent Alignment

Fields:

- `therapeutic_area_ml`
- `gbd_cause_id_3_ml`
- `is_rare_disease_ml`
- `phase_ml`
- `strategic_ambition_ml`
- text: `summary_ui`, `conditions_ui`

Adjustment asks:

- Does the phase fit the intended development question?
- Does regulatory intent match the therapeutic area, indication, rarity, and available design?
- Is the scenario exploratory, signal-seeking, dose-characterizing, or confirmatory in a coherent way?
- Is the development ambition plausible for the disease context?

Contradictory examples:

- Completion improves after lowering ambition from pivotal to signal-seeking, but strategic development fit may weaken if the exercise goal was registration-oriented.
- A rare-disease Phase III may have lower completion outlook but stronger strategic fit if the design preserves patient relevance and decision value.

Naming alternatives:

- `Development Intent Fit`
- `Phase & Intent Fit`
- `Development Rationale`
- `Strategic Trial Fit`
- `Indication Strategy Fit`

Current preferred concrete label: `Phase & Intent Fit`.

Typical scoring evidence:

- primary: `phase_ml`, `strategic_ambition_ml`, `therapeutic_area_ml`, `gbd_cause_id_3_ml`, `is_rare_disease_ml`
- supporting: `primary_purpose_ml`, `target_precedent_ml`, `innovation_tier_ml`, `therapeutic_modality_ml`, `sponsor_tier_ml`
- text: `summary_ui`, `conditions_ui`, `interventions_ui`
- iteration evidence: changes in phase, regulatory intent, indication, rarity, primary purpose, and score movement in Therapeutic Context

#### Scientific Challenge -> Endpoint & Evidence Strength

Fields:

- `target_precedent_ml`
- `target_pathway_class_ml`
- `therapeutic_modality_ml`
- `innovation_tier_ml`
- `intervention_model_ml`
- `primary_purpose_ml`
- `adaptive_design_ml`
- `endpoint_rigor_ml`
- `endpoint_structure_ml`
- `biomarker_stratification_ml`
- cross-pillar evidence fields: `masking_ml`, `allocation_ml`, `has_placebo_ml`, `comparator_benchmark_ml`, `primary_duration_months_ml`
- text: `primary_outcomes_ui`, `interventions_ui`, `summary_ui`, `biomarker_description`, `molecular_targets`

Adjustment asks:

- Does the endpoint/comparator setup produce interpretable evidence for the biological and clinical question?
- Is the endpoint mature enough for the disease and expected treatment effect?
- Is biomarker selection coherent with the target/modality and population?
- Does novelty increase uncertainty in a way the design addresses?
- Does adaptive design strengthen learning or just add complexity?

Contradictory examples:

- Completion improves after moving from hard clinical endpoint to surrogate endpoint, but evidence credibility may weaken.
- Completion worsens after adding randomization, masking, or an active comparator, but evidence credibility may improve.

Naming alternatives:

- `Endpoint & Evidence Strength`
- `Readout Credibility`
- `Evidence Interpretability`
- `Decision Evidence Strength`
- `Endpoint Confidence`

Current preferred concrete label: `Endpoint & Evidence Strength`.

Typical scoring evidence:

- primary: `endpoint_rigor_ml`, `endpoint_structure_ml`, `comparator_benchmark_ml`, `has_placebo_ml`, `masking_ml`, `allocation_ml`, `primary_duration_months_ml`
- supporting: `intervention_model_ml`, `adaptive_design_ml`, `biomarker_stratification_ml`, `target_precedent_ml`, `target_pathway_class_ml`, `therapeutic_modality_ml`, `innovation_tier_ml`, `number_of_arms_ml`
- text: `primary_outcomes_ui`, `summary_ui`, `interventions_ui`, `biomarker_description`, `molecular_targets`
- operational context: `planned_duration_months`, sometimes `planned_enrollment` when evidence precision or endpoint maturity is at issue
- iteration evidence: endpoint/comparator/bias-control changes and score movement in Scientific Challenge or Execution Framework

#### Patient Profile -> Target Population Alignment

Fields:

- `patient_severity_ml`
- `line_of_therapy_ml`
- `gender_ml`
- `healthy_volunteers_ml`
- `adult_ml`
- `child_ml`
- `older_adult_ml`
- cross-pillar context: `therapeutic_area_ml`, `gbd_cause_id_3_ml`, `is_rare_disease_ml`, `phase_ml`, `strategic_ambition_ml`, `biomarker_stratification_ml`
- text: `conditions_ui`, `summary_ui`, optionally `criteria_ui` if reintroduced

Adjustment asks:

- Does the population still represent the intended disease and unmet need?
- Does narrowing the population improve feasibility at the expense of relevance?
- Is severity/line of therapy appropriate for the phase and intervention?
- Are pediatric, geriatric, sex/gender, healthy-volunteer, or rare-disease choices clinically defensible?
- Is biomarker-based restriction justified by biology and development intent?

Contradictory examples:

- Completion improves after excluding children or severe/refractory patients, but patient/indication fit may weaken.
- Completion worsens after retaining a high-need population, but patient/indication fit may improve.

Naming alternatives:

- `Target Population Fit`
- `Patient Relevance`
- `Population Relevance`
- `Indication Population Fit`
- `Patient Need Fit`

Current preferred concrete label: `Target Population Fit`.

Typical scoring evidence:

- primary: `patient_severity_ml`, `line_of_therapy_ml`, `healthy_volunteers_ml`, `adult_ml`, `child_ml`, `older_adult_ml`, `gender_ml`
- supporting: `therapeutic_area_ml`, `gbd_cause_id_3_ml`, `is_rare_disease_ml`, `phase_ml`, `strategic_ambition_ml`, `biomarker_stratification_ml`, `endpoint_rigor_ml`
- text: `conditions_ui`, `summary_ui`, `criteria_ui` if reintroduced
- operational context: `planned_enrollment`, `planned_sites` when population restriction creates feasibility or representativeness trade-offs
- iteration evidence: demographic scope, severity, line-of-therapy, healthy-volunteer, indication, rarity, or biomarker-selection changes

#### Execution Framework -> Operational Burden Balance

Fields:

- `masking_ml`
- `allocation_ml`
- `has_dmc_ml`
- `has_placebo_ml`
- `comparator_benchmark_ml`
- `administration_complexity_ml`
- `number_of_arms_ml`
- `sponsor_tier_ml`
- `primary_duration_months_ml`
- operational assumptions: `planned_enrollment`, `planned_sites`, `planned_duration_months`
- cross-pillar context: `therapeutic_modality_ml`, `patient_severity_ml`, `phase_ml`, `strategic_ambition_ml`, `endpoint_rigor_ml`
- text: `interventions_ui`, `primary_outcomes_ui`, `summary_ui`

Adjustment asks:

- Is the operational footprint proportionate to the evidence ambition?
- Is the site/enrollment/duration plan credible for the design and patient population?
- Is oversight proportional to patient risk, modality, phase, and complexity?
- Does cost/complexity pressure appear justified by evidence or strategic gain?
- Does simplification reduce burden in a way that undermines decision value?

Contradictory examples:

- Completion improves after fewer arms, no placebo, shorter duration, and smaller enrollment, but delivery burden balance can worsen if the simplification makes the study less decision-useful.
- Completion worsens after adding arms, longer follow-up, or stronger oversight, but delivery burden balance can improve if the added burden is justified.

Naming alternatives:

- `Operational Burden Fit`
- `Execution Burden Balance`
- `Resource & Feasibility Fit`
- `Operational Proportionality`
- `Cost & Complexity Fit`

Current preferred concrete label: `Operational Burden Fit`.

Typical scoring evidence:

- primary operational assumptions: `planned_enrollment`, `planned_sites`, `planned_duration_months`
- primary structured fields: `administration_complexity_ml`, `number_of_arms_ml`, `primary_duration_months_ml`, `sponsor_tier_ml`, `has_dmc_ml`
- supporting: `masking_ml`, `allocation_ml`, `has_placebo_ml`, `comparator_benchmark_ml`, `intervention_model_ml`, `therapeutic_modality_ml`, `phase_ml`, `strategic_ambition_ml`, `patient_severity_ml`, `is_rare_disease_ml`, `endpoint_rigor_ml`
- text: `interventions_ui`, `primary_outcomes_ui`, `summary_ui`
- benchmark metadata: benchmark status/source/confidence for enrollment, sites, and duration
- iteration evidence: changes in enrollment, sites, duration, arms, delivery profile, endpoint duration, DMC/oversight, comparator, masking, and sponsor context

### Suggested Full Scenario Bar Semantics

Each current pillar can show current model subcategories plus the design-adjustment subcategory:

```text
Full pillar contribution = existing model subcategories + design-adjustment subcategory
```

Display example:

```text
Scientific Challenge
Biological Profile: +1.0
Protocol Architecture: +2.0
Endpoint & Evidence Strength adjustment: -1.5
Full scenario contribution: +1.5
Status: Trade-off
```

This gives participants a single familiar hierarchy while showing the contradictory stakes.

### Design Adjustment Evidence Rules

- A design adjustment should only move when the LLM cites supported packet evidence.
- Text-only concerns can move the adjustment only when they contradict or materially clarify structured fields.
- Operational assumptions can affect only `Operational Burden Balance` unless they directly create an endpoint/patient fit contradiction.
- Cost/complexity pressure can be positive only when the burden is proportionate and justified; lower cost is not automatically positive.
- Completion-improving simplification should be explicitly tested for evidence, patient-fit, and burden-balance trade-offs.

### Design Confidence Calibration Principle

Design Confidence should create thoughtful counterpressure, not a second dominant prediction model.

It should be able to partially help a trial with a difficult Completion Outlook when the difficulty appears related to legitimate development challenge, patient need, scientific ambition, or prudent risk governance rather than poor design quality.

Examples:

- A rare-disease or severe-population trial may have a lower Completion Outlook because recruitment and execution are difficult, but Design Confidence can add support if the target population, endpoint strategy, and operational assumptions are coherent.
- A complex modality or innovative mechanism may increase completion risk, but Design Confidence can recognize the design if the endpoint, biomarker logic, safety governance, and burden are proportionate.
- A longer or more controlled trial may be harder to complete, but Design Confidence can partially offset that when the added burden improves decision evidence.

It should also be able to challenge a very favorable Completion Outlook when the score appears favorable because the scenario became too easy, too narrow, too short, weakly controlled, or less decision-useful.

Examples:

- A high Completion Outlook from a smaller, shorter, less controlled scenario can be reduced if Endpoint & Evidence Strength weakens.
- A high Completion Outlook from a narrowed population can be reduced if Target Population Alignment weakens.
- A high Completion Outlook from lower operational burden can be reduced if the burden reduction is not proportionate to the evidence ambition.

Calibration guardrails:

- Design Confidence should normally be a bounded modifier, not an equal-weight replacement for Completion Outlook.
- Positive Design Confidence should not turn a very low Completion Outlook into a high Total Scenario Score by itself.
- Negative Design Confidence should not erase a strong Completion Outlook unless multiple supported concerns point to a material design shortcut.
- A neutral or typical operational profile should not add points by itself.
- A difficult design can receive positive adjustment only when the difficulty is justified by supported evidence, patient relevance, development logic, or risk governance.
- A favorable completion profile should be challenged only when supported evidence indicates reduced evidence value, patient relevance, strategic alignment, or burden proportionality.

Initial calibration hypothesis:

```text
Completion Outlook remains the dominant score.
Design Confidence adjustment is small-to-moderate.
Positive adjustment is harder to earn than a warning flag.
The main learning value is the trade-off explanation, not the numeric movement.
```

Recommended participant-facing state labels:

- `Favorable`
- `Watchlist`
- `High risk`
- `Trade-off`

These labels are easier for clinical operations and medical teams than abstract terms such as "quality" or "value". They can apply to each pillar and to the overall scenario.

Examples:

- Completion Outlook: `Favorable`, but Evidence Strength: `Watchlist`.
- Completion Outlook: `Watchlist`, but Strategic Fit: `Favorable`.
- Overall: `Trade-off`, because completion improved while evidence strength weakened.

## Proposed Trial Value Dimensions

### 1. Evidence Decisiveness

Question: would this scenario generate interpretable, decision-useful evidence?

Fields and evidence:

- `endpoint_rigor_ml`
- `endpoint_structure_ml`
- `comparator_benchmark_ml`
- `has_placebo_ml`
- `masking_ml`
- `allocation_ml`
- `adaptive_design_ml`
- `primary_duration_months_ml`
- `primary_outcomes_ui`
- `summary_ui`
- Completion Score movement for Scientific Challenge and Execution Framework, where available

What it should detect:

- stronger or weaker endpoint interpretability
- loss of assay sensitivity from weak comparator choices
- duration that no longer matches endpoint maturity
- simplification that raises completion likelihood but weakens the decision value of the readout
- text/structured contradictions around endpoints or comparator

Clinical anchors:

- ICH E8(R1) quality by design, critical-to-quality factors, meaningful endpoints, bias reduction, and fitness for purpose.
- ICH E9(R1) estimand thinking: alignment of objective, population, endpoint, treatment condition, intercurrent-event thinking, and interpretation.
- ICH E10 / control-group logic if comparator and placebo interpretation becomes a larger future module.

### 2. Population & Patient Relevance

Question: does the scenario still study the right patients for the intended development question?

Fields and evidence:

- `therapeutic_area_ml`
- `gbd_cause_id_3_ml`
- `is_rare_disease_ml`
- `patient_severity_ml`
- `line_of_therapy_ml`
- `healthy_volunteers_ml`
- `adult_ml`, `child_ml`, `older_adult_ml`, `gender_ml`
- `biomarker_stratification_ml`
- `conditions_ui`
- `summary_ui`
- `criteria_ui` if reintroduced later

What it should detect:

- narrowing that makes recruitment easier but weakens representativeness
- broadening that improves relevance but stresses operational feasibility or endpoint clarity
- pediatric, elderly, rare-disease, sex/gender, or severity mismatches
- biomarker choices that are either mechanistically justified or unnecessarily restrictive

Clinical anchors:

- ICH E8(R1) patient input and patient-relevant endpoints.
- FDA diversity and eligibility guidance as a context source.
- Disease-specific evidence packs for high-value TAs once available.

### 3. Scientific & Mechanistic Rationale

Question: is the intervention, target, modality, biomarker, and phase logic scientifically credible?

Fields and evidence:

- `therapeutic_modality_ml`
- `target_precedent_ml`
- `target_pathway_class_ml`
- `innovation_tier_ml`
- `biomarker_stratification_ml`
- `phase_ml`
- `primary_purpose_ml`
- `interventions_ui`
- `summary_ui`

What it should detect:

- novel target or modality needing stronger design support
- biomarker-positive strategy that fits a targeted mechanism
- biologic, gene therapy, cell therapy, vaccine, or complex modality needing proportional safety/operational support
- phase/design mismatch, such as confirmatory posture before adequate exploratory learning

Clinical anchors:

- ICH E8(R1) drug-development lifecycle and state-of-knowledge principle.
- FDA Project Optimus for oncology dose-optimization context where relevant.
- Future modality packs for ATMP, vaccines, immunology, neurology, rare disease, and oncology.

### 4. Regulatory & Development Strategy Fit

Question: does the trial fit the implied regulatory and development decision?

Fields and evidence:

- `phase_ml`
- `strategic_ambition_ml`
- `primary_purpose_ml`
- `sponsor_tier_ml`
- `adaptive_design_ml`
- `comparator_benchmark_ml`
- `endpoint_rigor_ml`
- `biomarker_stratification_ml`
- `summary_ui`

What it should detect:

- confirmatory/regulatory ambition paired with weak endpoint/comparator choices
- exploratory scenario presented as registration-enabling without support
- adaptive design that is appropriate versus decorative
- regional or US-regulatory implications when the available field supports it

Clinical anchors:

- ICH E8(R1) and E9(R1) alignment of objectives, design, analysis, and interpretation.
- FDA and EMA guidance context relevant to design elements, DCT/RWE/adaptive designs, diversity, and evidence standards.

### 5. Operational Deliverability & Risk Governance

Question: can the design be executed credibly without undermining evidence value or participant protection?

Fields and evidence:

- `planned_enrollment`
- `planned_sites`
- `planned_duration_months`
- operational benchmark status/source/confidence
- `administration_complexity_ml`
- `number_of_arms_ml`
- `has_dmc_ml`
- `intervention_model_ml`
- `masking_ml`
- `allocation_ml`
- `sponsor_tier_ml`
- `patient_severity_ml`
- `therapeutic_modality_ml`
- `interventions_ui`

What it should detect:

- ambitious enrollment/sites/duration that are supported versus implausible
- operationally easy scenarios that reduce evidence value
- oversight mismatch, such as high-risk population or complex modality without proportional risk governance
- excessive burden that may threaten completion or participant retention

Clinical anchors:

- ICH E6(R3) risk-proportionate GCP, fit-for-purpose trial conduct, data integrity, and participant protection.
- ICH E8(R1) critical-to-quality operational factors.

## Completion Movement Analysis

This should become a first-class prompt section and output object separate from the value dimensions.

Inputs:

- baseline score and decomposition
- previous score and decomposition
- current score and decomposition
- changed structured fields
- changed text fields
- changed operational assumptions, flagged as non-XGBoost
- feature impact movement
- subcategory impact movement
- pillar impact movement

Required reasoning:

1. Identify the model-facing fields that changed.
2. Separate direct changed-feature effects from broader subcategory/pillar movement.
3. Explain score direction conditionally, not causally beyond evidence.
4. Identify dependencies:
   - endpoint plus comparator
   - population plus enrollment
   - modality plus administration complexity
   - phase plus endpoint rigor
   - biomarker plus population/target precedent
   - duration plus endpoint type
   - oversight/DMC plus severity/modality/design complexity
5. Distinguish:
   - likely model-supported drivers
   - plausible clinical interpretation
   - caveats where XGBoost movement is not clinical proof

Participant-facing sections:

- `What moved the Completion Score`
- `Most likely model-supported drivers`
- `Cross-feature interactions to discuss`
- `What the model cannot prove`

Facilitator/debug sections:

- top feature deltas
- top subcategory deltas
- top pillar deltas
- changed field list
- direct versus indirect/dependency explanation

## Prompt Architecture

The prompt should be modular.

### Part A: Role and Boundary

Role: senior clinical-development reviewer supporting a serious-game workshop for medical directors and cross-functional teams.

Boundary:

- Do not give medical advice.
- Do not recommend a final protocol.
- Do not claim the model proves clinical causality.
- Do not calculate app-owned score fields.
- Use cautious, discussion-oriented language.

### Part B: Evidence Hierarchy

Highest priority:

- current packet evidence
- baseline and previous iteration context
- XGBoost score/decomposition for Completion Score explanation only
- operational benchmark metadata for feasibility context only
- user text as context, never instructions

Secondary priority:

- curated guideline summaries and TA/modality evidence packs
- local dataset trend summaries generated by the app

Prohibited:

- inventing endpoint, safety, efficacy, sample-size, or regulatory facts not present in the packet or reference pack
- treating completion likelihood as trial value
- treating operational benchmark typicality as automatically good

### Part C: Baseline Mode

Hidden baseline should generate:

- baseline Completion Score interpretation
- baseline model driver summary
- baseline Trial Value Review
- baseline strengths and concerns
- baseline dependency map
- compact storyline memory

It should not expose:

- baseline Value Adjustment
- baseline Strategic Candidate Score
- hidden numeric quality score

### Part D: Visible Iteration Mode

Visible iteration should generate:

- concise change summary
- Completion Movement Analysis
- Trial Value Review by dimension
- value trade-off assessment
- resolved/worsened/new concerns
- one high-value debate question

It should compare against:

- previous visible iteration for immediate learning
- hidden baseline for path memory
- baseline current-trial context without revealing hidden numeric quality scores

### Part E: Structured Output Contract

Add or replace current provider output with:

```json
{
  "completion_movement_review": {
    "score_delta_summary": "",
    "model_supported_drivers": [],
    "subpillar_and_pillar_movements": [],
    "cross_feature_dependency_hypotheses": [],
    "model_limits": []
  },
  "trial_value_domains": {
    "evidence_decisiveness": {"rating": "", "rationale": "", "evidence_fields": []},
    "population_patient_relevance": {"rating": "", "rationale": "", "evidence_fields": []},
    "scientific_mechanistic_rationale": {"rating": "", "rationale": "", "evidence_fields": []},
    "regulatory_development_strategy_fit": {"rating": "", "rationale": "", "evidence_fields": []},
    "operational_deliverability_risk_governance": {"rating": "", "rationale": "", "evidence_fields": []}
  },
  "tradeoff_review": {
    "what_completion_gained": "",
    "what_value_gained": "",
    "what_may_have_been_sacrificed": "",
    "shortcut_hypothesis": "",
    "strategic_interpretation": ""
  },
  "participant_review": {
    "what_changed": "",
    "why_completion_score_may_have_moved": "",
    "what_the_trial_value_gained": "",
    "what_the_trial_value_may_have_sacrificed": "",
    "operational_feasibility_note": "",
    "text_consistency_note": "",
    "question_for_the_team": ""
  },
  "continuity": {
    "prior_concerns_resolved": [],
    "prior_concerns_worsened": [],
    "prior_concerns_unchanged": [],
    "new_concerns": [],
    "storyline_update": ""
  },
  "trace": {
    "main_features_considered": [],
    "main_completion_drivers_considered": [],
    "main_value_dimensions_considered": [],
    "reference_pack_ids_used": [],
    "compared_against": ""
  }
}
```

## Scoring Recommendation

Use the Trial Value Review for explanation first. Add a numeric adjustment only after examples show it is stable.

If numeric scoring is kept:

- app calculates it, not the LLM
- each value dimension maps from validated rating plus supported evidence
- total adjustment should be bounded and secondary
- start with `-10` to `+8`, because serious simplification should be easier to penalize than excellence is to prove
- no positive adjustment for "typical" operational assumptions alone
- no positive adjustment for unchanged baseline design alone
- positive value requires participant-introduced improvement or a strongly defensible difficult design

Candidate rating scale:

- `value_strengthening`: +2
- `credible_support`: +1
- `neutral_or_balanced`: 0
- `unresolved_tension`: -1.5
- `material_value_risk`: -3

Shortcut-specific override:

- A scenario with a positive Completion Score delta and material negative value movement should receive an explicit trade-off flag.
- The app should not automatically punish every positive Completion Score delta. It should only flag shortcut risk when evidence fields support a loss in trial value.

## Reference and Evidence Assets

Create small curated reference packs, not a broad RAG system first.

Use a hybrid document strategy:

1. Always attach a very short `core_clinical_development_v1` summary pack.
2. Conditionally attach one or two feature-triggered packs only when the packet justifies them.

Do not attach full source documents to every prompt. Full guidance documents are too long, increase cost and latency, can drown out packet evidence, and may bias the review toward issues that are not actually present in the scenario.

Runtime prompt rule:

```text
Packet evidence first.
Core summary pack second.
Conditional packs third.
Never let a guidance pack override the actual scenario fields.
```

Free-access / usage rule:

- Use public, freely accessible official sources.
- Store short internal summaries and citations, not copied full guidance text.
- Keep summaries versioned and source-linked.
- Avoid long verbatim excerpts in prompts or app output.
- Treat guidance as non-binding context unless the source states a requirement.

Core always-on pack:

- ICH E8(R1) general considerations for clinical studies: quality by design, critical-to-quality factors, fit-for-purpose design, patient relevance, and study objective/design alignment.
- ICH E6(R3) GCP Principles and Annex 1: risk-proportionate trial conduct, participant protection, data integrity, operational feasibility, and sponsor oversight.
- ICH E9(R1) estimands: alignment of objective, population, treatment condition, endpoint, intercurrent events, analysis, and interpretation.
- FDA clinical trials guidance inventory: source index only, used to identify optional FDA packs without embedding all guidance by default.

This core pack should be phase-agnostic enough for Phase II and Phase III trials. The prompt should instruct the model to scale expectations by `phase_ml` and `strategic_ambition_ml`, so Phase II is not judged by automatic Phase III confirmatory standards and Phase III is not treated like exploratory signal seeking.

Conditional pack triggers:

- Endpoint / patient outcome pack: attach when endpoint text, endpoint rigor, PRO/subjective endpoint signals, or patient-relevance tension is present. Candidate source: FDA Patient-Focused Drug Development clinical outcome assessment guidance.
- Diversity / eligibility pack: attach when age, sex/gender, pediatric, older-adult, healthy-volunteer, rare-disease, or population-scope fields materially change. Candidate source: FDA eligibility/diversity guidance from the FDA clinical-trials guidance inventory.
- Decentralized / digital / operational modernization pack: attach only when future fields support decentralized elements, digital health technologies, or remote assessments. Do not attach by default today.
- Adaptive design pack: attach when `adaptive_design_ml` changes or when adaptive design is central to the scenario. Candidate sources: FDA adaptive design guidance and ICH E20 when mature.
- UCB focus pack: attach only for workshop personalization or disease-area context, not for scoring. Candidate sources: UCB public disease area and pipeline pages for neurology, immunology, rare disease, epilepsy/rare syndromes, dermatology, and rheumatology.

Avoid source packs that are too TA-specific unless the trial and workshop require them. The V1 product should remain usable across Phase II/III trials without assuming oncology, rare disease, or UCB-specific standards unless those are explicitly relevant.

Candidate pack IDs:

- `ich_e8_quality_by_design`
- `ich_e6_r3_risk_proportionate_gcp`
- `ich_e9_estimands`
- `fda_clinical_trials_guidance_index`
- `endpoint_patient_outcome_fit`
- `control_group_and_comparator_logic`
- `diversity_patient_relevance`
- `adaptive_design_context`
- `ucb_focus_context`

TA/modality packs to consider next:

- immunology/inflammation
- neurology
- rare disease
- oncology
- vaccines/infectious disease
- ATMP/cell/gene therapy

Each pack should contain:

- short human-authored summary
- source URLs
- applicability tags
- prompt-safe bullet principles
- "do not infer" cautions
- fields it can support

Do not let the LLM browse live during gameplay. Runtime should use fixed versioned packs.

## Local Data Enrichment

Use `data/data_clinpred.csv` to create empirical context summaries.

Initial data facts from planning inspection:

- 34,066 rows.
- `start_year` range: 2009 to 2026.
- key available fields include trial status, phase, TA, indication, modality, target, biomarker, endpoint, comparator, demographics, operational fields, text fields, and scientific-success labels.
- recent-year completion status is censored because many recent trials are ongoing, so raw completion-rate trends must be mature-cohort adjusted.

Planned analyses:

1. Mature-cohort completion/termination status by start year, phase, TA, modality, and endpoint/comparator profile.
2. Design-pattern evolution over time:
   - biomarker stratification
   - endpoint rigor
   - endpoint structure
   - comparator type
   - adaptive design
   - DMC use
   - placebo/control use
   - phase/modality shifts
3. Operational benchmark enrichment:
   - enrollment, site, duration distributions by phase/TA/rare/modality/strategic intent
   - compare user scenario to mature historical cohort, not immature recent cohorts
4. Failure-pattern summaries:
   - terminated/withdrawn patterns by field combination
   - `why_stopped` text clustering if sufficiently clean
5. Evidence-pack hooks:
   - generate compact field-specific statistics that the LLM can cite as local context
   - version the summaries and include them in prompt trace

Output artifact proposal:

- `frontend/data/narrative_context_stats_v1.json`
- `scripts/build_narrative_context_stats.py`
- `scripts/check_narrative_context_stats.py`

## Implementation Phases

### Phase 1: Decide the Rubric

Deliverables:

- finalize names for Trial Value dimensions
- decide whether to keep `Quality Review` or rename to `Trial Value Review`
- decide whether to keep numeric adjusted score in V1
- update `docs/architecture_narratives.md`

No code behavior change.

### Phase 2: Contract Fixtures

Deliverables:

- replace or extend narrative fixtures with scenarios that test the new dimensions
- include at least:
  - score improves but evidence value weakens
  - score improves and value improves
  - score declines but value improves
  - operational-only edit
  - endpoint text contradiction
  - biomarker/population mismatch
  - regulatory ambition versus weak design
  - modality/risk-governance mismatch

Verification:

- fixture checker
- py_compile
- `git diff --check`

### Phase 3: Packet Builder

Deliverables:

- add richer Completion Movement Analysis inputs
- ensure feature, subcategory, and pillar deltas are explicit and separated
- add dependency-context fields where deterministic
- add selected reference-pack IDs and local context-stat IDs

Verification:

- packet checker
- live snapshot flow checker
- fixture checker

### Phase 4: Prompt and Schema

Deliverables:

- modular provider prompt with baseline/visible modes
- new JSON schema for `completion_movement_review` and `trial_value_domains`
- stronger instructions for expert clinical-development language
- source hierarchy and "do not infer" rules

Verification:

- prompt checker
- provider schema checker
- mock provider compatibility
- one opt-in Gemini smoke on a fixed packet

### Phase 5: Deterministic Validation and Scoring

Deliverables:

- replace `quality_review_domains` with `trial_value_domains`, or provide a compatibility adapter
- app-owned mapping from value-domain ratings to optional Value Adjustment
- supported-evidence enforcement retained
- no app-owned score fields accepted from provider

Verification:

- scoring checker
- malformed/partial response checker
- no stale adjustment reuse

### Phase 6: Reference Packs and Data Context

Deliverables:

- cross-cutting reference packs
- local mature-cohort context stats
- version IDs included in packet and trace
- no runtime web browsing

Verification:

- JSON/schema checks
- deterministic rebuild check
- sample packet inspection

### Phase 7: UI Redesign

Deliverables:

- separate `Completion Movement` panel from `Trial Value Review`
- show dimensions as a compact value profile, not as XGBoost sub-pillars
- keep Completion Score drivers visually distinct from Value Review dimensions
- show trade-off flag when completion improves but value weakens
- keep facilitator/debug details behind an expander

Verification:

- Streamlit simulator smoke
- screenshot check on desktop
- representative trial scenarios

### Phase 8: Calibration and Playtesting

Deliverables:

- 10 to 20 representative scenarios
- expert review notes
- rating-to-adjustment calibration
- prompt refinement
- final decision on `Strategic Candidate Score`

Verification:

- repeatability across same packet/provider settings
- latency and token-cost diagnostics
- no impact on XGBoost parity path

## External Source Anchors Checked During Planning

- ICH E8(R1), adopted 2021, frames clinical-study quality as fitness for purpose, quality by design, critical-to-quality factors, patient-relevant endpoints, bias reduction, and operational feasibility.
- EMA page for ICH E6(R3), current in 2025/2026, emphasizes risk-based and proportionate GCP, fit-for-purpose conduct, trial design innovation, data integrity, and participant protection.
- FDA E9(R1) guidance frames estimands as a structured way to align objectives, design, conduct, analysis, and interpretation.
- FDA Project Optimus is a useful oncology-specific reference for dose optimization and benefit/risk trade-offs, especially where completion likelihood could conflict with tolerability or dose-selection quality.
- FDA clinical-trials guidance inventory provides current official anchors for diversity, decentralized trials, externally controlled trials, adaptive designs, eligibility, and other design topics.

## Open Decisions For Brainstorming

1. Should the participant-facing term be `Trial Value Review`, `Design Review`, `Strategic Review`, or keep `Quality Review`?
2. Should the adjusted number remain visible, or should V1 show Completion Score plus a nonnumeric value profile first?
3. Should the five proposed value dimensions be reduced to four for UI simplicity?
4. Which TA/modality reference packs matter first for UCB-like workshops?
5. Should cost be included in V1? Current recommendation: yes, but only as `Cost / Complexity Pressure`, not exact budget. Current fields support relative resource pressure from enrollment, sites, duration, arms, modality, administration complexity, oversight, masking, control structure, phase, sponsor type, and endpoint duration. True budget should wait until country, site, visit, procedure, and vendor assumptions exist.
6. Should regulatory strength be a standalone dimension or merged into development strategy fit? Current recommendation is standalone within `Regulatory & Development Strategy Fit`.
7. Should patient/participant burden be its own dimension? Current recommendation is to include it inside population relevance and operational/risk governance until more fields exist.

## Next Step

Use this plan as the brainstorming agenda. First decision to make: confirm or revise the five Trial Value dimensions before any code changes.
