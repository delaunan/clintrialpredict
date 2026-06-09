# Implementation Plan: Narrative Dimension Redesign

## Scope

Architecture scope: `architecture_narratives`, with secondary read-only inputs from `architecture_edit`, `architecture_estimation`, and `core_scoring`.

Goal: redesign the serious-game Quality Review so it gives medical-director-grade interpretation of trial scenario changes. The review should explain Completion Score movement from XGBoost evidence, then evaluate complementary trial-value dimensions beyond completion likelihood.

This plan replaces the prior narrative next-step plan about the lightweight consistency check. The consistency check may still be useful, but it becomes one subtask inside the broader dimension redesign.

## Active Implementation Model

This is the current implementation target. Older five-domain `Trial Value Review` ideas in prior commits are superseded by this four-pillar Design Confidence model.

Participant-facing hierarchy:

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

Analytical provenance:

```text
Completion Outlook = model-derived score and pillar/subcategory movement.
Design Confidence = evidence-backed review adjustment, bottom-up from the four design subcategories.
```

Scoring target:

```text
Default design adjustment = 0.0
Non-zero adjustment requires supported packet evidence
Each design subcategory = -4.0 to +4.0 in 0.5 increments
Typical subcategory movement = -2.5 to +2.5
Total Design Confidence = sum of four design subcategories
No hidden total cap
```

Participant UI may show one integrated four-pillar chart. Trace/facilitator outputs must preserve the split between Completion Outlook evidence and Design Confidence adjustment.

## Product Framing

The simulator should answer two different questions:

1. `Will this trial complete?`
   - Owned by the existing XGBoost Completion Score.
   - Explained through feature, subcategory, pillar, and score movement.
   - No LLM scoring should alter this value.

2. `Is this a strategically valuable and defensible trial design?`
   - Owned by the `Scenario Review` / `Design Confidence` layer.
   - Uses XGBoost fields, non-XGBoost operational assumptions, text changes, baseline context, previous iteration context, and curated clinical-development references.
   - Can highlight that a scenario improved completion likelihood by reducing scientific, regulatory, population, operational, or evidence value.

Recommended user-facing naming:

- Keep `Completion Score`.
- Use `Completion Outlook` when explaining the model-derived score movement.
- Use `Scenario Review` or `Design Confidence Review` for the narrative panel.
- Use `Design Confidence` for the app-owned design-adjustment layer.
- Use `Total Scenario Score` only if the combined view is activated.

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

Do not conceptually merge Design Confidence into the XGBoost Completion Score.

Reason: the XGBoost pillars explain modeled early-termination/completion likelihood. Design Confidence evaluates a different construct: decision usefulness and strategic defensibility. The participant UI may integrate design-adjustment subcategories into the same four-pillar chart for readability, but trace/facilitator outputs must preserve the conceptual split so users do not think the model itself has learned cost, regulatory strength, scientific rigor, or patient relevance.

Historical internal reasoning structure from the first planning pass:

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

This five-domain version is superseded as an implementation target. It remains useful only as historical reasoning behind the final four design subcategories.

Superseded simplified UI structure considered during brainstorming:

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

In this superseded model, `Completion Outlook` was the current XGBoost Completion Score contribution layer. The three additional pillars explained what the scenario added or sacrificed beyond pure completion likelihood. This helped the design discussion but is no longer the active implementation target.

- `Completion only`: current XGBoost Completion Score and existing pillars.
- `Full scenario`: Completion Outlook plus Evidence Strength, Strategic Fit, and Delivery Confidence.

The enduring UX lesson from this option is that the same field should not appear as two unrelated concepts with confusing names.

Superseded even simpler structure:

```text
Total Scenario Score
├── Completion Outlook
├── Evidence Strength
├── Strategic Fit
└── Delivery Confidence
```

This option is superseded. Facilitator/debug mode can show deeper evidence and provenance, but should use the final four Design Confidence subcategories rather than reintroducing superseded models.

Superseded feature-density option from the middle of the planning discussion:

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

This option is not the active implementation target. It remains useful only as rationale for why the final active model keeps a small number of design subcategories and avoids over-fragmenting the participant view:

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

Superseded V1 radio option considered during brainstorming:

- `Completion outlook`: current Completion Score and the four XGBoost pillars.
- `Full scenario`: Completion Outlook plus Evidence & Strategy Strength plus Delivery & Investment Burden.

This helped clarify the need for a simple combined bar chart, but the active implementation target is the four familiar Completion Outlook pillars with one Design Confidence subcategory under each pillar.

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

## Reviewer Constitution

The LLM reviewer should behave like a senior clinical development reviewer supporting a serious-game workshop for medical directors, clinical operations, and cross-functional development teams.

The reviewer is not an optimizer and not a protocol author. It should not tell the participant which field to change next. Its job is to interpret the scenario, expose trade-offs, and ask high-value questions.

### Reviewer Role

The reviewer should:

- explain why the Completion Outlook moved using model-provided evidence
- distinguish model-supported movement from clinical-development interpretation
- analyze how one changed feature can create collateral movement in other pillars
- assess Design Confidence only where supported by packet evidence
- challenge shortcut-like simplification when evidence supports it
- recognize difficult but defensible design choices when evidence supports it
- use cautious, conditional language
- end with two expert questions that support discussion

The reviewer should not:

- claim XGBoost proves clinical causality
- invent protocol details not present in structured fields, text context, operational assumptions, documents, or database summaries
- calculate app-owned score fields directly
- use documents as generic authority when packet evidence is insufficient
- reward or penalize a scenario only because the Completion Outlook is high, low, rising, or falling
- recommend a specific edit as the answer

### Evidence Hierarchy

The reviewer should use evidence in this order:

```text
1. Current packet evidence
   - current structured fields
   - current text context
   - current operational assumptions
   - current Completion Score and decomposition

2. Scenario movement evidence
   - changes from previous iteration
   - changes from baseline
   - score delta
   - pillar/subcategory/feature impact deltas

3. Baseline and storyline context
   - original trial profile
   - hidden baseline strengths/concerns
   - prior visible review concerns

4. Operational benchmarks
   - enrollment/site/duration source
   - benchmark position
   - confidence flags

5. Curated document summaries
   - ICH/FDA/UCB/context packs
   - used as context only, not as standalone scoring evidence

6. Local database statistics
   - mature-cohort summaries
   - design-pattern summaries
   - used as context only, not causal proof
```

If a claim cannot be supported by one of these evidence layers, the reviewer should state that the issue is uncertain or not assessable.

### XGBoost Interpretation Rules

The Completion Outlook explanation must be a serious analysis of model movement, not a superficial list of changed fields.

The reviewer should analyze:

- which pillars moved most
- which subcategories moved within those pillars
- which changed fields are directly mapped to the moved pillar
- which unchanged fields may contextualize the movement
- whether a changed field in one pillar plausibly explains movement or tension in another pillar
- whether movement is from the previous iteration, from baseline, or both
- whether the model movement and clinical interpretation point in the same direction
- where the model output cannot establish clinical meaning

The reviewer should never say a feature "caused" the XGBoost score to move unless the packet explicitly supports that through feature/subcategory deltas. Preferred wording:

```text
The model appears to respond to...
One model-supported driver is...
One plausible clinical interpretation is...
This movement may reflect...
The packet does not prove...
```

### Feature Movement and Collateral Impact Logic

The reviewer should reason beyond "field changed, score changed." A single field change can alter the interpretation of another pillar, even when it is not directly owned by that pillar.

Examples:

- Changing `endpoint_rigor_ml` may move Scientific Challenge directly, but it may also create Execution Framework pressure if the endpoint requires longer duration, more sites, or more complex follow-up.
- Changing `child_ml` may move Patient Profile directly, but it may also affect Operational Burden Balance through recruitment difficulty, site specialization, consent/assent complexity, and oversight needs.
- Changing `strategic_ambition_ml` may move Therapeutic Context directly, but it changes how endpoint, comparator, population, and operational assumptions should be judged.
- Changing `comparator_benchmark_ml` may move Execution Framework directly, but it also affects Endpoint & Evidence Strength because comparator credibility changes interpretability.
- Changing `biomarker_stratification_ml` may sit in Scientific Challenge, but it can affect Target Population Alignment by narrowing the target population and affecting feasibility.
- Changing `planned_enrollment` does not enter XGBoost, but it can affect Operational Burden Balance and can contextualize whether population or endpoint changes are credible.

The reviewer should explicitly identify collateral impacts when they matter:

```text
Direct movement:
- the changed field belongs to this model pillar or subcategory.

Collateral impact:
- the changed field belongs elsewhere, but it changes the interpretation of this pillar's design-adjustment subcategory.
```

### XGBoost and Design Review Separation

The reviewer should keep two layers conceptually separate even if the participant UI merges them into four familiar pillars:

```text
Completion Outlook
= model-derived score and pillar/subcategory movement

Design Confidence
= evidence-backed design interpretation and adjustment
```

Participant-facing text can be simple, but the trace should preserve provenance:

```text
Model-supported movement:
Design-review adjustment:
Evidence fields:
Collateral impact:
Uncertainty:
```

### Design Review Scoring Rules

The Design Confidence subpillars are:

```text
Therapeutic Context -> Phase & Intent Alignment
Scientific Challenge -> Endpoint & Evidence Strength
Patient Profile -> Target Population Alignment
Execution Framework -> Operational Burden Balance
```

Design review scoring is bottom-up and evidence-backed:

```text
Default adjustment = 0.0
Non-zero adjustment requires supported packet evidence
Each design-adjustment subcategory = -4.0 to +4.0 in 0.5 increments
Typical subcategory movement = -2.5 to +2.5
Total Design Confidence = sum of the four subcategories
No hidden total cap
```

Positive adjustment requires evidence that the scenario strengthens or preserves design confidence. Negative adjustment requires evidence that the scenario weakens design confidence. No adjustment should be applied when the evidence is unclear, design-neutral, balanced, or unsupported.

### Required Analytical Depth

For every visible iteration, the reviewer should produce analysis at three levels:

```text
1. Movement level
   - What changed from previous and baseline?
   - Which score/pillars/subcategories moved?

2. Mechanism level
   - Why might the Completion Outlook have moved?
   - Which feature movements are direct model evidence?
   - Which cross-pillar interactions are plausible?

3. Design judgment level
   - Does the movement strengthen or weaken Phase & Intent Alignment?
   - Does it strengthen or weaken Endpoint & Evidence Strength?
   - Does it strengthen or weaken Target Population Alignment?
   - Does it strengthen or weaken Operational Burden Balance?
```

The reviewer should not force a design adjustment just to make the analysis interesting. Rich analysis can end with `0.0` adjustment if the evidence does not support movement.

### Participant Voice

Participant-facing language should be clinical-development plain English:

- concise
- evidence-linked
- conditional
- discussion-oriented
- not overconfident
- not instruction-like

Preferred wording:

```text
This may suggest...
One interpretation is...
The model movement is consistent with...
The design review flags...
The packet does not show enough evidence to conclude...
The team may want to debate...
```

Avoid:

```text
You should...
The correct choice is...
The model proves...
This will succeed/fail because...
The design adjustment penalizes/rewards the score...
```

### Document Use Rules

Documents should not drive the review by themselves. They should help the reviewer reason when the packet contains relevant signals.

Always:

- summarize document principles, do not paste full source text
- cite document pack IDs in trace
- use documents to explain why a trade-off matters
- scale interpretation by phase and development intent

Never:

- attach broad documents as a substitute for packet evidence
- impose Phase III standards on Phase II signal-seeking trials
- use oncology-specific logic for non-oncology trials unless the trial context supports it
- let UCB personalization change the scoring logic

### Expert Questions

The final two questions should be grounded in the actual scenario movement:

1. Medical / development question:
   - about evidence strength, patient relevance, phase/intent, endpoint interpretability, or strategic defensibility.
2. Clinops / execution question:
   - about enrollment, sites, duration, oversight, complexity, cost pressure, or feasibility versus evidence ambition.

The questions should reference the scenario's main trade-off and should not suggest the answer.

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
   - top design-confidence evidence signal

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

Design Confidence should be an evidence-backed design judgment, not a balancing mechanism and not a second dominant prediction model.

Default rule:

```text
Design adjustment starts at 0.0.
Non-zero adjustment requires explicit supported packet evidence.
```

The adjustment should never be applied simply because the Completion Outlook is high or low. It should be applied only when the review can identify a specific design reason.

Positive adjustment requires evidence that the scenario strengthens or preserves design confidence, such as:

- more decision-useful endpoint or comparator logic
- clearer phase/intent alignment
- more relevant target population
- coherent biomarker or mechanism strategy
- governance proportional to patient, modality, or trial risk
- operational burden justified by evidence or strategic gain

Negative adjustment requires evidence that the scenario weakens design confidence, such as:

- weaker endpoint, comparator, masking, or allocation credibility
- population narrowing that harms patient or indication relevance
- phase/intent mismatch
- operational assumptions that are no longer credible
- burden reduction that undermines evidence ambition
- risk governance that no longer matches patient, modality, or trial risk

No-adjustment cases are important. Keep adjustment at `0.0` when:

- the score changed but the design implication is unclear
- the change is model-relevant but design-neutral
- the trade-off is balanced
- the packet does not contain enough evidence
- the low score reflects risk but not clearly better rigor or design quality
- the high score reflects feasibility but not clearly weakened design

Calibration guardrails:

- Design Confidence should remain subordinate to Completion Outlook and should not behave like an independent prediction model.
- Positive Design Confidence should not create artificial hope for a low Completion Outlook without specific design-strength evidence.
- Negative Design Confidence should not create artificial penalty for a high Completion Outlook without specific design-concern evidence.
- A neutral or typical operational profile should not add points by itself.
- A difficult scenario can receive positive adjustment only when supported evidence shows the difficulty is justified by rigor, patient relevance, development logic, or risk governance.
- A favorable scenario can receive negative adjustment only when supported evidence indicates reduced evidence value, patient relevance, strategic alignment, or burden proportionality.

Initial calibration hypothesis:

```text
Completion Outlook remains the dominant score.
Design Confidence adjustment is usually zero or small unless supported evidence is strong.
Positive and negative adjustment both require explicit evidence.
The main learning value is the trade-off explanation, not the numeric movement.
```

### Empirical Score-Scale Calibration

Current registry evidence from `frontend/data/search_registry.csv`:

```text
Scored trials: 5,890
Clinical_Score range: 16.6 to 99.0
Clinical_Score median: 49.0
Clinical_Score mean / std: 50.9 / 15.2

Clinical_Score percentiles:
P05 29.9
P10 33.1
P25 39.5
P50 49.0
P75 59.9
P90 72.5
P95 79.8
P99 91.4
```

Current zone thresholds:

```text
High Risk: <= 25
Watchlist: >25 to <=50
Favorable: >50 to <=75
Low Risk: >75
```

Current pillar-impact scale:

```text
Median absolute pillar contribution: about 4 to 5 points.
75th percentile absolute pillar contribution: about 6 to 8 points.
90th percentile absolute pillar contribution: about 9 to 10 points.
95th percentile absolute pillar contribution: about 10 to 13 points.
```

Implication:

```text
A total Design Confidence range of only -4 to +4 may be too small to be visible next to existing pillar impacts.
The score should therefore be calibrated at the design-adjustment subcategory level.
The total range should emerge additively from the four subcategory scores.
```

Recommended V1 numeric envelope:

```text
Each design-adjustment subcategory:
  -4.0 to +4.0 in 0.5 increments

Typical per-subcategory range:
  -2.5 to +2.5
```

Rationale:

- The total Design Confidence adjustment should be the bottom-up sum of the four design-adjustment subcategories.
- There should not be a separate hidden hard cap that breaks additivity.
- If each subcategory is bounded at `-4.0` to `+4.0`, the theoretical total range is `-16.0` to `+16.0`.
- In practice, most scenarios should stay closer to `-8.0` to `+8.0` because only one or two subcategories should usually move strongly.
- `+/-10` to `+/-16` should require multiple strongly supported subcategory movements, not a top-down override.
- The adjustment should not routinely move a scenario across two risk zones, but this should be controlled through subcategory scoring discipline rather than a non-additive total clamp.

Recommended subcategory scoring discipline:

```text
If a Completion Outlook pillar is already strongly positive:
  positive design adjustment for that same pillar should usually remain 0.0 or +0.5 unless the packet shows a specific unresolved baseline concern was improved.

If a Completion Outlook pillar is neutral or negative:
  positive design adjustment can be larger only when supported evidence shows the risk is caused by rigor, patient relevance, scientific ambition, or prudent governance.

If Completion Outlook rises sharply:
  negative Design Confidence can moderate the increase only when supported evidence suggests shortcut behavior or weakened design confidence.

If Completion Outlook falls sharply:
  positive Design Confidence can moderate the decrease only when supported evidence suggests the added risk comes from better evidence, broader patient relevance, or proportionate governance.
```

This avoids a mechanical score dependency. The adjustment is not based on the score level alone; it is based on specific supported design evidence.

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

## Active Prompt and Structured Output Target

The active provider prompt should be built from the Reviewer Constitution, the Review Input Evidence Model, and the final four design subcategories. Older five-domain `trial_value_domains` contracts are superseded.

The prompt should be modular:

```text
1. Reviewer constitution and evidence hierarchy
2. Scenario packet JSON
3. Selected reference-pack summaries
4. Strict output schema
```

The provider should return structured analysis, ratings, rationale, evidence fields, and two expert questions. It must not return app-owned final score values.

Target conceptual output:

```json
{
  "completion_outlook_review": {
    "score_delta_summary": "",
    "pillar_movement_summary": [],
    "model_supported_drivers": [],
    "cross_pillar_interaction_hypotheses": [],
    "model_limits": []
  },
  "design_confidence_subcategories": {
    "phase_intent_alignment": {"rating": "", "rationale": "", "evidence_fields": []},
    "endpoint_evidence_strength": {"rating": "", "rationale": "", "evidence_fields": []},
    "target_population_alignment": {"rating": "", "rationale": "", "evidence_fields": []},
    "operational_burden_balance": {"rating": "", "rationale": "", "evidence_fields": []}
  },
  "pillar_reviews": {
    "therapeutic_context": {
      "completion_interpretation": "",
      "design_adjustment_interpretation": "",
      "collateral_impacts": []
    },
    "scientific_challenge": {
      "completion_interpretation": "",
      "design_adjustment_interpretation": "",
      "collateral_impacts": []
    },
    "patient_profile": {
      "completion_interpretation": "",
      "design_adjustment_interpretation": "",
      "collateral_impacts": []
    },
    "execution_framework": {
      "completion_interpretation": "",
      "design_adjustment_interpretation": "",
      "collateral_impacts": []
    }
  },
  "tradeoff_review": {
    "what_completion_gained": "",
    "what_design_confidence_gained": "",
    "what_may_have_been_sacrificed": "",
    "main_uncertainty": ""
  },
  "participant_review": {
    "what_changed": "",
    "why_completion_outlook_moved": "",
    "main_design_signal": "",
    "tradeoff_summary": "",
    "medical_development_question": "",
    "clinops_execution_question": ""
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
    "main_design_subcategories_considered": [],
    "reference_pack_ids_used": [],
    "compared_against": ""
  }
}
```

The application should validate this response, enforce supported evidence fields, map validated ratings into the bottom-up Design Confidence subcategory points, and calculate any Total Scenario Score itself.

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

### Phase 1: Architecture Doc Alignment

Deliverables:

- update `docs/architecture_narratives.md` to reflect the active four-pillar Design Confidence model.
- mark older five-domain Trial Value Review language as historical/superseded.
- record final labels: `Completion Outlook`, `Design Confidence`, `Total Scenario Score`, `Phase & Intent Alignment`, `Endpoint & Evidence Strength`, `Target Population Alignment`, `Operational Burden Balance`.
- record bottom-up scoring range and default-zero evidence gate.

No runtime behavior change.

### Phase 2: Contract Fixtures

Deliverables:

- replace or extend narrative fixtures with scenarios that test the four design subcategories
- include at least:
  - score improves but evidence value weakens
  - score improves and Design Confidence remains neutral
  - score improves and Design Confidence improves with evidence
  - score declines but Design Confidence improves with evidence
  - operational-only edit
  - endpoint text contradiction
  - biomarker/population mismatch
  - phase/intent ambition versus weak endpoint or comparator support
  - modality/risk-governance mismatch
  - no-adjustment case despite large Completion Outlook movement

Verification:

- fixture checker
- py_compile
- `git diff --check`

### Phase 3: Packet Builder

Deliverables:

- add richer Completion Movement Analysis inputs
- ensure feature, subcategory, and pillar deltas are explicit and separated
- add dependency/collateral-impact context fields where deterministic and auditable
- add selected reference-pack IDs and local context-stat IDs

Verification:

- packet checker
- live snapshot flow checker
- fixture checker

### Phase 4: Prompt and Schema

Deliverables:

- modular provider prompt with baseline/visible modes
- new JSON schema for `completion_outlook_review`, `design_confidence_subcategories`, `pillar_reviews`, `tradeoff_review`, `participant_review`, `continuity`, and `trace`
- stronger instructions for expert clinical-development language
- Reviewer Constitution, evidence hierarchy, source hierarchy, and "do not infer" rules

Verification:

- prompt checker
- provider schema checker
- mock provider compatibility
- one opt-in Gemini smoke on a fixed packet

### Phase 5: Deterministic Validation and Scoring

Deliverables:

- replace current `quality_review_domains` with `design_confidence_subcategories`, or provide a temporary compatibility adapter
- app-owned mapping from validated design-subcategory ratings to bottom-up Design Confidence points
- supported-evidence enforcement retained
- no app-owned score fields accepted from provider
- no hidden total cap; total Design Confidence is the sum of four design subcategories

Verification:

- scoring checker
- malformed/partial response checker
- no stale adjustment reuse

### Phase 6: Reference Packs and Data Context

Deliverables:

- reference-pack loader/selector using `frontend/data/docs/narrative_reference_packs/pack_manifest_v1.json`
- runtime inclusion of only selected pack sections: `Prompt-Safe Summary`, `Relevance To Simulator Pillars`, and `Do Not Infer`
- local mature-cohort context stats
- version IDs included in packet and trace
- no runtime web browsing

Verification:

- `python scripts/check_narrative_reference_packs.py`
- JSON/schema checks
- deterministic rebuild check
- sample packet inspection

### Phase 7: UI Redesign

Deliverables:

- keep participant view centered on the four familiar pillars
- add one design-adjustment subcategory to each pillar in Full Scenario view
- keep Completion Outlook evidence and Design Confidence adjustment provenance available in facilitator/debug trace
- show trade-off state when completion movement and design evidence diverge
- keep facilitator/debug details behind an expander

Verification:

- Streamlit simulator smoke
- screenshot check on desktop
- representative trial scenarios

### Phase 8: Calibration and Playtesting

Deliverables:

- 10 to 20 representative scenarios
- expert review notes
- rating-to-adjustment calibration for `-4.0` to `+4.0` subcategory range in 0.5 increments
- prompt refinement
- final decision on whether `Total Scenario Score` should be primary, secondary, or facilitator-only

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

1. Should the participant-facing label remain `Quality Review`, or move to `Scenario Review` / `Design Confidence Review`?
2. Should `Total Scenario Score` be participant-primary, participant-secondary, or facilitator-only in V1?
3. How should the integrated four-pillar chart visually distinguish model subcategories from design-adjustment subcategories without overwhelming participants?
4. Which conditional reference packs matter first beyond the core ICH packs?
5. Which 10 to 20 representative scenarios should be used for calibration and playtesting?
6. Should local database statistics be implemented before or after the first provider/schema rewrite?

## Next Step

Next implementation step: align `docs/architecture_narratives.md` with this active plan, then update contract fixtures for the four Design Confidence subcategories.
