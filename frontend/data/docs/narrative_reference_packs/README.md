# Narrative Reference Packs

This folder contains short, versioned reference packs used by the CTPredict narrative layer.

The goal is not to reproduce regulatory guidance. The goal is to give the LLM concise, prompt-safe clinical development principles that can help produce better design-coherence narratives.

## Runtime Use

At runtime, the app should usually include:

1. `core_clinical_development_v1.md`
2. Zero, one, or two specialist packs, depending on the scenario evidence.

Do not include every pack in every prompt. The scenario packet remains the primary source of truth.

## Source Priority

The scenario packet wins over reference packs.

Reference packs should guide interpretation, not override:

* model outputs
* user-edited assumptions
* trial packet evidence
* benchmark evidence
* phase, indication, therapeutic area, and endpoint facts from the scenario

## Available Packs

### core_clinical_development_v1.md

Always-on pack.

Use for general clinical development reasoning:

* phase and intent alignment
* development-stage realism
* balance between ambition and feasibility
* evidence generation logic
* avoiding overly deterministic claims

### ich_e8_quality_by_design_v1.md

Use when the narrative concerns:

* quality by design
* critical-to-quality factors
* patient relevance
* study objective and design alignment
* operational feasibility
* whether the trial is designed to answer the question it claims to answer

This is likely the most frequently useful specialist pack for design-coherence narratives.

### ich_e6_r3_gcp_v1.md

Use when the narrative concerns:

* participant protection
* data integrity
* sponsor oversight
* investigator/site burden
* risk-proportionate trial conduct
* operational controls
* feasibility of trial execution

This pack should not turn the narrative into a compliance audit. It should only support operational and quality reasoning.

### ich_e9_statistical_principles_v1.md

Use when the narrative concerns:

* statistical design
* bias reduction
* randomisation
* blinding
* control group logic
* sample size logic
* analysis sets
* missing data
* interpretation of exploratory versus confirmatory evidence

### ich_e9_r1_estimands_v1.md

Use when the narrative concerns:

* endpoint clarity
* treatment effect definition
* estimands
* intercurrent events
* missing data versus events that change interpretation
* alignment between objective, endpoint, analysis, and interpretation
* sensitivity analysis

This pack is especially relevant when the simulator needs to comment on whether the endpoint and analysis logic are coherent with the trial objective.

## Do Not Infer

The LLM must not:

* claim regulatory acceptability
* claim that a trial will succeed
* claim that a design is invalid only because it differs from guidance
* impose Phase III confirmatory standards on early exploratory trials
* invent missing protocol details
* override scenario facts with generic guidance
* provide medical, regulatory, or legal advice

## Output Style Guidance

Narratives should be:

* conditional rather than absolute
* practical rather than academic
* connected to the scenario evidence
* clear about trade-offs
* concise enough for simulator feedback

Preferred wording:

* “This would strengthen…”
* “This may weaken…”
* “The design appears more coherent if…”
* “The main trade-off is…”
* “This should be interpreted as a design signal, not as proof of success.”

Avoid wording:

* “This proves…”
* “This guarantees…”
* “This is compliant…”
* “This is unacceptable…”
* “Regulators would reject…”
