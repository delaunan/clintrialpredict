# Pack ID: core_clinical_development_v1

## Source

* Title: Core Clinical Development Principles for CTPredict Narratives
* Organization: CTPredict internal synthesis
* Source basis: ICH E8(R1), ICH E6(R3), ICH E9, ICH E9(R1)
* Version/date: v1
* Access type: internal summary based on public guidance

## When To Use

Use this pack as the always-on clinical development background for CTPredict narratives.

Use it to support general reasoning about:

* phase and intent alignment
* evidence generation logic
* design coherence
* patient relevance
* feasibility
* operational burden
* uncertainty and trade-offs
* exploratory versus confirmatory trial logic

This pack should be included in most narrative prompts, unless the prompt must be extremely short.

## Key Principles

### 1. A clinical trial should answer a clear question

A trial design is more coherent when the objective, population, endpoint, intervention, comparator, duration, and analysis logic all point toward the same clinical question.

A design becomes weaker when the trial appears to combine incompatible aims, such as exploratory signal detection and confirmatory proof, without enough sample size, endpoint maturity, or operational control to support the stronger claim.

### 2. Phase and intent matter

Early-phase trials usually tolerate more uncertainty and may focus on safety, dose, feasibility, biological signal, or preliminary efficacy.

Later-phase trials usually require stronger alignment between endpoints, population, comparator, sample size, follow-up, statistical assumptions, and operational controls.

The same design choice may be reasonable in one phase and weak in another.

### 3. Coherence is not the same as probability of success

A coherent design is one that is logically structured to answer its intended question.

A coherent trial can still fail because the treatment does not work, recruitment is difficult, safety issues emerge, or external assumptions prove wrong.

The narrative must not imply that design coherence guarantees trial success.

### 4. Trial quality should be designed in, not added later

Important risks should be identified early and managed through the design.

The most important design risks are those that could affect:

* participant protection
* reliability of results
* interpretability of endpoints
* feasibility of execution
* ability of the trial to meet its objective

The narrative should focus on design choices that matter, not on minor procedural details.

### 5. Patient relevance strengthens the design

A trial is stronger when the population, endpoints, visit schedule, follow-up duration, and burden are meaningful and realistic for the patients being studied.

A design may look efficient statistically but still be weak if it is too burdensome, excludes the relevant population, or measures outcomes that are poorly connected to patient benefit.

### 6. Feasibility is part of scientific quality

A trial that cannot recruit, retain participants, collect endpoint data, or maintain protocol adherence may fail to answer its question even if the scientific concept is strong.

Operational burden should be interpreted as part of design coherence, not as a separate administrative issue.

### 7. Trade-offs should be explicit

Most trial design decisions involve trade-offs.

Examples:

* broader eligibility may improve generalisability but increase heterogeneity
* stricter eligibility may reduce noise but weaken real-world relevance
* longer follow-up may improve endpoint maturity but increase burden and attrition
* more sites may accelerate recruitment but increase oversight complexity
* fewer sites may improve control but slow recruitment
* ambitious endpoints may increase evidence value but reduce feasibility

The narrative should explain the trade-off rather than simply label a choice as good or bad.

### 8. Evidence strength depends on context

Evidence strength depends on whether the trial design is appropriate for its development stage and objective.

For exploratory trials, useful evidence may come from signal detection, feasibility, safety, or biomarker directionality.

For confirmatory trials, stronger evidence usually requires clearer endpoint hierarchy, adequate sample size, bias control, missing-data handling, and alignment between the clinical question and the analysis.

### 9. The simulator should not over-regulate

CTPredict is a learning and exploration tool.

The narrative should not read like a regulatory rejection letter.

It should help the user understand whether their design choices seem more or less coherent, more or less feasible, and more or less aligned with the stated development intent.

### 10. The scenario packet is the primary source of truth

Reference packs provide background principles only.

The model must not override scenario facts, user-edited values, benchmark evidence, model outputs, or trial packet evidence with generic assumptions from guidance.

## Relevance To Simulator Pillars

### Phase & Intent Alignment

Assess whether the design choices fit the stated phase and trial intent.

A design may be coherent if its endpoint ambition, sample size, duration, comparator, and operational complexity are proportionate to the development stage.

### Endpoint & Evidence Strength

Assess whether the endpoint strategy can plausibly answer the stated clinical question.

Consider whether the endpoint is interpretable, patient-relevant, feasible to collect, and appropriate for exploratory or confirmatory use.

### Target Population Alignment

Assess whether the selected population matches the intended clinical question.

Consider eligibility breadth, disease severity, prior treatment context, special populations, and whether the trial population is too narrow or too heterogeneous for its purpose.

### Operational Burden Balance

Assess whether enrollment, site count, duration, visit burden, data collection, and trial complexity appear proportionate to the expected evidence value.

Operational burden should be treated as a design trade-off, not automatically as a negative.

### Design Coherence

Assess whether the overall design forms a consistent evidence story.

A coherent design aligns:

* objective
* phase
* population
* endpoint
* comparator
* duration
* operational assumptions
* analysis logic
* expected decision use

## Do Not Infer

Do not infer that:

* a trial will succeed because it is coherent
* a trial will fail because it is operationally ambitious
* a design is invalid because it differs from guidance
* a Phase II trial must meet Phase III standards
* a small exploratory trial must prove efficacy
* a broad population is always better
* a narrow population is always better
* a larger sample is always better
* more sites are always better
* shorter duration is always better

Do not provide:

* regulatory advice
* medical advice
* legal advice
* claims of compliance
* claims of likely approval or rejection

## Prompt-Safe Summary

Clinical trial design coherence depends on whether the objective, phase, population, endpoint, comparator, duration, operational assumptions, and analysis logic are aligned. A coherent design does not guarantee success; it means the trial is structured in a way that can plausibly answer its intended question. Early-phase trials may accept more uncertainty and focus on safety, dose, feasibility, biological signal, or preliminary efficacy, while later-phase trials usually require stronger endpoint, population, comparator, sample-size, and analysis alignment. Patient relevance and operational feasibility are part of scientific quality because a trial that cannot recruit, retain participants, or collect interpretable endpoint data may fail to answer its question. Most design choices involve trade-offs: broader eligibility can improve relevance but increase heterogeneity; longer follow-up can improve endpoint maturity but increase burden; more sites can accelerate recruitment but increase oversight complexity. Use this pack to produce practical, conditional design-coherence feedback. The scenario packet remains the primary source of truth.
