# Pack ID: ich_e9_r1_estimands_v1

## Source

- Title: ICH E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials to the Guideline on Statistical Principles for Clinical Trials
- Organization: International Council for Harmonisation / European Medicines Agency
- URL: https://www.ema.europa.eu/en/documents/scientific-guideline/ich-e9-r1-addendum-estimands-and-sensitivity-analysis-clinical-trials-guideline-statistical-principles-clinical-trials-step-5_en.pdf
- Version/date: Step 5, final adoption by CHMP 30 January 2020, effective 30 July 2020
- Access type: public/free

## When To Use

Use this pack when scenario changes affect:

* endpoint clarity
* treatment effect definition
* estimands
* intercurrent events
* missing data
* treatment discontinuation
* rescue medication
* treatment switching
* death or other terminal events
* endpoint interpretation
* analysis alignment
* sensitivity analysis
* whether the trial can answer the clinical question it claims to answer

This pack is especially relevant when the narrative needs to explain whether the trial objective, endpoint, population, treatment comparison, and analysis logic are aligned.

## Key Principles

### 1. The trial objective should define the treatment effect of interest

A clinical trial should be clear about the treatment effect it is trying to estimate.

The same endpoint can support different clinical questions depending on how the trial handles treatment discontinuation, rescue medication, treatment switching, death, missing data, and follow-up.

For CTPredict, this means the narrative should not only ask “what endpoint is used?” It should also ask “what treatment effect is this endpoint meant to describe?”

### 2. Estimands connect objective, design, analysis, and interpretation

An estimand is a precise description of the treatment effect that corresponds to the clinical question of interest.

A coherent trial should align:

* objective
* treatment condition
* target population
* endpoint or variable
* handling of intercurrent events
* population-level summary
* estimator
* sensitivity analysis
* interpretation

The narrative should flag designs where these elements appear misaligned.

### 3. Planning should proceed in sequence

The clinical question should come first.

The design, data collection, main estimator, and sensitivity analysis should follow from the estimand.

A weaker design may appear when the analysis method or available data seem to define the question after the fact, instead of the clinical question guiding the design from the start.

### 4. Intercurrent events are not the same as missing data

Intercurrent events occur after treatment starts and affect either the interpretation or the existence of the measurement linked to the clinical question.

Examples include:

* treatment discontinuation
* use of rescue medication
* use of prohibited medication
* treatment switching
* changes in background therapy
* death
* other terminal events

Missing data are data that would be meaningful for the analysis of a given estimand but were not collected.

For CTPredict, this distinction is important. A post-treatment event may change what the endpoint means; it is not automatically just a missing-data problem.

### 5. Intercurrent events should be addressed explicitly

A trial is stronger when important intercurrent events are anticipated and handled consistently with the clinical question.

For example:

* If rescue medication is expected, the design should clarify whether post-rescue endpoint values remain relevant.
* If treatment discontinuation is common, the design should clarify whether the question concerns initial treatment assignment, treatment while continued, or a hypothetical scenario without discontinuation.
* If death prevents endpoint measurement, the design should clarify whether death is part of the outcome or changes the endpoint interpretation.

The narrative should identify when such issues may weaken interpretability.

### 6. Different strategies answer different questions

E9(R1) describes several strategies for handling intercurrent events. Each strategy corresponds to a different clinical question.

#### Treatment policy strategy

The endpoint value is used regardless of whether the intercurrent event occurs.

This can be useful when the question concerns the effect of assigning or initiating treatment in a real treatment policy context.

#### Hypothetical strategy

The question asks what would have happened if the intercurrent event had not occurred.

This may require stronger assumptions and should be clinically justified.

#### Composite variable strategy

The intercurrent event is incorporated into the endpoint itself.

For example, treatment discontinuation due to toxicity may count as failure, or death may be included as part of the outcome.

#### While on treatment strategy

The question focuses on outcomes before the intercurrent event occurs.

This may be relevant for exposure-related safety or symptom control while treatment continues.

#### Principal stratum strategy

The question focuses on a subgroup defined by whether an intercurrent event would or would not occur under treatment conditions.

This can require strong assumptions and should be used cautiously.

### 7. Endpoint interpretation depends on the strategy

The same endpoint can mean different things depending on how intercurrent events are handled.

For example, a symptom score after rescue medication may not mean the same thing as a symptom score without rescue medication.

A progression endpoint that includes death is different from one that only measures progression among survivors.

The narrative should comment on endpoint interpretability when the trial design creates ambiguity.

### 8. Missing-data handling should align with the estimand

A stronger trial distinguishes between:

* data that are relevant but missing
* data that do not exist
* data that exist but are no longer meaningful for the clinical question

Missing-data methods should be aligned with the estimand.

The narrative should avoid treating all missing or post-event data as the same problem.

### 9. Sensitivity analysis tests robustness of interpretation

Sensitivity analysis explores whether the main inference is robust to limitations in the data and deviations from the assumptions behind the main estimator.

A stronger design identifies the main estimator and plans sensitivity analyses that target the same estimand.

For CTPredict, robustness should be interpreted as confidence in the treatment-effect interpretation, not as certainty of trial success.

### 10. Exploratory objectives do not require the same estimand burden

E9(R1) is most important for treatment effects that support important decision-making.

Exploratory objectives may still benefit from clarity, but the narrative should not impose full confirmatory estimand expectations on every early exploratory endpoint.

The key is proportionality: the stronger the claim, the more precise the estimand logic should be.

### 11. Changes during the trial can weaken credibility

Changing the estimand during the trial can reduce credibility, especially if it happens after relevant trial information is available.

For CTPredict, this is useful when a simulated design change makes the endpoint or interpretation more flexible but less pre-specified.

The narrative should treat flexibility as a trade-off: it may support learning, but it may weaken confirmatory interpretability.

### 12. Multiple estimands may be needed

A trial can have more than one clinical question.

For example, one estimand may describe the effect regardless of rescue medication, while another may describe the effect before rescue medication.

Multiple estimands can improve clarity, but they also increase complexity and require careful interpretation.

The narrative should recognise when multiple perspectives enrich interpretation and when they create ambiguity.

## Relevance To Simulator Pillars

### Phase & Intent Alignment

Use E9(R1) to assess whether the treatment-effect question fits the development stage.

Early trials may use simpler or more exploratory endpoint logic. Later trials usually need clearer estimand, endpoint, analysis, and sensitivity-analysis alignment.

### Endpoint & Evidence Strength

Use E9(R1) to assess whether the endpoint really supports the intended clinical question.

Endpoint strength depends not only on what is measured, but also on how intercurrent events, missing data, and analysis assumptions affect interpretation.

### Target Population Alignment

Use E9(R1) to assess whether the population used for inference matches the clinical question.

The target population should be clear, especially when discontinuation, treatment switching, rescue medication, response status, or tolerance may define the relevant question.

### Operational Burden Balance

Use E9(R1) to assess whether data collection is sufficient and proportionate for the estimand.

Collecting post-discontinuation or post-rescue data may improve interpretation for some estimands but add burden. Not collecting such data may simplify the trial but create missing-data or interpretability problems.

### Design Coherence

Use E9(R1) to assess whether the trial objective, endpoint, population, intercurrent-event strategy, analysis, and interpretation form a coherent evidence chain.

A coherent design makes clear what treatment effect is being estimated and how the trial design supports that estimate.

## Do Not Infer

Do not infer that:

* every trial needs a fully detailed confirmatory estimand
* every exploratory endpoint requires extensive sensitivity analysis
* treatment discontinuation is always missing data
* rescue medication always invalidates endpoint interpretation
* death should always be treated as missing data
* a hypothetical strategy is automatically acceptable
* a treatment policy strategy is always preferable
* a composite endpoint is always stronger
* a while-on-treatment strategy is always weaker
* multiple estimands are always better
* the simulator can determine regulatory acceptability

Do not provide regulatory advice, legal advice, medical advice, or formal statistical validation.

## Prompt-Safe Summary

ICH E9(R1) supports CTPredict narratives when endpoint interpretation, treatment-effect definition, intercurrent events, missing data, analysis alignment, or sensitivity analysis matter. The core idea is that a trial objective should be translated into a precise clinical question through an estimand before the design, data collection, estimator, and sensitivity analysis are chosen. A coherent trial should align the treatment condition, population, endpoint, handling of intercurrent events, population-level summary, analysis method, and interpretation. Intercurrent events, such as treatment discontinuation, rescue medication, treatment switching, or death, are not the same as missing data; they may change the meaning or existence of endpoint measurements. Different strategies for intercurrent events answer different clinical questions. Use this pack to generate practical, conditional feedback on whether the endpoint and analysis logic can support the trial’s intended evidence claim.
