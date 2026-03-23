# GBD Data Integration Logic: Market Potential & Unmet Need (MPUN)

This document explains the technical logic used to transform IHME Global Burden of Disease (GBD) data into a professional valuation anchor for clinical trial success prediction.

## 1. Objective
The goal is to provide every clinical trial with an authoritative "Economic and Clinical Signature" based on the disease it targets. This allows the model to distinguish between high-volume chronic markets (Recurring Revenue) and high-urgency rescue markets (Pricing Power).

## 2. The "Decoupled" Architecture (v05)
A critical update in v05 is the **separation of Indication and Therapeutic Area (TA) logic**:
- **LLM Responsibility**: Gemini 1.5 Flash is responsible *only* for mapping the trial text to the most granular **GBD Cause ID** possible using the `prompts/gbd_codes.md`.
- **System Responsibility**: The Therapeutic Area is assigned deterministically in Step 2 via a master lookup table. This prevents "taxonomy drift" where the LLM might categorize the same disease into different TAs across different trials.

## 3. Data Sources & Metrics
- **Hierarchy (`reference/hier_gbd.csv`)**: Derived from IHME 2023. Maps diseases across four levels (L1: Category -> L2: TA/Safety Net -> L3: Indication -> L4: Sub-indication).
- **Core Metrics**: We extract **Rates** (per 100k) for **Global** and **High SDI** (Socio-demographic Index) regions:
    - **DALYs**: Disability-Adjusted Life Years (Total Burden).
    - **YLDs**: Years Lived with Disability (Morbidity/Chronic component).
    - **YLLs**: Years of Life Lost (Mortality/Acute component).

## 4. The "Bulletproof" Inheritance & Reconciliation (v6.0)

### Hierarchical Mapping Logic
To ensure 100% data coverage, the system employs a hierarchical fallback:
1. **L4 Match**: Highest granularity.
2. **L3 Fallback**: Used if L4 data is missing.
3. **L2 Safety Net**: If no L3/L4 match is found, the system assigns the trial to its parent Level 2 "Safety Net" ID (e.g., "Cardiovascular diseases" [ID: 491]).
4. **Final Fallback**: [ID: 0] (Global Mean) for unclassifiable cases.

### Mathematical Identity & Ratios
- **Identity Reconciliation**: We enforce $DALY = YLL + YLD$. If components are missing, they are derived using parent split ratios or mathematical subtraction.
- **Chronic Ratio**: $YLD / DALY$. A ratio near 1.0 indicates a chronic condition; near 0.0 indicates high mortality.
- **Market Skew Index**: $DALY_{High SDI} / DALY_{Global}$. Measures the concentration of burden in high-income markets relative to the global average.

## 5. Strategic Implementation
The LLM prompt is anchored by the `prompts/gbd_codes.md`. By forcing the model to select from a fixed list of IDs, we enable:
- **Deterministic Valuation**: Immediate lookup of epidemiological "Value" for any trial.
- **Bias Correction**: Comparing a trial's protocol sophistication against the "Hurdle Tier" of its specific GBD indication.
- **Zero-Shot Enrichment**: The `Cause ID` serves as the primary key connecting raw trial data to the high-alpha strategic features.
