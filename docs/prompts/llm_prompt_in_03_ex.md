# GOLD STANDARD EXAMPLES (v17.0 - Real World Evidence)

### **Example 1: Oncology Hard Clinical + Adaptive**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT04762069
[START_YEAR]: 2021
[PHASE]: Phase 2
[ASSIGNED_INDICATION]: Glioblastoma multiforme
[ASSIGNED_THERAPEUTIC_AREA]: Oncology
[AGENT]: Berubicin
[SPONSOR]: CNS Pharmaceuticals, Inc.
[OFFICIAL_TITLE]: Multicenter Study of Berubicin vs Lomustine in Glioblastoma (GBM)
[ASSIGNED_LINE_OF_THERAPY]: LATER_LINE
[ASSIGNED_PATIENT_SEVERITY]: ADVANCED_METASTATIC
[DESIGN]: Arms: 2, Allocation: RANDOMIZED, Model: PARALLEL, Purpose: TREATMENT, Masking: SINGLE
[PRIMARY_ENDPOINTS_DETAIL]: Overall Survival (Timeframe: 4 years)
[INTERVENTION_DETAILS_ENHANCED]:
NAME: Berubicin (DRUG)
DESC: Berubicin HCl is a novel synthetic anthracycline with a chemical structure similar to doxorubicin HCl.
---
NAME: Lomustine (DRUG)
DESC: Lomustine is an anti-cancer (antineoplastic or cytotoxic) chemotherapy drug.

[PROTOCOL_SUMMARY]:
This is an open-label, multicenter, randomized, parallel, 2-arm, efficacy and safety study. A pre-planned, non-binding futility analysis will be performed after approximately 30 to 50% of all planned patients have completed the primary endpoint at 6 months.
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT04762069",
  "strategist_logic": "(1) ROUTE-DECOUPLING: Berubicin is synthetic anthracycline (IV/Infusion); [STEP-1-RESULT: ROUTINE_INFUSION]. (2) OS-MORTALITY CHECK: Primary outcome is Overall Survival; [STEP-2-RESULT: HARD_CLINICAL]. (3) HISTORIAN RESET: Lomustine was a preferred standard cytotoxic for GBM in 2021; [STEP-3-RESULT: ACTIVE_MODERN_STANDARD]. (4) PRIMARY EFFICACY AUDIT: Targeting core GBM survival; [STEP-4-RESULT: PIVOTAL_INTENT]. (5) DECISION TRIGGER: Pre-planned futility analysis mentioned; [STEP-5-RESULT: ADAPTIVE].",
  "endpoint_rigor": "HARD_CLINICAL",
  "endpoint_structure": "SINGLE_GOAL",
  "comparator_benchmark": "ACTIVE_MODERN_STANDARD",
  "strategic_ambition": "PIVOTAL_INTENT",
  "administration_complexity": "ROUTINE_INFUSION",
  "innovation_tier": "FIRST_IN_CLASS",
  "adaptive_design": "ADAPTIVE"
}
```

---

### **Example 2: The Pill Rule + Safety Focus**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT00360867
[START_YEAR]: 2005
[PHASE]: Phase 1
[ASSIGNED_INDICATION]: Malignant neoplasm of liver
[ASSIGNED_THERAPEUTIC_AREA]: Oncology
[AGENT]: AMG 706
[SPONSOR]: Amgen
[OFFICIAL_TITLE]: An Open Label Treatment Extension Study of AMG 706
[ASSIGNED_LINE_OF_THERAPY]: LATER_LINE
[ASSIGNED_PATIENT_SEVERITY]: ADVANCED_METASTATIC
[DESIGN]: Arms: 1, Allocation: NON_RANDOMIZED, Model: SINGLE_GROUP, Masking: NONE
[PRIMARY_ENDPOINTS_DETAIL]: Safety (serious adverse events, adverse events, blood pressure and laboratory tests) (Timeframe: Until objective evidence of disease progression)
[INTERVENTION_DETAILS_ENHANCED]:
NAME: AMG 706 (DRUG)
DESC: The dose of AMG 706 will be no greater than 125 mg QD or 75 mg BID.

[PROTOCOL_SUMMARY]:
This is an extension study to provide ongoing treatment with AMG 706 monotherapy for subjects with solid tumors who have completed the planned duration of AMG 706 treatment on a separate Amgen protocol.
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT00360867",
  "strategist_logic": "(1) ROUTE-DECOUPLING: QD/BID dosing indicates oral; [STEP-1-RESULT: SIMPLE_ORAL]. (2) OS-MORTALITY CHECK: Safety/Labs are primary proxies for clinical state; [STEP-2-RESULT: SUBJECTIVE_PRO]. (3) HISTORIAN RESET: Single arm extension study in 2005; [STEP-3-RESULT: NO_CONTROL_GROUP]. (4) PRIMARY EFFICACY AUDIT: Extension focused on safety/tolerability; [STEP-4-RESULT: SAFETY_DOSING]. (5) DECISION TRIGGER: Fixed extension path; [STEP-5-RESULT: STATIC].",
  "endpoint_rigor": "SUBJECTIVE_PRO",
  "endpoint_structure": "SINGLE_GOAL",
  "comparator_benchmark": "NO_CONTROL_GROUP",
  "strategic_ambition": "SAFETY_DOSING",
  "administration_complexity": "SIMPLE_ORAL",
  "innovation_tier": "NEXT_GEN_OPTIMIZED",
  "adaptive_design": "STATIC"
}
```

---

### **Example 3: Oncology Surrogate (The Blacklist Rule)**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT05986851
[START_YEAR]: 2023
[PHASE]: Phase 2
[ASSIGNED_INDICATION]: Glioblastoma multiforme
[ASSIGNED_THERAPEUTIC_AREA]: Oncology
[AGENT]: Azeliragon
[SPONSOR]: Azelista Pharmaceuticals
[OFFICIAL_TITLE]: Phase II Study of Azeliragon in MGMT Unmethylated Glioblastoma
[ASSIGNED_LINE_OF_THERAPY]: FIRST_LINE
[ASSIGNED_PATIENT_SEVERITY]: ADVANCED_METASTATIC
[DESIGN]: Arms: 1, Allocation: nan, Model: SINGLE_GROUP, Purpose: TREATMENT, Masking: NONE
[PRIMARY_ENDPOINTS_DETAIL]: Progression-free survival (Timeframe: Up to 2 years)
[INTERVENTION_DETAILS_ENHANCED]:
NAME: Azeliragon (DRUG)
DESC: Oral capsule

[PROTOCOL_SUMMARY]:
This is a phase 2 study to evaluate the safety and preliminary evidence of effectiveness of azeliragon, in combination with radiation therapy, as an initial treatment of a form of glioblastoma.
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT05986851",
  "strategist_logic": "(1) ROUTE-DECOUPLING: Oral capsule; [STEP-1-RESULT: SIMPLE_ORAL]. (2) OS-MORTALITY CHECK: PFS in Oncology is blacklisted; [STEP-2-RESULT: SURROGATE]. (3) HISTORIAN RESET: Single arm phase 2 in 2023; [STEP-3-RESULT: NO_CONTROL_GROUP]. (4) PRIMARY EFFICACY AUDIT: Core disease progression; [STEP-4-RESULT: PIVOTAL_INTENT]. (5) DECISION TRIGGER: Fixed path; [STEP-5-RESULT: STATIC].",
  "endpoint_rigor": "SURROGATE",
  "endpoint_structure": "SINGLE_GOAL",
  "comparator_benchmark": "NO_CONTROL_GROUP",
  "strategic_ambition": "PIVOTAL_INTENT",
  "administration_complexity": "SIMPLE_ORAL",
  "innovation_tier": "FIRST_IN_CLASS",
  "adaptive_design": "STATIC"
}
```

---

### **Example 11: Statistical Adaptive (Simon Two-Stage)**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT04984668
[START_YEAR]: 2021
[PHASE]: Phase 1
[ASSIGNED_INDICATION]: Other malignant neoplasms
[ASSIGNED_THERAPEUTIC_AREA]: Oncology
[AGENT]: GT90001 + KN046
[SPONSOR]: Suzhou Kintor Pharmaceutical Inc.
[OFFICIAL_TITLE]: A Phase Ib/II Study of GT90001 Combined With KN046 in Solid Tumors
[ASSIGNED_LINE_OF_THERAPY]: LATER_LINE
[ASSIGNED_PATIENT_SEVERITY]: ADVANCED_METASTATIC
[DESIGN]: Arms: 1, Allocation: nan, Model: SINGLE_GROUP, Purpose: TREATMENT, Masking: NONE
[PRIMARY_ENDPOINTS_DETAIL]: Safety, tolerability, PK, and preliminary antitumor activity (Timeframe: 2 years)
[INTERVENTION_DETAILS_ENHANCED]:
NAME: GT90001+KN046 (DRUG)
DESC: GT90001 3mg/Kg, KN046 3mg/Kg.
[PROTOCOL_SUMMARY]:
Phase Ib dose de-escalation study. A Simon two-stage design is planned for each indication in order to minimize treated participants if there is minimal efficacy activity (futility).
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT04984668",
  "strategist_logic": "(1) ROUTE-DECOUPLING: Combo IV/Infusion; [STEP-1-RESULT: ROUTINE_INFUSION]. (2) OS-MORTALITY CHECK: Outcomes are safety and preliminary activity; [STEP-2-RESULT: SURROGATE]. (3) HISTORIAN RESET: No control group in 2021 Phase 1b/2 dose-finding combo study; [STEP-3-RESULT: NO_CONTROL_GROUP]. (4) PRIMARY EFFICACY AUDIT: Phase 1b/2 focused on safety and signal search; [STEP-4-RESULT: SAFETY_DOSING]. (5) DECISION TRIGGER: Simon two-stage design with explicit futility-based stopping rule; [STEP-5-RESULT: ADAPTIVE].",
  "endpoint_rigor": "SURROGATE",
  "endpoint_structure": "MULTI_COMPOSITE",
  "comparator_benchmark": "NO_CONTROL_GROUP",
  "strategic_ambition": "SAFETY_DOSING",
  "administration_complexity": "ROUTINE_INFUSION",
  "innovation_tier": "FIRST_IN_CLASS",
  "adaptive_design": "ADAPTIVE"
}
```

---

### **Example 4: Cardiovascular Multi-Drug Pill Rule**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT04158076
[START_YEAR]: 2020
[PHASE]: Phase 3
[ASSIGNED_INDICATION]: Essential hypertension
[ASSIGNED_THERAPEUTIC_AREA]: Cardiovascular
[AGENT]: Rosuvastatin/Ezetimibe and Telmisartan/Amlodipine
[SPONSOR]: Hanmi Pharmaceutical Company Limited
[OFFICIAL_TITLE]: Evaluation of Co-administered AD-2071 and AD-2073 in Primary Hypercholesterolemia and Essential Hypertension
[ASSIGNED_LINE_OF_THERAPY]: FIRST_LINE
[ASSIGNED_PATIENT_SEVERITY]: CHRONIC_STABLE
[DESIGN]: Arms: 3, Allocation: RANDOMIZED, Model: PARALLEL, Purpose: TREATMENT, Masking: DOUBLE
[PRIMARY_ENDPOINTS_DETAIL]: Low density lipoprotein cholesterol (LDL-C) (Timeframe: Baseline, Week 8) | Mean sitting systolic blood pressure (MSSBP) (Timeframe: Baseline, Week 8)
[INTERVENTION_DETAILS_ENHANCED]:
NAME: Ezetimibe/Rosuvastatin (DRUG)
DESC: PO, Once daily(QD)
---
NAME: Telmisartan (DRUG)
DESC: PO, Once daily(QD)
---
NAME: Telmisartan/Amlodipine (DRUG)
DESC: ORAL TABLET, PO

[PROTOCOL_SUMMARY]:
The purpose of this study is to evaluate the efficacy and safety of co-administrated Rosuvastatin/Ezetimibe and Telmisartan/Amlodipine in patients with primary hypercholesterolemia and essential hypertension.
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT04158076",
  "strategist_logic": "(1) ROUTE-DECOUPLING: All components are PO (Oral Tablet); [STEP-1-RESULT: SIMPLE_ORAL]. (2) OS-MORTALITY CHECK: LDL and BP are biological proxies; [STEP-2-RESULT: SURROGATE]. (3) HISTORIAN RESET: Statins and ARBs were standard \"Kings\" in 2020; [STEP-3-RESULT: ACTIVE_MODERN_STANDARD]. (4) PRIMARY EFFICACY AUDIT: Core disease targets; [STEP-4-RESULT: PIVOTAL_INTENT]. (5) DECISION TRIGGER: Fixed path; [STEP-5-RESULT: STATIC].",
  "endpoint_rigor": "SURROGATE",
  "endpoint_structure": "MULTI_COMPOSITE",
  "comparator_benchmark": "ACTIVE_MODERN_STANDARD",
  "strategic_ambition": "PIVOTAL_INTENT",
  "administration_complexity": "SIMPLE_ORAL",
  "innovation_tier": "NEXT_GEN_OPTIMIZED",
  "adaptive_design": "STATIC"
}
```

---

### **Example 5: Oncology Head-to-Head + Pill Rule**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT03053440
[START_YEAR]: 2017
[PHASE]: Phase 3
[ASSIGNED_INDICATION]: Non-Hodgkin lymphoma
[ASSIGNED_THERAPEUTIC_AREA]: Oncology
[AGENT]: BGB-3111 (Zanubrutinib)
[SPONSOR]: BeiGene
[OFFICIAL_TITLE]: Study Comparing Zanubrutinib and Ibrutinib in Waldenström's Macroglobulinemia
[ASSIGNED_LINE_OF_THERAPY]: LATER_LINE
[ASSIGNED_PATIENT_SEVERITY]: CHRONIC_PROGRESSIVE
[DESIGN]: Arms: 2, Allocation: RANDOMIZED, Model: PARALLEL, Masking: NONE
[PRIMARY_ENDPOINTS_DETAIL]: Percentage of Participants Achieving CR or VGPR (Timeframe: 2.5 years)
[INTERVENTION_DETAILS_ENHANCED]:
NAME: BGB-3111 (DRUG)
DESC: 160 mg PO BID.
---
NAME: Ibrutinib (DRUG)
DESC: 420 mg PO QD.

[PROTOCOL_SUMMARY]:
Active-controlled study of zanubrutinib vs ibrutinib; targeting complete/partial response.
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT03053440",
  "strategist_logic": "(1) ROUTE-DECOUPLING: Both drugs are oral (PO); [STEP-1-RESULT: SIMPLE_ORAL]. (2) OS-MORTALITY CHECK: Primary outcome is Response (CR/VGPR); [STEP-2-RESULT: SURROGATE]. (3) HISTORIAN RESET: Ibrutinib was the preferred BTK inhibitor \"King\" for WM in 2017; [STEP-3-RESULT: ACTIVE_MODERN_STANDARD]. (4) PRIMARY EFFICACY AUDIT: Core disease response; [STEP-4-RESULT: PIVOTAL_INTENT]. (5) DECISION TRIGGER: Fixed path; [STEP-5-RESULT: STATIC].",
  "endpoint_rigor": "SURROGATE",
  "endpoint_structure": "SINGLE_GOAL",
  "comparator_benchmark": "ACTIVE_MODERN_STANDARD",
  "strategic_ambition": "PIVOTAL_INTENT",
  "administration_complexity": "SIMPLE_ORAL",
  "innovation_tier": "NEXT_GEN_OPTIMIZED",
  "adaptive_design": "STATIC"
}
```

---

### **Example 6: Pediatric Ophthalmology + Composite**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT02132312
[START_YEAR]: 2014
[PHASE]: Phase 3
[ASSIGNED_INDICATION]: Cataract
[ASSIGNED_THERAPEUTIC_AREA]: Ophthalmology
[AGENT]: OMS302
[SPONSOR]: Omeros Corporation
[OFFICIAL_TITLE]: Study of OMS302 in Children Undergoing Unilateral Cataract Extraction
[ASSIGNED_LINE_OF_THERAPY]: LINE_NA
[ASSIGNED_PATIENT_SEVERITY]: CHRONIC_STABLE
[DESIGN]: Arms: 2, Allocation: RANDOMIZED, Model: PARALLEL, Masking: QUADRUPLE
[PRIMARY_ENDPOINTS_DETAIL]: Intraoperative Pupil Diameter; Acute Postoperative Pain (Timeframe: 24 hours)
[INTERVENTION_DETAILS_ENHANCED]:
NAME: OMS302 (DRUG)
DESC: Administered in irrigation solution.
---
NAME: Phenylephrine HCl (DRUG)
DESC: Active control in irrigation.

[PROTOCOL_SUMMARY]:
Evaluate effect on pupil diameter and acute pain in children birth through 3 years.
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT02132312",
  "strategist_logic": "(1) ROUTE-DECOUPLING: Irrigation solution (local/surgical); [STEP-1-RESULT: ROUTINE_INFUSION]. (2) OS-MORTALITY CHECK: Pupil diameter and pain are biological/subjective proxies; [STEP-2-RESULT: SURROGATE]. (3) HISTORIAN RESET: Phenylephrine was a standard \"King\" for pupil dilation in 2014; [STEP-3-RESULT: ACTIVE_MODERN_STANDARD]. (4) PRIMARY EFFICACY AUDIT: Core disease extraction support; [STEP-4-RESULT: PIVOTAL_INTENT]. (5) DECISION TRIGGER: Fixed path; [STEP-5-RESULT: STATIC].",
  "endpoint_rigor": "SURROGATE",
  "endpoint_structure": "MULTI_COMPOSITE",
  "comparator_benchmark": "ACTIVE_MODERN_STANDARD",
  "strategic_ambition": "PIVOTAL_INTENT",
  "administration_complexity": "ROUTINE_INFUSION",
  "innovation_tier": "NEXT_GEN_OPTIMIZED",
  "adaptive_design": "STATIC"
}
```

---

### **Example 7: Respiratory Nebulization**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT00262652
[START_YEAR]: 2006
[PHASE]: Phase 2
[ASSIGNED_INDICATION]: Asthma
[ASSIGNED_THERAPEUTIC_AREA]: Respiratory
[AGENT]: Sodium pyruvate
[SPONSOR]: Sodium Pyruvate Study Group
[OFFICIAL_TITLE]: Sodium Pyruvate Bronchodilation in Asthmatics
[ASSIGNED_LINE_OF_THERAPY]: FIRST_LINE
[ASSIGNED_PATIENT_SEVERITY]: CHRONIC_PROGRESSIVE
[DESIGN]: Arms: nan, Allocation: RANDOMIZED, Model: PARALLEL, Masking: QUADRUPLE
[PRIMARY_ENDPOINTS_DETAIL]: FEV1% predicted at multiple timepoints (Timeframe: 4 hours)
[INTERVENTION_DETAILS_ENHANCED]:
NAME: sodium pyruvate (DRUG)
DESC: Delivered by nebulization to the lungs.

[PROTOCOL_SUMMARY]:
Randomized, double-blind, placebo-controlled study to produce bronchodilation.
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT00262652",
  "strategist_logic": "(1) ROUTE-DECOUPLING: Nebulization; [STEP-1-RESULT: ROUTINE_INFUSION]. (2) OS-MORTALITY CHECK: FEV1 is a biological proxy; [STEP-2-RESULT: SURROGATE]. (3) HISTORIAN RESET: Placebo controlled study in 2006; [STEP-3-RESULT: PLACEBO]. (4) PRIMARY EFFICACY AUDIT: Targeting bronchodilation (symptom relief) in a short-term trial; [STEP-4-RESULT: SIGNAL_SEARCH]. (5) DECISION TRIGGER: Fixed path; [STEP-5-RESULT: STATIC].",
  "endpoint_rigor": "SURROGATE",
  "endpoint_structure": "MULTI_COMPOSITE",
  "comparator_benchmark": "PLACEBO",
  "strategic_ambition": "SIGNAL_SEARCH",
  "administration_complexity": "ROUTINE_INFUSION",
  "innovation_tier": "ESTABLISHED_COPY",
  "adaptive_design": "STATIC"
}
```

---

### **Example 8: Infections IV Treatment**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT01176058
[START_YEAR]: 2010
[PHASE]: Phase 3
[ASSIGNED_INDICATION]: Other infectious diseases
[ASSIGNED_THERAPEUTIC_AREA]: Infections
[AGENT]: Unknown
[SPONSOR]: Pfizer
[OFFICIAL_TITLE]: Study of Anidulafungin vs. Fluconazole in Candidemia
[ASSIGNED_LINE_OF_THERAPY]: FIRST_LINE
[ASSIGNED_PATIENT_SEVERITY]: ACUTE_CRITICAL
[DESIGN]: Arms: 1, Allocation: RANDOMIZED, Model: PARALLEL, Masking: NONE
[PRIMARY_ENDPOINTS_DETAIL]: Global Response at End of Intravenous Treatment (Timeframe: Day 42)
[INTERVENTION_DETAILS_ENHANCED]:
NAME: Anidulafungin/Fluconazole (DRUG)
DESC: Anidulafungin: IV, 100 mg daily. Fluconazole: IV/Oral, 400mg, QD.

[PROTOCOL_SUMMARY]:
Anidulafungin is compared to Fluconazole for efficacy and safety in candidemia.
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT01176058",
  "strategist_logic": "(1) ROUTE-DECOUPLING: Primary treatment is Intravenous (IV); [STEP-1-RESULT: ROUTINE_INFUSION]. (2) OS-MORTALITY CHECK: Global response is a clinical/biological proxy; [STEP-2-RESULT: SURROGATE]. (3) HISTORIAN RESET: Fluconazole was a standard antifungal \"King\" in 2010; [STEP-3-RESULT: ACTIVE_MODERN_STANDARD]. (4) PRIMARY EFFICACY AUDIT: Targeting core infection response; [STEP-4-RESULT: PIVOTAL_INTENT]. (5) DECISION TRIGGER: Fixed path; [STEP-5-RESULT: STATIC].",
  "endpoint_rigor": "SURROGATE",
  "endpoint_structure": "SINGLE_GOAL",
  "comparator_benchmark": "ACTIVE_MODERN_STANDARD",
  "strategic_ambition": "PIVOTAL_INTENT",
  "administration_complexity": "ROUTINE_INFUSION",
  "innovation_tier": "ESTABLISHED_COPY",
  "adaptive_design": "STATIC"
}
```

---

### **Example 9: Cardiovascular Oral Combination (Pill Rule)**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT00797862
[START_YEAR]: 2008
[PHASE]: Phase 3
[ASSIGNED_INDICATION]: Essential hypertension
[ASSIGNED_THERAPEUTIC_AREA]: Cardiovascular
[AGENT]: Unknown
[SPONSOR]: Novartis
[OFFICIAL_TITLE]: Study of Aliskiren/Amlodipine Combination in Essential Hypertension
[ASSIGNED_LINE_OF_THERAPY]: FIRST_LINE
[ASSIGNED_PATIENT_SEVERITY]: CHRONIC_STABLE
[DESIGN]: Arms: 3, Allocation: RANDOMIZED, Model: PARALLEL, Masking: DOUBLE
[PRIMARY_ENDPOINTS_DETAIL]: Change from Baseline in Mean Sitting Systolic Blood Pressure (msSBP) (Timeframe: 24 weeks)
[INTERVENTION_DETAILS_ENHANCED]:
NAME: Amlodipine (DRUG)
DESC: Capsules taken orally once daily.
---
NAME: hydrochlorothiazide (DRUG)
DESC: Hydrochlorothiazide capsules were taken orally once daily
---
NAME: Aliskiren (DRUG)
DESC: Aliskiren provided as film-coated tablets, taken orally once daily.

[PROTOCOL_SUMMARY]:
Compare initial combination treatment to sequential add-on strategies in hypertension.
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT00797862",
  "strategist_logic": "(1) ROUTE-DECOUPLING: All interventions are oral capsules/tablets; [STEP-1-RESULT: SIMPLE_ORAL]. (2) OS-MORTALITY CHECK: Blood pressure is a biological proxy; [STEP-2-RESULT: SURROGATE]. (3) HISTORIAN RESET: Amlodipine and HCTZ were frontline standards in 2008; [STEP-3-RESULT: ACTIVE_MODERN_STANDARD]. (4) PRIMARY EFFICACY AUDIT: Targeting core disease control (BP); [STEP-4-RESULT: PIVOTAL_INTENT]. (5) DECISION TRIGGER: Fixed path; [STEP-5-RESULT: STATIC].",
  "endpoint_rigor": "SURROGATE",
  "endpoint_structure": "SINGLE_GOAL",
  "comparator_benchmark": "ACTIVE_MODERN_STANDARD",
  "strategic_ambition": "PIVOTAL_INTENT",
  "administration_complexity": "SIMPLE_ORAL",
  "innovation_tier": "NEXT_GEN_OPTIMIZED",
  "adaptive_design": "STATIC"
}
```

---

### **Example 10: Oncology Dose Escalation + Adaptive**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT03435640
[START_YEAR]: 2018
[PHASE]: Phase 1
[ASSIGNED_INDICATION]: Other malignant neoplasms
[ASSIGNED_THERAPEUTIC_AREA]: Oncology
[AGENT]: NKTR-262 and bempegaldesleukin
[SPONSOR]: Nektar Therapeutics
[OFFICIAL_TITLE]: Phase 1/2 Study of NKTR-262 + Bempegaldesleukin +/- Nivolumab in Solid Tumors
[ASSIGNED_LINE_OF_THERAPY]: LATER_LINE
[ASSIGNED_PATIENT_SEVERITY]: ADVANCED_METASTATIC
[DESIGN]: Arms: 1, Allocation: nan, Model: SINGLE_GROUP, Purpose: TREATMENT, Masking: NONE
[PRIMARY_ENDPOINTS_DETAIL]: Dose-Limiting Toxicities (DLTs) (Timeframe: 21 days); Objective Response Rate (ORR) (Timeframe: 100 days)
[INTERVENTION_DETAILS_ENHANCED]:
NAME: NKTR-262 (DRUG)
DESC: Patients receive escalating doses of NKTR-262 IT (starting dose 0.03 mg).
---
NAME: bempegaldesleukin (DRUG)
DESC: Administered in 3-week treatment cycles.

[PROTOCOL_SUMMARY]:
Phase 1 dose escalation followed by dose expansion. Decision to start Phase 2 based on Phase 1 results.
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT03435640",
  "strategist_logic": "(1) ROUTE-DECOUPLING: NKTR-262 is intratumoral (IT) injection; [STEP-1-RESULT: ROUTINE_INFUSION]. (2) OS-MORTALITY CHECK: Primary outcomes are DLTs and ORR; [STEP-2-RESULT: SURROGATE]. (3) HISTORIAN RESET: No control group in 2018 dose-finding study; [STEP-3-RESULT: NO_CONTROL_GROUP]. (4) PRIMARY EFFICACY AUDIT: Phase 1/2 focused on safety/dosing and preliminary activity; [STEP-4-RESULT: SAFETY_DOSING]. (5) DECISION TRIGGER: Dose escalation and contingent Phase 2 start; [STEP-5-RESULT: ADAPTIVE].",
  "endpoint_rigor": "SURROGATE",
  "endpoint_structure": "MULTI_COMPOSITE",
  "comparator_benchmark": "NO_CONTROL_GROUP",
  "strategic_ambition": "SAFETY_DOSING",
  "administration_complexity": "ROUTINE_INFUSION",
  "innovation_tier": "FIRST_IN_CLASS",
  "adaptive_design": "ADAPTIVE"
}
```
