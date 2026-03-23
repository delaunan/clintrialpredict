# **Clinical Trial Enrichment: One-Shot Examples (v16.5 Master Protocol)**

**MANDATE**: These examples demonstrate **Pathophysiology-to-Meaning** mapping using the mandatory **L4 -> L3 -> L2 Staircase**.

---

### **Example 1: The "Vaccine Rule" (Target Intent)**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT04348240
[OFFICIAL_TITLE]: A Phase 3 Study to Evaluate the Efficacy and Safety of a SARS-CoV-2 RNA Vaccine (BNT162b2) in Healthy Individuals
[TRIAL_CONDITIONS_RAW]: Healthy Volunteers | Prevention of COVID-19
[PROTOCOL_SUMMARY]: This study evaluates a prophylactic vaccine against COVID-19...
[INTERVENTION_DETAILS]: NAME: BNT162b2 | DESC: Lipid nanoparticle-formulated RNA vaccine.
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT04348240",
  "clinical_evidence": "Prophylactic mRNA vaccine (BNT162b2) for COVID-19 prevention in healthy adults.",
  "mapping_logic": "(1) PATHOPHYSIOLOGY PROFILE: Prophylactic induction of adaptive immunity against SARS-CoV-2 viral infection. (2) BOTTOM-UP ESCALATION TRACE: Checked L4s; no COVID-19 specific code found. Escalated to L3; found exact clinical match 'COVID-19' [ID: 1048]. (3) RULE CHECK: Triggered 'The Vaccine Rule'. Mapped to Target Disease ID 1048. (4) PATIENT STATE: Vaccines map to PREVENTATIVE. Healthy population is UNCERTAIN_SEVERITY. Not a rare disease.",
  "gbd_cause_id": 1048,
  "gbd_indication_name": "COVID-19",
  "therapeutic_area": "Infections",
  "patient_severity": "UNCERTAIN_SEVERITY",
  "line_of_therapy": "PREVENTATIVE",
  "is_rare_disease": false
}
```

---

### **Example 2: Oncology Specificity (L4 Hierarchy)**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT00003028
[OFFICIAL_TITLE]: Fludarabine and Rituximab in Treating Patients With Previously Untreated Chronic Lymphocytic Leukemia
[TRIAL_CONDITIONS_RAW]: Leukemia | CLL
[INTERVENTION_DETAILS]: NAME: Rituximab | DESC: Anti-CD20 monoclonal antibody.
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT00003028",
  "clinical_evidence": "Previously untreated (first-line) Chronic Lymphocytic Leukemia (CLL).",
  "mapping_logic": "(1) PATHOPHYSIOLOGY PROFILE: Progressive malignancy of the bone marrow and blood (Chronic Lymphoid Leukemia). (2) BOTTOM-UP ESCALATION TRACE: Scanned L4s under Neoplasms. Found exact clinical subtype match 'Chronic lymphoid leukemia' [ID: 846]. (3) RULE CHECK: No special rules. (4) PATIENT STATE: CLL is CHRONIC_PROGRESSIVE. 'Previously untreated' in eligibility confirms FIRST_LINE. Not a rare disease.",
  "gbd_cause_id": 846,
  "gbd_indication_name": "Chronic lymphoid leukemia",
  "therapeutic_area": "Oncology",
  "patient_severity": "CHRONIC_PROGRESSIVE",
  "line_of_therapy": "FIRST_LINE",
  "is_rare_disease": false
}
```

---

### **Example 3: The "Procedure Rule" (Root Cause)**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT02934711
[OFFICIAL_TITLE]: Randomized Controlled Trial of Physical Therapy Following Total Knee Arthroplasty
[ELIGIBILITY_CRITERIA_FULL]: Patients with end-stage osteoarthritis of the knee undergoing TKA...
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT02934711",
  "clinical_evidence": "Post-surgical rehabilitation following Total Knee Arthroplasty (TKA) for end-stage knee osteoarthritis.",
  "mapping_logic": "(1) PATHOPHYSIOLOGY PROFILE: Chronic degenerative joint disease (Osteoarthritis). (2) BOTTOM-UP ESCALATION TRACE: Scanned L4s under Musculoskeletal. Found 'Osteoarthritis knee' [ID: 1015] as exact clinical match. (3) RULE CHECK: Triggered 'The Procedure Rule'. Mapped to underlying cause ID 1015. (4) PATIENT STATE: Osteoarthritis is CHRONIC_PROGRESSIVE. Line is LINE_NA for post-surgical rehab. Not a rare disease.",
  "gbd_cause_id": 1015,
  "gbd_indication_name": "Osteoarthritis knee",
  "therapeutic_area": "Musculoskeletal",
  "patient_severity": "CHRONIC_PROGRESSIVE",
  "line_of_therapy": "LINE_NA",
  "is_rare_disease": false
}
```

---

### **Example 4: Targeted Healthy Volunteer (Pipeline Alpha)**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT01234567
[OFFICIAL_TITLE]: A Phase 1 Study of ABC-123 in Healthy Volunteers
[TRIAL_CONDITIONS_RAW]: Healthy Subjects | Alzheimer's Disease
[INTERVENTION_DETAILS]: NAME: ABC-123 | DESC: Monoclonal antibody targeting beta-amyloid.
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT01234567",
  "clinical_evidence": "Phase 1 safety/PK study of a beta-amyloid antibody (ABC-123) for Alzheimer's Disease.",
  "mapping_logic": "(1) PATHOPHYSIOLOGY PROFILE: Progressive neurodegeneration associated with beta-amyloid plaque accumulation. (2) BOTTOM-UP ESCALATION TRACE: Checked L4s under Neurology; no exact match for Alzheimer's subtype. Escalated to L3; found exact match 'Alzheimer's disease and other dementias' [ID: 543]. (3) RULE CHECK: Triggered 'The Healthy Volunteer Rule'. Intent is target-specific; mapping to target ID 543 instead of UNKNOWN. (4) PATIENT STATE: CHRONIC_PROGRESSIVE. Phase 1 in healthy subjects is LINE_NA. Not rare.",
  "gbd_cause_id": 543,
  "gbd_indication_name": "Alzheimer's disease and other dementias",
  "therapeutic_area": "Neurology",
  "patient_severity": "CHRONIC_PROGRESSIVE",
  "line_of_therapy": "LINE_NA",
  "is_rare_disease": false
}
```

---

### **Example 5: Pathophysiological Mapping (DVT/PE)**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT00643201
[OFFICIAL_TITLE]: Efficacy and Safety of Apixaban in symptomatic Deep Vein Thrombosis or Pulmonary Embolism
[PROTOCOL_SUMMARY]: Evaluating an oral anticoagulant for treatment of venous thromboembolic events (VTE)...
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT00643201",
  "clinical_evidence": "Acute treatment of symptomatic Deep Vein Thrombosis (DVT) and Pulmonary Embolism (PE).",
  "mapping_logic": "(1) PATHOPHYSIOLOGY PROFILE: Acute circulatory obstruction caused by venous blood clots (VTE). (2) BOTTOM-UP ESCALATION TRACE: Checked Cardiovascular L4s; no exact match for venous clots found at L4. Escalated to L3; found 'Other cardiovascular and circulatory diseases' [ID: 507] where description explicitly includes VTE. (3) RULE CHECK: No special rules. (4) PATIENT STATE: Acute VTE is ACUTE_CRITICAL. New event treatment is FIRST_LINE. Not a rare disease.",
  "gbd_cause_id": 507,
  "gbd_indication_name": "Other cardiovascular and circulatory diseases",
  "therapeutic_area": "Cardiovascular",
  "patient_severity": "ACUTE_CRITICAL",
  "line_of_therapy": "FIRST_LINE",
  "is_rare_disease": false
}
```

---

### **Example 6: IBD Granularity (L4 Priority)**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT09876543
[OFFICIAL_TITLE]: A Phase 3 Study of Drug-X in Patients with Moderate-to-Severe Crohn's Disease
[TRIAL_CONDITIONS_RAW]: Inflammatory Bowel Disease | IBD
[ELIGIBILITY_CRITERIA_FULL]: Diagnosis of Crohn's disease for at least 6 months...
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "nct_id": "NCT09876543",
  "clinical_evidence": "Moderate-to-severe Crohn's Disease failing anti-TNF therapy.",
  "mapping_logic": "(1) PATHOPHYSIOLOGY PROFILE: Chronic transmural inflammation of the gastrointestinal tract (Crohn's). (2) BOTTOM-UP ESCALATION TRACE: Checked L4s under Digestive. Found exact clinical match 'Crohn's disease' [ID: 1025]. Although [TRIAL_CONDITIONS_RAW] says L3 'IBD', [ELIGIBILITY_CRITERIA_FULL] confirms the L4 subtype. (3) RULE CHECK: No special rules. (4) PATIENT STATE: Crohn's is CHRONIC_PROGRESSIVE. Previous failure maps to LATER_LINE. Not a rare disease.",
  "gbd_cause_id": 1025,
  "gbd_indication_name": "Crohn's disease",
  "therapeutic_area": "Gastrointestinal",
  "patient_severity": "CHRONIC_PROGRESSIVE",
  "line_of_therapy": "LATER_LINE",
  "is_rare_disease": false
}
```
