# **Run 4: Examples (The Steel Shield - REAL DATA)**
---

### **Example 1: Canonical Roche Identity (Real: NCT00773331)**
[NCT_ID]: NCT00773331
[START_YEAR]: 2009
[OFFICIAL_TITLE]: An Open Label Randomized Controlled Study to Compare the Efficacy and Safety of Once Every 4 Weeks Administration of Mircera Versus Short-acting Epoetin for the Maintenance of Hemoglobin Levels in Dialysis Patients With Chronic Renal Anemia.
[RAW_SPONSOR]: Hoffmann-La Roche
[AGENT]: methoxy polyethylene glycol-epoetin beta

[PRIMARY_ENDPOINTS]:
TITLE: Change in hemoglobin concentration between baseline and the EEP | TIMEFRAME: Week 20-28

```json
{
  "nct_id": "NCT00773331",
  "structural_forensic_monologue": "(1) RAW IDENTITY: Hoffmann-La Roche; [STEP-1-RESULT: Hoffmann-La Roche] (2) TEMPORAL AUDIT: 2009 is current identity; [STEP-2-RESULT: ROCHE/2009] (3) STANDARDIZATION: Mapping to ROCHE anchor; [STEP-3-RESULT: ROCHE] (4) TIER CLASSIFICATION: Global Top 10; [STEP-4-RESULT: TIER 1] (5) EVIDENCE HARVEST: Week 20-28; [STEP-5-RESULT: Week 20-28] (6) MATHEMATICAL COMPARISON: Single range, 28 is max; [STEP-6-RESULT: 28 > 20] (7) FINAL DURATION: 28 WEEKS; [STEP-7-RESULT: 28 WEEKS]",
  "lead_sponsor_canonical": "ROCHE",
  "sponsor_tier": "TIER 1",
  "primary_duration_value": 28,
  "primary_duration_unit": "WEEKS"
}
```

---

### **Example 2: Pre-Acquisition Naming & Tier Guardrail (Real: NCT02472964)**
[NCT_ID]: NCT02472964
[START_YEAR]: 2012
[OFFICIAL_TITLE]: A Randomized, Open-Label, 2-Way Crossover Study to Evaluate the Bioequivalence of Mylan's Buprenorphine and Naloxone Sublingual Tablets 8 mg/2 mg and Suboxone® 8 mg/2 mg Sublingual Tablets in Healthy Volunteers
[RAW_SPONSOR]: Mylan Inc.
[AGENT]: Buprenorphine/Naloxone

[PRIMARY_ENDPOINTS]:
TITLE: Area Under the Curve (AUC) | TIMEFRAME: Day 14

```json
{
  "nct_id": "NCT02472964",
  "structural_forensic_monologue": "(1) RAW IDENTITY: Mylan Inc.; [STEP-1-RESULT: Mylan Inc.] (2) TEMPORAL AUDIT: 2012 < 2020 (Viatris merger); [STEP-2-RESULT: PRE-ACQUISITION/2012] (3) STANDARDIZATION: Mapping to MYLAN (VIATRIS); [STEP-3-RESULT: MYLAN (VIATRIS)] (4) TIER CLASSIFICATION: Mylan was Mid-Cap in 2012; [STEP-4-RESULT: MID_CAP] (5) EVIDENCE HARVEST: Day 14; [STEP-5-RESULT: Day 14] (6) MATHEMATICAL COMPARISON: Single timeframe; [STEP-6-RESULT: 14] (7) FINAL DURATION: 14 DAYS; [STEP-7-RESULT: 14 DAYS]",
  "lead_sponsor_canonical": "MYLAN (VIATRIS)",
  "sponsor_tier": "MID_CAP",
  "primary_duration_value": 14,
  "primary_duration_unit": "DAYS"
}
```

---

### **Example 3: J&J Short-Form Anchor (Real: NCT01008618)**
[NCT_ID]: NCT01008618
[START_YEAR]: 2009
[OFFICIAL_TITLE]: A Phase II, Double-blind, Randomized, Placebo-controlled, Dose-finding Study of JNJ-27018966 in Patients with Diarrhea-predominant Irritable Bowel Syndrome (IBS-D)
[RAW_SPONSOR]: Janssen Pharmaceutical K.K.
[AGENT]: JNJ-27018966

[PRIMARY_ENDPOINTS]:
TITLE: Safety Assessment | TIMEFRAME: Day 1 up to Day 85 and Day 92

```json
{
  "nct_id": "NCT01008618",
  "structural_forensic_monologue": "(1) RAW IDENTITY: Janssen Pharmaceutical K.K.; [STEP-1-RESULT: Janssen Pharmaceutical K.K.] (2) TEMPORAL AUDIT: 2009 current; [STEP-2-RESULT: J&J/2009] (3) STANDARDIZATION: Janssen maps to J&J anchor; [STEP-3-RESULT: J&J] (4) TIER CLASSIFICATION: Global Top 10; [STEP-4-RESULT: TIER 1] (5) EVIDENCE HARVEST: Day 85, Day 92; [STEP-5-RESULT: Day 85, Day 92] (6) MATHEMATICAL COMPARISON: Day 92 > Day 85; [STEP-6-RESULT: 92 > 85] (7) FINAL DURATION: 92 DAYS; [STEP-7-RESULT: 92 DAYS]",
  "lead_sponsor_canonical": "J&J",
  "sponsor_tier": "TIER 1",
  "primary_duration_value": 92,
  "primary_duration_unit": "DAYS"
}
```

---

### **Example 4: Chinese Canonical Anchor (Real: NCT01970046)**
[NCT_ID]: NCT01970046
[START_YEAR]: 2013
[OFFICIAL_TITLE]: A Multi-center, Randomized, Double-blind, Placebo-controlled, Phase II Trial to Evaluate the Efficacy and Safety of Apatinib in Patients With Non-triple-negative Metastatic Breast Cancer
[RAW_SPONSOR]: Jiangsu HengRui Medicine Co., Ltd.
[AGENT]: Apatinib

[PRIMARY_ENDPOINTS]:
TITLE: Progression Free Survival | TIMEFRAME: Week 24

```json
{
  "nct_id": "NCT01970046",
  "structural_forensic_monologue": "(1) RAW IDENTITY: Jiangsu HengRui Medicine Co., Ltd.; [STEP-1-RESULT: Jiangsu HengRui Medicine Co., Ltd.] (2) TEMPORAL AUDIT: 2013; [STEP-2-RESULT: HENGRUI/2013] (3) STANDARDIZATION: Mapping to HENGRUI anchor; [STEP-3-RESULT: HENGRUI] (4) TIER CLASSIFICATION: Top 25 Global; [STEP-4-RESULT: TIER 1] (5) EVIDENCE HARVEST: Week 24; [STEP-5-RESULT: Week 24] (6) MATHEMATICAL COMPARISON: Single timeframe; [STEP-6-RESULT: 24] (7) FINAL DURATION: 24 WEEKS; [STEP-7-RESULT: 24 WEEKS]",
  "lead_sponsor_canonical": "HENGRUI",
  "sponsor_tier": "TIER 1",
  "primary_duration_value": 24,
  "primary_duration_unit": "WEEKS"
}
```

---

### **Example 5: Pre-Acquisition Historian Rule (Real: NCT05132582)**
[NCT_ID]: NCT05132582
[START_YEAR]: 2022
[OFFICIAL_TITLE]: A Phase 1/2 Trial of SGN-B7H4V in Advanced Solid Tumors
[RAW_SPONSOR]: Seagen Inc.
[AGENT]: SGN-B7H4V

[PRIMARY_ENDPOINTS]:
TITLE: Dose-Limiting Toxicities (DLTs) | TIMEFRAME: Approximately 3 years

```json
{
  "nct_id": "NCT05132582",
  "structural_forensic_monologue": "(1) RAW IDENTITY: Seagen Inc.; [STEP-1-RESULT: Seagen Inc.] (2) TEMPORAL AUDIT: START_YEAR 2022 vs 2023 Pfizer acq; 2022 < 2023 is TRUE; [STEP-2-RESULT: PRE-ACQUISITION/2022] (3) STANDARDIZATION: Mapping to SEAGEN (PFIZER); [STEP-3-RESULT: SEAGEN (PFIZER)] (4) TIER CLASSIFICATION: Established specialty; [STEP-4-RESULT: MID_CAP] (5) EVIDENCE HARVEST: Approximately 3 years; [STEP-5-RESULT: 3 years] (6) MATHEMATICAL COMPARISON: Single timeframe; [STEP-6-RESULT: 3 years] (7) FINAL DURATION: 3 YEARS; [STEP-7-RESULT: 3 YEARS]",
  "lead_sponsor_canonical": "SEAGEN (PFIZER)",
  "sponsor_tier": "MID_CAP",
  "primary_duration_value": 3,
  "primary_duration_unit": "YEARS"
}
```

---

### **Example 6: Complex Estimate & Biotech Tier (Real: NCT03758287)**
[NCT_ID]: NCT03758287
[START_YEAR]: 2016
[OFFICIAL_TITLE]: A Phase Ib Study of Ningetinib in Combination With Gefitinib in NSCLC Patients
[RAW_SPONSOR]: Sunshine Lake Pharma Co., Ltd.
[AGENT]: Ningetinib

[PRIMARY_ENDPOINTS]:
TITLE: Part 1: MTD | TIMEFRAME: Cycle 1 Day 1 to Cycle 1 Day 28
---
TITLE: Part 2: ORR | TIMEFRAME: up to approximately 36 months

```json
{
  "nct_id": "NCT03758287",
  "structural_forensic_monologue": "(1) RAW IDENTITY: Sunshine Lake Pharma Co., Ltd.; [STEP-1-RESULT: Sunshine Lake Pharma Co., Ltd.] (2) TEMPORAL AUDIT: 2016 current; [STEP-2-RESULT: SUNSHINE/2016] (3) STANDARDIZATION: Mapping to SUNSHINE LAKE PHARMA; [STEP-3-RESULT: SUNSHINE LAKE PHARMA] (4) TIER CLASSIFICATION: Emerging innovator; [STEP-4-RESULT: BIOTECH] (5) EVIDENCE HARVEST: 28 Days, 36 Months; [STEP-5-RESULT: 28 Days, 36 Months] (6) MATHEMATICAL COMPARISON: 36 Months > 28 Days; [STEP-6-RESULT: 36 Months] (7) FINAL DURATION: 36 MONTHS; [STEP-7-RESULT: 36 MONTHS]",
  "lead_sponsor_canonical": "SUNSHINE LAKE PHARMA",
  "sponsor_tier": "BIOTECH",
  "primary_duration_value": 36,
  "primary_duration_unit": "MONTHS"
}
```

---

### **Example 7: Unit Rounding (Real: NCT04532710)**
[NCT_ID]: NCT04532710
[START_YEAR]: 2020
[OFFICIAL_TITLE]: Clinical Evaluation of Tacrosolv Eye Drops
[RAW_SPONSOR]: Marinomed Biotech AG
[AGENT]: Tacrolimus

[PRIMARY_ENDPOINTS]:
TITLE: Total Ocular Symptom Score | TIMEFRAME: 0-4 hours

```json
{
  "nct_id": "NCT04532710",
  "structural_forensic_monologue": "(1) RAW IDENTITY: Marinomed Biotech AG; [STEP-1-RESULT: Marinomed Biotech AG] (2) TEMPORAL AUDIT: 2020 current; [STEP-2-RESULT: MARINOMED/2020] (3) STANDARDIZATION: Mapping to MARINOMED BIOTECH; [STEP-3-RESULT: MARINOMED BIOTECH] (4) TIER CLASSIFICATION: Biotech innovator; [STEP-4-RESULT: BIOTECH] (5) EVIDENCE HARVEST: 4 hours; [STEP-5-RESULT: 4 hours] (6) MATHEMATICAL COMPARISON: Sub-day duration; [STEP-6-RESULT: 4h < 24h] (7) FINAL DURATION: Rounding to 1 DAY per safety protocol; [STEP-7-RESULT: 1 DAYS]",
  "lead_sponsor_canonical": "MARINOMED BIOTECH",
  "sponsor_tier": "BIOTECH",
  "primary_duration_value": 1,
  "primary_duration_unit": "DAYS"
}
```

---

### **Example 9: CTTQ Chinese Anchor (Real: NCT03806686)**
[NCT_ID]: NCT03806686
[START_YEAR]: 2019
[OFFICIAL_TITLE]: A Phase I Clinical Study of TQB3473 in Healthy Subjects
[RAW_SPONSOR]: Chia Tai Tianqing Pharmaceutical Group Co., Ltd.
[AGENT]: TQB3473

[PRIMARY_ENDPOINTS]:
TITLE: Number of Participants With Adverse Events (AEs) | TIMEFRAME: Up to 28 days

```json
{
  "nct_id": "NCT03806686",
  "structural_forensic_monologue": "(1) RAW IDENTITY: Chia Tai Tianqing Pharmaceutical Group Co.; Ltd.; [STEP-1-RESULT: Chia Tai Tianqing Pharmaceutical Group Co.; Ltd.] (2) TEMPORAL AUDIT: 2019 current; [STEP-2-RESULT: CTTQ/2019] (3) STANDARDIZATION: Mapping to CTTQ anchor; [STEP-3-RESULT: CTTQ] (4) TIER CLASSIFICATION: Top 25 Global; [STEP-4-RESULT: TIER 1] (5) EVIDENCE HARVEST: 28 days; [STEP-5-RESULT: 28 days] (6) MATHEMATICAL COMPARISON: Single timeframe; [STEP-6-RESULT: 28] (7) FINAL DURATION: 28 DAYS; [STEP-7-RESULT: 28 DAYS]",
  "lead_sponsor_canonical": "CTTQ",
  "sponsor_tier": "TIER 1",
  "primary_duration_value": 28,
  "primary_duration_unit": "DAYS"
}
```

---

### **Example 10: Abbott Historian Rule (Real: NCT00851890)**
[NCT_ID]: NCT00851890
[START_YEAR]: 2009
[OFFICIAL_TITLE]: A Blinded, Randomized, Placebo-controlled Study to Evaluate the Safety, Tolerability, Pharmacokinetics, and Antiviral Activity of Multiple Doses of ABT-333 Alone and in Combination With Pegylated Interferon (pegIFN) and Ribavirin (RBV) in Subjects With Genotype 1 Chronic Hepatitis C Virus (HCV) Infection
[RAW_SPONSOR]: AbbVie (prior sponsor, Abbott)
[AGENT]: ABT-333

[PRIMARY_ENDPOINTS]:
TITLE: Mean Maximal Change From Baseline in Hepatitis C Virus Ribonucleic Acid (HCV RNA) Levels During ABT-333 Monotherapy Treatment | TIMEFRAME: Prior to the first dose on Day 1 to before first dose on Day 3

```json
{
  "nct_id": "NCT00851890",
  "structural_forensic_monologue": "(1) RAW IDENTITY: AbbVie (prior sponsor; Abbott); [STEP-1-RESULT: AbbVie (prior sponsor; Abbott)] (2) TEMPORAL AUDIT: START_YEAR 2009 vs 2013 AbbVie spin-off; 2009 < 2013 is TRUE; [STEP-2-RESULT: PRE-ACQUISITION/2009] (3) STANDARDIZATION: Mapping to ABBOTT (ABBVIE); [STEP-3-RESULT: ABBOTT (ABBVIE)] (4) TIER CLASSIFICATION: Abbott was Tier 1; [STEP-4-RESULT: TIER 1] (5) EVIDENCE HARVEST: 3 days; 28 days; [STEP-5-RESULT: 3 days; 28 days] (6) MATHEMATICAL COMPARISON: 28 days > 3 days; [STEP-6-RESULT: 28] (7) FINAL DURATION: 28 DAYS; [STEP-7-RESULT: 28 DAYS]",
  "lead_sponsor_canonical": "ABBOTT (ABBVIE)",
  "sponsor_tier": "TIER 1",
  "primary_duration_value": 28,
  "primary_duration_unit": "DAYS"
}
```
