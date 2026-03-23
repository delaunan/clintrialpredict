# FEW-SHOT EXAMPLES: MOLECULAR BLUEPRINT (v15.0)

### EXAMPLE 1: ONCOLOGY (SIMPLE MONOTHERAPY)
**Context**: [START_YEAR]: 2018 | [ASSIGNED_INDICATION]: Non-small cell lung cancer | [INTERVENTION]: NAME: Pembrolizumab (Aliases: Keytruda) [BIOLOGICAL]

**Output**:
```json
{
  "nct_id": "NCT00000001",
  "pharmacology_logic": "(0) ALPHA_DRUG: Pembrolizumab. (1) IDENTITY: Pembrolizumab is the alpha investigational agent. (2) MOA (PROTEIN REFERENCE): Pembrolizumab targets the Programmed Death Receptor 1 (CD274), shorthand symbol PD-1. (3) HISTORIAN CHECK: As of 2018, Pembrolizumab was already approved for NSCLC (first approved in 2015). (4) PRECISION: No specific biomarker stratification mentioned in this arm. (5) JUSTIFICATION: It is a monoclonal antibody (BIOLOGIC_MAB) in the IMMUNO_ONCOLOGY pathway. (6) FINAL VERIFICATION: I found an approval for Pembrolizumab in NSCLC in 2015, which is before 2018, so I am selecting PRECEDENT_IN_INDICATION.",
  "alpha_drug_name": "Pembrolizumab",
  "therapeutic_modality": "BIOLOGIC_MAB",
  "molecular_targets": "PD-1",
  "target_pathway_class": "IMMUNO_ONCOLOGY",
  "target_precedent": "PRECEDENT_IN_INDICATION",
  "biomarker_stratification": false,
  "biomarker_description": "N/A"
}
```

---

### EXAMPLE 2: BIOSIMILAR (INHERITED PRECEDENT)
**Context**: [START_YEAR]: 2016 | [ASSIGNED_INDICATION]: Non-Hodgkin Lymphoma | [INTERVENTION]: NAME: ABP 798 (Aliases: Rituximab Biosimilar) [BIOLOGICAL]

**Output**:
```json
{
  "nct_id": "NCT02747043",
  "pharmacology_logic": "(1) IDENTITY: ABP 798 (biosimilar to rituximab) is the alpha agent. (2) MOA (PROTEIN REFERENCE): Both ABP 798 and Rituximab target the Cluster of Differentiation 20 (CD20) protein. (3) HISTORIAN CHECK: By 2016, Rituximab was already approved for Non-Hodgkin Lymphoma. Per the Biosimilar Rule, the target inherits this precedent. Target has PRECEDENT_IN_INDICATION. (4) PRECISION: No molecular biomarker stratification beyond CD20 positivity. (5) JUSTIFICATION: It is a monoclonal antibody (BIOLOGIC_MAB) targeting the OTHER_PATHWAY (B-cell depletion).",
  "alpha_drug_name": "ABP 798",
  "therapeutic_modality": "BIOLOGIC_MAB",
  "molecular_targets": "CD20",
  "target_pathway_class": "OTHER_PATHWAY",
  "target_precedent": "PRECEDENT_IN_INDICATION",
  "biomarker_stratification": false,
  "biomarker_description": "N/A"
}
```

---

### EXAMPLE 3: RARE DISEASE (RNA THERAPY)
**Context**: [START_YEAR]: 2016 | [ASSIGNED_INDICATION]: Hereditary transthyretin-mediated amyloidosis | [INTERVENTION]: NAME: Patisiran (Aliases: ALN-TTR02) [RNA_THERAPY]

**Output**:
```json
{
  "nct_id": "NCT00000002",
  "pharmacology_logic": "(1) IDENTITY: Patisiran is the alpha agent. (2) MOA (PROTEIN REFERENCE): Patisiran targets the Transthyretin (TTR) protein mRNA via RNA interference (RNAi). (3) HISTORIAN CHECK: In 2016, no RNAi therapy for hATTR was approved. Landmark Check: Patisiran was the first-ever RNAi drug for this indication, approved in 2018. Target has NO_PRECEDENT. (4) PRECISION: No molecular biomarker stratification; inclusion based on clinical diagnosis. (5) JUSTIFICATION: It is an siRNA (RNA_THERAPY) targeting the METABOLIC_REPROGRAMMING pathway.",
  "alpha_drug_name": "Patisiran",
  "therapeutic_modality": "RNA_THERAPY",
  "molecular_targets": "TTR",
  "target_pathway_class": "METABOLIC_REPROGRAMMING",
  "target_precedent": "NO_PRECEDENT",
  "biomarker_stratification": false,
  "biomarker_description": "N/A"
}
```

---

### EXAMPLE 4: METABOLIC (MULTI-TARGET PEPTIDE)
**Context**: [START_YEAR]: 2021 | [ASSIGNED_INDICATION]: Type 2 Diabetes | [INTERVENTION]: NAME: Tirzepatide (Aliases: LY3298176) [PEPTIDE]

**Output**:
```json
{
  "nct_id": "NCT00000003",
  "pharmacology_logic": "(1) IDENTITY: Tirzepatide is the alpha agent. (2) MOA (PROTEIN REFERENCE): Tirzepatide is a dual agonist of the Glucagon-like peptide-1 receptor (GLP-1R) and Glucose-dependent insulinotropic polypeptide receptor (GIPR). (3) HISTORIAN CHECK: As of 2021, GLP-1R agonists (e.g., Liraglutide) were already approved for Type 2 Diabetes. Although no dual GLP-1/GIP agonist was approved, the target GLP-1R has PRECEDENT_IN_INDICATION. (4) PRECISION: No molecular biomarker stratification. (5) JUSTIFICATION: It is a peptide hormone (PEPTIDE_HORMONES) in the METABOLIC_REPROGRAMMING pathway.",
  "alpha_drug_name": "Tirzepatide",
  "therapeutic_modality": "PEPTIDE_HORMONES",
  "molecular_targets": "GLP-1R | GIPR",
  "target_pathway_class": "METABOLIC_REPROGRAMMING",
  "target_precedent": "PRECEDENT_IN_INDICATION",
  "biomarker_stratification": false,
  "biomarker_description": "N/A"
}
```

---

### EXAMPLE 5: CARDIOVASCULAR (SMALL MOLECULE COCKTAIL)
**Context**: [START_YEAR]: 2014 | [ASSIGNED_INDICATION]: Heart Failure | [INTERVENTION]: NAME: LCZ696 (Aliases: Sacubitril/Valsartan, Entresto) [SMALL_MOLECULE]

**Output**:
```json
{
  "nct_id": "NCT00000004",
  "pharmacology_logic": "(1) IDENTITY: LCZ696 is the alpha agent. (2) MOA (PROTEIN REFERENCE): Sacubitril inhibits Neprilysin (Neutral Endopeptidase), and Valsartan is an Angiotensin II Receptor Type 1 (AT1) antagonist. (3) HISTORIAN CHECK: In 2014, Valsartan (AT1 blocker) was already approved for heart failure. While Neprilysin was a novel target, the presence of an approved AT1 blocker means the mechanism has PRECEDENT_IN_INDICATION. (4) PRECISION: Patients selected based on NT-proBNP elevation (molecular biomarker of heart strain). (5) JUSTIFICATION: It is a small molecule combination (SMALL_MOLECULE) targeting the ENZYME_MODULATOR pathway.",
  "alpha_drug_name": "LCZ696",
  "therapeutic_modality": "SMALL_MOLECULE",
  "molecular_targets": "Neprilysin | AT1",
  "target_pathway_class": "ENZYME_MODULATOR",
  "target_precedent": "PRECEDENT_IN_INDICATION",
  "biomarker_stratification": true,
  "biomarker_description": "NT-proBNP elevation"
}
```

---

### EXAMPLE 6: CELL THERAPY (HEMATOLOGY)
**Context**: [START_YEAR]: 2019 | [ASSIGNED_INDICATION]: Multiple Myeloma | [INTERVENTION]: NAME: Idecabtagene vicleucel (Aliases: bb2121, Abecma) [CELL_THERAPY]

**Output**:
```json
{
  "nct_id": "NCT00000005",
  "pharmacology_logic": "(1) IDENTITY: Idecabtagene vicleucel is the alpha agent. (2) MOA (PROTEIN REFERENCE): It is a CAR T-cell therapy targeting B-cell maturation antigen (CD269), shorthand symbol BCMA. (3) HISTORIAN CHECK: As of 2019, several CAR-T therapies were approved for CD19, but NO therapy was approved for BCMA. Target has NO_PRECEDENT. (4) PRECISION: Biomarker stratification is true as it requires BCMA expression. (5) JUSTIFICATION: It is a CAR-T cell therapy (CELL_GENE_THERAPY) in the IMMUNO_ONCOLOGY pathway.",
  "alpha_drug_name": "Idecabtagene vicleucel",
  "therapeutic_modality": "CELL_GENE_THERAPY",
  "molecular_targets": "BCMA",
  "target_pathway_class": "IMMUNO_ONCOLOGY",
  "target_precedent": "NO_PRECEDENT",
  "biomarker_stratification": true,
  "biomarker_description": "BCMA expression"
}
```

---

### EXAMPLE 7: ONCOLOGY (ANTIBODY-DRUG CONJUGATE)
**Context**: [START_YEAR]: 2019 | [ASSIGNED_INDICATION]: Breast Cancer | [INTERVENTION]: NAME: Trastuzumab deruxtecan (Aliases: DS-8201, Enhertu) [BIOLOGIC_ADC]

**Output**:
```json
{
  "nct_id": "NCT00000007",
  "pharmacology_logic": "(1) IDENTITY: Trastuzumab deruxtecan is the alpha agent. (2) MOA (PROTEIN REFERENCE): It is an ADC targeting Human Epidermal Growth Factor Receptor 2 (CD340), shorthand symbol HER2, with a topoisomerase I inhibitor payload. (3) HISTORIAN CHECK: By 2019, T-DM1 was already approved for HER2+ breast cancer. Target (HER2) has PRECEDENT_IN_INDICATION. (4) PRECISION: Biomarker stratification is true; patients must be HER2-positive (IHC 3+ or FISH+). (5) JUSTIFICATION: It is an ADC (BIOLOGIC_ADC) in the KINASE_INHIBITOR pathway.",
  "alpha_drug_name": "Trastuzumab deruxtecan",
  "therapeutic_modality": "BIOLOGIC_ADC",
  "molecular_targets": "HER2",
  "target_pathway_class": "KINASE_INHIBITOR",
  "target_precedent": "PRECEDENT_IN_INDICATION",
  "biomarker_stratification": true,
  "biomarker_description": "HER2+"
}
```

---

### EXAMPLE 8: RARE DISEASE (GENE EDITING)
**Context**: [START_YEAR]: 2022 | [ASSIGNED_INDICATION]: Sickle Cell Disease | [INTERVENTION]: NAME: Exagamglogene autotemcel (Aliases: exa-cel, Casgevy) [CELL_GENE_THERAPY]

**Output**:
```json
{
  "nct_id": "NCT00000008",
  "pharmacology_logic": "(1) IDENTITY: Exa-cel is the alpha agent. (2) MOA (PROTEIN REFERENCE): It uses CRISPR/Cas9 to edit the BCL11A (B-cell lymphoma/leukemia 11A) gene. (3) HISTORIAN CHECK: As of 2022, no CRISPR gene-edited therapy was approved. Target has NO_PRECEDENT. (4) PRECISION: Patients selected based on genetic diagnosis of Sickle Cell Disease (HBB mutation). (5) JUSTIFICATION: It is a gene-edited cell therapy (CELL_GENE_THERAPY) in the OTHER_PATHWAY (Hemoglobin regulation).",
  "alpha_drug_name": "Exa-cel",
  "therapeutic_modality": "CELL_GENE_THERAPY",
  "molecular_targets": "BCL11A",
  "target_pathway_class": "OTHER_PATHWAY",
  "target_precedent": "NO_PRECEDENT",
  "biomarker_stratification": true,
  "biomarker_description": "HBB mutation / SCD diagnosis"
}
```

---

### EXAMPLE 9: ONCOLOGY (RADIOPHARMACEUTICAL)
**Context**: [START_YEAR]: 2021 | [ASSIGNED_INDICATION]: Prostate Cancer | [INTERVENTION]: NAME: 177Lu-PSMA-617 (Aliases: Pluvicto) [RADIOPHARMACEUTICAL]

**Output**:
```json
{
  "nct_id": "NCT00000009",
  "pharmacology_logic": "(1) IDENTITY: 177Lu-PSMA-617 is the alpha agent. (2) MOA (PROTEIN REFERENCE): It targets Prostate-Specific Membrane Antigen (Folate hydrolase 1), shorthand symbol PSMA, delivering a radioactive isotope. (3) HISTORIAN CHECK: By 2021, no PSMA-targeted radiopharmaceutical was approved for therapy. Target has NO_PRECEDENT. (4) PRECISION: Patients must be PSMA-positive on a PET scan. (5) JUSTIFICATION: It is a targeted radioactive agent (RADIOPHARMACEUTICAL) in the OTHER_PATHWAY.",
  "alpha_drug_name": "177Lu-PSMA-617",
  "therapeutic_modality": "RADIOPHARMACEUTICAL",
  "molecular_targets": "PSMA",
  "target_pathway_class": "OTHER_PATHWAY",
  "target_precedent": "NO_PRECEDENT",
  "biomarker_stratification": true,
  "biomarker_description": "PSMA-positive PET"
}
```

---

### EXAMPLE 10: INFECTIOUS VACCINE (TAXONOMIC DISCIPLINE)
**Context**: [START_YEAR]: 2015 | [ASSIGNED_INDICATION]: Influenza | [INTERVENTION]: NAME: IL-YANG Flu Vaccine QIV [BIOLOGICAL]

**Output**:
```json
{
  "nct_id": "NCT02480998",
  "pharmacology_logic": "(1) IDENTITY: IL-YANG Flu Vaccine QIV is the investigational agent. (2) MOA (PROTEIN REFERENCE): The vaccine targets influenza virus antigens (Hemagglutinin/Neuraminidase) to induce an immune response. (3) HISTORIAN CHECK: As of 2015, influenza vaccines were already approved. Target has PRECEDENT_IN_INDICATION. (4) PRECISION: No molecular biomarker stratification. (5) JUSTIFICATION: It is a vaccine (VACCINE) for an infectious disease, thus categorized as OTHER_PATHWAY per the rulebook.",
  "alpha_drug_name": "IL-YANG Flu Vaccine QIV",
  "therapeutic_modality": "VACCINE",
  "molecular_targets": "Influenza Virus Antigens",
  "target_pathway_class": "OTHER_PATHWAY",
  "target_precedent": "PRECEDENT_IN_INDICATION",
  "biomarker_stratification": false,
  "biomarker_description": "N/A"
}
```

---

### EXAMPLE 11: MULTI-TARGET COMBINATION (ALPHA-FIRST ORDERING)
**Context**: [START_YEAR]: 2024 | [ASSIGNED_INDICATION]: Pancreatic Cancer | [INTERVENTION]: NAME: Chiauranib [SMALL_MOLECULE] combined with Paclitaxel and Gemcitabine.

**Output**:
```json
{
  "nct_id": "NCT06492915",
  "pharmacology_logic": "(1) IDENTITY: Chiauranib is the alpha investigational agent; paclitaxel and gemcitabine are background chemotherapy. (2) MOA (PROTEIN REFERENCE): Chiauranib inhibits VEGFR, Aurora B kinase, and CSF-1R. Paclitaxel targets Tubulin. (3) HISTORIAN CHECK: By 2024, VEGFR inhibitors (e.g., Sunitinib) and CSF-1R inhibitors (e.g., Pexidartinib) were approved for other indications, but not for pancreatic adenocarcinoma. Target has PRECEDENT_IN_OTHER. (4) PRECISION: No molecular biomarker stratification. (5) JUSTIFICATION: Chiauranib is a small molecule (SMALL_MOLECULE) targeting the KINASE_INHIBITOR pathway.",
  "alpha_drug_name": "Chiauranib",
  "therapeutic_modality": "SMALL_MOLECULE",
  "molecular_targets": "VEGFR | Aurora B | CSF-1R | Tubulin",
  "target_pathway_class": "KINASE_INHIBITOR",
  "target_precedent": "PRECEDENT_IN_OTHER",
  "biomarker_stratification": false,
  "biomarker_description": "N/A"
}
```

---

### EXAMPLE 12: ONCOLOGY (PRECEDENT IN OTHER)
**Context**: [START_YEAR]: 2015 | [ASSIGNED_INDICATION]: Head and Neck Squamous Cell Carcinoma | [INTERVENTION]: NAME: Pembrolizumab [BIOLOGICAL]

**Output**:
```json
{
  "nct_id": "NCT02358031",
  "pharmacology_logic": "(0) ALPHA_DRUG: Pembrolizumab. (1) IDENTITY: Pembrolizumab is the alpha agent. (2) MOA (PROTEIN REFERENCE): Pembrolizumab targets the Programmed Death Receptor 1 (CD274), shorthand symbol PD-1. (3) HISTORIAN CHECK: By the end of 2015, Pembrolizumab was approved for Melanoma (approved 2014), but it was NOT yet approved for Head and Neck cancer (approved 2016). (4) PRECISION: Biomarker stratification is true; patients are selected based on PD-L1 expression (CPS score). (5) JUSTIFICATION: It is a monoclonal antibody (BIOLOGIC_MAB) in the IMMUNO_ONCOLOGY pathway. (6) FINAL VERIFICATION: I found an approval in Melanoma (2014) but not for HNSCC (2016), so I am selecting PRECEDENT_IN_OTHER per the decision matrix.",
  "alpha_drug_name": "Pembrolizumab",
  "therapeutic_modality": "BIOLOGIC_MAB",
  "molecular_targets": "PD-1",
  "target_pathway_class": "IMMUNO_ONCOLOGY",
  "target_precedent": "PRECEDENT_IN_OTHER",
  "biomarker_stratification": true,
  "biomarker_description": "PD-L1 expression (CPS)"
}
```

---

### EXAMPLE 13: INFECTIOUS DISEASE (THERAPEUTIC mAb vs VACCINE BOUNDARY)
**Context**: [START_YEAR]: 2015 | [ASSIGNED_INDICATION]: Influenza A | [INTERVENTION]: NAME: MEDI8852 [BIOLOGICAL]

**Output**:
```json
{
  "nct_id": "NCT02603952",
  "pharmacology_logic": "(1) IDENTITY: MEDI8852 is the alpha investigational agent. (2) MOA (PROTEIN REFERENCE): MEDI8852 is a monoclonal antibody targeting the hemagglutinin (HA) protein of influenza A viruses. (3) HISTORIAN CHECK: As of 2015, influenza vaccines were approved, but NO therapeutic monoclonal antibody targeting the influenza HA protein was approved. Intent Boundary: Prophylactic vaccine history does NOT establish precedent for a therapeutic mAb. Target has NO_PRECEDENT. (4) PRECISION: No molecular biomarker stratification. (5) JUSTIFICATION: It is a monoclonal antibody (BIOLOGIC_MAB) in the OTHER_PATHWAY.",
  "alpha_drug_name": "MEDI8852",
  "therapeutic_modality": "BIOLOGIC_MAB",
  "molecular_targets": "Hemagglutinin",
  "target_pathway_class": "OTHER_PATHWAY",
  "target_precedent": "NO_PRECEDENT",
  "biomarker_stratification": false,
  "biomarker_description": "N/A"
}
```

---

### EXAMPLE 14: PIPELINE CODE RESOLUTION (KNOWLEDGE RETRIEVAL)
**Context**: [START_YEAR]: 2023 | [ASSIGNED_INDICATION]: Solid Tumors | [INTERVENTION]: NAME: LBL-024 [BIOLOGICAL]

**Output**:
```json
{
  "nct_id": "NCT05161039",
  "pharmacology_logic": "(1) IDENTITY: LBL-024 is the alpha investigational agent. (2) MOA (PROTEIN REFERENCE): LBL-024 is a bispecific antibody targeting Programmed Death Ligand 1 (CD274), shorthand PD-L1, and 4-1BB (CD137). (3) HISTORIAN CHECK: By 2023, PD-L1 inhibitors were approved, but no bispecific antibody targeting PD-L1 and 4-1BB was approved. Target has NO_PRECEDENT for this bispecific mechanism. (4) PRECISION: No specific biomarker stratification mentioned in this arm. (5) JUSTIFICATION: It is a bispecific antibody (BIOLOGIC_MAB) in the IMMUNO_ONCOLOGY pathway.",
  "alpha_drug_name": "LBL-024",
---

### EXAMPLE 15: CODE RESOLUTION (FACTOR Xa)
**Context**: [START_YEAR]: 2008 | [ASSIGNED_INDICATION]: Venous Thromboembolism | [INTERVENTION]: NAME: PD 0348292 [SMALL_MOLECULE]

**Output**:
```json
{
  "nct_id": "NCT00306254",
  "pharmacology_logic": "(0) ALPHA_DRUG: PD 0348292. (1) IDENTITY: PD 0348292 is the alpha investigational agent. (2) MOA (PROTEIN REFERENCE): PD 0348292 is the development code for Apixaban, which targets Coagulation Factor Xa, shorthand symbol Factor Xa. (3) HISTORIAN CHECK: As of 2008, no Factor Xa inhibitor was approved for VTE (Apixaban was approved in 2011). Target has NO_PRECEDENT. (4) PRECISION: No molecular biomarker stratification. (5) JUSTIFICATION: It is a small molecule (SMALL_MOLECULE) targeting the ENZYME_MODULATOR pathway. (6) FINAL VERIFICATION: I resolved the code PD 0348292 to Apixaban; no global approvals existed in 2008, so I am selecting NO_PRECEDENT.",
  "alpha_drug_name": "Apixaban (PD 0348292)",
  "therapeutic_modality": "SMALL_MOLECULE",
  "molecular_targets": "Factor Xa",
  "target_pathway_class": "ENZYME_MODULATOR",
  "target_precedent": "NO_PRECEDENT",
  "biomarker_stratification": false,
  "biomarker_description": "N/A"
}
```

---

### EXAMPLE 16: IO ANCHOR (COMBINATION THERAPY)
**Context**: [START_YEAR]: 2024 | [ASSIGNED_INDICATION]: NSCLC | [INTERVENTION]: NAME: IN10018 and D-1553 combined with Pembrolizumab.

**Output**:
```json
{
  "nct_id": "NCT07174908",
  "pharmacology_logic": "(0) ALPHA_DRUG: IN10018 and D-1553. (1) IDENTITY: IN10018 (FAK inhibitor) and D-1553 (KRAS G12C inhibitor) are the alpha agents; Pembrolizumab (anti-PD-1) is background. (2) MOA (PROTEIN REFERENCE): IN10018 targets FAK; D-1553 targets Kras G12C; Pembrolizumab targets PD-1. (3) HISTORIAN CHECK: By 2024, KRAS G12C inhibitors and PD-1 inhibitors were approved for NSCLC. Target has PRECEDENT_IN_INDICATION. (4) PRECISION: Patients must have KRAS G12C mutation. (5) JUSTIFICATION: Although small molecules are involved, the presence of PD-1 targets triggers the IMMUNO_ONCOLOGY anchor rule. (6) FINAL VERIFICATION: IO Anchor Rule applied due to PD-1 targeting.",
  "alpha_drug_name": "IN10018 | D-1553",
  "therapeutic_modality": "SMALL_MOLECULE",
  "molecular_targets": "Fak | Kras G12C | PD-1",
  "target_pathway_class": "IMMUNO_ONCOLOGY",
  "target_precedent": "PRECEDENT_IN_INDICATION",
  "biomarker_stratification": true,
  "biomarker_description": "KRAS G12C mutation"
}
```

