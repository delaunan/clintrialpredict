# **Trial Enrichment Taxonomy: The Rulebook (v14 Lean)**

This document defines the allowed categories and standardized logic for the clinical enrichment pipeline.

---

## **Part 1: Categorical Indicator Definitions**

### **1. Therapeutic Modality**
- `SMALL_MOLECULE`: Traditional oral drugs, kinase inhibitors (`-tinib`), etc.
- `BIOLOGIC_MAB`: Monoclonal Antibodies and ADCs.
- `BIOLOGIC_OTHER`: Recombinant enzymes, cytokines, fusion proteins.
- `CELL_GENE_THERAPY`: CAR-T, AAV/Lentiviral vectors, etc.
- `RNA_THERAPY`: mRNA, siRNA, ASOs.
- `VACCINE`: Prophylactic or therapeutic immune stimulators.
- `PEPTIDE_HORMONES`: Insulins, GLP-1 agonists (`-tide`), etc.
- `OTHER_MODALITY`: Devices, imaging agents, supplements.

### **2. Clinical Line of Therapy**
- `FIRST_LINE`: Primary treatment / Treatment-naive.
- `LATER_LINE`: 2nd line or higher (post-SoC failure).
- `REFRACTORY_RELAPSED`: Specifically for resistant/returned disease.
- `ADJUVANT_NEOADJUVANT`: Peri-surgical treatment.
- `PREVENTATIVE`: Prophylaxis/Vaccines for healthy or at-risk populations.
- `LINE_NA`: Fallback for acute/lifestyle conditions.

### **3. Innovation Tier**
- `FIRST_IN_CLASS`: First agent to target this specific mechanism.
- `NEXT_GEN`: Improvements on existing classes or unapproved drugs for validated targets.
- `BIOSIMILAR_GENERIC`: Equivalents of already approved drugs.
- `REPURPOSED_SUPPLEMENTAL`: Known approved drugs tested for NEW indications. **Rule**: Must have approval prior to [START_YEAR].

### **4. Patient Severity**
- `ACUTE_CRITICAL`: Sudden, life-threatening (Sepsis, MI, Seizures).
- `CHRONIC_PROGRESSIVE`: Long-term worsening (Alzheimer's, CKD, MS).
- `CHRONIC_STABLE`: Long-term manageable (Hypertension, T2D).
- `ADVANCED_METASTATIC`: End-stage oncology spreading.
- `UNCERTAIN_SEVERITY`: Fallback for neutral baseline.

### **5. Endpoint Rigor Tier**
- `HARD_CLINICAL`: Survival (OS/PFS), MACE, Organ failure.
- `SURROGATE`: Lab markers (HbA1c, LDL, Viral Load, ORR).
- `SUBJECTIVE_PRO`: Human perception (Pain, QoL, Depression scales).

### **6. Target Pathway Class**
- `CHECKPOINT_INHIBITOR`: PD-1, PD-L1, CTLA-4, LAG-3, TIGIT.
- `KINASE_INHIBITOR`: EGFR, HER2, VEGFR, JAK, BTK, etc.
- `INTERLEUKIN_CYTOKINE`: IL-1, IL-4, IL-6, IL-17, IL-23, etc.
- `GPCR_TARGET`: Serotonin, Dopamine, CGRP, Opioid receptors.
- `ENZYME_MODULATOR`: Proteasome, ACE, DPP-4, SGLT2, HMG-CoA.
- `EPIGENETIC_REGULATOR`: HDAC, DNMT, Chromatin.
- `NUCLEAR_RECEPTOR`: ER, PR, AR, GR.
- `PROTEIN_DEGRADER`: PROTACs, Molecular Glues.
- `ION_CHANNEL`: Sodium, Calcium, GABA-A, NMDA.
- `DNA_REPAIR`: PARP, Topoisomerase, Antimetabolites.
- `METABOLIC_PATHWAY`: Modulating glucose/lipid flux.
- `OTHER_PATHWAY`: Fallback (Amyloid, CD20, CD19).

### **7. Allowed Therapeutic Areas**
- `Oncology`: Cancers, Tumors, Leukemias, etc.
- `Cardiovascular`: Heart, Stroke, Hypertension, etc.
- `Metabolic`: Diabetes, Obesity, NASH, etc.
- `Neurology`: Alzheimer, Parkinson, MS, Epilepsy, etc.
- `Infections`: Viral, Bacterial, Vaccines, Sepsis, etc.
- `Immunology`: Autoimmune, Arthritis, IBD, Psoriasis, etc.
- `Gastrointestinal`: Liver, GERD, Gastric, etc.
- `Renal/Urology`: Kidney, CKD, Bladder, etc.
- `Psychiatry`: Depression, Anxiety, Schizophrenia, etc.
- `Dermatology`: Skin, Acne, Rosacea, etc.
- `Respiratory`: COPD, Asthma, Cystic Fibrosis, etc.
- `Ophthalmology`: Eye, Glaucoma, Macular, etc.
- `Musculoskeletal`: Osteoarthritis, Bone, Spine, etc.
- `Hematology`: Anemia, Blood, Hemophilia, etc.
- `Reproductive`: Contraception, Infertility, Endometriosis, etc.
- `Genetic`: Rare genetic disorders, Duchenne, etc.
- `Dental`: Gingivitis, Periodontitis, etc.
- `Ear/Nose/Throat`: Otitis, Tinnitus, Sinusitis, etc.
- `Unclassified`: The fallback for everything else.

---

## **Part 2: The Industrial Pipeline Logic**

### **1. Molecular Target Extraction (The "Human-to-Hugo" Bridge)**
- **Requirement**: List the clinical names of the targets for **EVERY** drug in the intervention.
- **Format**: `Target A | Target B` (e.g., `PD-1 | VEGF`).
- **Precision**: If the drug is a combination, do not omit the background therapy targets.

### **2. GBD Mapping (Name Match)**
- **Requirement**: Provide the **GBD_INDICATION_NAME** exactly as it appears in the menu.
- **ID Offloading**: The Python pipeline will handle the mapping to the numeric ID.

### **3. Duration Logic (Math Gate)**
- **Requirement**: Provide the `raw_duration_value` and `raw_duration_unit`.
- **Unit Options**: `years`, `months`, `weeks`, `days`.
- **Calculation**: Python will apply the 4.33 constant and 0.5 month floor.

---

## **Part 3: Scientific Integrity & Anti-Hallucination**

1. **Negative Constraint**: If you do not know the target, output `Unknown`. **NEVER** guess a target (like CD19) just because of the disease name (like B-CLL).
2. **Temporal Anchoring**: All novelty and hurdle tiers MUST be judged based on the SoCs available in **[START_YEAR]**.