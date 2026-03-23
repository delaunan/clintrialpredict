import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, TargetEncoder

# ==============================================================================
# 1. UNIFIED PIPELINE REGISTRY (The Single Source of Truth)
# ==============================================================================
# Embedded directly for joblib portability, type safety, and zero-path errors.

PIPELINE_REGISTRY = {
    "FIELDS": {
        "therapeutic_area_ml": {
            "ui": {
                "label": "Therapeutic Area",
                "pillar": "Therapeutic Context",
                "subgroup": "Therapeutic Area Profile",
                "priority": 0,
                "options": [
                    ["CARDIOVASCULAR", "Cardiovascular"],
                    ["DENTAL", "Dental"],
                    ["DERMATOLOGY", "Dermatology"],
                    ["EAR/NOSE/THROAT", "Ear/Nose/Throat"],
                    ["GASTROINTESTINAL", "Gastrointestinal"],
                    ["GENETIC", "Genetic"],
                    ["HEMATOLOGY", "Hematology"],
                    ["IMMUNOLOGY", "Immunology"],
                    ["INFECTIONS", "Infections"],
                    ["METABOLIC", "Metabolic"],
                    ["MUSCULOSKELETAL", "Musculoskeletal"],
                    ["NEUROLOGY", "Neurology"],
                    ["ONCOLOGY", "Oncology"],
                    ["OPHTHALMOLOGY", "Ophthalmology"],
                    ["PSYCHIATRY", "Psychiatry"],
                    ["RENAL/UROLOGY", "Renal/Urology"],
                    ["REPRODUCTIVE", "Reproductive"],
                    ["RESPIRATORY", "Respiratory"],
                    ["UNCLASSIFIED", "Other / Unclassified"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "DENTAL": [0, "Dental"],
                "DERMATOLOGY": [1, "Dermatology"],
                "METABOLIC": [2, "Metabolic"],
                "UNCLASSIFIED": [3, "Other / Unclassified"],
                "UNKNOWN": [3, "Other / Unclassified"],
                "RESPIRATORY": [4, "Respiratory"],
                "INFECTIONS": [5, "Infections"],
                "OPHTHALMOLOGY": [6, "Ophthalmology"],
                "PSYCHIATRY": [7, "Psychiatry"],
                "REPRODUCTIVE": [8, "Reproductive"],
                "MUSCULOSKELETAL": [9, "Musculoskeletal"],
                "EAR/NOSE/THROAT": [10, "Ear/Nose/Throat"],
                "GASTROINTESTINAL": [11, "Gastrointestinal"],
                "RENAL/UROLOGY": [12, "Renal/Urology"],
                "CARDIOVASCULAR": [13, "Cardiovascular"],
                "NEUROLOGY": [14, "Neurology"],
                "GENETIC": [15, "Genetic"],
                "IMMUNOLOGY": [16, "Immunology"],
                "HEMATOLOGY": [17, "Hematology"],
                "ONCOLOGY": [18, "Oncology"]
            }
        },
        "gbd_cause_id_3_ml": {
            "ui": {
                "label": "Indication",
                "pillar": "Therapeutic Context",
                "subgroup": "Therapeutic Area Profile",
                "priority": 1
            },
            "encoding": "target"
        },
        "is_rare_disease_ml": {
            "ui": {
                "label": "Rare Condition",
                "pillar": "Therapeutic Context",
                "subgroup": "Therapeutic Area Profile",
                "priority": 2,
                "options": [
                    ["1", "Yes"],
                    ["0", "Unlikely"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                0: [0, "Unlikely"],
                1: [1, "Yes"],
                "0": [0, "Unlikely"],
                "1": [1, "Yes"],
                "F": [0, "Unlikely"],
                "T": [1, "Yes"],
                "FALSE": [0, "Unlikely"],
                "TRUE": [1, "Yes"],
                "NO": [0, "Unlikely"],
                "YES": [1, "Yes"],
                "UNKNOWN": [0, "Unlikely"]
            }
        },
        "phase_ml": {
            "ui": {
                "label": "Clinical Phase",
                "pillar": "Therapeutic Context",
                "subgroup": "Development Phase and Goal",
                "priority": 3,
                "options": [
                    ["PHASE1/PHASE2", "Phase 1/2"],
                    ["PHASE2", "Phase 2"],
                    ["PHASE2/PHASE3", "Phase 2/3"],
                    ["PHASE3", "Phase 3"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "PHASE1/PHASE2": [1, "Phase 1/2"],
                "PHASE2": [2, "Phase 2"],
                "PHASE2/PHASE3": [3, "Phase 2/3"],
                "PHASE3": [4, "Phase 3"],
                "UNKNOWN": [2, "Not Specified"]
            }
        },
        "strategic_ambition_ml": {
            "ui": {
                "label": "Regulatory Intent",
                "pillar": "Therapeutic Context",
                "subgroup": "Development Phase and Goal",
                "priority": 4,
                "options": [
                    ["SAFETY_DOSING", "Early Phase / Dose Finding"],
                    ["SIGNAL_SEARCH", "Efficacy / Signal Detection"],
                    ["PIVOTAL_INTENT", "Confirmatory / Registration"],
                    ["UNKNOWN", "Unknown Intent"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "SIGNAL_SEARCH": [1, "Efficacy / Signal Detection"],
                "PIVOTAL_INTENT": [2, "Confirmatory / Registration"],
                "SAFETY_DOSING": [3, "Early Phase / Dose Finding"],
                "UNKNOWN": [2, "Unknown Intent"]
            }
        },
        "target_precedent_ml": {
            "ui": {
                "label": "Target Precedent",
                "pillar": "Scientific Attempt",
                "subgroup": "Biological Profile",
                "priority": 5,
                "options": [
                    ["UNKNOWN", "Not Specified"],
                    ["NO_PRECEDENT", "Novel Target (No Prior Approvals)"],
                    ["PRECEDENT_IN_OTHER", "Established in Other"],
                    ["PRECEDENT_IN_INDICATION", "Established in Indication"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "UNKNOWN": [0, "Not Specified"],
                "NO_PRECEDENT": [0, "Novel Target (No Prior Approvals)"],
                "PRECEDENT_IN_OTHER": [1, "Established in Other"],
                "PRECEDENT_IN_INDICATION": [2, "Established in Indication"]
            }
        },
        "target_pathway_class_ml": {
            "ui": {
                "label": "Pathway Profile",
                "pillar": "Scientific Attempt",
                "subgroup": "Biological Profile",
                "priority": 6,
                "options": [
                    ["METABOLIC_REPROGRAMMING", "Metabolic Reprogramming"],
                    ["ION_CHANNEL", "Ion Channel"],
                    ["GPCR_TARGET", "GPCR Target"],
                    ["ENZYME_MODULATOR", "Enzyme Modulator"],
                    ["NUCLEAR_RECEPTOR", "Nuclear Receptor"],
                    ["INTERLEUKIN_CYTOKINE", "Interleukin Cytokine"],
                    ["KINASE_INHIBITOR", "Kinase Inhibitor"],
                    ["DNA_REPAIR", "DNA Repair"],
                    ["IMMUNO_ONCOLOGY", "Immuno Oncology"],
                    ["PROTEIN_DEGRADER", "Protein Degrader"],
                    ["EPIGENETIC_REGULATOR", "Epigenetic Regulator"],
                    ["OTHER_PATHWAY", "Other Pathway / Unknown"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "METABOLIC_REPROGRAMMING": [0, "Metabolic Reprogramming"],
                "ION_CHANNEL": [1, "Ion Channel"],
                "GPCR_TARGET": [2, "GPCR Target"],
                "ENZYME_MODULATOR": [3, "Enzyme Modulator"],
                "OTHER_PATHWAY": [4, "Other Pathway / Unknown"],
                "UNKNOWN": [4, "Other Pathway / Unknown"],
                "NUCLEAR_RECEPTOR": [5, "Nuclear Receptor"],
                "INTERLEUKIN_CYTOKINE": [6, "Interleukin Cytokine"],
                "KINASE_INHIBITOR": [7, "Kinase Inhibitor"],
                "DNA_REPAIR": [8, "DNA Repair"],
                "IMMUNO_ONCOLOGY": [9, "Immuno Oncology"],
                "PROTEIN_DEGRADER": [10, "Protein Degrader"],
                "EPIGENETIC_REGULATOR": [11, "Epigenetic Regulator"]
            }
        },
        "therapeutic_modality_ml": {
            "ui": {
                "label": "Therapeutic Modality",
                "pillar": "Scientific Attempt",
                "subgroup": "Biological Profile",
                "priority": 7,
                "options": [
                    ["SMALL_MOLECULE", "Small Molecule"],
                    ["BIOLOGIC_MAB", "Biologic Mab"],
                    ["BIOLOGIC_ADC", "Biologic Adc"],
                    ["BIOLOGIC_OTHER", "Biologic Other"],
                    ["CELL_GENE_THERAPY", "Cell Gene Therapy"],
                    ["RNA_THERAPY", "Rna Therapy"],
                    ["VACCINE", "Vaccine"],
                    ["RADIOPHARMACEUTICAL", "Radiopharmaceutical"],
                    ["PEPTIDE_HORMONES", "Peptide Hormones"],
                    ["OTHER_MODALITY", "Other Modality"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "VACCINE": [0, "Vaccine"],
                "PEPTIDE_HORMONES": [1, "Peptide Hormones"],
                "OTHER_MODALITY": [2, "Other Modality"],
                "UNKNOWN": [2, "Other Modality"],
                "SMALL_MOLECULE": [3, "Small Molecule"],
                "BIOLOGIC_MAB": [4, "Biologic Mab"],
                "BIOLOGIC_OTHER": [5, "Biologic Other"],
                "CELL_GENE_THERAPY": [6, "Cell Gene Therapy"],
                "RNA_THERAPY": [7, "Rna Therapy"],
                "BIOLOGIC_ADC": [8, "Biologic Adc"],
                "RADIOPHARMACEUTICAL": [9, "Radiopharmaceutical"]
            }
        },
        "innovation_tier_ml": {
            "ui": {
                "label": "Innovation Rank",
                "pillar": "Scientific Attempt",
                "subgroup": "Biological Profile",
                "priority": 8,
                "options": [
                    ["ESTABLISHED_COPY", "Established / Copy"],
                    ["NEXT_GEN_OPTIMIZED", "Next-Gen / Optimized"],
                    ["FIRST_IN_CLASS", "First-in-Class (Novel)"],
                    ["UNKNOWN", "Not Specified"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "ESTABLISHED_COPY": [0, "Established / Copy"],
                "NEXT_GEN_OPTIMIZED": [1, "Next-Gen / Optimized"],
                "FIRST_IN_CLASS": [2, "First-in-Class (Novel)"],
                "UNKNOWN": [0, "Not Specified"]
            }
        },
        "intervention_model_ml": {
            "ui": {
                "label": "Intervention Model",
                "pillar": "Scientific Attempt",
                "subgroup": "Protocol Architecture",
                "priority": 9,
                "options": [
                    ["SINGLE_GROUP", "Single Group"],
                    ["SEQUENTIAL", "Sequential"],
                    ["CROSSOVER", "Crossover"],
                    ["FACTORIAL", "Factorial"],
                    ["PARALLEL", "Parallel"],
                    ["UNKNOWN", "Not Specified"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "SINGLE_GROUP": [0, "Single Group"],
                "SEQUENTIAL": [1, "Sequential"],
                "CROSSOVER": [2, "Crossover"],
                "FACTORIAL": [3, "Factorial"],
                "PARALLEL": [4, "Parallel"],
                "UNKNOWN": [0, "Not Specified"]
            }
        },
        "primary_purpose_ml": {
            "ui": {
                "label": "Primary Purpose",
                "pillar": "Scientific Attempt",
                "subgroup": "Protocol Architecture",
                "priority": 10,
                "options": [
                    ["PREVENTION", "Prevention"],
                    ["TREATMENT", "Treatment"],
                    ["SUPPORTIVE_CARE", "Supportive Care"],
                    ["UNKNOWN", "Other Purpose / Unknown"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "PREVENTION": [0, "Prevention"],
                "UNKNOWN": [1, "Other Purpose / Unknown"],
                "OTHER": [1, "Other Purpose / Unknown"],
                "DIAGNOSTIC": [1, "Other Purpose / Unknown"],
                "BASIC_SCIENCE": [1, "Other Purpose / Unknown"],
                "SCREENING": [1, "Other Purpose / Unknown"],
                "HEALTH_SERVICES_RESEARCH": [1, "Other Purpose / Unknown"],
                "DEVICE_FEASIBILITY": [1, "Other Purpose / Unknown"],
                "TREATMENT": [2, "Treatment"],
                "SUPPORTIVE_CARE": [3, "Supportive Care"]
            }
        },
        "adaptive_design_ml": {
            "ui": {
                "label": "Design Flexibility",
                "pillar": "Scientific Attempt",
                "subgroup": "Protocol Architecture",
                "priority": 11,
                "options": [
                    ["STATIC", "Static Design"],
                    ["ADAPTIVE", "Adaptive Design"],
                    ["UNKNOWN", "Not Specified"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "STATIC": [0, "Static Design"],
                "ADAPTIVE": [1, "Adaptive Design"],
                "UNKNOWN": [0, "Not Specified"]
            }
        },
        "endpoint_rigor_ml": {
            "ui": {
                "label": "Primary Endpoint Type",
                "pillar": "Scientific Attempt",
                "subgroup": "Protocol Architecture",
                "priority": 12,
                "options": [
                    ["HARD_CLINICAL", "Hard Clinical (Survival/Death)"],
                    ["SUBJECTIVE_PRO", "Subjective / Patient Reported"],
                    ["SURROGATE", "Surrogate / Biomarker"],
                    ["UNKNOWN", "Not Specified / Unknown"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "UNKNOWN": [0, "Not Specified / Unknown"],
                "SUBJECTIVE_PRO": [1, "Subjective / Patient Reported"],
                "SURROGATE": [2, "Surrogate / Biomarker"],
                "HARD_CLINICAL": [3, "Hard Clinical (Survival/Death)"]
            }
        },
        "endpoint_structure_ml": {
            "ui": {
                "label": "Primary Endpoints",
                "pillar": "Scientific Attempt",
                "subgroup": "Protocol Architecture",
                "priority": 13,
                "options": [
                    ["MULTI_COMPOSITE", "Multi/Composite"],
                    ["SINGLE_GOAL", "Single Goal"],
                    ["UNKNOWN", "Not Specified"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "MULTI_COMPOSITE": [1, "Multi/Composite"],
                "SINGLE_GOAL": [0, "Single Goal"],
                "UNKNOWN": [0, "Not Specified"]
            }
        },
        "biomarker_stratification_ml": {
            "ui": {
                "label": "Biomarker Patient Selection",
                "pillar": "Scientific Attempt",
                "subgroup": "Protocol Architecture",
                "priority": 14,
                "options": [
                    ["0", "No"],
                    ["1", "Yes"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                0: [0, "No"],
                1: [1, "Yes"],
                "0": [0, "No"],
                "1": [1, "Yes"],
                "F": [0, "No"],
                "T": [1, "Yes"],
                "FALSE": [0, "No"],
                "TRUE": [1, "Yes"],
                "NO": [0, "No"],
                "YES": [1, "Yes"],
                "UNKNOWN": [0, "No"]
            }
        },
        "sponsor_tier_ml": {
            "ui": {
                "label": "Sponsor Type",
                "pillar": "Execution Framework",
                "subgroup": "Sponsor Type",
                "priority": 15,
                "options": [
                    ["TIER 1", "Top-Tier Pharma"],
                    ["MID_CAP", "Mid-Cap Pharma"],
                    ["BIOTECH", "Biotech and Emerging Pharma"],
                    ["UNKNOWN", "Unknown Sponsor Tier"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "BIOTECH": [3, "Biotech and Emerging Pharma"],
                "TIER 1": [1, "Top-Tier Pharma"],
                "MID_CAP": [2, "Mid-Cap Pharma"],
                "UNKNOWN": [3, "Unknown Sponsor Tier"]
            }
        },
        "masking_ml": {
            "ui": {
                "label": "Bias Control",
                "pillar": "Execution Framework",
                "subgroup": "Methodological Setup",
                "priority": 16,
                "options": [
                    ["UNKNOWN", "Open Label or Not Specified"],
                    ["SINGLE", "Single Blind"],
                    ["DOUBLE", "Double Blind"],
                    ["TRIPLE", "Triple Blind"],
                    ["QUADRUPLE", "Quadruple Blind"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "UNKNOWN": [0, "Open Label or Not Specified"],
                "NONE": [0, "Open Label or Not Specified"],
                "NONE (OPEN LABEL)": [0, "Open Label or Not Specified"],
                "SINGLE": [1, "Single Blind"],
                "DOUBLE": [2, "Double Blind"],
                "TRIPLE": [3, "Triple Blind"],
                "QUADRUPLE": [4, "Quadruple Blind"]
            }
        },
        "allocation_ml": {
            "ui": {
                "label": "Allocation Method",
                "pillar": "Execution Framework",
                "subgroup": "Methodological Setup",
                "priority": 17,
                "options": [
                    ["RANDOMIZED", "Randomized"],
                    ["NON-RANDOMIZED", "Non-Randomized"],
                    ["UNKNOWN", "Not Specified / Not Applicable"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "RANDOMIZED": [1, "Randomized"],
                "NON-RANDOMIZED": [0, "Non-Randomized"],
                "NON_RANDOMIZED": [0, "Non-Randomized"],
                "NA": [0, "Not Specified / Not Applicable"],
                "N/A": [0, "Not Specified / Not Applicable"],
                "UNKNOWN": [0, "Not Specified / Not Applicable"]
            }
        },
        "has_dmc_ml": {
            "ui": {
                "label": "Data Monitoring Committee",
                "pillar": "Execution Framework",
                "subgroup": "Methodological Setup",
                "priority": 18,
                "options": [
                    ["0", "No"],
                    ["1", "Yes"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                0: [0, "No"],
                1: [1, "Yes"],
                "0": [0, "No"],
                "1": [1, "Yes"],
                "F": [0, "No"],
                "T": [1, "Yes"],
                "FALSE": [0, "No"],
                "TRUE": [1, "Yes"],
                "NO": [0, "No"],
                "YES": [1, "Yes"],
                "UNKNOWN": [0, "No"]
            }
        },
        "has_placebo_ml": {
            "ui": {
                "label": "Includes Placebo Control",
                "pillar": "Execution Framework",
                "subgroup": "Methodological Setup",
                "priority": 19,
                "options": [
                    ["0", "No"],
                    ["1", "Yes"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                0: [0, "No"],
                1: [1, "Yes"],
                "0": [0, "No"],
                "1": [1, "Yes"],
                "F": [0, "No"],
                "T": [1, "Yes"],
                "FALSE": [0, "No"],
                "TRUE": [1, "Yes"],
                "NO": [0, "No"],
                "YES": [1, "Yes"],
                "UNKNOWN": [0, "No"]
            }
        },
        "comparator_benchmark_ml": {
            "ui": {
                "label": "Benchmark Comparator",
                "pillar": "Execution Framework",
                "subgroup": "Methodological Setup",
                "priority": 20,
                "options": [
                    ["NO_CONTROL_GROUP", "No Control Group or Not Specified"],
                    ["PLACEBO", "Placebo Control"],
                    ["ACTIVE_LEGACY_STANDARD", "Active (Legacy Standard)"],
                    ["ACTIVE_MODERN_STANDARD", "Active (Modern Standard)"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "UNKNOWN": [0, "No Control Group or Not Specified"],
                "NO_CONTROL_GROUP": [0, "No Control Group or Not Specified"],
                "PLACEBO": [1, "Placebo Control"],
                "ACTIVE_LEGACY_STANDARD": [2, "Active (Legacy Standard)"],
                "ACTIVE_MODERN_STANDARD": [3, "Active (Modern Standard)"]
            }
        },
        "primary_duration_months_ml": {
            "ui": {
                "label": "Maximum Primary Endpoint Duration",
                "pillar": "Execution Framework",
                "subgroup": "Trial Complexity Footprint",
                "priority": 21
            },
            "encoding": "numeric"
        },
        "number_of_arms_ml": {
            "ui": {
                "label": "Number of Arms",
                "pillar": "Execution Framework",
                "subgroup": "Trial Complexity Footprint",
                "priority": 22
            },
            "encoding": "numeric"
        },
        "administration_complexity_ml": {
            "ui": {
                "label": "Delivery Profile",
                "pillar": "Execution Framework",
                "subgroup": "Trial Complexity Footprint",
                "priority": 23,
                "options": [
                    ["SIMPLE_ORAL", "Simple Oral (Pill/Tablet)"],
                    ["ROUTINE_INFUSION", "Routine (Injection/IV)"],
                    ["INTENSIVE_MANAGEMENT", "Intensive (Hospitalized)"],
                    ["UNKNOWN", "Not Specified"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "SIMPLE_ORAL": [1, "Simple Oral (Pill/Tablet)"],
                "ROUTINE_INFUSION": [2, "Routine (Injection/IV)"],
                "INTENSIVE_MANAGEMENT": [3, "Intensive (Hospitalized)"],
                "UNKNOWN": [2, "Not Specified"]
            }
        },
        "patient_severity_ml": {
            "ui": {
                "label": "Patient Severity",
                "pillar": "Patient Profile",
                "subgroup": "Clinical Severity",
                "priority": 24,
                "options": [
                    ["UNKNOWN", "Uncertain Severity"],
                    ["CHRONIC_STABLE", "Chronic Stable"],
                    ["CHRONIC_PROGRESSIVE", "Chronic Progressive"],
                    ["ACUTE_CRITICAL", "Acute / Critical"],
                    ["ADVANCED_METASTATIC", "Advanced / Metastatic"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "UNKNOWN": [0, "Uncertain Severity"],
                "UNCERTAIN_SEVERITY": [0, "Uncertain Severity"],
                "CHRONIC_STABLE": [1, "Chronic Stable"],
                "CHRONIC_PROGRESSIVE": [2, "Chronic Progressive"],
                "ACUTE_CRITICAL": [3, "Acute / Critical"],
                "ADVANCED_METASTATIC": [4, "Advanced / Metastatic"]
            }
        },
        "line_of_therapy_ml": {
            "ui": {
                "label": "Line of Therapy",
                "pillar": "Patient Profile",
                "subgroup": "Clinical Severity",
                "priority": 25,
                "options": [
                    ["PREVENTATIVE", "Preventative / Prophylaxis"],
                    ["FIRST_LINE", "First-Line"],
                    ["ADJUVANT_NEOADJUVANT", "Adjuvant / Neoadjuvant"],
                    ["LATER_LINE", "Later-Line (2nd+)"],
                    ["REFRACTORY_RELAPSED", "Refractory / Relapsed"],
                    ["UNKNOWN", "Not Applicable or Not Specified"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "PREVENTATIVE": [1, "Preventative / Prophylaxis"],
                "FIRST_LINE": [2, "First-Line"],
                "ADJUVANT_NEOADJUVANT": [3, "Adjuvant / Neoadjuvant"],
                "LATER_LINE": [4, "Later-Line (2nd+)"],
                "REFRACTORY_RELAPSED": [5, "Refractory / Relapsed"],
                "LINE_NA": [0, "Not Applicable or Not Specified"],
                "UNKNOWN": [0, "Not Applicable or Not Specified"]
            }
        },
        "gender_ml": {
            "ui": {
                "label": "Patient Gender",
                "pillar": "Patient Profile",
                "subgroup": "Population Scope",
                "priority": 26,
                "options": [
                    ["ALL", "All (Male & Female)"],
                    ["FEMALE", "Female Only"],
                    ["MALE", "Male Only"],
                    ["UNKNOWN", "Not Specified"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                "ALL": [0, "All (Male & Female)"],
                "FEMALE": [1, "Female Only"],
                "MALE": [2, "Male Only"],
                "UNKNOWN": [0, "Not Specified"]
            }
        },
        "healthy_volunteers_ml": {
            "ui": {
                "label": "Population Setting",
                "pillar": "Patient Profile",
                "subgroup": "Population Scope",
                "priority": 27,
                "options": [
                    ["0", "Patients Only"],
                    ["1", "Healthy Volunteers Accepted"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                0: [0, "Patients Only"],
                1: [1, "Healthy Volunteers Accepted"],
                "0": [0, "Patients Only"],
                "1": [1, "Healthy Volunteers Accepted"],
                "UNKNOWN": [0, "Patients Only"]
            }
        },
        "adult_ml": {
            "ui": {
                "label": "Adult Profile",
                "pillar": "Patient Profile",
                "subgroup": "Population Scope",
                "priority": 28,
                "options": [
                    ["1", "Included"],
                    ["0", "Excluded"],
                    ["UNKNOWN", "Not Specified"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                1: [1, "Included"],
                0: [0, "Excluded"],
                "1": [1, "Included"],
                "0": [0, "Excluded"],
                "UNKNOWN": [1, "Not Specified"]
            }
        },
        "child_ml": {
            "ui": {
                "label": "Pediatric Profile",
                "pillar": "Patient Profile",
                "subgroup": "Population Scope",
                "priority": 29,
                "options": [
                    ["1", "Included"],
                    ["0", "Excluded"],
                    ["UNKNOWN", "Not Specified"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                1: [1, "Included"],
                0: [0, "Excluded"],
                "1": [1, "Included"],
                "0": [0, "Excluded"],
                "UNKNOWN": [0, "Not Specified"]
            }
        },
        "older_adult_ml": {
            "ui": {
                "label": "Geriatric Profile",
                "pillar": "Patient Profile",
                "subgroup": "Population Scope",
                "priority": 30,
                "options": [
                    ["1", "Included"],
                    ["0", "Excluded"],
                    ["UNKNOWN", "Not Specified"]
                ]
            },
            "encoding": "ordinal",
            "mapping": {
                1: [1, "Included"],
                0: [0, "Excluded"],
                "1": [1, "Included"],
                "0": [0, "Excluded"],
                "UNKNOWN": [1, "Not Specified"]
            }
        },
        "gbd_hierarchy_level_ml": {
            "ui": {
                "label": "Mapping Depth",
                "pillar": "Metadata",
                "subgroup": "System",
                "priority": 31,
                "options": [
                    ["1", "Level 1 (Unclassified Condition)"],
                    ["2", "Level 2 (Condition High-Level Category)"],
                    ["3", "Level 3 (Condition Group)"],
                    ["4", "Level 4 (Specific Condition)"],
                    ["0", "Unknown Depth"]
                ]
            },
            "encoding": None,
            "mapping": {
                0: [0, "Unknown Depth"],
                1: [1, "Level 1 (Unclassified Condition)"],
                2: [2, "Level 2 (Condition High-Level Category)"],
                3: [3, "Level 3 (Condition Group)"],
                4: [4, "Level 4 (Specific Condition)"],
                "0": [0, "Unknown Depth"],
                "1": [1, "Level 1 (Unclassified Condition)"],
                "2": [2, "Level 2 (Condition High-Level Category)"],
                "3": [3, "Level 3 (Condition Group)"],
                "4": [4, "Level 4 (Specific Condition)"],
                "UNKNOWN": [0, "Unknown Depth"]
            }
        },
        "is_duration_unknown_ml": {
            "ui": {
                "label": "Duration Known",
                "pillar": "Metadata",
                "subgroup": "System",
                "priority": 32,
                "options": [
                    ["1", "Duration Missing"],
                    ["0", "Known"]
                ]
            },
            "encoding": None,
            "mapping": {
                0: [0, "Known"],
                1: [1, "Duration Missing"],
                "0": [0, "Known"],
                "1": [1, "Duration Missing"],
                "F": [0, "Known"],
                "T": [1, "Duration Missing"],
                "NO": [0, "Known"],
                "YES": [1, "Duration Missing"],
                "UNKNOWN": [1, "Duration Missing"]
            }
        },
        "target": {
            "ui": {
                "label": "Outcome",
                "pillar": "Metadata",
                "subgroup": "System",
                "priority": 33,
                "options": [
                    ["1.0", "Failure"],
                    ["0.0", "Success"]
                ]
            },
            "encoding": None,
            "mapping": {
                "0.0": [0, "Success"],
                "1.0": [1, "Failure"],
                0.0: [0, "Success"],
                1.0: [1, "Failure"],
                "UNKNOWN": [np.nan, "Not Specified"]
            }
        },
        "ui_acronym": {
            "ui": {
                "label": "Study Acronym",
                "pillar": "Metadata",
                "subgroup": "Identity",
                "priority": 34
            }
        }
    }
}

# ==============================================================================
# 2. CORE PIPELINE LOGIC (Internal Helpers)
# ==============================================================================

def _build_feature_registry():
    """Generates FEATURE_REGISTRY and UI_SCHEMA from PIPELINE_REGISTRY."""
    feature_registry = {}
    ui_schema = {}
    for field_name, field_meta in PIPELINE_REGISTRY["FIELDS"].items():
        if "ui" in field_meta:
            ui_schema[field_name] = field_meta["ui"]
        if "encoding" in field_meta or field_name.endswith('_ml'):
            # Only include defined keys to avoid triggering empty-mapping logic
            feature_registry[field_name] = {
                k: field_meta[k] for k in ["encoding", "mapping"] if k in field_meta
            }
    return feature_registry, ui_schema

FEATURE_REGISTRY, UI_SCHEMA = _build_feature_registry()

# ==============================================================================
# 3. CUSTOM ML TRANSFORMERS
# ==============================================================================

class RegistryImputer(BaseEstimator, TransformerMixin):
    """
    Joblib-safe Custom Imputer.
    Bakes the Registry fallback codes (0, 1, 2) into the transformer state.
    """
    def __init__(self):
        self.fill_values_ = {}

    def fit(self, X, y=None):
        for feat, meta in FEATURE_REGISTRY.items():
            fill_val = 0
            if 'mapping' in meta and 'UNKNOWN' in meta['mapping']:
                fill_val = meta['mapping']['UNKNOWN'][0]
            self.fill_values_[feat] = fill_val
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            cols_to_fix = [c for c in X.columns if c in self.fill_values_]
            if cols_to_fix:
                subset_map = {c: self.fill_values_[c] for c in cols_to_fix}
                return X.fillna(value=subset_map)
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features

def identity_transform(x):
    return x

# ==============================================================================
# 4. PIPELINE BUILDER
# ==============================================================================

def preprocessor():
    """
    Returns a dynamic ColumnTransformer based on the FEATURE_REGISTRY.
    """
    ORDINAL_COLS = []
    TARGET_COLS  = []

    DISABLED_COLS = [
        'includes_us_ml', 'is_fda_regulated_drug_ml', 'gbd_cause_id_ml',
        'gbd_cause_id_2_ml', 'gbd_cause_id_4_ml', 'gbd_hierarchy_level_ml',
        'is_duration_unknown_ml', 'target',  'masking_ml',
        'therapeutic_area_ml', 'strategic_ambition_ml', 'intervention_model_ml'
        ]

    for feat, meta in FEATURE_REGISTRY.items():
        if feat in DISABLED_COLS: continue
        enc = meta.get('encoding')
        if enc == 'ordinal': ORDINAL_COLS.append(feat)
        elif enc == 'target': TARGET_COLS.append(feat)

    NUM_ARMS_COL = ['number_of_arms_ml']
    NUM_DURATION_COL = ['primary_duration_months_ml']

    pipe_ordinal = Pipeline([
        ('imputer', RegistryImputer()),
        ('passthrough', FunctionTransformer(identity_transform, feature_names_out="one-to-one"))
    ])

    pipe_target = Pipeline([
        ('imputer', RegistryImputer()),
        ('encoder', TargetEncoder(target_type='binary', smooth=200.0, random_state=42))
    ])

    pipe_arms = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])

    pipe_duration = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    return ColumnTransformer(
        transformers=[
            ('ordinal',      pipe_ordinal,      ORDINAL_COLS),
            ('target',       pipe_target,       TARGET_COLS),
            ('num_arms',     pipe_arms,         NUM_ARMS_COL),
            ('num_duration', pipe_duration,     NUM_DURATION_COL)
        ],
        remainder='drop',
        verbose_feature_names_out=True
    )
