import pandas as pd
import json
import re
import os
import numpy as np

# 1. THE GOLD STANDARD: RISK-SORTED & MERGED LOGIC
# val: Risk (Higher = More Failure Risk)
# order: UI Dropdown sequence
RAW_ORDINAL_DATA = {
    "phase_ml": [
        {"key": "UNKNOWN",       "val": 0, "label": "Not Specified", "order": 0},
        {"key": "PHASE3",        "val": 1, "label": "Phase 3", "order": 1},
        {"key": "PHASE2/PHASE3", "val": 2, "label": "Phase 2/3", "order": 2},
        {"key": "PHASE2",        "val": 3, "label": "Phase 2", "order": 3},
        {"key": "PHASE1/PHASE2", "val": 4, "label": "Phase 1/2", "order": 4},
        {"key": "PHASE1",        "val": 5, "label": "Phase 1", "order": 5},
    ],
    "masking_ml": [
        {"key": "UNKNOWN",           "val": 0, "label": "Open Label / Not Specified", "order": 0},
        {"key": "NONE (OPEN LABEL)", "val": 0, "label": "Open Label / Not Specified", "order": 1},
        {"key": "SINGLE",            "val": 1, "label": "Single Blind", "order": 2},
        {"key": "DOUBLE",            "val": 2, "label": "Double Blind", "order": 3},
        {"key": "TRIPLE",            "val": 3, "label": "Triple Blind", "order": 4},
        {"key": "QUADRUPLE",         "val": 4, "label": "Quadruple Blind", "order": 5},
    ],
    "allocation_ml": [
        {"key": "UNKNOWN",        "val": 0, "label": "Non-Randomized / Not Specified", "order": 0},
        {"key": "NON-RANDOMIZED", "val": 0, "label": "Non-Randomized / Not Specified", "order": 1},
        {"key": "RANDOMIZED",     "val": 1, "label": "Randomized", "order": 2},
    ],
    "comparator_benchmark_ml": [
        {"key": "UNKNOWN",                 "val": 0, "label": "No Control / Not Specified", "order": 0},
        {"key": "NO_CONTROL_GROUP",        "val": 0, "label": "No Control / Not Specified", "order": 1},
        {"key": "PLACEBO",                 "val": 1, "label": "Placebo Control", "order": 2},
        {"key": "ACTIVE_LEGACY_STANDARD",  "val": 2, "label": "Active (Legacy Standard)", "order": 3},
        {"key": "ACTIVE_MODERN_STANDARD",  "val": 3, "label": "Active (Modern Standard)", "order": 4},
    ],
    "intervention_model_ml": [
        {"key": "UNKNOWN",      "val": 0, "label": "Single Group / Not Specified", "order": 0},
        {"key": "SINGLE_GROUP", "val": 0, "label": "Single Group / Not Specified", "order": 1},
        {"key": "SEQUENTIAL",   "val": 1, "label": "Sequential", "order": 2},
        {"key": "CROSSOVER",    "val": 2, "label": "Crossover", "order": 3},
        {"key": "FACTORIAL",    "val": 3, "label": "Factorial", "order": 4},
        {"key": "PARALLEL",     "val": 4, "label": "Parallel", "order": 5},
    ],
    "patient_severity_ml": [
        {"key": "UNKNOWN",            "val": 0, "label": "Uncertain Severity", "order": 0},
        {"key": "UNCERTAIN_SEVERITY", "val": 0, "label": "Uncertain Severity", "order": 1},
        {"key": "CHRONIC_STABLE",     "val": 1, "label": "Chronic Stable", "order": 2},
        {"key": "CHRONIC_PROGRESSIVE","val": 2, "label": "Chronic Progressive", "order": 3},
        {"key": "ACUTE_CRITICAL",     "val": 3, "label": "Acute / Critical", "order": 4},
        {"key": "ADVANCED_METASTATIC","val": 4, "label": "Advanced / Metastatic", "order": 5},
    ],
    "line_of_therapy_ml": [
        {"key": "UNKNOWN",              "val": 0, "label": "Line N/A", "order": 0},
        {"key": "LINE_NA",              "val": 0, "label": "Line N/A", "order": 1},
        {"key": "PREVENTATIVE",         "val": 1, "label": "Preventative / Prophylaxis", "order": 2},
        {"key": "FIRST_LINE",           "val": 2, "label": "First-Line", "order": 3},
        {"key": "ADJUVANT_NEOADJUVANT", "val": 3, "label": "Adjuvant / Neoadjuvant", "order": 4},
        {"key": "LATER_LINE",           "val": 4, "label": "Later-Line (2nd+)", "order": 5},
        {"key": "REFRACTORY_RELAPSED",  "val": 5, "label": "Refractory / Relapsed", "order": 6},
    ],
    "adult_ml": [
        {"key": "UNKNOWN", "val": 1, "label": "Yes (Default)", "order": 0},
        {"key": "1",       "val": 1, "label": "Yes", "order": 1},
        {"key": "0",       "val": 0, "label": "No", "order": 2},
    ],
    "older_adult_ml": [
        {"key": "UNKNOWN", "val": 1, "label": "Yes (Default)", "order": 0},
        {"key": "1",       "val": 1, "label": "Yes", "order": 1},
        {"key": "0",       "val": 0, "label": "No", "order": 2},
    ],
    "child_ml": [
        {"key": "UNKNOWN", "val": 0, "label": "No (Default)", "order": 0},
        {"key": "1",       "val": 1, "label": "Yes", "order": 1},
        {"key": "0",       "val": 0, "label": "No", "order": 2},
    ],
    "healthy_volunteers_ml": [
        {"key": "UNKNOWN", "val": 0, "label": "Patients Only (Default)", "order": 0},
        {"key": "1",       "val": 1, "label": "Healthy Volunteers", "order": 1},
        {"key": "0",       "val": 0, "label": "Patients Only", "order": 2},
    ],
    "strategic_ambition_ml": [
        {"key": "UNKNOWN",       "val": 0, "label": "Unknown Intent", "order": 0},
        {"key": "PIVOTAL_INTENT", "val": 1, "label": "Pivotal / Registration", "order": 1},
        {"key": "SAFETY_DOSING", "val": 2, "label": "Safety / Dosing", "order": 2},
        {"key": "SIGNAL_SEARCH", "val": 3, "label": "Signal Searching", "order": 3},
    ],
    "target_precedent_ml": [
        {"key": "UNKNOWN",                  "val": 0, "label": "Unknown Precedent", "order": 0},
        {"key": "PRECEDENT_IN_INDICATION",  "val": 1, "label": "Precedent in Indication", "order": 1},
        {"key": "PRECEDENT_IN_OTHER",       "val": 2, "label": "Precedent in Other Indication", "order": 2},
        {"key": "NO_PRECEDENT",             "val": 3, "label": "Novel Target / No Precedent", "order": 3},
    ],
    "administration_complexity_ml": [
        {"key": "UNKNOWN",              "val": 0, "label": "Unknown Complexity", "order": 0},
        {"key": "SIMPLE_ORAL",          "val": 1, "label": "Simple Oral (Pill/Tablet)", "order": 1},
        {"key": "ROUTINE_INFUSION",     "val": 2, "label": "Routine (Injection/IV)", "order": 2},
        {"key": "INTENSIVE_MANAGEMENT", "val": 3, "label": "Intensive (Hospitalized)", "order": 3},
    ],
    "innovation_tier_ml": [
        {"key": "UNKNOWN",            "val": 0, "label": "Unknown Tier", "order": 0},
        {"key": "ESTABLISHED_COPY",   "val": 1, "label": "Established / Copy", "order": 1},
        {"key": "NEXT_GEN_OPTIMIZED", "val": 2, "label": "Next-Gen / Optimized", "order": 2},
        {"key": "FIRST_IN_CLASS",     "val": 3, "label": "First-in-Class (Novel)", "order": 3},
    ]
}

# 2. CATEGORICAL REFERENCE LISTS
TAS = ['PSYCHIATRY', 'ONCOLOGY', 'DERMATOLOGY', 'MUSCULOSKELETAL', 'NEUROLOGY', 'INFECTIONS', 'METABOLIC', 'RESPIRATORY', 'CARDIOVASCULAR', 'IMMUNOLOGY', 'GASTROINTESTINAL', 'RENAL/UROLOGY', 'OPHTHALMOLOGY', 'HEMATOLOGY', 'REPRODUCTIVE', 'GENETIC', 'DENTAL', 'EAR/NOSE/THROAT', 'UNCLASSIFIED']
PATHWAYS = ['IMMUNO_ONCOLOGY', 'KINASE_INHIBITOR', 'METABOLIC_REPROGRAMMING', 'INTERLEUKIN_CYTOKINE', 'GPCR_TARGET', 'ENZYME_MODULATOR', 'EPIGENETIC_REGULATOR', 'NUCLEAR_RECEPTOR', 'PROTEIN_DEGRADER', 'ION_CHANNEL', 'DNA_REPAIR', 'OTHER_PATHWAY']
MODALITIES = ['SMALL_MOLECULE', 'BIOLOGIC_MAB', 'BIOLOGIC_ADC', 'BIOLOGIC_OTHER', 'CELL_GENE_THERAPY', 'RNA_THERAPY', 'VACCINE', 'RADIOPHARMACEUTICAL', 'PEPTIDE_HORMONES', 'OTHER_MODALITY']
PURPOSE_WHITELIST = ['TREATMENT', 'PREVENTION', 'SUPPORTIVE_CARE']

# 3. METADATA & STATS
METADATA_FIELDS = {
    "nct_id": {"label": "Trial ID", "pillar": "UI Metadata", "subgroup": "Identity"},
    "ui_brief_title": {"label": "Brief Title", "pillar": "UI Metadata", "subgroup": "Identity"},
    "lead_sponsor_canonical": {"label": "Lead Sponsor", "pillar": "UI Metadata", "subgroup": "Identity"},
    "ui_summary": {"label": "Protocol Summary", "pillar": "UI Metadata", "subgroup": "Narrative"},
    "ui_criteria": {"label": "Eligibility Criteria", "pillar": "UI Metadata", "subgroup": "Narrative"},
    "why_stopped": {"label": "Termination Reason", "pillar": "UI Metadata", "subgroup": "Narrative"},
    "start_year": {"label": "Project Year", "pillar": "UI Metadata", "subgroup": "Temporal"},
    "enrollment": {"label": "Patient Enrollment", "pillar": "UI Metadata", "subgroup": "Metrics"},
    "min_p_value": {"label": "P-Value", "pillar": "UI Metadata", "subgroup": "Results"},
    "overall_status": {"label": "Status", "pillar": "UI Metadata", "subgroup": "Results"},
    "gbd_indication_name": {"label": "Indication", "pillar": "UI Metadata", "subgroup": "Epidemiology"},
}

def smart_ui_sort(options):
    if not options: return []
    unique_map = {}
    for key, label in options:
        if label not in unique_map: unique_map[label] = key
        else:
            if key in ['0', '1', 'UNKNOWN'] and unique_map[label] not in ['0', '1', 'UNKNOWN']:
                unique_map[label] = key
    deduped = [[v, k] for k, v in unique_map.items()]
    standard, others, unknowns = [], [], []
    for opt in deduped:
        k, lab = opt[0], opt[1]
        if k.upper() in ['UNKNOWN', 'UNCLASSIFIED', 'NA', 'BASELINE'] or 'NOT SPECIFIED' in lab.upper(): unknowns.append(opt)
        elif 'OTHER' in k.upper() or 'OTHER' in label.upper(): others.append(opt)
        else: standard.append(opt)
    standard.sort(key=lambda x: x[1].lower()); others.sort(key=lambda x: x[1].lower())
    return standard + others + unknowns

def reconstruct_taxonomy():
    df = pd.read_excel('data_clinpred_summary.xlsx')
    DISABLED_FIELDS = ['gbd_cause_id_ml', 'gbd_cause_id_2_ml', 'gbd_cause_id_4_ml', 'gbd_hierarchy_level_ml', 'is_duration_unknown_ml', 'target_ml', 'includes_us_ml', 'is_fda_regulated_drug_ml']
    
    unified_registry = {}
    for idx, row in df.iterrows():
        f = row['Field Name']
        if pd.isna(f) or f.startswith(('crit_', 'sci_', 'endp_')): continue
        
        fo = {'ui': {'label': row['UI Label'], 'pillar': row['Pillar'] if f not in DISABLED_FIELDS else 'Metadata', 'subgroup': row['Subgroup'] if f not in DISABLED_FIELDS else 'System', 'priority': int(idx)}}
        
        if f in RAW_ORDINAL_DATA:
            dl = RAW_ORDINAL_DATA[f]
            fo['encoding'] = 'ordinal'
            fo['mapping'] = {d['key']: [d['val'], d['label']] for d in dl}
            fo['ui']['options'] = [[d['key'], d['label']] for d in sorted(dl, key=lambda x: x['order'])]
            seen = set(); unique_opts = []
            for k, l in fo['ui']['options']:
                if l not in seen: unique_opts.append([k, l]); seen.add(l)
            fo['ui']['options'] = unique_opts
        
        elif f == 'primary_purpose_ml':
            fo['encoding'] = 'target'
            fo['mapping'] = {k: [i+1 if k != 'UNKNOWN' else 0, k.replace('_',' ').title()] for i, k in enumerate(PURPOSE_WHITELIST + ['UNKNOWN'])}
            fo['ui']['options'] = [[k, k.replace('_',' ').title()] for k in PURPOSE_WHITELIST] + [['UNKNOWN', 'Other Purpose / Unknown']]
            
        elif f == 'therapeutic_area_ml':
            fo['encoding'] = 'target'
            fo['mapping'] = {k: [i+1 if k != 'UNCLASSIFIED' else 0, k.replace('_',' ').title()] for i, k in enumerate(TAS)}
            fo['mapping']['UNKNOWN'] = [0, 'Other / Unclassified']
            fo['ui']['options'] = smart_ui_sort([[k, k.replace('_',' ').title()] for k in TAS])

        elif f == 'therapeutic_modality_ml':
            fo['encoding'] = 'target'
            fo['mapping'] = {k: [i+1 if 'OTHER' not in k else 0, k.replace('_',' ').title()] for i, k in enumerate(MODALITIES)}
            fo['mapping']['UNKNOWN'] = [0, 'Other Modality / Unknown']
            fo['ui']['options'] = smart_ui_sort([[k, k.replace('_',' ').title()] for k in MODALITIES + ['UNKNOWN']])

        elif f == 'target_pathway_class_ml':
            fo['encoding'] = 'target'
            fo['mapping'] = {k: [i+1 if 'OTHER' not in k else 0, k.replace('_',' ').title()] for i, k in enumerate(PATHWAYS)}
            fo['mapping']['UNKNOWN'] = [0, 'Other / Unknown']
            fo['ui']['options'] = smart_ui_sort([[k, k.replace('_',' ').title()] for k in PATHWAYS + ['UNKNOWN']])

        elif row['Status'] == 'In XGBoost' or f.endswith('_ml'):
            fo['encoding'] = row['Encoding Strategy'] if (pd.notna(row['Encoding Strategy']) and f not in DISABLED_FIELDS) else None
            val_ex = str(row['Values / Examples'])
            if ' -> ' in val_ex:
                m, o = {}, []
                for p in val_ex.split(' | '):
                    if ' -> ' in p:
                        k, v_p = p.split(' -> ', 1)
                        match = re.match(r'(\d+)\s*\((.*)\)', v_p.strip())
                        if match:
                            m[k.strip().upper()] = [int(match.group(1)), match.group(2).strip()]
                            o.append([k.strip().upper(), match.group(2).strip()])
                if m: fo['mapping'] = m; fo['ui']['options'] = smart_ui_sort(o)
        
        unified_registry[f] = fo

    for f, m in METADATA_FIELDS.items():
        if f not in unified_registry: unified_registry[f] = {'ui': {'label': m['label'], 'pillar': m['pillar'], 'subgroup': m['subgroup'], 'priority': 999}, 'encoding': None}

    hi_stats = {'daly_high_income': 'DALY (High Income)', 'yld_high_income': 'YLD (High Income)', 'yll_high_income': 'YLL (High Income)', 'chronic_ratio_high_income': 'Chronic Ratio (High Income)'}
    for f, lab in hi_stats.items(): unified_registry[f] = {'ui': {'label': lab, 'pillar': 'Therapeutic Context', 'subgroup': 'Epidemiological Burden', 'priority': 100}, 'encoding': 'numeric'}

    os.makedirs('models', exist_ok=True)
    with open('models/taxonomy_01.json', 'w') as f_out: json.dump({'FIELDS': unified_registry}, f_out, indent=4)
    print("✅ ABSOLUTE FINAL PRODUCTION TAXONOMY RECONSTRUCTED.")

if __name__ == "__main__":
    reconstruct_taxonomy()
