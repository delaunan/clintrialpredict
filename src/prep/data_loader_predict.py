import pandas as pd
import numpy as np
import os
import csv
import re

from src.prep.text_cleaning import day_zero_reconstructor
from src.prep.text_cleaning_ui import ui_clean_text


class ClinicalTrialLoaderPredict:

    def __init__(self, data_path):
        self.data_path = data_path
        self.df_drugs = pd.DataFrame()

        # --- STRATEGY A: PERFECT ---
        self.params_perfect = {
            "sep": "|", "dtype": str, "header": 0, "quotechar": '"',
            "quoting": csv.QUOTE_MINIMAL, "low_memory": False, "on_bad_lines": "warn"
        }

        # --- STRATEGY B: ROBUST ---
        self.params_robust = {
            "sep": "|", "dtype": str, "header": 0, "quotechar": '"',
            "quoting": 3, "low_memory": False, "on_bad_lines": "warn"
        }

    def _safe_load(self, filename, cols=None):
        full_path = os.path.join(self.data_path, filename)
        if not os.path.exists(full_path):
            print(f"   [!] Warning: File not found {filename}. Features will be empty.")
            return pd.DataFrame()

        try:
            return pd.read_csv(full_path, usecols=cols, **self.params_perfect)
        except Exception as e:
            print(f"   [!] Formatting error in {filename}. Switching to Robust Mode...")
            try:
                return pd.read_csv(full_path, usecols=cols, **self.params_robust)
            except Exception as e2:
                print(f"   [x] CRITICAL: Could not load {filename}. Error: {e2}")
                return pd.DataFrame()

    def load_and_clean(self):

        print(">>> 1. Loading Studies & Applying Filters (Predict Universe)...")

        # 0. Load Core Columns + Metadata for Dashboard
        cols_studies = ['nct_id', 'overall_status', 'study_type', 'phase',
                        'start_date', 'number_of_arms',
                        'official_title', 'brief_title', 'why_stopped',
                        'has_dmc', 'is_fda_regulated_drug',
                        'enrollment', 'enrollment_type']

        df = self._safe_load('studies.txt', cols=cols_studies)

        if df.empty:
            raise ValueError("Critical Error: 'studies.txt' failed to load.")


        # --- FALLBACK LOGIC : official -> brief title ---
        df['official_title'] = df['official_title'].fillna(df['brief_title'])
        df['brief_title'] = df['brief_title'].fillna(df['official_title'])


        # 1. INDUSTRY FILTER
        df_sponsors = self._safe_load('sponsors.txt', cols=['nct_id', 'lead_or_collaborator', 'agency_class'])
        if not df_sponsors.empty:
            industry_ids = df_sponsors[
                (df_sponsors['lead_or_collaborator'].str.upper() == 'LEAD') &
                (df_sponsors['agency_class'].str.upper() == 'INDUSTRY')
            ]['nct_id'].unique()
            df = df[df['nct_id'].isin(industry_ids)]
            print(f"    [Filter] Kept {len(df)} Industry-led trials.")


        # 2. Filter: Interventional Only
        if 'study_type' in df.columns:
            df = df[df['study_type'].str.upper() == 'INTERVENTIONAL'].copy()

        # 3. Filter: Drugs/Biologics/Genetic Only & Load Names
        cols_int = ['nct_id', 'intervention_type', 'name']
        df_int = self._safe_load('interventions.txt', cols=cols_int)

        if not df_int.empty:
            target_types = ['DRUG', 'BIOLOGICAL', 'GENETIC']
            drug_ids = df_int[df_int['intervention_type'].str.upper().isin(target_types)]['nct_id'].unique()
            df = df[df['nct_id'].isin(drug_ids)]
            self.df_drugs = df_int[df_int['intervention_type'].str.upper().isin(target_types + ['DIETARY SUPPLEMENT', 'OTHER'])].copy()
        else:
            self.df_drugs = pd.DataFrame(columns=['nct_id', 'name', 'intervention_type'])

        # 4. Filter: Status (KEEP UNFILTERED for Evaluation)
        # In this predict-mode loader, we do not filter by status.

        # 5. Filter: Phase (DROP PHASE 0 - Focus on Valley of Death)
        excluded_phases = ['EARLY_PHASE1', 'PHASE1','PHASE4', 'NA']
        df = df[~df['phase'].str.upper().isin(excluded_phases)]
        df = df.dropna(subset=['phase'])
        df = df[df['phase'].str.strip() != '']

        # 6. Filter: COVID Sanitizer (Remove Pandemic Failures)
        if 'why_stopped' in df.columns:
            covid_keywords = ['covid', 'pandemic', 'coronavirus', 'sars-cov-2','travel restrictions', 'quarantine', 'lockdown', 'sars-cov']
            mask_covid = df['why_stopped'].fillna('').astype(str).str.lower().apply(
                lambda x: any(k in x for k in covid_keywords)
            )
            if mask_covid.sum() > 0:
                print(f"    [Sanitizer] Dropping {mask_covid.sum()} trials terminated due to COVID/Logistics.")
                df = df[~mask_covid]

        # 7. Filter: Date Range (2005-2026 for production universe)
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['start_year'] = df['start_date'].dt.year
        df = df[df['start_year'].between(2005, 2026)]

        # 8. Create Segment & Target
        def determine_segment(status):
            s = str(status).upper()
            if s in ['COMPLETED', 'TERMINATED', 'WITHDRAWN']:
                return 'HISTORICAL'
            return 'ONGOING'

        df['trial_segment'] = df['overall_status'].apply(determine_segment)
        
        def determine_target(status):
            s = str(status).upper()
            if s == 'COMPLETED': return 0
            if s in ['TERMINATED', 'WITHDRAWN']: return 1
            return -1 # Placeholder for Ongoing

        df['target'] = df['overall_status'].apply(determine_target)

        print(f"    Full Universe: {len(df)} trials (Phase 2/3 focus, 2005-2026).")
        return df.copy()

    def add_features(self, df):

        df = df.copy()
        print(">>> 2. Engineering Features...")

        # 1. Phase Grouping
        df = self._engineer_phase_groups(df)

        # 2. Geography
        df_countries = self._safe_load('countries.txt', cols=['nct_id', 'name'])
        if not df_countries.empty:
            us_trials = df_countries[df_countries['name'] == 'United States']['nct_id'].unique()
            df['includes_us'] = df['nct_id'].isin(us_trials).astype(int)
        else:
            df['includes_us'] = 0


        # 3. Merge (Designs & Calculated Values)
        df = self._merge_file(df, 'designs.txt', ['nct_id', 'allocation', 'intervention_model', 'masking', 'primary_purpose'])
        
        # Metadata Enrichment: Adding number_of_facilities
        df = self._merge_file(df, 'calculated_values.txt', ['nct_id', 'number_of_primary_outcomes_to_measure', 'number_of_facilities'])

        df = self._engineer_clinical_setting(df)
        df = self._engineer_comparator_architecture(df)
        df = self._engineer_protocol_duration(df)


        # 4. Sponsor Engineering
        df = self._engineer_sponsor_features(df)

        # 5. Complexity Engineering
        df = self._engineer_complexity(df)

        # 6. Medical Hierarchy & Competition
        df = self._attach_medical_hierarchy(df)

        # 7A. UI-only text fields
        df = self._add_ui_text_fields(df)

        # 7B. NLP pillars for embeddings/training (txt_*) 
        df = self._prepare_text(df)

        # 8. Agent Type
        df = self._engineer_agent_type(df)

        # 9. Competition
        df = self._calculate_competition(df)

        # 10. Smart Patterns
        df = self._engineer_smart_patterns(df)

        # 11. Safe Features
        df = self._engineer_safe_features(df)

        # 12. Attach Embeddings (BioBERT)
        df = self._attach_embeddings(df)

        # 13. Attach P-Values
        df = self._attach_p_values(df)


        # ----------------------------------------------------------------------
        # FINAL TYPE ENFORCEMENT
        # ----------------------------------------------------------------------
        print("    -> Finalizing Data Types...")

        if 'number_of_primary_outcomes_to_measure' in df.columns:
            df.rename(columns={'number_of_primary_outcomes_to_measure': 'num_primary_endpoints'}, inplace=True)

        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')

        numeric_cols = [
            'target', 'start_year', 'includes_us', 'num_primary_endpoints',
            'is_acute', 'is_refractory', 'is_severe', 'is_critical_setting',
            'has_placebo', 'has_active_comparator', 'duration_months',
            'design_rigor_score', 'eligibility_strictness_score', 'is_sick_only',
            'criteria_len_log', 'competition_broad', 'competition_niche', 'competition_agent',
            'has_dmc', 'is_fda_regulated_drug', 'child', 'adult', 'older_adult',
            'min_p_value', 'scientific_success', 'enrollment', 'number_of_facilities'
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['num_primary_endpoints'] = df['num_primary_endpoints'].fillna(1)

        return df

    def _engineer_clinical_setting(self, df):
        print("    -> Engineering Clinical Setting...")
        cols = ['nct_id', 'name']
        df_cond = self._safe_load('conditions.txt', cols=cols)
        if df_cond.empty:
            df_cond = self._safe_load('browse_conditions.txt', cols=['nct_id', 'mesh_term'])
            if not df_cond.empty:
                df_cond.rename(columns={'mesh_term': 'name'}, inplace=True)

        df_sum = self._safe_load('brief_summaries.txt', cols=['nct_id', 'description'])
        df_cond = df_cond[df_cond['nct_id'].isin(df['nct_id'])].copy()
        cond_agg = df_cond.groupby('nct_id')['name'].apply(lambda x: " ".join(str(v) for v in x)).reset_index()

        temp = df[['nct_id', 'official_title']].merge(cond_agg, on='nct_id', how='left')
        temp = temp.merge(df_sum, on='nct_id', how='left')

        temp['full_text'] = (
            temp['official_title'].fillna('') + " " + 
            temp['name'].fillna('') + " " + 
            temp['description'].fillna('')
        ).str.lower()

        patterns = {
            'is_acute': r'\b(?:acute|emergency|critical|sepsis|shock|stroke|infarction|crisis|sudden|urgent|icu|trauma)\b',
            'is_refractory': r'\b(?:refractory|resistant|relapsed|recur|second line|third line|salvage|unresponsive|previously treated)\b',
            'is_severe': r'\b(?:severe|advanced|metastatic|stage (?:iv|4|iii|3)|end-stage|decompensated|invasive|terminal)\b'
        }

        for col, pat in patterns.items():
            mask = temp['full_text'].str.contains(pat, regex=True, na=False)
            df[col] = df['nct_id'].map(dict(zip(temp['nct_id'], mask.astype(int)))).fillna(0).astype(int)

        df['is_critical_setting'] = df[['is_acute', 'is_refractory', 'is_severe']].max(axis=1)
        return df

    def _engineer_comparator_architecture(self, df):
        print("    -> Engineering Comparator Architecture...")
        cols = ['nct_id', 'group_type', 'title', 'description']
        df_groups = self._safe_load('design_groups.txt', cols=cols)

        if df_groups.empty:
            df['has_placebo'] = 0
            df['has_active_comparator'] = 0
            return df

        df_groups = df_groups[df_groups['nct_id'].isin(df['nct_id'])].copy()
        df_groups['group_type'] = df_groups['group_type'].fillna('UNKNOWN').str.upper()
        df_groups['text'] = (df_groups['title'].fillna('') + " " + df_groups['description'].fillna('')).str.lower()

        is_placebo = (
            df_groups['group_type'].str.contains('PLACEBO') |
            df_groups['text'].str.contains(r'\b(?:placebo|sham|vehicle|sugar pill)\b', regex=True)
        )

        is_active = (
            df_groups['group_type'].str.contains(r'ACTIVE[ _]COMPARATOR', regex=True) |
            df_groups['text'].str.contains(r'\b(?:standard of care|soc|active control|reference|marketed|standard therapy)\b', regex=True)
        )

        placebo_ids = df_groups[is_placebo]['nct_id'].unique()
        active_ids = df_groups[is_active]['nct_id'].unique()

        df['has_placebo'] = df['nct_id'].isin(placebo_ids).astype(int)
        df['has_active_comparator'] = df['nct_id'].isin(active_ids).astype(int)
        return df

    def _engineer_protocol_duration(self, df):
        print("    -> Engineering Protocol Duration...")
        cols = ['nct_id', 'outcome_type', 'time_frame']
        df_out = self._safe_load('outcomes.txt', cols=cols)

        if not df_out.empty:
            primary = df_out[df_out['outcome_type'].str.lower().str.contains('primary', na=False)].copy()
            prim_agg = primary.groupby('nct_id')['time_frame'].apply(lambda x: " ".join(x.dropna().astype(str))).reset_index()
            prim_agg.rename(columns={'time_frame': 'txt_outcome'}, inplace=True)
        else:
            prim_agg = pd.DataFrame(columns=['nct_id', 'txt_outcome'])

        df_sum = self._safe_load('brief_summaries.txt', cols=['nct_id', 'description'])
        if not df_sum.empty:
            df_sum.rename(columns={'description': 'txt_summary'}, inplace=True)

        df_dur = df[['nct_id', 'official_title']].copy()
        df_dur = df_dur.merge(prim_agg, on='nct_id', how='left')
        df_dur = df_dur.merge(df_sum, on='nct_id', how='left')

        def extract_duration(text):
            if pd.isna(text) or len(str(text)) < 3: return np.nan
            text = str(text).lower()
            if re.search(r'\b(overall survival|os|pfs|time to progression|until progression)\b', text):
                return 24.0
            text = text.replace('approx.', '').replace('approximately', '').strip()
            max_months = 0.0
            unit_pattern = r'(week|wk|month|mth|year|yr|day|hour|hr)'
            p1 = re.findall(r'(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)?\s*' + unit_pattern, text)
            p2 = re.findall(unit_pattern + r's?\s+(\d+(?:\.\d+)?)', text)

            def to_months(val, unit):
                u = unit.lower()
                if 'year' in u or 'yr' in u: return val * 12
                if 'week' in u or 'wk' in u: return val / 4.33
                if 'day' in u: return val / 30.4
                if 'hour' in u or 'hr' in u: return val / 730.0
                return val

            for m in p1:
                v1 = float(m[0]); v2 = float(m[1]) if m[1] else 0
                val = max(v1, v2)
                m_val = to_months(val, m[2])
                if m_val > max_months: max_months = m_val
            for m in p2:
                m_val = to_months(float(m[1]), m[0])
                if m_val > max_months: max_months = m_val
            if max_months > 0.5: return min(max_months, 120.0)
            if re.search(r'\b(single[ -]dose|one[ -]time|baseline|day 0|day 1|pre-dose)\b', text):
                return 0.1
            return max_months if max_months > 0 else np.nan

        def resolve(row):
            d = extract_duration(row['txt_outcome'])
            if pd.notna(d): return d
            d = extract_duration(row['official_title'])
            if pd.notna(d): return d
            d = extract_duration(row['txt_summary'])
            if pd.notna(d): return d
            return np.nan

        df_dur['duration_months'] = df_dur.apply(resolve, axis=1)
        df = df.merge(df_dur[['nct_id', 'duration_months']], on='nct_id', how='left')
        median_dur = df['duration_months'].median()
        if pd.isna(median_dur): median_dur = 12.0
        df['duration_months'] = df['duration_months'].fillna(median_dur).clip(upper=60)
        return df

    def _engineer_smart_patterns(self, df):
        print("    -> Engineering Smart Patterns...")
        def get_masking_score(val):
            val = str(val).lower()
            if 'quadruple' in val: return 3
            if 'double' in val: return 2
            if 'single' in val: return 1
            return 0
        def get_allocation_score(val):
            return 1 if 'randomized' in str(val).lower() else 0
        def get_model_score(val):
            val = str(val).lower()
            return 1 if 'crossover' in val or 'factorial' in val else 0

        df['score_masking'] = df['masking'].apply(get_masking_score)
        df['score_allocation'] = df['allocation'].apply(get_allocation_score)
        df['score_model'] = df['intervention_model'].apply(get_model_score)
        df['design_rigor_score'] = df['score_masking'] + df['score_allocation'] + df['score_model']

        df['is_gender_restricted'] = df['gender'].apply(lambda x: 0 if str(x).lower() == 'all' else 1)
        df['is_sick_only'] = df['healthy_volunteers'].apply(lambda x: 1 if x == 0 else 0)
        for col in ['child', 'adult', 'older_adult']:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: 1 if x.lower() in ['true', '1', 'yes'] else 0)
            else:
                df[col] = 1
        df['eligibility_strictness_score'] = df['is_gender_restricted'] + df['is_sick_only'] + (1 - df['child']) + (1 - df['older_adult'])
        return df

    def _engineer_agent_type(self, df):
        print("    -> Engineering Agent Type...")
        if self.df_drugs.empty:
            df['agent_category'] = 'UNKNOWN'
            return df
        df_int = self.df_drugs.copy()
        df_int['name_clean'] = df_int['name'].str.lower().fillna('')
        priority_map = {
            'RNA_GENE_THERAPY': 1, 'CELL_THERAPY': 1, 'ANTIBODY_DRUG_CONJUGATE': 1,
            'BISPECIFIC_ANTIBODY': 1, 'MONOCLONAL_ANTIBODY': 1, 'GLP1_PEPTIDE': 1,
            'PI3K_INHIBITOR': 1, 'BTK_INHIBITOR': 1, 'JAK_INHIBITOR': 1,
            'PARP_INHIBITOR': 1, 'BCL2_INHIBITOR': 1, 'SGLT2_INHIBITOR': 1,
            'KINASE_INHIBITOR_TYROSINE': 1, 'TARGETED_KINASE_INHIBITOR': 1,
            'ANTIVIRAL': 2, 'ANTIBIOTIC': 2, 'CORTICOSTEROID': 2, 'NSAID_PAIN': 2,
            'DIABETES_ORAL': 2, 'CHEMOTHERAPY': 2, 'HORMONAL_THERAPY': 2,
            'STATIN_CHOLESTEROL': 2, 'ENZYME_INHIBITOR': 2,
            'BIOLOGIC_OTHER': 3, 'SMALL_MOLECULE_GENERIC': 3, 'SMALL_MOLECULE_OTHER': 3,
            'PLACEBO_CTRL': 4
        }
        patterns = {
            'CELL_THERAPY': r'\b(car-t|chimeric antigen|autologous|allogeneic|t-cell|nk cell|stem cell|mesenchymal|t-lymphocytes|islet cell)\b|.*cel$',
            'RNA_GENE_THERAPY': r'\b(crispr|cas9|mrna|sirna|antisense|oligonucleotide|plasmid|vector|aav|rnai|gene transfer|gene therapy)\b',
            'ANTIBODY_DRUG_CONJUGATE': r'\b(adc|conjugate)\b.*mab|mab.*\b(adc|conjugate)\b',
            'BISPECIFIC_ANTIBODY': r'\b(bispecific|dual-targeting|bi-specific|engager)\b',
            'MONOCLONAL_ANTIBODY': r'.*mab\b',
            'GLP1_PEPTIDE': r'.*(tide|glutide)\b',
            'PI3K_INHIBITOR': r'.*lisib\b',
            'BTK_INHIBITOR': r'.*brutinib\b',
            'JAK_INHIBITOR': r'.*citinib\b',
            'PARP_INHIBITOR': r'.*parib\b',
            'BCL2_INHIBITOR': r'.*clax\b',
            'SGLT2_INHIBITOR': r'.*gliflozin\b',
            'KINASE_INHIBITOR_TYROSINE': r'.*tinib\b',
            'TARGETED_KINASE_INHIBITOR': r'.*ib\b',
            'CHEMOTHERAPY': r'.*(platin|taxel|rubicin|fluorouracil|gemcitabine|cyclophosphamide|methotrexate|etoposide|vincristine|vinblastine|irinotecan|oxaliplatin|ifosfamide|pemetrexed)\b',
            'HORMONAL_THERAPY': r'\b(tamoxifen|anastrozole|letrozole|exemestane|fulvestrant|bicalutamide|enzalutamide|abiraterone)\b',
            'STATIN_CHOLESTEROL': r'.*vastatin\b',
            'ENZYME_INHIBITOR': r'.*stat\b',
            'ANTIVIRAL': r'.*(vir|virine|viro)\b',
            'ANTIBIOTIC': r'.*(cillin|mycin|micin|oxacin|cycline)\b',
            'NSAID_PAIN': r'.*(fenac|profen|coxib)\b',
            'CORTICOSTEROID': r'.*(sone|solone|nide)\b',
            'DIABETES_ORAL': r'.*(gliptin|glitazone|formin)\b',
            'SMALL_MOLECULE_GENERIC': r'.*(ine|ide|one|ole|ate|ant|ent|ril|lol|tan|vir|micin|mycin|acin|xaban|stat|pril|sartan|prazole|nib|mab|cept|tide)\b',
            'PLACEBO_CTRL': r'\b(placebo|sham|control|comparator)\b'
        }
        def classify(row):
            name = row['name_clean']
            itype = str(row.get('intervention_type', '')).upper()
            for cat, pattern in patterns.items():
                if cat == 'PLACEBO_CTRL': continue
                if re.search(pattern, name): return cat
            if re.search(patterns['PLACEBO_CTRL'], name): return 'PLACEBO_CTRL'
            return 'BIOLOGIC_OTHER' if itype == 'BIOLOGICAL' else 'SMALL_MOLECULE_OTHER'

        df_int['agent_category'] = df_int.apply(classify, axis=1)
        df_int['priority'] = df_int['agent_category'].map(priority_map).fillna(3)
        best_agent = df_int.sort_values(['nct_id', 'priority']).drop_duplicates('nct_id')
        df = df.merge(best_agent[['nct_id', 'agent_category']], on='nct_id', how='left')
        df['agent_category'] = df['agent_category'].fillna('SMALL_MOLECULE_OTHER')
        return df

    def _engineer_safe_features(self, df):
        print("    -> Engineering Gated Protocol Features...")
        if 'has_dmc' in df.columns:
            df['has_dmc'] = df['has_dmc'].astype(str).apply(lambda x: 1 if x.lower() in ['true', 't', '1', 'yes'] else 0)
        else:
            df['has_dmc'] = 0
        if 'is_fda_regulated_drug' in df.columns:
            raw_signal = df['is_fda_regulated_drug'].astype(str).apply(lambda x: 1 if x.lower() in ['true', 't', '1', 'yes'] else 0)
            df['is_fda_regulated_drug'] = ((raw_signal == 1) & (df['start_year'] >= 2017)).astype(int)
        else:
            df['is_fda_regulated_drug'] = 0
        return df

    def _attach_medical_hierarchy(self, df):
        print("    -> Attaching Medical Hierarchy...")
        cols_bridge = ['nct_id', 'mesh_term']
        df_bridge = self._safe_load('browse_conditions.txt', cols=cols_bridge)
        if df_bridge.empty:
            df_bridge = self._safe_load('conditions.txt', cols=['nct_id', 'name'])
            if not df_bridge.empty: df_bridge.rename(columns={'name': 'mesh_term'}, inplace=True)
        if df_bridge.empty:
            df['mesh_term'], df['therapeutic_area'] = np.nan, np.nan
            return df
        mesh_path = os.path.join(self.data_path, 'mesh_lookup.csv')
        df_dictionary = pd.DataFrame()
        if os.path.exists(mesh_path):
            try:
                df_dictionary = pd.read_csv(mesh_path, sep='|', on_bad_lines='skip')
                if 'mesh_term' in df_dictionary.columns and 'therapeutic_area' in df_dictionary.columns:
                    df_dictionary = df_dictionary[['mesh_term', 'therapeutic_area']].drop_duplicates()
            except: pass
        if not df_dictionary.empty: df_full_mesh = df_bridge.merge(df_dictionary, on='mesh_term', how='left')
        else: df_full_mesh = df_bridge.copy(); df_full_mesh['therapeutic_area'] = np.nan
        bad_values = ['Unclassified', 'Other/Unclassified', 'Pathological Conditions', 'None', '', 'nan']
        df_full_mesh['therapeutic_area'] = df_full_mesh['therapeutic_area'].replace(bad_values, np.nan)
        df_full_mesh['mesh_term'] = df_full_mesh['mesh_term'].astype(str)
        fallbacks = {
            'Oncology': r'\b(?:cancer|tumor|carcinoma|lymphoma|leukemia|melanoma|neoplasm|oncology|solid|malignant|adenocarcinoma|sarcoma|myeloma|glioma|metastatic|advanced|recurrent|squamous|her2|kras|glioblastoma|prostate|myelodysplastic|myelofibrosis|polycythemia|cachexia)\b',
            'Cardiovascular': r'\b(?:heart|cardiac|vascular|stent|hypertension|myocardial|atrial|coronary|stroke|embolism|arrhythmia|cholesterol|angina|infarction|hfpef|hfrer|tachycardia|atherosclerosis|venous thromboembolism|ischemia|thrombosis|thrombocytopenia|hemorrhage|bleeding)\b',
            'Metabolic': r'\b(?:diabetes|diabetic|insulin|obesity|hyperlipidemia|metabolic|glucose|nash|steatohepatitis|dyslipidemia|t2dm|hypoglycemia|weight loss|fatty liver|endocrine|hypercholesterolemia|hypertriglyceridemia|growth hormone|hyperparathyroidism|hyperkalemia|hyperphosphatemia|overweight|cushing|adrenal hyperplasia|hyperuricemia)\b',
            'Neurology': r'\b(?:alzheimer|parkinson|brain|neurology|epilepsy|sclerosis|migraine|cns|neurodegenerative|dementia|als|huntington|seizure|neuropathic|neuropathy|pain|fibromyalgia|myasthenia gravis|restless legs|tourette|tremor|dystonia|rett syndrome|ataxia|autism)\b',
            'Infections': r'\b(?:infection|virus|hiv|bacterial|fungal|antibiotic|covid|hepatitis|influenza|pneumonia|sepsis|septic|tuberculosis|vaccine|antiviral|hiv-1|sars-cov-2|pneumococcal|tetanus|diphtheria|malaria|poliomyelitis|lice|candidiasis|helicobacter|meningococcal|aspergillosis|bacteremia|haemophilus|pathogen)\b',
            'Immunology': r'\b(?:arthritis|lupus|autoimmune|inflammation|crohn|psoriasis|rheumatoid|ulcerative colitis|dermatitis|eczema|asthma|atopic|sjogren|ankylosing|celiac|dermatomyositis|graft versus host|spondyloarthritis)\b',
            'Gastrointestinal': r'\b(?:gastric|gi|bowel|stomach|liver|hepatic|cirrhosis|gerd|colitis|ibs|digestive|peptic|esophagitis|constipation|biliary cholangitis|dyspepsia|gastritis|gastroparesis|ileus|reflux|gastroesophageal|esophagus|vomiting|nausea|heartburn|diarrhea)\b',
            'Renal/Urology': r'\b(?:kidney|renal|nephropathy|urology|bladder|ckd|dialysis|urinary|prostatitis|erectile|benign prostatic hyperplasia|nocturia)\b',
            'Psychiatry': r'\b(?:depression|depressive|anxiety|schizophrenia|bipolar|psychiatric|adhd|autism|ptsd|major depressive|mental|insomnia|attention deficit|alcohol|eating disorder|binge eating|mood disorder|opioid use|opioid dependence)\b',
            'Dermatology': r'\b(?:skin|dermatology|acne|urticaria|rosacea|alopecia|vitiligo|pruritus|hidradenitis|actinic keratosis|tinea|baldness|pyoderma|onychomycosis|prurigo|cellulite)\b',
            'Respiratory': r'\b(?:copd|asthma|pulmonary|lung|respiratory|bronchitis|cf|cystic fibrosis|rhinitis|cough|sinusitis|rhinoconjunctivitis)\b',
            'Ophthalmology': r'\b(?:eye|glaucoma|cataract|macular|retina|ocular|vision|cornea|conjunctivitis|uveitis|presbyopia|blepharitis|diabetic retinopathy|retinopathy|vitrectomy|hypotrichosis)\b',
            'Musculoskeletal': r'\b(?:spine|back pain|osteoarthritis|bone|muscle|fracture|orthopedic|knee|hip|osteoporosis|gout|duchenne|spondyloarthritis|disc disease|atrophy)\b',
            'Hematology': r'\b(?:anemia|blood|hemophilia|sickle|platelet|bleeding|purpura|thrombocytopenia)\b',
            'Reproductive': r'\b(?:contraception|infertility|endometriosis|uterine|menopause|hot flash|hypogonadism|libido|contraceptive|ovarian|sperm|erectile|fibroids|premature ejaculation|vaginal atrophy|cervical ripening|dysmenorrhea|vulvovaginal)\b',
            'Genetic': r'\b(?:duchenne|fabry|hereditary|genetic|mutation|cystic fibrosis|huntington|spinal muscular|hemophilia|fragile x|pompe|gaucher|prader-willi|friedreich|down syndrome)\b',
            'Pain/Anesthesia': r'\b(?:pain|neuralgia|analgesic|anesthesia|postoperative|neuropathic|migraine|headache|nociceptive|opioid|sedation|blockade)\b',
            'Aesthetic': r'\b(?:glabellar|canthal|wrinkle|botulinum|filler|facial lines|aesthetic|cosmetic|acne|submental fat|subcutaneous fat|crow\'s feet)\b',
            'Healthy': r'\b(?:healthy|bioequivalence|pharmacokinetics|safety study|phase 1|hygiene|smoking)\b',
            'Dental': r'\b(?:dental|gingivitis|periodontitis|tooth|teeth|plaque|oral health|caries)\b',
            'Sleep': r'\b(?:sleep apnea|insomnia|narcolepsy|sleep disorder|restless legs)\b',
            'Ear/Nose/Throat': r'\b(?:otitis|tinnitus|sinusitis|nasal polyps|rhinitis)\b'
        }
        for area, pattern in fallbacks.items():
            missing_mask = df_full_mesh['therapeutic_area'].isna()
            if not missing_mask.any(): break
            matches = df_full_mesh.loc[missing_mask, 'mesh_term'].str.contains(pattern, case=False, regex=True, na=False)
            if matches.any(): df_full_mesh.loc[matches[matches].index, 'therapeutic_area'] = area
        df_full_mesh['therapeutic_area'] = df_full_mesh['therapeutic_area'].fillna('Unclassified')
        df_full_mesh['therapeutic_subgroup_name'] = df_full_mesh['mesh_term']
        bad_mesh_mask = (df_full_mesh['therapeutic_subgroup_name'].isna()) | (df_full_mesh['therapeutic_subgroup_name'].astype(str).str.lower() == 'nan')
        df_full_mesh.loc[bad_mesh_mask, 'therapeutic_subgroup_name'] = df_full_mesh.loc[bad_mesh_mask, 'therapeutic_area']
        df_full_mesh['is_good'] = (df_full_mesh['therapeutic_area'] != 'Unclassified').astype(int)
        df_full_mesh.sort_values(['nct_id', 'is_good'], ascending=[True, False], inplace=True)
        df_best_match = df_full_mesh.drop_duplicates(subset=['nct_id'], keep='first')
        df = df.merge(df_best_match[['nct_id', 'mesh_term', 'therapeutic_area', 'therapeutic_subgroup_name']], on='nct_id', how='left')
        unification_map = {
            'Infections (Bacterial/Fungal)': 'Infections', 'Digestive': 'Gastrointestinal', 'Ophthalmology (Eye)': 'Ophthalmology',
            'Urology (Male)': 'Renal/Urology', 'Congenital/Genetic': 'Genetic', 'Stomatognathic': 'Dental',
            'ENT (Ear/Nose/Throat)': 'Ear/Nose/Throat', 'Wounds': 'Musculoskeletal', 'Pathological Conditions': 'Unclassified',
            'Other/Unclassified': 'Unclassified', 'Chemically Induced': 'Unclassified', 'Animal Diseases': 'Unclassified'
        }
        df['therapeutic_area'] = df['therapeutic_area'].replace(unification_map)
        mask_still_missing = (df['therapeutic_area'] == 'Unclassified') | (df['therapeutic_area'].isna())
        if mask_still_missing.sum() > 0:
            df['official_title'] = df['official_title'].astype(str)
            for area, pattern in fallbacks.items():
                current_missing_mask = (df['therapeutic_area'] == 'Unclassified') | (df['therapeutic_area'].isna())
                if not current_missing_mask.any(): break
                matches = df.loc[current_missing_mask, 'official_title'].str.contains(pattern, case=False, regex=True, na=False)
                if matches.any(): df.loc[matches[matches].index, 'therapeutic_area'] = area
        df['therapeutic_area'] = df['therapeutic_area'].fillna('Unclassified')
        df['therapeutic_subgroup_name'] = df['therapeutic_subgroup_name'].fillna('Unclassified')
        return df

    def _engineer_sponsor_features(self, df):
        print("    -> Engineering Sponsor Tiers...")
        cols_needed = ['nct_id', 'lead_or_collaborator', 'name', 'agency_class']
        df_sponsors = self._safe_load('sponsors.txt', cols=cols_needed)
        if df_sponsors.empty:
            df['sponsor_tier'] = 'TIER_2_OTHER'; df['sponsor_clean'] = 'UNKNOWN'; df['agency_class'] = 'UNKNOWN'
            return df
        leads = df_sponsors[df_sponsors['lead_or_collaborator'].str.lower() == 'lead'][['nct_id', 'name', 'agency_class']]
        leads = leads.rename(columns={'name': 'lead_sponsor'}).drop_duplicates('nct_id')
        df = df.merge(leads, on='nct_id', how='left')
        df['lead_sponsor'] = df['lead_sponsor'].fillna('UNKNOWN')
        df['agency_class'] = df['agency_class'].fillna('UNKNOWN')
        clean_col = df['lead_sponsor'].astype(str).str.lower().str.strip()
        legal_pattern = r'[.,]|\binc\b|\bltd\b|\bllc\b|\bcorp\b|\bgmbh\b|\bsa\b|\bplc\b'
        clean_col = clean_col.str.replace(legal_pattern, '', regex=True).str.strip()
        mappings = {
            'Pfizer': ['pfizer', 'wyeth', 'hospira'], 'GSK': ['glaxo', 'gsk', 'smithkline'], 'Novartis': ['novartis', 'sandoz'],
            'AstraZeneca': ['astrazeneca', 'medimmune'], 'Merck': ['merck', 'msd'], 'Roche': ['roche', 'genentech', 'hoffmann'],
            'Sanofi': ['sanofi', 'aventis', 'genzyme'], 'J&J': ['johnson & johnson', 'janssen'], 'Bayer': ['bayer', 'monsanto'],
            'Boehringer': ['boehringer'], 'BMS': ['bristol-myers', 'squibb', 'celgene'], 'Lilly': ['lilly'],
            'Abbott': ['abbott', 'abbvie'], 'Amgen': ['amgen'], 'Takeda': ['takeda', 'shire'], 'Gilead': ['gilead'],
            'Novo Nordisk': ['novo nordisk'], 'NIH': ['national cancer institute', 'nci', 'national institutes of health', 'nih']
        }
        final_names = clean_col.copy()
        for std, keys in mappings.items():
            pattern = '|'.join(keys)
            mask = clean_col.str.contains(pattern, case=False, regex=True)
            final_names.loc[mask] = std
        df['sponsor_clean'] = final_names
        df['sponsor_tier'] = df['sponsor_clean'].apply(lambda x: 'TIER_1_GIANT' if x in mappings.keys() else 'TIER_2_OTHER')
        return df

    def _engineer_complexity(self, df):
        print("    -> Engineering Protocol Complexity...")
        cols_needed = ['nct_id', 'criteria', 'gender', 'healthy_volunteers', 'minimum_age', 'maximum_age']
        df_elig = self._safe_load('eligibilities.txt', cols=cols_needed)
        if df_elig.empty:
            df['criteria_len_log'] = 0
            for c in ['gender', 'healthy_volunteers', 'adult', 'child', 'older_adult']: df[c] = 0
            return df
        df_elig = df_elig.drop_duplicates('nct_id')
        df = df.merge(df_elig, on='nct_id', how='left')
        df['criteria_len_log'] = np.log1p(df['criteria'].astype(str).str.len().fillna(0))
        df['healthy_volunteers'] = df['healthy_volunteers'].astype(str).str.lower().apply(lambda x: 0 if x in ['f', 'false', '0', 'no', 'nan', 'none'] else 1)
        df['gender'] = df['gender'].fillna('UNKNOWN').str.upper()

        def parse_age_to_years(val, default_val):
            if pd.isna(val) or str(val).lower() in ['n/a', 'nan', '', 'none']: return default_val
            try:
                match = re.search(r'(\d+(?:\.\d+)?)', str(val))
                if not match: return default_val
                num = float(match.group(1)); text = str(val).lower()
                if 'month' in text: num /= 12.0
                elif 'week' in text: num /= 52.0
                elif 'day' in text: num /= 365.0
                return num
            except: return default_val

        df['min_age_years'] = df['minimum_age'].apply(lambda x: parse_age_to_years(x, 0.0))
        df['max_age_years'] = df['maximum_age'].apply(lambda x: parse_age_to_years(x, 100.0))
        df['child'] = (df['min_age_years'] < 18).astype(int)
        df['adult'] = ((df['max_age_years'] >= 18) & (df['min_age_years'] < 65)).astype(int)
        df['older_adult'] = (df['max_age_years'] > 65).astype(int)
        df.drop(columns=['minimum_age', 'maximum_age', 'min_age_years', 'max_age_years'], inplace=True, errors='ignore')
        return df

    def _calculate_competition(self, df):
        print("    -> Engineering Smart Competition...")
        try:
            req_cols = ['start_year', 'therapeutic_area', 'therapeutic_subgroup_name', 'agent_category']
            for col in req_cols:
                if col not in df.columns: df[col] = 'UNKNOWN'
            def get_rolling_density(dataframe, group_col):
                counts = dataframe.groupby(['start_year', group_col]).size().reset_index(name='yr_count')
                lookup = dict(zip(zip(counts['start_year'], counts[group_col]), counts['yr_count']))
                def calc(row):
                    val = row[group_col]
                    generics = ['UNKNOWN', 'Unclassified', 'SMALL_MOLECULE_OTHER', 'BIOLOGIC_OTHER', 'SMALL_MOLECULE_GENERIC', 'PLACEBO_CTRL']
                    if val in generics: return 0
                    if group_col == 'therapeutic_subgroup_name' and val == row['therapeutic_area']: return 0
                    y = row['start_year']
                    return lookup.get((y, val), 0) + lookup.get((y-1, val), 0)
                return dataframe.apply(calc, axis=1)
            df['competition_broad'] = get_rolling_density(df, 'therapeutic_area')
            df['competition_niche'] = get_rolling_density(df, 'therapeutic_subgroup_name')
            df['competition_agent'] = get_rolling_density(df, 'agent_category')
        except:
            df['competition_broad'] = 0; df['competition_niche'] = 0; df['competition_agent'] = 0
        return df

    def _add_ui_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        print("    -> Adding UI-friendly text fields...")
        df = df.copy()
        if "official_title" not in df.columns: df["official_title"] = ""
        if "brief_title" not in df.columns: df["brief_title"] = df["official_title"]
        df_summaries = self._safe_load("brief_summaries.txt", cols=["nct_id", "description"])
        if not df_summaries.empty: df = df.merge(df_summaries.rename(columns={"description": "ui_summary_raw"}), on="nct_id", how="left")
        else: df["ui_summary_raw"] = ""
        if "criteria" not in df.columns: df["criteria"] = ""
        df["ui_title"] = df["official_title"].fillna("").astype(str).apply(ui_clean_text)
        df["ui_brief_title"] = df["brief_title"].fillna("").astype(str).apply(ui_clean_text)
        df["ui_summary"] = df["ui_summary_raw"].fillna("").astype(str).apply(ui_clean_text)
        df["ui_criteria"] = df["criteria"].fillna("").astype(str).apply(ui_clean_text)
        df.drop(columns=["ui_summary_raw"], inplace=True, errors="ignore")
        return df

    def _prepare_text(self, df):
        print("    -> Engineering NLP Text Pillars...")
        df_summaries = self._safe_load('brief_summaries.txt', cols=['nct_id', 'description'])
        df_outcomes = self._safe_load('design_outcomes.txt', cols=['nct_id', 'measure', 'outcome_type'])
        if not df_summaries.empty: df = df.merge(df_summaries.rename(columns={'description': 'temp_summary'}), on='nct_id', how='left')
        if not df_outcomes.empty:
            primaries = df_outcomes[df_outcomes['outcome_type'].str.lower().str.contains('primary', na=False)].copy()
            endpoints = primaries.groupby('nct_id')['measure'].apply(lambda x: " [SEP] ".join(x.dropna().astype(str))).reset_index(name='txt_primary_endpoints')
            df = df.merge(endpoints, on='nct_id', how='left')
        df['official_title'] = df['official_title'].fillna("No title provided").astype(str)
        df['temp_summary'] = df.get('temp_summary', pd.Series([""*len(df)])).fillna("No summary provided").astype(str)
        df['txt_primary_endpoints'] = df.get('txt_primary_endpoints', pd.Series([""*len(df)])).fillna("No endpoints provided").astype(str)
        df['criteria'] = df['criteria'].fillna("No criteria provided").astype(str)
        df['txt_scientific_essence'] = df['official_title'].str.strip() + " [SEP] " + df['temp_summary'].str.strip()
        df['txt_criteria'] = df['criteria']
        df.drop(columns=['official_title', 'temp_summary', 'criteria', 'why_stopped'], inplace=True, errors='ignore')
        return df

    def _engineer_phase_groups(self, df):
        print("    -> Grouping Phases...")
        df['phase'] = df['phase'].astype(str).str.upper().str.strip()
        phase_map = {'PHASE1/PHASE2': 'Early_Efficacy', 'PHASE2': 'Early_Efficacy', 'PHASE2/PHASE3': 'Confirmatory', 'PHASE3': 'Confirmatory'}
        df['phase_group'] = df['phase'].map(phase_map).fillna('Confirmatory')
        return df

    def _attach_embeddings(self, df):
        print("    -> Attaching BioBERT Embeddings...")
        path_key = os.path.join(self.data_path, 'project_data_nlp_light.csv')
        path_crit = os.path.join(self.data_path, 'biobert_crit_raw.npy')
        path_sci = os.path.join(self.data_path, 'biobert_sci_raw.npy')
        path_endp = os.path.join(self.data_path, 'biobert_endp_raw.npy')
        if not os.path.exists(path_key) or not all(os.path.exists(p) for p in [path_crit, path_sci, path_endp]):
            print("       [!] Missing embedding files. Skipping."); return df
        try:
            df_key = pd.read_csv(path_key, usecols=['nct_id'], dtype=str)
            ids = df_key['nct_id'].values; n_samples = len(ids)
            def load_and_merge(npy_path, prefix, target_df):
                data = np.load(npy_path)
                if len(data) != n_samples: return target_df
                cols = [f"{prefix}_{i}" for i in range(data.shape[1])]
                df_emb = pd.DataFrame(data, columns=cols); df_emb['nct_id'] = ids
                return target_df.merge(df_emb.drop_duplicates('nct_id'), on='nct_id', how='left')
            df = load_and_merge(path_crit, "crit", df)
            df = load_and_merge(path_sci, "sci", df)
            df = load_and_merge(path_endp, "endp", df)
            return df
        except: return df

    def _attach_p_values(self, df):
        print("    -> Attaching P-Values...")
        cols_out = ['nct_id', 'id', 'outcome_type']
        df_out = self._safe_load('outcomes.txt', cols=cols_out)
        cols_ana = ['nct_id', 'outcome_id', 'p_value', 'p_value_modifier']
        df_ana = self._safe_load('outcome_analyses.txt', cols=cols_ana)
        if df_out.empty or df_ana.empty:
            df['min_p_value'] = np.nan; df['scientific_success'] = 0; return df
        df_out['id'] = df_out['id'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_ana['outcome_id'] = df_ana['outcome_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        merged = df_ana.merge(df_out, left_on=['nct_id', 'outcome_id'], right_on=['nct_id', 'id'], how='inner')
        primary = merged[merged['outcome_type'].astype(str).str.lower().str.contains('primary', na=False)].copy()
        if primary.empty:
            df['min_p_value'] = np.nan; df['scientific_success'] = 0; return df
        primary['p_val_num'] = pd.to_numeric(primary['p_value'].astype(str).str.replace(',', '.'), errors='coerce')
        def adjust_p_value(row):
            val = row['p_val_num']; mod = str(row['p_value_modifier']).strip()
            if pd.isna(val): return np.nan
            if '<' in mod: return val - 0.000001
            if '>' in mod: return val + 0.000001
            return val
        primary['adjusted_p'] = primary.apply(adjust_p_value, axis=1)
        trial_stats = primary.groupby('nct_id')['adjusted_p'].min().reset_index()
        trial_stats.rename(columns={'adjusted_p': 'min_p_value'}, inplace=True)
        df = df.merge(trial_stats, on='nct_id', how='left')
        df['scientific_success'] = df['min_p_value'].apply(lambda x: 1 if pd.notna(x) and x <= 0.05 else 0)
        return df

    def _merge_file(self, df, filename, cols, filter_col=None, filter_val=None):
        try:
            aux = self._safe_load(filename, cols=cols + ([filter_col] if filter_col else []))
            if aux.empty: return df
            if filter_col: aux = aux[aux[filter_col] == filter_val].drop(columns=[filter_col])
            aux = aux.drop_duplicates('nct_id')
            return df.merge(aux, on='nct_id', how='left')
        except: return df

    def save(self, df, filename='project_data_predict.csv'):
        out_path = os.path.join(self.data_path, filename)
        df.to_csv(out_path, index=False)
        print(f">>> Saved {len(df)} rows to {out_path}")
