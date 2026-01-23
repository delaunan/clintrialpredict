import pandas as pd
import numpy as np
import os
import csv
import re

from src.prep.text_cleaning import day_zero_reconstructor
from src.prep.text_cleaning_ui import ui_clean_text


class ClinicalTrialLoader:

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

        # --- FIRST MAIN FUNCTION CALLED ---

    def load_and_clean(self):

        print(">>> 1. Loading Studies & Applying Filters...")

        # 0. Load Core Columns (Added safety columns: has_dmc, is_fda_regulated_drug)
        cols_studies = ['nct_id', 'overall_status', 'study_type', 'phase',
                        'start_date', 'number_of_arms',
                        'official_title', 'brief_title', 'why_stopped',
                        'has_dmc', 'is_fda_regulated_drug']

        df = self._safe_load('studies.txt', cols=cols_studies)

        if df.empty:
            raise ValueError("Critical Error: 'studies.txt' failed to load.")


        # --- FALLBACK LOGIC : official -> brief title ---
        # 1. Fill missing Official Title with Brief Title (for NLP & Model)
        df['official_title'] = df['official_title'].fillna(df['brief_title'])

        # 2. Ensure Brief Title is never null (fill with Official Title if Brief is missing)
        # This ensures your UI always has a short name to display
        df['brief_title'] = df['brief_title'].fillna(df['official_title'])


        # 1. INDUSTRY FILTER (Based on Audit: 'INDUSTRY' is uppercase)
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
            # We include GENETIC here to capture gene therapies
            target_types = ['DRUG', 'BIOLOGICAL', 'GENETIC']
            drug_ids = df_int[df_int['intervention_type'].str.upper().isin(target_types)]['nct_id'].unique()
            df = df[df['nct_id'].isin(drug_ids)]
            # Save for Agent Engineering
            self.df_drugs = df_int[df_int['intervention_type'].str.upper().isin(target_types + ['DIETARY SUPPLEMENT', 'OTHER'])].copy()
        else:
            self.df_drugs = pd.DataFrame(columns=['nct_id', 'name', 'intervention_type'])

        # 4. Filter: Status (Completed vs Failed)
        allowed_statuses = ['COMPLETED', 'TERMINATED', 'WITHDRAWN']
        df = df[df['overall_status'].isin(allowed_statuses)]

        # 5. Filter: Phase (DROP PHASE 0 - Focus on Valley of Death)
        # We keep Phase 2, Phase 2/3, and Phase 3.
        excluded_phases = ['EARLY_PHASE1', 'PHASE1','PHASE4', 'NA']
        df = df[~df['phase'].str.upper().isin(excluded_phases)]
        df = df.dropna(subset=['phase'])
        df = df[df['phase'].str.strip() != '']

        # 6. Filter: COVID Sanitizer (Remove Pandemic Failures)
        if 'why_stopped' in df.columns:
            covid_keywords = ['covid', 'pandemic', 'coronavirus', 'sars-cov-2','travel restrictions', 'quarantine', 'lockdown', 'sars-cov']
            # Convert to string, lower case, check for keywords
            mask_covid = df['why_stopped'].fillna('').astype(str).str.lower().apply(
                lambda x: any(k in x for k in covid_keywords)
            )
            if mask_covid.sum() > 0:
                print(f"    [Sanitizer] Dropping {mask_covid.sum()} trials terminated due to COVID/Logistics.")
                df = df[~mask_covid]

        # 7. Filter: Date Range (2005-2025 for training)
        # Avoids Right-Censoring (trials still running) and COVID era bias
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['start_year'] = df['start_date'].dt.year
        df = df[df['start_year'].between(2005, 2025)]

        # 8. Create Target
        df['target'] = df['overall_status'].apply(lambda x: 0 if x.upper() == 'COMPLETED' else 1)

        print(f"    Core Cohort: {len(df)} trials (Phase 1/2/3, 2005-2025 training window and 2005-2025 for production).")
        return df.copy()

    # --- SECOND MAIN FUNCTION CALLED ---

    def add_features(self, df):

        df = df.copy()
        print(">>> 2. Engineering Features...")

        # 1. Phase Grouping
        df = self._engineer_phase_groups(df)

        # 2. Geography (SAFE: Only is_us, removed country counts to prevent leakage)
        df_countries = self._safe_load('countries.txt', cols=['nct_id', 'name'])
        if not df_countries.empty:
            us_trials = df_countries[df_countries['name'] == 'United States']['nct_id'].unique()
            df['includes_us'] = df['nct_id'].isin(us_trials).astype(int)
        else:
            df['includes_us'] = 0


        # 3. Merge (Designs & Calculated Values)
        df = self._merge_file(df, 'designs.txt', ['nct_id', 'allocation', 'intervention_model', 'masking', 'primary_purpose'])
        df = self._merge_file(df, 'calculated_values.txt', ['nct_id', 'number_of_primary_outcomes_to_measure'])

        df = self._engineer_clinical_setting(df)        # Patient State (Deep Search)
        df = self._engineer_comparator_architecture(df) # Scientific Hurdle (Robust)
        df = self._engineer_protocol_duration(df)       # Operational Horizon (Waterfall)


        # 4. Sponsor Engineering
        df = self._engineer_sponsor_features(df)

        # 5. Complexity Engineering
        df = self._engineer_complexity(df)

        # 6. Medical Hierarchy & Competition
        df = self._attach_medical_hierarchy(df)

        # 7A. UI-only text fields (keep latest readable text, remove struck content)
        df = self._add_ui_text_fields(df)

        # 7B. NLP pillars for embeddings/training (txt_*)
        df = self._prepare_text(df)

        # 8. Agent Type (The Bulletproof Classifier)
        df = self._engineer_agent_type(df)

        # 9. with have agent_category AND therapeutic_subgroup_name, we can calculate competition
        df = self._calculate_competition(df)

        # 10. Smart Patterns (Rigor & Strictness)
        df = self._engineer_smart_patterns(df)

        # 11. Safe Features (DMC, Responsible Party)
        df = self._engineer_safe_features(df)

        # 12. Attach Embeddings (BioBERT)
        # LOGIC: If .npy files exist (Scenario B), load them.
        # If not (Scenario A), skip so we can generate the CSV for Colab.
        df = self._attach_embeddings(df)

        # 13. Attach P-Values (Analysis Only)
        df = self._attach_p_values(df)


        # ----------------------------------------------------------------------
        # FINAL TYPE ENFORCEMENT (Schema Validation)
        # ----------------------------------------------------------------------
        print("    -> Finalizing Data Types...")

        # 1. Handle Primary Endpoints (Renaming)
        if 'number_of_primary_outcomes_to_measure' in df.columns:
            df.rename(columns={'number_of_primary_outcomes_to_measure': 'num_primary_endpoints'}, inplace=True)

        # 2. Enforce Datetime (Critical for Sorting)
        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')

        # 3. Enforce Numerics (Float/Int) - Critical for .between()
        # We explicitly list columns that MUST be numeric for the model/split
        numeric_cols = [
            'target', 'start_year', 'includes_us', 'num_primary_endpoints',
            'is_acute', 'is_refractory', 'is_severe', 'is_critical_setting',
            'has_placebo', 'has_active_comparator', 'duration_months',
            'design_rigor_score', 'eligibility_strictness_score', 'is_sick_only',
            'criteria_len_log', 'competition_broad', 'competition_niche', 'competition_agent',
            'has_dmc', 'is_fda_regulated_drug', 'child', 'adult', 'older_adult',
            'min_p_value', 'scientific_success'
        ]

        for col in numeric_cols:
            if col in df.columns:
                # errors='coerce' turns "2010" (str) -> 2010 (int) and "bad" -> NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill specific missing values that should be 1 or 0
        df['num_primary_endpoints'] = df['num_primary_endpoints'].fillna(1)

        return df



    # --- FEATURE ENGINEERING METHODS ---

    def _engineer_clinical_setting(self, df):
        """
        Determines Patient State (Acute, Refractory, Severe).
        Strategy: Deep Search across Official Title, Conditions, and Brief Summary.
        Update: Adds 'is_critical_setting' as a composite flag.
        """
        print("    -> Engineering Clinical Setting (Deep Search & Composite)...")

        # 1. Load Conditions
        cols = ['nct_id', 'name']
        df_cond = self._safe_load('conditions.txt', cols=cols)
        if df_cond.empty:
            df_cond = self._safe_load('browse_conditions.txt', cols=['nct_id', 'mesh_term'])
            if not df_cond.empty:
                df_cond.rename(columns={'mesh_term': 'name'}, inplace=True)

        # 2. Load Summaries
        df_sum = self._safe_load('brief_summaries.txt', cols=['nct_id', 'description'])

        # 3. Aggregate Text per Trial
        # Filter to current cohort
        df_cond = df_cond[df_cond['nct_id'].isin(df['nct_id'])].copy()
        cond_agg = df_cond.groupby('nct_id')['name'].apply(lambda x: " ".join(str(v) for v in x)).reset_index()

        # Merge everything into a temporary dataframe
        temp = df[['nct_id', 'official_title']].merge(cond_agg, on='nct_id', how='left')
        temp = temp.merge(df_sum, on='nct_id', how='left')

        # Create the "Deep Search" Blob
        temp['full_text'] = (
            temp['official_title'].fillna('') + " " +
            temp['name'].fillna('') + " " +
            temp['description'].fillna('')
        ).str.lower()

        # 4. Regex Patterns (Tightened for precision)
        patterns = {
            # Acute: Focus on speed/urgency
            'is_acute': r'\b(?:acute|emergency|critical|sepsis|shock|stroke|infarction|crisis|sudden|urgent|icu|trauma)\b',

            # Refractory: Focus on treatment failure (Added 'resistant to')
            'is_refractory': r'\b(?:refractory|resistant|relapsed|recur|second line|third line|salvage|unresponsive|previously treated)\b',

            # Severe: Focus on disease burden
            'is_severe': r'\b(?:severe|advanced|metastatic|stage (?:iv|4|iii|3)|end-stage|decompensated|invasive|terminal)\b'
        }

        # Apply Regex
        for col, pat in patterns.items():
            mask = temp['full_text'].str.contains(pat, regex=True, na=False)
            df[col] = df['nct_id'].map(dict(zip(temp['nct_id'], mask.astype(int)))).fillna(0).astype(int)

        # 5. Create Composite Feature (The "Union" you requested)
        # If ANY of the above are 1, this flag becomes 1.
        df['is_critical_setting'] = df[['is_acute', 'is_refractory', 'is_severe']].max(axis=1)

        return df

    def _engineer_comparator_architecture(self, df):
        """
        Determines Scientific Hurdle (Placebo vs Active).
        Strategy: Robust Regex on Group Types and Descriptions.
        Logic: 'ANY' aggregation (If any arm is Active, the trial is Active).
        """
        print("    -> Engineering Comparator Architecture (Robust)...")

        cols = ['nct_id', 'group_type', 'title', 'description']
        df_groups = self._safe_load('design_groups.txt', cols=cols)

        if df_groups.empty:
            df['has_placebo'] = 0
            df['has_active_comparator'] = 0
            return df

        # Filter to current cohort
        df_groups = df_groups[df_groups['nct_id'].isin(df['nct_id'])].copy()

        # Normalize Text
        df_groups['group_type'] = df_groups['group_type'].fillna('UNKNOWN').str.upper()
        df_groups['text'] = (df_groups['title'].fillna('') + " " + df_groups['description'].fillna('')).str.lower()

        # Logic: Check Group Type OR Free Text
        is_placebo = (
            df_groups['group_type'].str.contains('PLACEBO') |
            df_groups['text'].str.contains(r'\b(?:placebo|sham|vehicle|sugar pill)\b', regex=True)
        )

        # Note: Catches "ACTIVE_COMPARATOR" (underscore) and "ACTIVE COMPARATOR" (space)
        is_active = (
            df_groups['group_type'].str.contains(r'ACTIVE[ _]COMPARATOR', regex=True) |
            df_groups['text'].str.contains(r'\b(?:standard of care|soc|active control|reference|marketed|standard therapy)\b', regex=True)
        )

        # Get IDs (Set Detection)
        placebo_ids = df_groups[is_placebo]['nct_id'].unique()
        active_ids = df_groups[is_active]['nct_id'].unique()

        # Map to Main DF
        df['has_placebo'] = df['nct_id'].isin(placebo_ids).astype(int)
        df['has_active_comparator'] = df['nct_id'].isin(active_ids).astype(int)

        return df

    def _engineer_protocol_duration(self, df):
        """
        Determines Operational Horizon (Duration in Months).
        Strategy: Waterfall (Outcome -> Title -> Summary).
        Logic: 'MAX' aggregation (Duration is determined by the longest endpoint).
        """
        print("    -> Engineering Protocol Duration (Waterfall)...")

        # 1. Load Sources
        cols = ['nct_id', 'outcome_type', 'time_frame']
        df_out = self._safe_load('outcomes.txt', cols=cols)

        # Prepare Primary Outcomes (Concatenate all primary timeframes)
        if not df_out.empty:
            primary = df_out[df_out['outcome_type'].str.lower().str.contains('primary', na=False)].copy()
            prim_agg = primary.groupby('nct_id')['time_frame'].apply(lambda x: " ".join(x.dropna().astype(str))).reset_index()
            prim_agg.rename(columns={'time_frame': 'txt_outcome'}, inplace=True)
        else:
            prim_agg = pd.DataFrame(columns=['nct_id', 'txt_outcome'])

        # Prepare Summaries
        df_sum = self._safe_load('brief_summaries.txt', cols=['nct_id', 'description'])
        if not df_sum.empty:
            df_sum.rename(columns={'description': 'txt_summary'}, inplace=True)

        # Merge Sources into a temporary DF
        df_dur = df[['nct_id', 'official_title']].copy()
        df_dur = df_dur.merge(prim_agg, on='nct_id', how='left')
        df_dur = df_dur.merge(df_sum, on='nct_id', how='left')

        # 2. The Extraction Logic (Helper Function)
        def extract_duration(text):
            if pd.isna(text) or len(str(text)) < 3: return np.nan
            text = str(text).lower()

            # A. PROXY: Long-Term Survival (Overrides everything)
            if re.search(r'\b(overall survival|os|pfs|time to progression|until progression)\b', text):
                return 24.0

            # B. CLEANUP
            text = text.replace('approx.', '').replace('approximately', '').strip()

            max_months = 0.0

            # C. REGEX ENGINES (Updated for abbreviations: wk, mth, yr)
            # We define the unit pattern once to ensure consistency
            # Matches: week, weeks, wk, wks, month, mth, year, yr, etc.
            unit_pattern = r'(week|wk|month|mth|year|yr|day|hour|hr)'

            # Pattern 1: "12-24 wks", "up to 12 yr" (Number ... Unit)
            p1 = re.findall(r'(\d+(?:\.\d+)?)\s*(?:-|to)?\s*(\d+(?:\.\d+)?)?\s*' + unit_pattern, text)

            # Pattern 2: "Wk 12", "Yr 2" (Unit ... Number)
            p2 = re.findall(unit_pattern + r's?\s+(\d+(?:\.\d+)?)', text)

            def to_months(val, unit):
                u = unit.lower()
                if 'year' in u or 'yr' in u: return val * 12
                if 'week' in u or 'wk' in u: return val / 4.33
                if 'day' in u: return val / 30.4
                if 'hour' in u or 'hr' in u: return val / 730.0
                return val # Default is Month

            for m in p1:
                v1 = float(m[0])
                v2 = float(m[1]) if m[1] else 0
                val = max(v1, v2)
                m_val = to_months(val, m[2])
                if m_val > max_months: max_months = m_val

            for m in p2:
                m_val = to_months(float(m[1]), m[0])
                if m_val > max_months: max_months = m_val

            # D. LOGIC CHECK
            # If we found a meaningful duration (> 0.5 months), return it.
            if max_months > 0.5:
                return min(max_months, 120.0) # Cap at 10 years

            # E. INSTANT FALLBACK
            # Only if NO duration was found, check for "Baseline/Day 1"
            if re.search(r'\b(single[ -]dose|one[ -]time|baseline|day 0|day 1|pre-dose)\b', text):
                return 0.1

            return max_months if max_months > 0 else np.nan

        # 3. Apply Waterfall Strategy
        def resolve(row):
            # Try Outcome first
            d = extract_duration(row['txt_outcome'])
            if pd.notna(d): return d
            # Try Title second
            d = extract_duration(row['official_title'])
            if pd.notna(d): return d
            # Try Summary last
            d = extract_duration(row['txt_summary'])
            if pd.notna(d): return d
            return np.nan

        df_dur['duration_months'] = df_dur.apply(resolve, axis=1)

        # 4. Merge & Impute
        df = df.merge(df_dur[['nct_id', 'duration_months']], on='nct_id', how='left')

        # Impute missing with Median
        median_dur = df['duration_months'].median()
        if pd.isna(median_dur): median_dur = 12.0
        df['duration_months'] = df['duration_months'].fillna(median_dur)

        # Final Clip (Cap at 60 months)
        df['duration_months'] = df['duration_months'].clip(upper=60)

        return df

    def _engineer_smart_patterns(self, df):
        print("    -> Engineering Smart Patterns (Rigor & Strictness)...")

        # A. Design Rigor Index (Quality Gradient)
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

        # B. Eligibility Strictness (Narrowness)
        df['is_gender_restricted'] = df['gender'].apply(lambda x: 0 if str(x).lower() == 'all' else 1)
        # If healthy_volunteers is 0 (No), then it is a 'Sick Only' trial (1)
        df['is_sick_only'] = df['healthy_volunteers'].apply(lambda x: 1 if x == 0 else 0)

        for col in ['child', 'adult', 'older_adult']:
            if col in df.columns:
                # Handle boolean or string variations
                df[col] = df[col].astype(str).apply(lambda x: 1 if x.lower() in ['true', '1', 'yes'] else 0)
            else:
                df[col] = 1 # Assume included if missing

        df['eligibility_strictness_score'] = (
            df['is_gender_restricted'] +
            df['is_sick_only'] +
            (1 - df['child']) +
            (1 - df['older_adult'])
        )
        return df

    def _engineer_agent_type(self, df):
            print("    -> Engineering Agent Type (Prioritizing Active Molecules)...")

            if self.df_drugs.empty:
                df['agent_category'] = 'UNKNOWN'
                return df

            # 1. Prepare Intervention Data
            df_int = self.df_drugs.copy()
            df_int['name_clean'] = df_int['name'].str.lower().fillna('')


            # Priority 1: High-Tech / Targeted
            # Priority 2: Established / Generic
            # Priority 3: Placebo / Control

            # 2. Define Priority (Lower number = Higher Priority)
            priority_map = {
                # --- LEVEL 1: HIGH-TECH / TARGETED (Wins everything) ---
                'RNA_GENE_THERAPY': 1, 'CELL_THERAPY': 1, 'ANTIBODY_DRUG_CONJUGATE': 1,
                'BISPECIFIC_ANTIBODY': 1, 'MONOCLONAL_ANTIBODY': 1, 'GLP1_PEPTIDE': 1,
                'PI3K_INHIBITOR': 1, 'BTK_INHIBITOR': 1, 'JAK_INHIBITOR': 1,
                'PARP_INHIBITOR': 1, 'BCL2_INHIBITOR': 1, 'SGLT2_INHIBITOR': 1,
                'KINASE_INHIBITOR_TYROSINE': 1, 'TARGETED_KINASE_INHIBITOR': 1,

                # --- LEVEL 2: SPECIFIC ESTABLISHED (Wins over Generics) ---
                'ANTIVIRAL': 2,
                'ANTIBIOTIC': 2,
                'CORTICOSTEROID': 2,
                'NSAID_PAIN': 2,
                'DIABETES_ORAL': 2,
                'CHEMOTHERAPY': 2,
                'HORMONAL_THERAPY': 2,
                'STATIN_CHOLESTEROL': 2,
                'ENZYME_INHIBITOR': 2,

                # --- LEVEL 3: BROAD GENERICS (Only wins over Placebo) ---
                'BIOLOGIC_OTHER': 3,
                'SMALL_MOLECULE_GENERIC': 3,
                'SMALL_MOLECULE_OTHER': 3,

                # --- LEVEL 4: CONTROL (Loser) ---
                'PLACEBO_CTRL': 4
            }

            # 3. Define Patterns
            patterns = {
            # --- LEVEL 1: HIGH-TECH / ADVANCED THERAPIES ---
            'CELL_THERAPY': r'\b(car-t|chimeric antigen|autologous|allogeneic|t-cell|nk cell|stem cell|mesenchymal|t-lymphocytes|islet cell)\b|.*cel$',
            'RNA_GENE_THERAPY': r'\b(crispr|cas9|mrna|sirna|antisense|oligonucleotide|plasmid|vector|aav|rnai|gene transfer|gene therapy)\b',

            # --- LEVEL 2: COMPLEX BIOLOGICS ---
            'ANTIBODY_DRUG_CONJUGATE': r'\b(adc|conjugate)\b.*mab|mab.*\b(adc|conjugate)\b',
            'BISPECIFIC_ANTIBODY': r'\b(bispecific|dual-targeting|bi-specific|engager)\b',
            'MONOCLONAL_ANTIBODY': r'.*mab\b',
            'GLP1_PEPTIDE': r'.*(tide|glutide)\b', # Massive metabolic signal (Semaglutide, etc.)

            # --- LEVEL 3: TARGETED SMALL MOLECULES (SPECIFIC) ---
            'PI3K_INHIBITOR': r'.*lisib\b',
            'BTK_INHIBITOR': r'.*brutinib\b',
            'JAK_INHIBITOR': r'.*citinib\b',
            'PARP_INHIBITOR': r'.*parib\b',
            'BCL2_INHIBITOR': r'.*clax\b',
            'SGLT2_INHIBITOR': r'.*gliflozin\b', # Major cardio-renal signal
            'KINASE_INHIBITOR_TYROSINE': r'.*tinib\b',
            'TARGETED_KINASE_INHIBITOR': r'.*ib\b',

            # --- LEVEL 4: ESTABLISHED CLASSES ---
            'CHEMOTHERAPY': r'.*(platin|taxel|rubicin|fluorouracil|gemcitabine|cyclophosphamide|methotrexate|etoposide|vincristine|vinblastine|irinotecan|oxaliplatin|ifosfamide|pemetrexed)\b',
            'HORMONAL_THERAPY': r'\b(tamoxifen|anastrozole|letrozole|exemestane|fulvestrant|bicalutamide|enzalutamide|abiraterone)\b',
            'STATIN_CHOLESTEROL': r'.*vastatin\b',
            'ENZYME_INHIBITOR': r'.*stat\b',
            'ANTIVIRAL': r'.*(vir|virine|viro)\b',
            'ANTIBIOTIC': r'.*(cillin|mycin|micin|oxacin|cycline)\b',
            'NSAID_PAIN': r'.*(fenac|profen|coxib)\b',
            'CORTICOSTEROID': r'.*(sone|solone|nide)\b',
            'DIABETES_ORAL': r'.*(gliptin|glitazone|formin)\b',

            # --- LEVEL 5: GENERIC & PLACEBO ---
            'SMALL_MOLECULE_GENERIC': r'.*(ine|ide|one|ole|ate|ant|ent|ril|lol|tan|vir|micin|mycin|acin|xaban|stat|pril|sartan|prazole|nib|mab|cept|tide)\b',
            'PLACEBO_CTRL': r'\b(placebo|sham|control|comparator)\b'


            }


            def classify(row):
                name = row['name_clean']
                itype = str(row.get('intervention_type', '')).upper()

                # Check high-tech and established classes first
                for cat, pattern in patterns.items():
                    if cat == 'PLACEBO_CTRL': continue # Skip placebo for now
                    if re.search(pattern, name): return cat

                # Check Placebo only if no drug class matched
                if re.search(patterns['PLACEBO_CTRL'], name): return 'PLACEBO_CTRL'

                # Final Fallbacks
                return 'BIOLOGIC_OTHER' if itype == 'BIOLOGICAL' else 'SMALL_MOLECULE_OTHER'

            # 4. Apply classification and map priority
            df_int['agent_category'] = df_int.apply(classify, axis=1)
            df_int['priority'] = df_int['agent_category'].map(priority_map).fillna(3)

            # 5. SORT BY PRIORITY (1 wins over 3)
            # We sort by nct_id and then priority (ascending).
            # This puts Priority 1 at the top for each trial.
            best_agent = df_int.sort_values(['nct_id', 'priority']).drop_duplicates('nct_id')

            df = df.merge(best_agent[['nct_id', 'agent_category']], on='nct_id', how='left')
            df['agent_category'] = df['agent_category'].fillna('SMALL_MOLECULE_OTHER')

            return df

    def _engineer_safe_features(self, df):

        print("    -> Engineering Gated Protocol Features (FDA & DMC)...")

        # 1. Handle 'has_dmc' (Standard Cleaning)
        if 'has_dmc' in df.columns:
            df['has_dmc'] = df['has_dmc'].astype(str).apply(
                lambda x: 1 if x.lower() in ['true', 't', '1', 'yes'] else 0
            )
        else:
            df['has_dmc'] = 0

        # 2. Handle 'is_fda_regulated_drug' (Cleaning + Gating)
        if 'is_fda_regulated_drug' in df.columns:
            # Step A: Clean the raw signal into 1s and 0s
            raw_signal = df['is_fda_regulated_drug'].astype(str).apply(
                lambda x: 1 if x.lower() in ['true', 't', '1', 'yes'] else 0
            )

            # Step B: Apply the Gate
            # Only keep the '1' if the trial started in 2017 or later.
            # This removes the historical noise identified in the audit.
            df['is_fda_regulated_drug'] = (
                (raw_signal == 1) & (df['start_year'] >= 2017)
            ).astype(int)
        else:
            df['is_fda_regulated_drug'] = 0

        return df


    def _attach_medical_hierarchy(self, df):
        print("    -> Attaching Medical Hierarchy (Vectorized Optimization)...")

        # 1. Load the "Bridge" (nct_id -> mesh_term)
        cols_bridge = ['nct_id', 'mesh_term']
        df_bridge = self._safe_load('browse_conditions.txt', cols=cols_bridge)

        # Fallback to conditions.txt if browse_conditions is empty
        if df_bridge.empty:
            df_bridge = self._safe_load('conditions.txt', cols=['nct_id', 'name'])
            if not df_bridge.empty:
                df_bridge.rename(columns={'name': 'mesh_term'}, inplace=True)

        if df_bridge.empty:
            # Early exit if no data
            df['mesh_term'], df['therapeutic_area'] = np.nan, np.nan
            return df

        # 2. Load the "Dictionary" (mesh_term -> therapeutic_area)
        mesh_path = os.path.join(self.data_path, 'mesh_lookup.csv')
        df_dictionary = pd.DataFrame()

        if os.path.exists(mesh_path):
            try:
                df_dictionary = pd.read_csv(mesh_path, sep='|', on_bad_lines='skip')
                # Ensure we only keep relevant columns and drop duplicates to prevent merge explosion
                if 'mesh_term' in df_dictionary.columns and 'therapeutic_area' in df_dictionary.columns:
                    df_dictionary = df_dictionary[['mesh_term', 'therapeutic_area']].drop_duplicates()
            except Exception as e:
                print(f"   [!] Warning: Could not load mesh_lookup.csv. Error: {e}")

        # 3. Merge Bridge + Dictionary
        # We use a left merge. Rows without a match in df_dictionary get NaN automatically.
        if not df_dictionary.empty:
            df_full_mesh = df_bridge.merge(df_dictionary, on='mesh_term', how='left')
        else:
            df_full_mesh = df_bridge.copy()
            df_full_mesh['therapeutic_area'] = np.nan

        # --- OPTIMIZATION: Standardize "Bad" Values to NaN immediately ---
        bad_values = ['Unclassified', 'Other/Unclassified', 'Pathological Conditions', 'None', '', 'nan']
        df_full_mesh['therapeutic_area'] = df_full_mesh['therapeutic_area'].replace(bad_values, np.nan)

        # Ensure mesh_term is string for regex operations
        df_full_mesh['mesh_term'] = df_full_mesh['mesh_term'].astype(str)

        # 4. Expanded Regex Fallback Dictionary (Updated with Non-Capturing Groups (?:...))
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

        # 5. Apply Vectorized Regex (High Speed)
        print("       Applying Vectorized Regex (High Speed)...")

        for area, pattern in fallbacks.items():
            # 1. Identify rows that are STILL missing therapeutic_area
            missing_mask = df_full_mesh['therapeutic_area'].isna()

            # Optimization: If nothing is missing, stop the loop entirely
            if not missing_mask.any():
                break

            # 2. Find matches ONLY within the missing rows
            # na=False ensures NaN mesh_terms don't crash the regex
            matches = df_full_mesh.loc[missing_mask, 'mesh_term'].str.contains(pattern, case=False, regex=True, na=False)

            # 3. Update the main DataFrame using the index of the matches
            if matches.any():
                matched_indices = matches[matches].index
                df_full_mesh.loc[matched_indices, 'therapeutic_area'] = area

        # Fill remaining NaNs with 'Unclassified'
        df_full_mesh['therapeutic_area'] = df_full_mesh['therapeutic_area'].fillna('Unclassified')

        # --- STEP B: Determine Child (Vectorized) ---
        # Logic: If mesh_term is valid, use it. Else use therapeutic_area (parent).
        df_full_mesh['therapeutic_subgroup_name'] = df_full_mesh['mesh_term']

        # Where mesh_term is NaN or 'nan' string, fill with parent
        bad_mesh_mask = (df_full_mesh['therapeutic_subgroup_name'].isna()) | \
                        (df_full_mesh['therapeutic_subgroup_name'].astype(str).str.lower() == 'nan')

        df_full_mesh.loc[bad_mesh_mask, 'therapeutic_subgroup_name'] = df_full_mesh.loc[bad_mesh_mask, 'therapeutic_area']

        # 7. Flag "Good" Rows (Prioritize Classified over Unclassified)
        df_full_mesh['is_good'] = (df_full_mesh['therapeutic_area'] != 'Unclassified').astype(int)

        # 8. Sort: Put "Good" rows at the top for each ID
        df_full_mesh.sort_values(['nct_id', 'is_good'], ascending=[True, False], inplace=True)

        # 9. Fast Deduplication (Optimization over GroupBy)
        # Since we sorted, the first occurrence of nct_id is the "best" one.
        df_best_match = df_full_mesh.drop_duplicates(subset=['nct_id'], keep='first')

        # Select only necessary columns to merge back
        cols_to_merge = ['nct_id', 'mesh_term', 'therapeutic_area', 'therapeutic_subgroup_name']
        df = df.merge(df_best_match[cols_to_merge], on='nct_id', how='left')

        # 10. Final Unification Layer
        unification_map = {
            'Infections (Bacterial/Fungal)':   'Infections',
            'Digestive':                       'Gastrointestinal',
            'Ophthalmology (Eye)':             'Ophthalmology',
            'Urology (Male)':                  'Renal/Urology',
            'Congenital/Genetic':              'Genetic',
            'Stomatognathic':                  'Dental',
            'ENT (Ear/Nose/Throat)':           'Ear/Nose/Throat',
            'Wounds':                          'Musculoskeletal',
            'Pathological Conditions':         'Unclassified',
            'Other/Unclassified':              'Unclassified',
            'Chemically Induced':              'Unclassified',
            'Animal Diseases':                 'Unclassified'
        }

        df['therapeutic_area'] = df['therapeutic_area'].replace(unification_map)

        # --- FINAL SAFETY FILL: Vectorized Title Scan ---
        # If still Unclassified, try one last Regex on the TITLE
        mask_still_missing = (df['therapeutic_area'] == 'Unclassified') | (df['therapeutic_area'].isna())
        missing_count = mask_still_missing.sum()

        if missing_count > 0:
            print(f"       Running Final Title Scan on {missing_count} unclassified trials (Vectorized)...")

            # Ensure title is string
            df['official_title'] = df['official_title'].astype(str)

            for area, pattern in fallbacks.items():
                # Update the mask inside the loop: We only look at rows that are STILL unclassified
                current_missing_mask = (df['therapeutic_area'] == 'Unclassified') | (df['therapeutic_area'].isna())

                if not current_missing_mask.any():
                    break

                # Check title against regex
                matches = df.loc[current_missing_mask, 'official_title'].str.contains(pattern, case=False, regex=True, na=False)

                if matches.any():
                    matched_indices = matches[matches].index
                    df.loc[matched_indices, 'therapeutic_area'] = area

        # Final fill for anything that survived everything
        df['therapeutic_area'] = df['therapeutic_area'].fillna('Unclassified')
        df['therapeutic_subgroup_name'] = df['therapeutic_subgroup_name'].fillna('Unclassified')

        print("    -> Hierarchy Attachment Complete.")
        return df

    def _engineer_sponsor_features(self, df):
        print("    -> Engineering Sponsor Tiers...")
        cols_needed = ['nct_id', 'lead_or_collaborator', 'name', 'agency_class']
        df_sponsors = self._safe_load('sponsors.txt', cols=cols_needed)

        if df_sponsors.empty:
            df['sponsor_tier'] = 'TIER_2_OTHER'
            df['sponsor_clean'] = 'UNKNOWN'
            df['agency_class'] = 'UNKNOWN'
            return df

        leads = df_sponsors[df_sponsors['lead_or_collaborator'].str.lower() == 'lead'][['nct_id', 'name', 'agency_class']]
        leads = leads.rename(columns={'name': 'lead_sponsor'})
        leads = leads.drop_duplicates('nct_id')

        df = df.merge(leads, on='nct_id', how='left')
        df['lead_sponsor'] = df['lead_sponsor'].fillna('UNKNOWN')
        df['agency_class'] = df['agency_class'].fillna('UNKNOWN')

        # Clean Names
        clean_col = df['lead_sponsor'].astype(str).str.lower().str.strip()
        legal_pattern = r'[.,]|\binc\b|\bltd\b|\bllc\b|\bcorp\b|\bgmbh\b|\bsa\b|\bplc\b'
        clean_col = clean_col.str.replace(legal_pattern, '', regex=True).str.strip()

        # Map Big Pharma
        mappings = {
            'Pfizer': ['pfizer', 'wyeth', 'hospira'],
            'GSK': ['glaxo', 'gsk', 'smithkline'],
            'Novartis': ['novartis', 'sandoz'],
            'AstraZeneca': ['astrazeneca', 'medimmune'],
            'Merck': ['merck', 'msd'],
            'Roche': ['roche', 'genentech', 'hoffmann'],
            'Sanofi': ['sanofi', 'aventis', 'genzyme'],
            'J&J': ['johnson & johnson', 'janssen'],
            'Bayer': ['bayer', 'monsanto'],
            'Boehringer': ['boehringer'],
            'BMS': ['bristol-myers', 'squibb', 'celgene'],
            'Lilly': ['lilly'],
            'Abbott': ['abbott', 'abbvie'],
            'Amgen': ['amgen'],
            'Takeda': ['takeda', 'shire'],
            'Gilead': ['gilead'],
            'Novo Nordisk': ['novo nordisk'],
            'NIH': ['national cancer institute', 'nci', 'national institutes of health', 'nih']
        }

        final_names = clean_col.copy()
        for std, keys in mappings.items():
            pattern = '|'.join(keys)
            mask = clean_col.str.contains(pattern, case=False, regex=True)
            final_names.loc[mask] = std

        # --- ORIGINAL LOGIC: NO GROUPING ---
        # The 'final_names' variable now contains standardized names for Tier 1
        # and the cleaned original names for all others.
        # This is the logic that produced the highest Test AUC (0.6727).
        # -----------------------------------

        df['sponsor_clean'] = final_names

        def get_tier(name):
            if name in mappings.keys(): return 'TIER_1_GIANT'
            return 'TIER_2_OTHER'

        df['sponsor_tier'] = df['sponsor_clean'].apply(get_tier)
        return df

    def _engineer_complexity(self, df):
        print("    -> Engineering Protocol Complexity (Calculating Age Flags)...")

        # 1. CHANGED: Added 'minimum_age' and 'maximum_age' to the load list
        cols_needed = ['nct_id', 'criteria', 'gender', 'healthy_volunteers', 'minimum_age', 'maximum_age']
        df_elig = self._safe_load('eligibilities.txt', cols=cols_needed)

        if df_elig.empty:
            df['criteria_len_log'] = 0
            for c in ['gender', 'healthy_volunteers', 'adult', 'child', 'older_adult']: df[c] = 0
            return df

        df_elig = df_elig.drop_duplicates('nct_id')
        df = df.merge(df_elig, on='nct_id', how='left')

        # 2. Criteria Length
        df['criteria_len_log'] = np.log1p(df['criteria'].astype(str).str.len().fillna(0))

        # 3. Healthy Volunteers (Standardize to 1/0)
        # 1 = Accepts Healthy Volunteers, 0 = Does not
        df['healthy_volunteers'] = df['healthy_volunteers'].astype(str).str.lower().apply(
            lambda x: 0 if x in ['f', 'false', '0', 'no', 'nan', 'none'] else 1
        )

        # 4. Gender (Keep as string, it is categorical with 3+ values: ALL, MALE, FEMALE)
        df['gender'] = df['gender'].fillna('UNKNOWN').str.upper()

        # 5. AGE CALCULATION (THE FIX) -----------------------------------------
        # We parse "18 Years", "6 Months" -> Years (float) to fix the 0s bug
        def parse_age_to_years(val, default_val):
            if pd.isna(val) or str(val).lower() in ['n/a', 'nan', '', 'none']:
                return default_val
            try:
                # Extract the first number found
                match = re.search(r'(\d+(\.\d+)?)', str(val))
                if not match: return default_val
                num = float(match.group(1))

                # Normalize units to Years
                text = str(val).lower()
                if 'month' in text: num /= 12.0
                elif 'week' in text: num /= 52.0
                elif 'day' in text: num /= 365.0
                elif 'hour' in text: num /= 8760.0
                return num
            except:
                return default_val

        # Parse (Default Min = 0, Default Max = 100)
        df['min_age_years'] = df['minimum_age'].apply(lambda x: parse_age_to_years(x, 0.0))
        df['max_age_years'] = df['maximum_age'].apply(lambda x: parse_age_to_years(x, 100.0))

        # Generate Flags based on standard clinical definitions
        # Child: Can enroll < 18
        df['child'] = (df['min_age_years'] < 18).astype(int)

        # Adult: Can enroll 18-65 (Overlap check)
        df['adult'] = ((df['max_age_years'] >= 18) & (df['min_age_years'] < 65)).astype(int)

        # Older Adult: Can enroll > 65
        df['older_adult'] = (df['max_age_years'] > 65).astype(int)

        # Cleanup intermediate columns
        df.drop(columns=['minimum_age', 'maximum_age', 'min_age_years', 'max_age_years'], inplace=True, errors='ignore')
        # ----------------------------------------------------------------------

        return df

    def _calculate_competition(self, df):
        print("    -> Engineering Smart Competition (Disease & Molecule density)...")
        try:
            # 1. Ensure required columns exist
            req_cols = ['start_year', 'therapeutic_area', 'therapeutic_subgroup_name', 'agent_category']
            for col in req_cols:
                if col not in df.columns:
                    df[col] = 'UNKNOWN'

            # 2. Helper function to calculate rolling 2-year density
            def get_rolling_density(dataframe, group_col):
                # Group by year and the target column
                counts = dataframe.groupby(['start_year', group_col]).size().reset_index(name='yr_count')
                # Create a lookup dictionary: {(year, category): count}
                lookup = dict(zip(zip(counts['start_year'], counts[group_col]), counts['yr_count']))

                def calc(row):
                    val = row[group_col]

                    # --- THE GATE: PREVENT FAKE HIGH SCORES ---
                    # A. Check for generic/fallback strings
                    generics = ['UNKNOWN',
                                'Unclassified',
                                'SMALL_MOLECULE_OTHER',
                                'BIOLOGIC_OTHER',
                                'SMALL_MOLECULE_GENERIC', # <--- ADD THIS NEW CATEGORY
                                'PLACEBO_CTRL'
                                ]
                    if val in generics:
                        return 0

                    # B. Check for Identity Match (The Parent-Child Dilution)
                    # If the niche is just the broad area, it is not a specific niche.
                    if group_col == 'therapeutic_subgroup_name' and val == row['therapeutic_area']:
                        return 0

                    y = row['start_year']
                    # Sum current year + previous year to capture market "heat"
                    return lookup.get((y, val), 0) + lookup.get((y-1, val), 0)

                return dataframe.apply(calc, axis=1)

            # 3. Calculate the three levels of competition
            df['competition_broad'] = get_rolling_density(df, 'therapeutic_area')
            df['competition_niche'] = get_rolling_density(df, 'therapeutic_subgroup_name')
            df['competition_agent'] = get_rolling_density(df, 'agent_category')

        except Exception as e:
            print(f"       [!] Competition calculation failed: {e}")
            df['competition_broad'] = 0
            df['competition_niche'] = 0
            df['competition_agent'] = 0

        return df

    def _add_ui_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        UI-only fields:
        - remove struck-through CONTENT (not just tags)
        - remove HTML/format artifacts
        These columns are for display only and must not be used for training.
        """
        print("    -> Adding UI-friendly text fields (latest text only)...")
        df = df.copy()

        # Ensure the columns exist (defensive)
        # official_title already exists in df at this stage
        if "official_title" not in df.columns:
            df["official_title"] = ""

        # Defensive check for brief_title (in case load_and_clean didn't run properly)
        if "brief_title" not in df.columns:
            df["brief_title"] = df["official_title"]


        # Bring in summary text (UI needs it, but _prepare_text will also bring it later)
        df_summaries = self._safe_load("brief_summaries.txt", cols=["nct_id", "description"])
        if not df_summaries.empty:
            df = df.merge(
                df_summaries.rename(columns={"description": "ui_summary_raw"}),
                on="nct_id",
                how="left",
            )
        else:
            df["ui_summary_raw"] = ""

        # Criteria already exists because _engineer_complexity merged eligibilities.txt
        if "criteria" not in df.columns:
            df["criteria"] = ""

        # Fill NaNs then clean
        df["ui_title"] = df["official_title"].fillna("").astype(str).apply(ui_clean_text)
        df["ui_brief_title"] = df["brief_title"].fillna("").astype(str).apply(ui_clean_text) # <--- NEW
        df["ui_summary"] = df["ui_summary_raw"].fillna("").astype(str).apply(ui_clean_text)
        df["ui_criteria"] = df["criteria"].fillna("").astype(str).apply(ui_clean_text)

        # Optional: drop the temporary raw summary helper (keeps schema clean)
        df.drop(columns=["ui_summary_raw"], inplace=True, errors="ignore")

        return df


    def _prepare_text(self, df):

        print("    -> Engineering NLP Text Pillars (Scientific & Operational)...")

        # 1. Load Intermediate Data
        df_summaries = self._safe_load('brief_summaries.txt', cols=['nct_id', 'description'])
        df_outcomes = self._safe_load('design_outcomes.txt', cols=['nct_id', 'measure', 'outcome_type'])
        #df_keys = self._safe_load('keywords.txt', cols=['nct_id', 'name'])

        # 2. Merge Summary
        if not df_summaries.empty:
            df = df.merge(df_summaries.rename(columns={'description': 'temp_summary'}), on='nct_id', how='left')

        # 3. Merge Endpoints
        if not df_outcomes.empty:
            primaries = df_outcomes[df_outcomes['outcome_type'].str.lower().str.contains('primary', na=False)].copy()
            endpoints = primaries.groupby('nct_id')['measure'].apply(
                lambda x: " [SEP] ".join(x.dropna().astype(str))
            ).reset_index(name='txt_primary_endpoints')
            df = df.merge(endpoints, on='nct_id', how='left')


         # 4. CRITICAL: NaN-Proofing (Title and Summary have 100% coverage, but we stay defensive)
        df['official_title'] = df['official_title'].fillna("No title provided").astype(str)
        df['temp_summary'] = df.get('temp_summary', pd.Series([""]*len(df))).fillna("No summary provided").astype(str)
        df['txt_primary_endpoints'] = df.get('txt_primary_endpoints', pd.Series([""]*len(df))).fillna("No endpoints provided").astype(str)
        df['criteria'] = df['criteria'].fillna("No criteria provided").astype(str)


        # 5. Final NLP txt Construction

        # NLP field A: Scientific Intent (Title + Summary)
        # We use a simple join because we know both fields are now non-null strings
        df['txt_scientific_essence'] = (
            df['official_title'].str.strip() +
            " [SEP] " +
            df['temp_summary'].str.strip()
        )


        # NLP field B: Operational Rigor (Criteria)
        df['txt_criteria'] = df['criteria']


        # 6. Final Cleanup
        # We drop the raw columns and the temporary summary
        cols_to_drop = ['official_title', 'temp_summary', 'criteria', 'why_stopped']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')

        return df

    def _engineer_phase_groups(self, df):
        print("    -> Grouping Phases into Efficacy Tiers...")
        # Ensure phase strings are standardized for mapping
        df['phase'] = df['phase'].astype(str).str.upper().str.strip()

        phase_map = {
            'PHASE1/PHASE2': 'Early_Efficacy',
            'PHASE2': 'Early_Efficacy',
            'PHASE2/PHASE3': 'Confirmatory',
            'PHASE3': 'Confirmatory'
        }

        df['phase_group'] = df['phase'].map(phase_map).fillna('Confirmatory')
        return df

    def _attach_embeddings(self, df):
        print("    -> Attaching BioBERT Embeddings (Criteria, Scientific, Endpoints)...")

        # 1. Define File Paths
        # The "Key" file that guarantees row alignment
        path_key  = os.path.join(self.data_path, 'project_data_nlp_light.csv')

        # The Raw Embedding Matrices
        path_crit = os.path.join(self.data_path, 'biobert_crit_raw.npy')
        path_sci  = os.path.join(self.data_path, 'biobert_sci_raw.npy')
        path_endp = os.path.join(self.data_path, 'biobert_endp_raw.npy')

        # 2. Validation Checks
        if not os.path.exists(path_key):
            print("       [!] CRITICAL: 'project_data_nlp_light.csv' not found.")
            print("           Cannot align embeddings without this key file. Skipping.")
            return df

        if not all(os.path.exists(p) for p in [path_crit, path_sci, path_endp]):
            print("       [!] Warning: One or more .npy files are missing. Skipping embeddings.")
            return df

        try:
            # 3. Load the Key (IDs only)
            # We only need the nct_id column to label the rows
            df_key = pd.read_csv(path_key, usecols=['nct_id'], dtype=str)
            ids = df_key['nct_id'].values
            n_samples = len(ids)
            print(f"       Loaded Key: {n_samples} IDs from project_data_nlp_light.csv")

            # 4. Helper Function to Load & Label
            def load_and_merge(npy_path, prefix, target_df):
                # Load the raw matrix
                data = np.load(npy_path)

                # Safety Check: Row counts must match exactly
                if len(data) != n_samples:
                    print(f"       [!] Mismatch: {prefix} has {len(data)} rows, but Key has {n_samples}. Skipping.")
                    return target_df

                # Create labeled DataFrame
                # Columns: crit_0, crit_1 ... crit_767
                cols = [f"{prefix}_{i}" for i in range(data.shape[1])]
                df_emb = pd.DataFrame(data, columns=cols)
                df_emb['nct_id'] = ids  # <--- The Critical Step: Assigning IDs

                # Merge into main DF
                # We use 'left' join so we only keep embeddings for trials currently in our filtered df
                # We drop duplicates in case the ID file has them
                df_emb = df_emb.drop_duplicates('nct_id')
                return target_df.merge(df_emb, on='nct_id', how='left')

            # 5. Execute Merges
            df = load_and_merge(path_crit, "crit", df)
            df = load_and_merge(path_sci,  "sci",  df)
            df = load_and_merge(path_endp, "endp", df)

            # 6. Fill Missing
            # Trials in your main DF that were NOT in the NLP run (if any) get 0s
            # This is safe because we use Mean Imputation + PCA later
            emb_cols = [c for c in df.columns if c.startswith(('crit_', 'sci_', 'endp_'))]
            if emb_cols:
                print(f"       Successfully attached {len(emb_cols)} embedding dimensions.")

            return df

        except Exception as e:
            print(f"       [!] Error attaching embeddings: {e}")
            return df

    def _attach_p_values(self, df):
        print("    -> Attaching P-Values (Scientific Success Logic)...")

        # 1. Load Data
        cols_out = ['nct_id', 'id', 'outcome_type']
        df_out = self._safe_load('outcomes.txt', cols=cols_out)

        cols_ana = ['nct_id', 'outcome_id', 'p_value', 'p_value_modifier']
        df_ana = self._safe_load('outcome_analyses.txt', cols=cols_ana)

        if df_out.empty or df_ana.empty:
            df['min_p_value'] = np.nan
            df['scientific_success'] = 0
            return df

        # 2. Robust ID Cleaning (Fixes the Float/String mismatch)
        df_out['id'] = df_out['id'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_ana['outcome_id'] = df_ana['outcome_id'].astype(str).str.replace(r'\.0$', '', regex=True)

        # 3. Merge & Filter for PRIMARY Outcomes
        merged = df_ana.merge(df_out, left_on=['nct_id', 'outcome_id'], right_on=['nct_id', 'id'], how='inner')
        primary = merged[merged['outcome_type'].astype(str).str.lower().str.contains('primary', na=False)].copy()

        if primary.empty:
            df['min_p_value'] = np.nan
            df['scientific_success'] = 0
            return df

        # 4. Clean P-Values & Handle Modifiers
        # Replace commas for European formats
        primary['p_val_num'] = pd.to_numeric(primary['p_value'].astype(str).str.replace(',', '.'), errors='coerce')

        def adjust_p_value(row):
            val = row['p_val_num']
            mod = str(row['p_value_modifier']).strip()
            if pd.isna(val): return np.nan
            # Logic: < 0.05 is Significant (0.0499), > 0.05 is Not (0.0501)
            if '<' in mod: return val - 0.000001
            if '>' in mod: return val + 0.000001
            return val

        primary['adjusted_p'] = primary.apply(adjust_p_value, axis=1)

        # 5. Aggregation: Best Result per Trial (Minimum P-Value)
        trial_stats = primary.groupby('nct_id')['adjusted_p'].min().reset_index()
        trial_stats.rename(columns={'adjusted_p': 'min_p_value'}, inplace=True)

        # 6. Merge back to main DF
        df = df.merge(trial_stats, on='nct_id', how='left')

        # 7. Create Success Flag (Strict 0.05 cutoff)
        df['scientific_success'] = df['min_p_value'].apply(lambda x: 1 if pd.notna(x) and x <= 0.05 else 0)

        # --- AUDIT PRINTS (Requested) ---
        n_total = len(df)
        n_with_p = df['min_p_value'].notna().sum()
        n_success = df['scientific_success'].sum()

        print(f"       [Audit] Trials with P-values: {n_with_p} ({n_with_p/n_total:.1%})")
        print(f"       [Audit] Scientific Successes (p<=0.05): {n_success} ({n_success/n_total:.1%})")

        # Check distribution by Target (Completed vs Terminated)
        if 'target' in df.columns:
            print("       [Audit] P-Value Availability by Status:")
            stats = df.groupby('target')['min_p_value'].count()
            print(f"          - Completed (0): {stats.get(0, 0)} found")
            print(f"          - Terminated (1): {stats.get(1, 0)} found (Expect low)")

        return df

    def _merge_file(self, df, filename, cols, filter_col=None, filter_val=None):
        try:
            aux = self._safe_load(filename, cols=cols + ([filter_col] if filter_col else []))
            if aux.empty: return df
            if filter_col:
                aux = aux[aux[filter_col] == filter_val].drop(columns=[filter_col])
            aux = aux.drop_duplicates('nct_id')
            return df.merge(aux, on='nct_id', how='left')
        except:
            return df

    def check_and_export_nlp(self, df):
        """
        Smart Export Logic with "Day Zero" Sanitization:
        1. If ALL .npy files exist -> Locked (Training Mode).
        2. If ANY .npy file is missing -> Apply Anti-Leakage Cleaning -> Save CSV (Prep Mode).
        """
        print(">>> 3. Checking NLP Status...")

        # Define Paths
        path_key  = os.path.join(self.data_path, 'project_data_nlp_light.csv')

        # Define all 3 embedding paths
        path_crit = os.path.join(self.data_path, 'biobert_crit_raw.npy')
        path_sci  = os.path.join(self.data_path, 'biobert_sci_raw.npy')
        path_endp = os.path.join(self.data_path, 'biobert_endp_raw.npy')

        # --- SCENARIO 1: TRAINING MODE (All Embeddings Exist) ---
        # We strictly require ALL three to be present to consider it "Locked"
        if all(os.path.exists(p) for p in [path_crit, path_sci, path_endp]):
            print("    [Locked] All embedding files found.")
            print("    -> Saving final feature store with embeddings.")

            # ✅ SAVE FULL FEATURE STORE ONLY NOW
            self.save(df, filename="project_data.csv")

            print("    -> Feature store locked. Ready for training.")
            return

        # --- SCENARIO 2: PREP MODE (Generate Input for Colab) ---
        print("    [Prep Mode] One or more embeddings missing. Starting Sanitization Pipeline...")


        # A) Sanity check: ensure the imported cleaner supports return_stats
        import inspect

        params = inspect.signature(day_zero_reconstructor).parameters
        if "return_stats" not in params:
            raise RuntimeError(
                "day_zero_reconstructor does not support return_stats. "
                f"Loaded from: {inspect.getsourcefile(day_zero_reconstructor)} "
                f"with signature: {inspect.signature(day_zero_reconstructor)}"
            )


        # B. Define Columns to Export
        txt_cols = ['txt_scientific_essence', 'txt_criteria', 'txt_primary_endpoints']
        cols_to_export = ['nct_id', 'target'] + [c for c in txt_cols if c in df.columns]
        df_export = df[cols_to_export].copy()

        # C. Apply Sanitization (The Anti-Leakage Step)
        print("    [Sanitizing] Removing dates and leakage from text...")

        criteria_stats_rows = None  # will become a list of dicts only if txt_criteria exists

        for col in txt_cols:
            if col not in df_export.columns:
                continue

            label = col.replace('txt_', '')
            series = df_export[col].astype(str)

            # For criteria, also collect cleanup stats in the same pass
            if col == "txt_criteria":
                results = series.apply(lambda x: day_zero_reconstructor(x, label, return_stats=True))
                df_export[col] = results.apply(lambda t: t[0])

                stats_objs = results.apply(lambda t: t[1])
                criteria_stats_rows = [
                    {
                        "nct_id": nct_id,
                        "original_len": st.original_len,
                        "cleaned_len": st.cleaned_len,
                        "removed_chars": st.removed_chars,
                        "removed_admin_spans": st.removed_admin_spans,
                        "removed_struck_blocks": st.removed_struck_blocks,
                    }
                    for nct_id, st in zip(df_export["nct_id"].astype(str), stats_objs)
                ]
            else:
                df_export[col] = series.apply(lambda x: day_zero_reconstructor(x, label))


        # D) Save sanitized NLP CSV (this is the KEY FILE for embeddings)
        df_export.to_csv(path_key, index=False)
        print(f"    [Success] Sanitized & Saved {len(df_export)} rows to: {path_key}")
        print("    -> ACTION REQUIRED: Upload this file to Colab and run the Embedding Script.")

        # E) Save criteria cleanup stats (AUDIT ONLY)
        if criteria_stats_rows is not None:
            stats_path = os.path.join(self.data_path, "criteria_cleaning_stats.csv")
            pd.DataFrame(criteria_stats_rows).to_csv(stats_path, index=False)
            print(f"    [Audit] Saved criteria cleanup stats to: {stats_path}")

    def save(self, df, filename='project_data.csv'):
        out_path = os.path.join(self.data_path, filename)
        df.to_csv(out_path, index=False)
        print(f">>> Saved {len(df)} rows to {out_path}")
