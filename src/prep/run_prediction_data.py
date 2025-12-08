import pandas as pd
import numpy as np
import os
import csv
import re

class ClinicalTrialLoader:
    """
    A specialized loader for generating the PROSPECTIVE prediction dataset (2024-2025).
    It ensures feature consistency (especially for time-sensitive features like competition)
    and excludes all post-hoc/leakage features.
    """
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

    # --- STEP 1: LOAD FULL COHORT FOR TIME-SENSITIVE FEATURES ---

    def load_full_cohort_for_competition(self, min_year=2005, max_year=2025):
        """
        Loads the entire relevant dataset (2005-2025) to calculate time-sensitive
        features (Competition) on the complete historical context before splitting.
        """
        print(f">>> 1. Loading Full Cohort ({min_year}-{max_year}) for Competition Calculation...")

        # 1. Load Core Columns
        cols_studies = ['nct_id', 'overall_status', 'study_type', 'phase',
                        'start_date', 'number_of_arms',
                        'official_title', 'why_stopped',
                        'has_dmc', 'is_fda_regulated_drug', 'brief_title']

        df = self._safe_load('studies.txt', cols=cols_studies)
        if df.empty:
            raise ValueError("Critical Error: 'studies.txt' failed to load.")

        # 2. Apply Core Filters (Interventional, Drug, Phase 2/3)
        df = df[df['study_type'].str.upper() == 'INTERVENTIONAL'].copy()

        cols_int = ['nct_id', 'intervention_type', 'name']
        df_int = self._safe_load('interventions.txt', cols=cols_int)
        if not df_int.empty:
            target_types = ['DRUG', 'BIOLOGICAL', 'GENETIC']
            drug_ids = df_int[df_int['intervention_type'].str.upper().isin(target_types)]['nct_id'].unique()
            df = df[df['nct_id'].isin(drug_ids)]
            self.df_drugs = df_int[df_int['intervention_type'].str.upper().isin(target_types + ['DIETARY SUPPLEMENT', 'OTHER'])].copy()

        excluded_phases = ['EARLY_PHASE1', 'PHASE1', 'PHASE4', 'NA']
        df = df[~df['phase'].astype(str).str.upper().isin(excluded_phases)]
        df = df.dropna(subset=['phase'])

        # 3. Apply Full Date Range Filter (2005-2025)
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['start_year'] = df['start_date'].dt.year
        df = df[df['start_year'].between(min_year, max_year)].copy()

        # 4. Calculate Time-Sensitive Features (Hierarchy and Competition)
        df = self._attach_medical_hierarchy(df)
        df = self._calculate_competition(df)

        print(f"    Full Cohort Size: {len(df)} trials.")
        return df.copy()


    # --- STEP 2: FILTER AND FINALIZE PREDICTION SET ---

    def load_and_clean_prediction_set(self, df_full_cohort, start_year_predict=2024):
        """
        Filters the full cohort for the Prediction Set: Active/Pending Statuses, Year >= 2024.
        """
        df = df_full_cohort.copy()
        print(f">>> 2. Filtering for Prediction Set ({start_year_predict}-2025, Active/Pending Statuses)...")

        # 1. Filter: Date Range (2024-2025)
        df = df[df['start_year'] >= start_year_predict].copy()

        # 2. Filter: Status (Active/Pending Only)
        # We only want trials that are still running or about to start.
        ACTIVE_STATUSES = [
            'RECRUITING', 'NOT_YET_RECRUITING', 'ACTIVE_NOT_RECRUITING',
            'ENROLLING_BY_INVITATION'
        ]

        df_upper = df['overall_status'].astype(str).str.upper()
        df = df[df_upper.isin(ACTIVE_STATUSES)].copy()

        # 3. Filter: COVID Sanitizer (Kept for consistency, though few will remain)
        if 'why_stopped' in df.columns:
            covid_keywords = ['covid', 'pandemic', 'coronavirus', 'sars-cov-2', 'logistical reasons']
            mask_covid = df['why_stopped'].fillna('').astype(str).str.lower().apply(
                lambda x: any(k in x for k in covid_keywords)
            )
            if mask_covid.sum() > 0:
                print(f"    [Sanitizer] Dropping {mask_covid.sum()} trials terminated due to COVID/Logistics.")
                df = df[~mask_covid]

        # 4. Target: The prediction set has NO target. (Ensured by not creating it)
        if 'target' in df.columns:
             df.drop(columns=['target'], inplace=True)

        print(f"    Prediction Cohort: {len(df)} trials (Active/Pending Phase 2/3, {start_year_predict}-2025)")

        # 5. Add remaining features and user-facing text
        df = self.add_features_for_prediction(df)

        return df.copy()

    # --- NEW PREDICTION FEATURE FUNCTION (STEP 3) ---

    def add_features_for_prediction(self, df):
        """
        Adds remaining features (non-time-sensitive) and user-facing fields.
        Competition and Hierarchy features are assumed to be pre-calculated.
        """
        df = df.copy()
        print(">>> 3. Engineering Remaining Features for Prediction...")

        # 1. Operational Flags
        df['covid_exposure'] = df['start_year'].between(2019, 2022).astype(int)

        # 2. Geography (SAFE: Only is_us)
        df_countries = self._safe_load('countries.txt', cols=['nct_id', 'name'])
        if not df_countries.empty:
            us_trials = df_countries[df_countries['name'] == 'United States']['nct_id'].unique()
            df['includes_us'] = df['nct_id'].isin(us_trials).astype(int)
        else:
            df['includes_us'] = 0

        # 3. Merge Standard Metadata
        df = self._merge_file(df, 'designs.txt', ['nct_id', 'allocation', 'intervention_model', 'masking', 'primary_purpose'])
        df = self._merge_file(df, 'calculated_values.txt', ['nct_id', 'number_of_primary_outcomes_to_measure'])

        # 4. Sponsor Engineering
        df = self._engineer_sponsor_features(df)

        # 5. Complexity Engineering (Loads eligibility.txt and 'criteria' column)
        df = self._engineer_complexity(df)

        # 6. Agent Type
        df = self._engineer_agent_type(df)

        # 7. Smart Patterns (Rigor & Strictness)
        df = self._engineer_smart_patterns(df)

        # 8. Safe Features (DMC)
        df = self._engineer_safe_features(df)

        # 9. Text Features (Creates txt_tags, txt_criteria)
        df = self._prepare_text(df)

        # 10. ATTACH EMBEDDINGS (TEMPORARILY SKIPPED FOR COLLEAGUE'S WORKFLOW)
        # df = self._attach_embeddings(df)
        # Once 'embeddings_with_nctid.csv' is generated, uncomment the line above
        # and re-run this pipeline to include the features for model scoring.
        print("    -> Skipping embedding attachment for parallel workflow.")

        # 11. Attach User-Facing Text Fields
        df = self._load_user_facing_text(df)

        # Cleanup
        if 'number_of_primary_outcomes_to_measure' in df.columns:
            df.rename(columns={'number_of_primary_outcomes_to_measure': 'num_primary_endpoints'}, inplace=True)
        df['num_primary_endpoints'] = pd.to_numeric(df.get('number_of_primary_outcomes_to_measure', 1), errors='coerce').fillna(1)

        return df

    # --- NEW HELPER FUNCTION FOR UI/USER-FACING DATA ---

    def _load_user_facing_text(self, df):
        """
        Loads text fields that are useful for the UI but not necessarily for the model.
        """
        print("    -> Loading User-Facing Text (Brief Summary, Detailed Description)...")

        # 1. Brief Summary
        cols_brief = ['nct_id', 'description']
        df_brief = self._safe_load('brief_summaries.txt', cols=cols_brief)
        if not df_brief.empty:
            df_brief = df_brief.rename(columns={'description': 'brief_summary_text'})
            df = df.merge(df_brief.drop_duplicates('nct_id'), on='nct_id', how='left')
        else:
            df['brief_summary_text'] = ""

        # 2. Detailed Description
        cols_detailed = ['nct_id', 'description']
        df_detailed = self._safe_load('detailed_descriptions.txt', cols=cols_detailed)
        if not df_detailed.empty:
            df_detailed = df_detailed.rename(columns={'description': 'detailed_description_text'})
            df = df.merge(df_detailed.drop_duplicates('nct_id'), on='nct_id', how='left')
        else:
            df['detailed_description_text'] = ""

        return df

    # --- ALL OTHER NECESSARY HELPER METHODS (KEPT FOR FUNCTIONALITY) ---

    def _engineer_smart_patterns(self, df):
        print("    -> Engineering Smart Patterns (Rigor & Strictness)...")
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
        df['is_sick_only'] = df['healthy_volunteers'].apply(lambda x: 1 if str(x).lower() == 'no' else 0)
        for col in ['child', 'adult', 'older_adult']:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: 1 if x.lower() in ['true', '1', 'yes'] else 0)
            else:
                df[col] = 1
        df['eligibility_strictness_score'] = (
            df['is_gender_restricted'] +
            df['is_sick_only'] +
            (1 - df['child']) +
            (1 - df['older_adult'])
        )
        return df

    def _engineer_agent_type(self, df):
        print("    -> Engineering Agent Type (Bulletproof Classifier)...")
        if self.df_drugs.empty:
            df['agent_category'] = 'UNKNOWN'
            return df
        type_priority = {'GENETIC': 1,'BIOLOGICAL': 2,'DRUG': 3,'DIETARY SUPPLEMENT': 4,'OTHER': 5}
        df_int = self.df_drugs.copy()
        df_int['type_upper'] = df_int['intervention_type'].str.upper()
        df_int['priority'] = df_int['type_upper'].map(lambda x: type_priority.get(x, 5))
        df_int = df_int.sort_values('priority')
        best_types = df_int.drop_duplicates('nct_id')[['nct_id', 'type_upper', 'name']]
        df = df.merge(best_types, on='nct_id', how='left')
        def classify_molecule(row):
            itype = str(row['type_upper'])
            name = str(row['name']).lower()
            if 'placebo' in name: return 'PLACEBO_CTRL'
            if itype == 'GENETIC': return 'GENE_THERAPY'
            if any(x in name for x in ['car-t', 'chimeric antigen', 'autologous', 'allogeneic', 't-cell', 'nk cell']):
                return 'CELL_THERAPY'
            if name.endswith('cel'): return 'CELL_THERAPY'
            if any(x in name for x in ['crispr', 'cas9', 'mrna', 'sirna', 'antisense', 'oligonucleotide', 'plasmid', 'vector', 'aav']):
                return 'RNA_GENE_THERAPY'
            if itype == 'BIOLOGICAL': return 'BIOLOGIC'
            if 'mab' in name:
                if 'adc' in name or 'conjugate' in name: return 'ANTIBODY_DRUG_CONJUGATE'
                return 'MONOCLONAL_ANTIBODY'
            if name.endswith('cept'): return 'BIOLOGIC_FUSION'
            if 'vaccine' in name: return 'VACCINE'
            if any(x in name for x in ['interferon', 'interleukin', 'cytokine']): return 'IMMUNOTHERAPY'
            if name.endswith('ib') or 'ib ' in name:
                if 'tinib' in name: return 'KINASE_INHIBITOR_TYROSINE'
                if 'parib' in name: return 'PARP_INHIBITOR'
                if 'lisib' in name: return 'PI3K_INHIBITOR'
                return 'TARGETED_KINASE_INHIBITOR'
            if 'vastatin' in name: return 'STATIN_CHOLESTEROL'
            if name.endswith('stat') or 'stat ' in name: return 'ENZYME_INHIBITOR'
            if name.endswith('degib'): return 'HEDGEHOG_INHIBITOR'
            if name.endswith('clax'): return 'BCL2_INHIBITOR'
            chemo_stems = ['platin', 'taxel', 'rubicin', 'fluorouracil', 'gemcitabine',
                           'cyclophosphamide', 'methotrexate', 'etoposide', 'vincristine', 'vinblastine']
            if any(x in name for x in chemo_stems):
                return 'CHEMOTHERAPY'
            return 'SMALL_MOLECULE_OTHER'
        df['agent_category'] = df.apply(classify_molecule, axis=1)
        df.drop(columns=['type_upper', 'priority', 'name_y'], inplace=True, errors='ignore')
        if 'name_x' in df.columns: df.rename(columns={'name_x': 'official_title'}, inplace=True)
        return df

    def _engineer_safe_features(self, df):
        print("    -> Engineering Safe Protocol Features...")
        for col in ['has_dmc', 'is_fda_regulated_drug']:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda x: 1 if x.lower() in ['true', 't', '1', 'yes'] else 0
                )
            else:
                df[col] = 0
        return df

    def _attach_medical_hierarchy(self, df):
        print("    -> Attaching Medical Hierarchy (Bridge: nct_id -> mesh_term -> area)...")
        cols_bridge = ['nct_id', 'mesh_term']
        df_bridge = self._safe_load('browse_conditions.txt', cols=cols_bridge)
        if df_bridge.empty:
            cols_cond = ['nct_id', 'name']
            df_bridge = self._safe_load('conditions.txt', cols=cols_cond)
            if not df_bridge.empty:
                df_bridge.rename(columns={'name': 'mesh_term'}, inplace=True)
        mesh_path = os.path.join(self.data_path, 'mesh_lookup.csv')
        df_dictionary = pd.DataFrame()
        if os.path.exists(mesh_path):
            try:
                df_dictionary = pd.read_csv(mesh_path, sep='|', on_bad_lines='skip')
                if 'mesh_term' in df_dictionary.columns and 'therapeutic_area' in df_dictionary.columns:
                    df_dictionary = df_dictionary[['mesh_term', 'therapeutic_area']].drop_duplicates()
                    df_dictionary.rename(columns={'therapeutic_area': 'lookup_area'}, inplace=True)
                else:
                    df_dictionary = pd.DataFrame()
            except Exception as e:
                print(f"       [!] Error reading mesh_lookup.csv: {e}")
        if not df_bridge.empty:
            if not df_dictionary.empty:
                df_full_mesh = df_bridge.merge(df_dictionary, on='mesh_term', how='left')
            else:
                df_full_mesh = df_bridge
                df_full_mesh['lookup_area'] = np.nan
            df_grouped = df_full_mesh.groupby('nct_id').agg({
                'mesh_term': 'first',
                'lookup_area': 'first'
            }).reset_index()
            df = df.merge(df_grouped, on='nct_id', how='left')
        else:
            df['mesh_term'] = np.nan
            df['lookup_area'] = np.nan
        smart_path = os.path.join(self.data_path, 'smart_pathology_lookup.csv')
        if os.path.exists(smart_path):
            try:
                df_smart = pd.read_csv(smart_path)
                if 'nct_id' in df_smart.columns:
                    df = df.merge(df_smart, on='nct_id', how='left')
            except:
                pass
        if 'best_pathology' not in df.columns: df['best_pathology'] = np.nan
        if 'therapeutic_area' not in df.columns: df['therapeutic_area'] = np.nan
        def get_final_category(row):
            if pd.notna(row.get('best_pathology')) and str(row.get('best_pathology')) != 'Unknown':
                return row['best_pathology']
            if pd.notna(row.get('lookup_area')) and str(row.get('lookup_area')) != 'Other/Unclassified':
                return row['lookup_area']
            if pd.notna(row.get('mesh_term')):
                return row['mesh_term']
            if pd.notna(row.get('therapeutic_area')) and str(row.get('therapeutic_area')) != 'Other':
                return row['therapeutic_area']
            return 'Unclassified Condition'
        df['therapeutic_subgroup_name'] = df.apply(get_final_category, axis=1)
        df.drop(columns=['mesh_term', 'lookup_area'], inplace=True, errors='ignore')
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
        clean_col = df['lead_sponsor'].astype(str).str.lower().str.strip()
        legal_pattern = r'[.,]|\binc\b|\bltd\b|\bllc\b|\bcorp\b|\bgmbh\b|\bsa\b|\bplc\b'
        clean_col = clean_col.str.replace(legal_pattern, '', regex=True).str.strip()
        mappings = {
            'Pfizer': ['pfizer', 'wyeth', 'hospira'], 'GSK': ['glaxo', 'gsk', 'smithkline'],
            'Novartis': ['novartis', 'sandoz'], 'AstraZeneca': ['astrazeneca', 'medimmune'],
            'Merck': ['merck', 'msd'], 'Roche': ['roche', 'genentech', 'hoffmann'],
            'Sanofi': ['sanofi', 'aventis', 'genzyme'], 'J&J': ['johnson & johnson', 'janssen'],
            'Bayer': ['bayer', 'monsanto'], 'Boehringer': ['boehringer'],
            'BMS': ['bristol-myers', 'squibb', 'celgene'], 'Lilly': ['lilly'],
            'Abbott': ['abbott', 'abbvie'], 'Amgen': ['amgen'],
            'Takeda': ['takeda', 'shire'], 'Gilead': ['gilead'],
            'Novo Nordisk': ['novo nordisk'], 'NIH': ['national cancer institute', 'nci', 'national institutes of health', 'nih']
        }
        final_names = clean_col.copy()
        for std, keys in mappings.items():
            pattern = '|'.join(keys)
            mask = clean_col.str.contains(pattern, case=False, regex=True)
            final_names.loc[mask] = std
        df['sponsor_clean'] = final_names
        def get_tier(name):
            if name in mappings.keys(): return 'TIER_1_GIANT'
            return 'TIER_2_OTHER'
        df['sponsor_tier'] = df['sponsor_clean'].apply(get_tier)
        return df

    def _engineer_complexity(self, df):
        print("    -> Engineering Protocol Complexity (Calculating Age Flags)...")
        cols_needed = ['nct_id', 'criteria', 'gender', 'healthy_volunteers', 'minimum_age', 'maximum_age']
        df_elig = self._safe_load('eligibilities.txt', cols=cols_needed)
        if df_elig.empty:
            df['criteria_len_log'] = 0
            for c in ['gender', 'healthy_volunteers', 'adult', 'child', 'older_adult']: df[c] = 0
            return df
        df_elig = df_elig.drop_duplicates('nct_id')
        df = df.merge(df_elig, on='nct_id', how='left')
        df['criteria_len_log'] = np.log1p(df['criteria'].astype(str).str.len().fillna(0))
        df['healthy_volunteers'] = df['healthy_volunteers'].astype(str).str.lower().apply(
            lambda x: 'no' if x in ['f', 'false', '0', 'no', 'nan'] else 'yes'
        )
        df['gender'] = df['gender'].fillna('UNKNOWN')
        def parse_age_to_years(val, default_val):
            if pd.isna(val) or str(val).lower() in ['n/a', 'nan', '', 'none']: return default_val
            try:
                match = re.search(r'(\d+(\.\d+)?)', str(val))
                if not match: return default_val
                num = float(match.group(1))
                text = str(val).lower()
                if 'month' in text: num /= 12.0
                elif 'week' in text: num /= 52.0
                elif 'day' in text: num /= 365.0
                elif 'hour' in text: num /= 8760.0
                return num
            except:
                return default_val
        df['min_age_years'] = df['minimum_age'].apply(lambda x: parse_age_to_years(x, 0.0))
        df['max_age_years'] = df['maximum_age'].apply(lambda x: parse_age_to_years(x, 100.0))
        df['child'] = (df['min_age_years'] < 18).astype(int)
        df['adult'] = ((df['max_age_years'] >= 18) & (df['min_age_years'] < 65)).astype(int)
        df['older_adult'] = (df['max_age_years'] > 65).astype(int)
        df.drop(columns=['minimum_age', 'maximum_age', 'min_age_years', 'max_age_years'], inplace=True, errors='ignore')
        return df

    def _calculate_competition(self, df):
        print("    -> Calculating Competition...")
        try:
            req_cols = ['start_year', 'therapeutic_area', 'phase']
            if not all(col in df.columns for col in req_cols):
                df['competition_broad'] = 0
                df['competition_niche'] = 0
                return df
            grid = df.groupby(['start_year', 'therapeutic_area']).size().reset_index(name='count')
            lookup = dict(zip(zip(grid['start_year'], grid['therapeutic_area']), grid['count']))
            def get_comp(row):
                y, area = row['start_year'], row['therapeutic_area']
                return lookup.get((y, area), 0) + lookup.get((y-1, area), 0)
            df['competition_broad'] = df.apply(get_comp, axis=1)
            if 'therapeutic_subgroup_name' in df.columns:
                grid_niche = df.groupby(['start_year', 'therapeutic_subgroup_name']).size().reset_index(name='count')
                lookup_niche = dict(zip(zip(grid_niche['start_year'], grid_niche['therapeutic_subgroup_name']), grid_niche['count']))
                def get_niche(row):
                    y, sub = row['start_year'], row['therapeutic_subgroup_name']
                    return lookup_niche.get((y, sub), 0) + lookup_niche.get((y-1, sub), 0)
                df['competition_niche'] = df.apply(get_niche, axis=1)
            else:
                df['competition_niche'] = 0
        except:
            df['competition_broad'] = 0
            df['competition_niche'] = 0
        return df

    def _prepare_text(self, df):
        print("    -> Preparing Text Features...")
        df_keys = self._safe_load('keywords.txt', cols=['nct_id', 'name'])
        if not df_keys.empty:
            keys_grouped = df_keys.groupby('nct_id')['name'].apply(lambda x: " ".join(x.dropna().astype(str))).reset_index(name='txt_keywords')
            df = df.merge(keys_grouped, on='nct_id', how='left')
        else:
            df['txt_keywords'] = ""
        if not self.df_drugs.empty:
            int_names = self.df_drugs.groupby('nct_id')['name'].apply(lambda x: " ".join(x.dropna().astype(str))).reset_index(name='txt_int_names')
            df = df.merge(int_names, on='nct_id', how='left')
        else:
            df['txt_int_names'] = ""
        text_cols = ['official_title', 'txt_keywords', 'txt_int_names', 'criteria']
        for c in text_cols:
            if c in df.columns: df[c] = df[c].fillna("")
        df['txt_tags'] = (df['official_title'] + " " + df['txt_keywords'] + " " + df['txt_int_names'])
        if 'criteria' in df.columns:
            df['txt_criteria'] = df['criteria']
            # Keep official_title and criteria/txt_criteria for UI/Embedding
            df.drop(columns=['txt_keywords', 'txt_int_names'], inplace=True, errors='ignore')
        return df

    # NOTE: _attach_embeddings is now a placeholder function
    def _attach_embeddings(self, df):
        print("    -> Skipping embedding attachment for parallel workflow.")
        return df

    # NOTE: _attach_p_values is ONLY called in the original training pipeline (not in prediction)
    def _attach_p_values(self, df):
        print("    -> Attaching P-Values (Scientific Success Logic)...")
        # ... (implementation remains the same, but this function is not called in the prediction pipeline)
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

    def save(self, df, filename='data_predict.csv'):
        out_path = os.path.join(self.data_path, filename)
        df.to_csv(out_path, index=False)
        print(f">>> Saved {len(df)} rows to {out_path}")
