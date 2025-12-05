import pandas as pd
import numpy as np
import os
import csv
import re

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

    def load_and_clean(self):
        print(">>> 1. Loading Studies & Applying Filters...")

        cols_studies = ['nct_id', 'overall_status', 'study_type', 'phase',
                        'start_date', 'start_date_type', 'number_of_arms',
                        'official_title', 'why_stopped']

        df = self._safe_load('studies.txt', cols=cols_studies)

        if df.empty:
            raise ValueError("Critical Error: 'studies.txt' failed to load.")

        # Filter: Interventional
        if 'study_type' in df.columns:
            df = df[df['study_type'].str.upper() == 'INTERVENTIONAL'].copy()

        # Filter: Drugs/Biologics Only & Load Names
        cols_int = ['nct_id', 'intervention_type', 'name']
        df_int = self._safe_load('interventions.txt', cols=cols_int)

        if not df_int.empty:
            drug_ids = df_int[df_int['intervention_type'].str.upper().isin(['DRUG', 'BIOLOGICAL'])]['nct_id'].unique()
            df = df[df['nct_id'].isin(drug_ids)]
            self.df_drugs = df_int[df_int['intervention_type'].str.upper().isin(['DRUG', 'BIOLOGICAL'])].copy()
        else:
            self.df_drugs = pd.DataFrame(columns=['nct_id', 'name'])

        # Filter: Status & Phase
        allowed_statuses = ['COMPLETED', 'TERMINATED', 'WITHDRAWN', 'SUSPENDED']
        excluded_phases = ['EARLY_PHASE1', 'PHASE4', 'NA']

        df = df[df['overall_status'].isin(allowed_statuses)]
        df = df[~df['phase'].isin(excluded_phases)]

        # Target & Dates
        df['target'] = df['overall_status'].apply(lambda x: 0 if x == 'COMPLETED' else 1)
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['start_year'] = df['start_date'].dt.year

        current_year = pd.Timestamp.now().year
        df = df[df['start_year'].between(2000, 2015)]

        print(f"    Core Cohort: {len(df)} trials")
        return df.copy()

    def add_features(self, df):
        df = df.copy()
        print(">>> 2. Engineering Features...")


        # --- DEBUG START ---
        print("    [DEBUG] Unique Phases before mapping:")
        print(df['phase'].value_counts(dropna=False))
        # --- DEBUG END ---

        # 1. Phase Ordinal & Sample Weights
        # CHANGE: Integers, mapping mixed phases to the higher phase
        phase_map = {
            'PHASE1': 1,
            'PHASE1/PHASE2': 2,  # Treat as Phase 2
            'PHASE2': 2,
            'PHASE2/PHASE3': 3,  # Treat as Phase 3
            'PHASE3': 3
        }

        # Map, fill NaNs with 0, and force to Integer
        df['phase_ordinal'] = df['phase'].map(phase_map).fillna(0).astype(int)

        # Filter out rows that didn't match (0)
        df = df[df['phase_ordinal'] > 0]

        # Weight = 2.0 for Phase 2/3 (Logic remains the same)
        #df['sample_weight'] = df['phase'].apply(lambda x: 2.0 if '2' in str(x) or '3' in str(x) else 1.0)

        # 2. Operational Flags
        df['covid_exposure'] = df['start_year'].between(2019, 2021).astype(int)

        # 3. Geography
        df_countries = self._safe_load('countries.txt', cols=['nct_id', 'name'])
        if not df_countries.empty:
            country_stats = df_countries.groupby('nct_id')['name'].agg(
                cnt='nunique',
                includes_us=lambda x: 1 if 'United States' in x.values else 0
            ).reset_index()
            df = df.merge(country_stats, on='nct_id', how='left')
            df['is_international'] = (df['cnt'] > 1).astype(int)
            df['includes_us'] = df['includes_us'].fillna(0).astype(int)
            df.drop(columns=['cnt'], inplace=True)
        else:
            df['is_international'] = 0
            df['includes_us'] = 0

        # 4. Merge Standard Metadata
        df = self._merge_file(df, 'designs.txt', ['nct_id', 'allocation', 'intervention_model', 'masking', 'primary_purpose'])
        df = self._merge_file(df, 'calculated_values.txt', ['nct_id', 'number_of_primary_outcomes_to_measure'])

        # 5. Sponsor Engineering (LEAKAGE FIXED)
        df = self._engineer_sponsor_features(df)

        # 6. Complexity Engineering
        df = self._engineer_complexity(df)

        # 7. Medical Hierarchy & Competition
        df = self._attach_medical_hierarchy(df)
        df = self._calculate_competition(df)

        # 8. Text Features
        df = self._prepare_text(df)

        # --- NEW: 9. Attach Embeddings ---
        df = self._attach_embeddings(df)


        # Cleanup
        if 'number_of_primary_outcomes_to_measure' in df.columns:
            df.rename(columns={'number_of_primary_outcomes_to_measure': 'num_primary_endpoints'}, inplace=True)
        df['num_primary_endpoints'] = pd.to_numeric(df.get('num_primary_endpoints', 1), errors='coerce').fillna(1)

        return df

   # --- NEW METHOD TO PASTE INTO CLASS ---
    def _attach_embeddings(self, df):
        print("    -> Attaching Vectorized Text Embeddings...")
        # Path to the specific file
        emb_path = os.path.join(self.data_path, 'embeddings_with_nctid.csv')

        if not os.path.exists(emb_path):
            print("    [!] Warning: embeddings_with_nctid.csv not found. Skipping.")
            return df

        try:
            # Load embeddings
            df_emb = pd.read_csv(emb_path)

            # Identify the embedding columns (All columns that are NOT nct_id)
            # This makes it robust regardless of whether they are named '0', '1', 'emb_0', etc.
            emb_cols = [c for c in df_emb.columns if c != 'nct_id']

            if not emb_cols:
                print("    [!] Warning: No embedding columns found in file.")
                return df

            # Merge Left (Keep all trials, even if they don't have embeddings)
            df = df.merge(df_emb, on='nct_id', how='left')

            # Fill Missing Embeddings with 0.0
            # (If a trial has no text embedding, the vector should be zero, not NaN)
            df[emb_cols] = df[emb_cols].fillna(0.0)

            print(f"       Attached {len(emb_cols)} dimensions (e.g., {emb_cols[0]}...)")
            return df

        except Exception as e:
            print(f"    [!] Error attaching embeddings: {e}")
            return df
    # -------------------------------------



    def _engineer_sponsor_features(self, df):
        print("    -> Engineering Sponsor Tiers (Static List Only)...")

        cols_needed = ['nct_id', 'lead_or_collaborator', 'name', 'agency_class']
        df_sponsors = self._safe_load('sponsors.txt', cols=cols_needed)

        if df_sponsors.empty:
            df['sponsor_tier'] = 'TIER_2_OTHER'
            df['sponsor_clean'] = 'UNKNOWN'
            df['agency_class'] = 'UNKNOWN'
            return df

        # Filter Lead
        leads = df_sponsors[df_sponsors['lead_or_collaborator'].str.lower() == 'lead'][['nct_id', 'name', 'agency_class']]
        leads = leads.rename(columns={'name': 'lead_sponsor'})
        leads = leads.drop_duplicates('nct_id')

        df = df.merge(leads, on='nct_id', how='left')
        df['lead_sponsor'] = df['lead_sponsor'].fillna('UNKNOWN')
        df['agency_class'] = df['agency_class'].fillna('UNKNOWN')

        # 1. Clean Names (Regex)
        clean_col = df['lead_sponsor'].astype(str).str.lower().str.strip()
        legal_pattern = r'[.,]|\binc\b|\bltd\b|\bllc\b|\bcorp\b|\bgmbh\b|\bsa\b|\bplc\b'
        clean_col = clean_col.str.replace(legal_pattern, '', regex=True).str.strip()

        # 2. Map Big Pharma (Static Knowledge - NO LEAKAGE)
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
            'NIH': ['national cancer institute', 'nci', 'national institutes of health', 'nih'],
            'Mayo Clinic': ['mayo clinic']
        }

        final_names = clean_col.copy()
        for std, keys in mappings.items():
            pattern = '|'.join(keys)
            mask = clean_col.str.contains(pattern, case=False, regex=True)
            final_names.loc[mask] = std

        df['sponsor_clean'] = final_names

        # 3. Create Tiers (STATIC ONLY)
        # If in mapping -> Tier 1. Else -> Tier 2.
        # We removed the dynamic counting logic to prevent leakage.
        def get_tier(name):
            if name in mappings.keys(): return 'TIER_1_GIANT'
            return 'TIER_2_OTHER'

        df['sponsor_tier'] = df['sponsor_clean'].apply(get_tier)

        # REMOVED: sponsor_experience_log (Leakage risk)

        return df

    def _engineer_complexity(self, df):
        print("    -> Engineering Protocol Complexity & Eligibility...")

        cols_needed = ['nct_id', 'criteria', 'gender', 'healthy_volunteers', 'adult', 'child', 'older_adult']
        df_elig = self._safe_load('eligibilities.txt', cols=cols_needed)

        if df_elig.empty:
            df['criteria_len_log'] = 0
            for c in ['gender', 'healthy_volunteers', 'adult', 'child', 'older_adult']:
                df[c] = 'UNKNOWN'
            return df

        df_elig = df_elig.drop_duplicates('nct_id')
        df = df.merge(df_elig, on='nct_id', how='left')

        df['criteria_len_log'] = np.log1p(df['criteria'].astype(str).str.len().fillna(0))
        df['healthy_volunteers'] = df['healthy_volunteers'].fillna('No')
        df['gender'] = df['gender'].fillna('UNKNOWN')

        return df

    def _attach_medical_hierarchy(self, df):
        try:
            smart_path = os.path.join(self.data_path, 'smart_pathology_lookup.csv')
            if os.path.exists(smart_path):
                df_smart = pd.read_csv(smart_path)
                df = df.merge(df_smart, on='nct_id', how='left')

            df['therapeutic_area'] = df['therapeutic_area'].fillna('Other')
            df['best_pathology'] = df['best_pathology'].fillna('Unknown')

            mesh_path = os.path.join(self.data_path, 'mesh_lookup.csv')
            if os.path.exists(mesh_path):
                df_lookup = pd.read_csv(mesh_path, sep='|')
                code_to_name = pd.Series(df_lookup.mesh_term.values, index=df_lookup.tree_number).to_dict()
                def get_subgroup(tree_num):
                    if pd.isna(tree_num): return 'Unknown'
                    short_tree = str(tree_num)[:7] if len(str(tree_num)) >= 7 else str(tree_num)
                    return code_to_name.get(short_tree, 'Unknown Subgroup')
                df['therapeutic_subgroup_name'] = df['tree_number'].apply(get_subgroup)
            else:
                df['therapeutic_subgroup_name'] = 'Unknown Subgroup'
            df.drop(columns=['tree_number'], inplace=True, errors='ignore')
        except:
            df['therapeutic_area'] = 'Unknown'
            df['therapeutic_subgroup_name'] = 'Unknown'
        return df

    def _calculate_competition(self, df):
        try:
            req_cols = ['start_year', 'therapeutic_area', 'phase']
            if not all(col in df.columns for col in req_cols):
                df['competition_broad'] = 0
                df['competition_niche'] = 0
                return df

            # 1. Broad Competition (Therapeutic Area)
            grid = df.groupby(['start_year', 'therapeutic_area']).size().reset_index(name='count')
            # Create a lookup dictionary: (Year, Area) -> Count
            lookup = dict(zip(zip(grid['start_year'], grid['therapeutic_area']), grid['count']))

            def get_comp(row):
                y, area = row['start_year'], row['therapeutic_area']
                # FIX: Only look at Current Year (y) and Previous Year (y-1).
                # REMOVED: y+1 and y+2 (Future Data Leakage)
                return lookup.get((y, area), 0) + lookup.get((y-1, area), 0)

            df['competition_broad'] = df.apply(get_comp, axis=1)

            # 2. Niche Competition (Subgroup)
            if 'therapeutic_subgroup_name' in df.columns:
                grid_niche = df.groupby(['start_year', 'therapeutic_subgroup_name']).size().reset_index(name='count')
                lookup_niche = dict(zip(zip(grid_niche['start_year'], grid_niche['therapeutic_subgroup_name']), grid_niche['count']))

                def get_niche(row):
                    y, sub = row['start_year'], row['therapeutic_subgroup_name']
                    # FIX: Only look at Current Year (y) and Previous Year (y-1).
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
            if c in df.columns:
                df[c] = df[c].fillna("")

        df['txt_tags'] = (df['official_title'] + " " + df['txt_keywords'] + " " + df['txt_int_names'])
        if 'criteria' in df.columns:
            df['txt_criteria'] = df['criteria']
            df.drop(columns=['official_title', 'txt_keywords', 'txt_int_names', 'criteria'], inplace=True, errors='ignore')

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

    def save(self, df, filename='project_data.csv'):
        out_path = os.path.join(self.data_path, filename)
        df.to_csv(out_path, index=False)
        print(f">>> Saved {len(df)} rows to {out_path}")
