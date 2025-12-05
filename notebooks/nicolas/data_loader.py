import pandas as pd
import numpy as np
import os
import csv
# this is a test
class ClinicalTrialLoader:
    def __init__(self, data_path):
        self.data_path = data_path

        # --- STRATEGY A: PERFECT (Preferred) ---
        # Respects quotes. Captures "Drug A | Drug B" correctly.
        # RISK: Crashes on unbalanced quotes.
        self.params_perfect = {
            "sep": "|", "dtype": str, "header": 0, "quotechar": '"',
            "quoting": csv.QUOTE_MINIMAL, "low_memory": False, "on_bad_lines": "warn"
        }

        # --- STRATEGY B: ROBUST (Fallback) ---
        # Ignores quotes. Survives unbalanced quotes.
        # COST: Skips rows where pipe is inside text (~55 rows lost).
        self.params_robust = {
            "sep": "|", "dtype": str, "header": 0, "quotechar": '"',
            "quoting": 3, "low_memory": False, "on_bad_lines": "warn"
        }

    def _safe_load(self, filename, cols=None):
        """
        Tries to load with 'Perfect' settings first.
        If it crashes, falls back to 'Robust' settings.
        """
        full_path = os.path.join(self.data_path, filename)

        if not os.path.exists(full_path):
            print(f":x: Error: File not found {filename}")
            return pd.DataFrame() # Return empty if file missing

        # 1. Try Perfect Load
        try:
            # print(f"   -> Loading {filename} (Strategy A: Perfect)...")
            return pd.read_csv(full_path, usecols=cols, **self.params_perfect)

        except Exception as e:
            # 2. Fallback to Robust Load
            print(f"   -> :warning: CRASH detected on {filename} ({str(e)[:50]}...)")
            print(f"   -> SWITCHING to Strategy B (Robust/Ignore Quotes)...")
            try:
                df_robust = pd.read_csv(full_path, usecols=cols, **self.params_robust)
                print(f"   -> :white_check_mark: Recovered {len(df_robust)} rows (Safe Mode).")
                return df_robust
            except Exception as e2:
                print(f"   -> :x: FATAL: Robust load also failed for {filename}: {e2}")
                return pd.DataFrame()

    def load_and_clean(self):
        print(">>> 1. Loading Studies & Applying Filters...")

        # A. Load Studies (Using Safe Load)
        # FIX: Changed from pd.read_csv to self._safe_load
        cols_studies = ['nct_id', 'overall_status', 'study_type', 'phase',
                        'start_date', 'start_date_type', 'number_of_arms', 'official_title', 'why_stopped']

        df = self._safe_load('studies.txt', cols=cols_studies)

        if df.empty:
            raise ValueError("Critical Error: Studies file failed to load.")

        # B. Filter: Interventional
        df = df[df['study_type'] == 'INTERVENTIONAL'].copy()

        # C. Filter: Drugs (Using Safe Load)
        # FIX: Changed from pd.read_csv to self._safe_load
        df_int_filter = self._safe_load('interventions.txt', cols=['nct_id', 'intervention_type'])

        # Guard against empty interventions
        if not df_int_filter.empty:
            drug_ids = df_int_filter[df_int_filter['intervention_type'].str.upper().isin(['DRUG', 'BIOLOGICAL'])]['nct_id'].unique()
            df = df[df['nct_id'].isin(drug_ids)]

        # Load 'name' separately for the text features later.
        try:
            # FIX: Changed from pd.read_csv to self._safe_load
            df_int_names = self._safe_load('interventions.txt', cols=['nct_id', 'intervention_type', 'name'])

            if not df_int_names.empty:
                self.df_drugs = df_int_names[df_int_names['intervention_type'].str.upper().isin(['DRUG', 'BIOLOGICAL'])].copy()
            else:
                self.df_drugs = pd.DataFrame(columns=['nct_id', 'name'])
        except Exception as e:
            print(f"WARNING: Could not load intervention names for text features: {e}")
            self.df_drugs = pd.DataFrame(columns=['nct_id', 'name'])

        # D. Filter: Status & Phase
        allowed_statuses = ['COMPLETED', 'TERMINATED', 'WITHDRAWN', 'SUSPENDED']
        excluded_phases = ['EARLY_PHASE1', 'PHASE4', 'NA']
        df = df[df['overall_status'].isin(allowed_statuses)]
        df = df[~df['phase'].isin(excluded_phases)]

        # E. Target & Dates
        df['target'] = df['overall_status'].apply(lambda x: 0 if x == 'COMPLETED' else 1)
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['start_year'] = df['start_date'].dt.year

        current_year = pd.Timestamp.now().year
        df = df[df['start_year'].between(2000, current_year - 1)]

        print(f"    Core Cohort: {len(df)} trials")

        return df.copy()

    def add_features(self, df):
        df = df.copy()
        print(">>> 2. Engineering Features...")

        # 1. Phase Ordinal
        phase_map = {'PHASE1': 1, 'PHASE1/PHASE2': 1.5, 'PHASE2': 2, 'PHASE2/PHASE3': 2.5, 'PHASE3': 3}
        df['phase_ordinal'] = df['phase'].map(phase_map).fillna(0)
        df = df[df['phase_ordinal'] > 0]

        # 2. Operational Flags
        df['covid_exposure'] = df['start_year'].between(2019, 2021).astype(int)

        # 3. Geography (International/US)
        # FIX: Changed from pd.read_csv to self._safe_load
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
        # FIX: _merge_file now uses _safe_load internally
        df = self._merge_file(df, 'sponsors.txt', ['nct_id', 'agency_class'], filter_col='lead_or_collaborator', filter_val='lead')
        df = self._merge_file(df, 'designs.txt', ['nct_id', 'allocation', 'intervention_model', 'masking', 'primary_purpose'])
        df = self._merge_file(df, 'eligibilities.txt', ['nct_id', 'criteria', 'gender', 'healthy_volunteers', 'adult', 'child', 'older_adult'])
        df = self._merge_file(df, 'calculated_values.txt', ['nct_id', 'number_of_primary_outcomes_to_measure'])

        # Rename and Clean
        df.rename(columns={'number_of_primary_outcomes_to_measure': 'num_primary_endpoints'}, inplace=True)
        df['num_primary_endpoints'] = pd.to_numeric(df['num_primary_endpoints'], errors='coerce').fillna(1)

        # 5. Medical Hierarchy
        df = self._attach_medical_hierarchy(df)

        # 6. Competition/Crowding Logic
        df = self._calculate_competition(df)

        # 7. Text Features
        df = self._prepare_text(df)

        return df

    def _attach_medical_hierarchy(self, df):
        print("    -> Attaching Medical Hierarchy (Smart Lookup)...")
        try:
            # Note: smart_pathology_lookup is usually a standard CSV (comma), not pipe.
            # We keep standard pd.read_csv here. If your file is pipe, change to sep='|'.
            smart_path = os.path.join(self.data_path, 'smart_pathology_lookup.csv')
            if os.path.exists(smart_path):
                df_smart = pd.read_csv(smart_path)
                df = df.merge(df_smart, on='nct_id', how='left')

            df['therapeutic_area'] = df['therapeutic_area'].fillna('Other/Unclassified')
            df['best_pathology'] = df['best_pathology'].fillna('Unknown')

            df['therapeutic_subgroup'] = df['tree_number'].astype(str).apply(
                lambda x: x[:7] if pd.notna(x) and len(x) >= 7 else 'Unknown'
            )

            # Mesh Lookup (handling pipe separator explicitly as per previous context)
            mesh_path = os.path.join(self.data_path, 'mesh_lookup.csv')
            if os.path.exists(mesh_path):
                df_lookup = pd.read_csv(mesh_path, sep='|')
                code_to_name = pd.Series(df_lookup.mesh_term.values, index=df_lookup.tree_number).to_dict()
                df['therapeutic_subgroup_name'] = df['therapeutic_subgroup'].map(code_to_name).fillna('Unknown Subgroup')
            else:
                 df['therapeutic_subgroup_name'] = 'Unknown Subgroup'

            df.drop(columns=['tree_number', 'therapeutic_subgroup'], inplace=True, errors='ignore')

        except Exception as e:
            print(f"    [Warning] Medical Hierarchy failed: {e}. Creating dummy columns.")
            df['therapeutic_area'] = 'Unknown'
            df['therapeutic_subgroup_name'] = 'Unknown'

        return df

    def _calculate_competition(self, df):
        print("    -> Calculating Competition (Broad & Niche)...")

        phase_group_map = {
            'PHASE1': 'PHASE1', 'PHASE1/PHASE2': 'PHASE2',
            'PHASE2': 'PHASE2', 'PHASE2/PHASE3': 'PHASE3', 'PHASE3': 'PHASE3'
        }
        df['phase_group'] = df['phase'].map(phase_group_map).fillna('UNKNOWN')

        # Broad
        grid_broad = df.groupby(['start_year', 'therapeutic_area', 'phase_group']).size().reset_index(name='count')
        dict_broad = dict(zip(zip(grid_broad['start_year'], grid_broad['therapeutic_area'], grid_broad['phase_group']), grid_broad['count']))

        def get_broad(row):
            y, area, ph = row['start_year'], row['therapeutic_area'], row['phase_group']
            if pd.isna(y): return 0
            return dict_broad.get((y, area, ph), 0) + dict_broad.get((y+1, area, ph), 0) + dict_broad.get((y+2, area, ph), 0)

        df['competition_broad'] = df.apply(get_broad, axis=1)

        # Niche
        grid_niche = df.groupby(['start_year', 'therapeutic_subgroup_name', 'phase_group']).size().reset_index(name='count')
        dict_niche = dict(zip(zip(grid_niche['start_year'], grid_niche['therapeutic_subgroup_name'], grid_niche['phase_group']), grid_niche['count']))

        def get_niche(row):
            y, sub, ph = row['start_year'], row['therapeutic_subgroup_name'], row['phase_group']
            if pd.isna(y) or sub == 'Unknown': return 0
            return dict_niche.get((y, sub, ph), 0) + dict_niche.get((y+1, sub, ph), 0) + dict_niche.get((y+2, sub, ph), 0)

        df['competition_niche'] = df.apply(get_niche, axis=1)
        df.drop(columns=['phase_group'], inplace=True)
        return df

    def _prepare_text(self, df):
        print("    -> Preparing Text Features...")

        # FIX: Changed from pd.read_csv to self._safe_load
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
        df['txt_criteria'] = df['criteria']

        df.drop(columns=['official_title', 'txt_keywords', 'txt_int_names', 'criteria'], inplace=True, errors='ignore')
        return df

    def _merge_file(self, df, filename, cols, filter_col=None, filter_val=None):
        try:
            # FIX: Changed from pd.read_csv to self._safe_load
            # This ensures sponsors.txt, designs.txt etc don't crash the script
            aux = self._safe_load(filename, cols=cols + ([filter_col] if filter_col else []))

            if aux.empty:
                return df

            if filter_col:
                aux = aux[aux[filter_col] == filter_val]
                aux = aux.drop(columns=[filter_col])
            aux = aux.drop_duplicates('nct_id')

            df = df.merge(aux, on='nct_id', how='left')
            return df
        except Exception as e:
            print(f"Warning: Could not merge {filename}. Error: {e}")
            return df

    def save(self, df, filename='project_data.csv'):
        output_path = os.path.join(self.data_path, filename)

        # SAVE AS STANDARD CSV
        # We rely on pandas default quoting to handle commas/newlines safely.
        df.to_csv(output_path, index=False)

        print("    Save Complete.")
        print(f">>> Saved {len(df)} rows to {output_path}")

        print("\n--- DATA DISTRIBUTION ---")
        if 'target' in df.columns:
            counts = df['target'].value_counts(normalize=True) * 100
            pct_success = counts.get(0, 0.0)
            pct_failure = counts.get(1, 0.0)
            print(f"Class 0 (Success/Completed):  {pct_success:.2f}%")
            print(f"Class 1 (Failure/Terminated): {pct_failure:.2f}%")
        else:
            print("Target column not found.")
        print("-------------------------")
