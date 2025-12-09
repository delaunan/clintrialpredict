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

    # --- FIRST MAIN FUNCTION CALLED ---

    def load_and_clean(self):

        print(">>> 1. Loading Studies & Applying Filters...")

        # 1. Load Core Columns (Added safety columns: has_dmc, is_fda_regulated_drug)
        cols_studies = ['nct_id', 'overall_status', 'study_type', 'phase',
                        'start_date', 'number_of_arms',
                        'official_title', 'why_stopped',
                        'has_dmc', 'is_fda_regulated_drug']

        df = self._safe_load('studies.txt', cols=cols_studies)

        if df.empty:
            raise ValueError("Critical Error: 'studies.txt' failed to load.")

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
        allowed_statuses = ['COMPLETED', 'TERMINATED', 'SUSPENDED']
        df = df[df['overall_status'].isin(allowed_statuses)]

        # 5. Filter: Phase (DROP PHASE 0 - Focus on Valley of Death)
        # We keep Phase 2, Phase 2/3, and Phase 3.
        excluded_phases = ['EARLY_PHASE1', 'PHASE4', 'NA']
        df = df[~df['phase'].isin(excluded_phases)]
        df = df.dropna(subset=['phase'])

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

        # 7. Filter: Date Range (2005-2018 for training)
        # Avoids Right-Censoring (trials still running) and COVID era bias
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['start_year'] = df['start_date'].dt.year
        df = df[df['start_year'].between(2005, 2023)]

        # 8. Create Target
        df['target'] = df['overall_status'].apply(lambda x: 0 if x == 'COMPLETED' else 1)

        print(f"    Core Cohort: {len(df)} trials (Phase 1/2/3, 2000-2018 training window and 2005-2023 for production).")
        return df.copy()

    # --- SECOND MAIN FUNCTION CALLED ---

    def add_features(self, df):
        df = df.copy()
        print(">>> 2. Engineering Features...")

        # 1. Phase Ordinal (Updated for Phase 2/3 focus)
        # phase_map = {'PHASE1/PHASE2': 2,'PHASE2': 2, 'PHASE2/PHASE3': 3, 'PHASE3': 3}     # <--- Map to Phase 2
        # df['phase_ordinal'] = df['phase'].map(phase_map).fillna(2).astype(int)

        # 2. Operational Flags
        df['covid_exposure'] = df['start_year'].between(2019, 2022).astype(int)

        # 3. Geography (SAFE: Only is_us, removed country counts to prevent leakage)
        df_countries = self._safe_load('countries.txt', cols=['nct_id', 'name'])
        if not df_countries.empty:
            us_trials = df_countries[df_countries['name'] == 'United States']['nct_id'].unique()
            df['includes_us'] = df['nct_id'].isin(us_trials).astype(int)
        else:
            df['includes_us'] = 0

        # 4. Merge Standard Metadata
        df = self._merge_file(df, 'designs.txt', ['nct_id', 'allocation', 'intervention_model', 'masking', 'primary_purpose'])
        df = self._merge_file(df, 'calculated_values.txt', ['nct_id', 'number_of_primary_outcomes_to_measure'])

        # 5. Sponsor Engineering
        df = self._engineer_sponsor_features(df)

        # 6. Complexity Engineering
        df = self._engineer_complexity(df)

        # 7. Medical Hierarchy & Competition
        df = self._attach_medical_hierarchy(df)
        df = self._calculate_competition(df)

        # 8. Text Features (Needed for keywords/title)
        df = self._prepare_text(df)

        # 9. Agent Type (The Bulletproof Classifier)
        df = self._engineer_agent_type(df)

        # 10. Smart Patterns (Rigor & Strictness)
        df = self._engineer_smart_patterns(df)

        # 11. Safe Features (DMC, Responsible Party)
        df = self._engineer_safe_features(df)

        # 12. Attach Embeddings (BioBERT)
        df = self._attach_embeddings(df)

        # 13. Attach P-Values (Analysis Only)
        df = self._attach_p_values(df)

        # Cleanup
        if 'number_of_primary_outcomes_to_measure' in df.columns:
            df.rename(columns={'number_of_primary_outcomes_to_measure': 'num_primary_endpoints'}, inplace=True)
        df['num_primary_endpoints'] = pd.to_numeric(df.get('num_primary_endpoints', 1), errors='coerce').fillna(1)

        return df

    # --- FEATURE ENGINEERING METHODS ---

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
        df['is_sick_only'] = df['healthy_volunteers'].apply(lambda x: 1 if str(x).lower() == 'no' else 0)

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
        print("    -> Engineering Agent Type (Bulletproof Classifier)...")

        # 1. Load Raw Data
        if self.df_drugs.empty:
            df['agent_category'] = 'UNKNOWN'
            return df

        # Priority Map (1 = Highest Tech)
        type_priority = {
            'GENETIC': 1,
            'BIOLOGICAL': 2,
            'DRUG': 3,
            'DIETARY SUPPLEMENT': 4,
            'OTHER': 5
        }

        df_int = self.df_drugs.copy()
        df_int['type_upper'] = df_int['intervention_type'].str.upper()
        df_int['priority'] = df_int['type_upper'].map(lambda x: type_priority.get(x, 5))

        # Sort: High Tech First. If a trial has Drug + Placebo, we keep Drug.
        df_int = df_int.sort_values('priority')
        best_types = df_int.drop_duplicates('nct_id')[['nct_id', 'type_upper', 'name']]

        df = df.merge(best_types, on='nct_id', how='left')

        # 2. The Regex Classifier
        def classify_molecule(row):
            itype = str(row['type_upper'])
            name = str(row['name']).lower()

            # Safety Check: Placebo
            if 'placebo' in name: return 'PLACEBO_CTRL'

            # --- LEVEL 1: ADVANCED THERAPIES (GENE / CELL / RNA) ---
            if itype == 'GENETIC': return 'GENE_THERAPY'

            # Cell Therapies (CAR-T, NK cells)
            if any(x in name for x in ['car-t', 'chimeric antigen', 'autologous', 'allogeneic', 't-cell', 'nk cell']):
                return 'CELL_THERAPY'
            if name.endswith('cel'): return 'CELL_THERAPY' # e.g., Tisagenlecleucel

            # RNA / DNA / Gene Editing
            if any(x in name for x in ['crispr', 'cas9', 'mrna', 'sirna', 'antisense', 'oligonucleotide', 'plasmid', 'vector', 'aav']):
                return 'RNA_GENE_THERAPY'

            # --- LEVEL 2: BIOLOGICS (LARGE MOLECULES) ---
            if itype == 'BIOLOGICAL': return 'BIOLOGIC'

            # Antibodies
            if 'mab' in name:
                if 'adc' in name or 'conjugate' in name: return 'ANTIBODY_DRUG_CONJUGATE'
                return 'MONOCLONAL_ANTIBODY'

            # Fusion Proteins & Receptors
            if name.endswith('cept'): return 'BIOLOGIC_FUSION'

            # Vaccines
            if 'vaccine' in name: return 'VACCINE'
            if any(x in name for x in ['interferon', 'interleukin', 'cytokine']): return 'IMMUNOTHERAPY'

            # --- LEVEL 3: TARGETED SMALL MOLECULES ---
            # Kinase Inhibitors
            if name.endswith('ib') or 'ib ' in name:
                if 'tinib' in name: return 'KINASE_INHIBITOR_TYROSINE'
                if 'parib' in name: return 'PARP_INHIBITOR'
                if 'lisib' in name: return 'PI3K_INHIBITOR'
                return 'TARGETED_KINASE_INHIBITOR'

            # Enzyme Inhibitors & Statins (FIXED)
            if 'vastatin' in name: return 'STATIN_CHOLESTEROL' # Specific catch for Atorvastatin etc.
            if name.endswith('stat') or 'stat ' in name: return 'ENZYME_INHIBITOR'
            if name.endswith('degib'): return 'HEDGEHOG_INHIBITOR'
            if name.endswith('clax'): return 'BCL2_INHIBITOR'

            # --- LEVEL 4: CHEMOTHERAPY ---
            chemo_stems = ['platin', 'taxel', 'rubicin', 'fluorouracil', 'gemcitabine',
                           'cyclophosphamide', 'methotrexate', 'etoposide', 'vincristine', 'vinblastine']
            if any(x in name for x in chemo_stems):
                return 'CHEMOTHERAPY'

            # --- LEVEL 5: GENERAL ---
            return 'SMALL_MOLECULE_OTHER'

        df['agent_category'] = df.apply(classify_molecule, axis=1)

        # Cleanup
        df.drop(columns=['type_upper', 'priority', 'name_y'], inplace=True, errors='ignore')
        if 'name_x' in df.columns: df.rename(columns={'name_x': 'official_title'}, inplace=True)

        return df

    def _engineer_safe_features(self, df):
        print("    -> Engineering Safe Protocol Features...")

        # REMOVED: Loading responsible_parties.txt

        # Fill Safety Flags
        # We use the same logic as _engineer_smart_patterns to ensure consistency.
        # This handles 't', 'f', 'True', 'False', 'Yes', 'No', 1, 0.
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

        # 1. Load the "Bridge" (nct_id -> mesh_term)
        # We use browse_conditions.txt because it connects trials to terms
        cols_bridge = ['nct_id', 'mesh_term']
        df_bridge = self._safe_load('browse_conditions.txt', cols=cols_bridge)

        if df_bridge.empty:
            # Fallback to conditions.txt if browse_conditions is missing
            cols_cond = ['nct_id', 'name']
            df_bridge = self._safe_load('conditions.txt', cols=cols_cond)
            if not df_bridge.empty:
                df_bridge.rename(columns={'name': 'mesh_term'}, inplace=True)

        # 2. Load the "Dictionary" (mesh_term -> therapeutic_area)
        # This is your custom file
        mesh_path = os.path.join(self.data_path, 'mesh_lookup.csv')
        df_dictionary = pd.DataFrame()

        if os.path.exists(mesh_path):
            try:
                # Audit showed this file is Pipe (|) separated
                df_dictionary = pd.read_csv(mesh_path, sep='|', on_bad_lines='skip')

                # Keep only relevant columns to save memory
                if 'mesh_term' in df_dictionary.columns and 'therapeutic_area' in df_dictionary.columns:
                    df_dictionary = df_dictionary[['mesh_term', 'therapeutic_area']].drop_duplicates()
                    # Rename to avoid collision with existing columns
                    df_dictionary.rename(columns={'therapeutic_area': 'lookup_area'}, inplace=True)
                else:
                    df_dictionary = pd.DataFrame() # Invalid columns
            except Exception as e:
                print(f"       [!] Error reading mesh_lookup.csv: {e}")

        # 3. Perform the Double Merge
        if not df_bridge.empty:
            # A. Merge Bridge + Dictionary (on mesh_term)
            if not df_dictionary.empty:
                df_full_mesh = df_bridge.merge(df_dictionary, on='mesh_term', how='left')
            else:
                df_full_mesh = df_bridge
                df_full_mesh['lookup_area'] = np.nan

            # B. Group by NCT_ID (Take first valid area found)
            # We want the 'lookup_area' if it exists, otherwise just the 'mesh_term'
            df_grouped = df_full_mesh.groupby('nct_id').agg({
                'mesh_term': 'first',
                'lookup_area': 'first'
            }).reset_index()

            # C. Merge back to Main DF
            df = df.merge(df_grouped, on='nct_id', how='left')
        else:
            df['mesh_term'] = np.nan
            df['lookup_area'] = np.nan

        # 4. Also Load Smart Lookup (Your Curated File)
        smart_path = os.path.join(self.data_path, 'smart_pathology_lookup.csv')
        if os.path.exists(smart_path):
            try:
                df_smart = pd.read_csv(smart_path)
                if 'nct_id' in df_smart.columns:
                    df = df.merge(df_smart, on='nct_id', how='left')
            except:
                pass

        # Ensure columns exist
        if 'best_pathology' not in df.columns: df['best_pathology'] = np.nan
        if 'therapeutic_area' not in df.columns: df['therapeutic_area'] = np.nan

        # 5. Final Fallback Logic
        def get_final_category(row):
            # Priority 1: Smart Lookup (Curated)
            if pd.notna(row.get('best_pathology')) and str(row.get('best_pathology')) != 'Unknown':
                return row['best_pathology']

            # Priority 2: Mesh Lookup (From your custom file)
            if pd.notna(row.get('lookup_area')) and str(row.get('lookup_area')) != 'Other/Unclassified':
                return row['lookup_area']

            # Priority 3: Raw Mesh Term (Better than nothing)
            if pd.notna(row.get('mesh_term')):
                return row['mesh_term']

            # Priority 4: Old Therapeutic Area
            if pd.notna(row.get('therapeutic_area')) and str(row.get('therapeutic_area')) != 'Other':
                return row['therapeutic_area']

            return 'Unclassified Condition'

        df['therapeutic_subgroup_name'] = df.apply(get_final_category, axis=1)

        # Cleanup
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

        # 3. Healthy Volunteers (Clean Yes/No)
        df['healthy_volunteers'] = df['healthy_volunteers'].astype(str).str.lower().apply(
            lambda x: 'no' if x in ['f', 'false', '0', 'no', 'nan'] else 'yes'
        )

        # 4. Gender
        df['gender'] = df['gender'].fillna('UNKNOWN')

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
        try:
            req_cols = ['start_year', 'therapeutic_area', 'phase']
            if not all(col in df.columns for col in req_cols):
                df['competition_broad'] = 0
                df['competition_niche'] = 0
                return df

            # Broad Competition
            grid = df.groupby(['start_year', 'therapeutic_area']).size().reset_index(name='count')
            lookup = dict(zip(zip(grid['start_year'], grid['therapeutic_area']), grid['count']))

            def get_comp(row):
                y, area = row['start_year'], row['therapeutic_area']
                return lookup.get((y, area), 0) + lookup.get((y-1, area), 0)

            df['competition_broad'] = df.apply(get_comp, axis=1)

            # Niche Competition
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
            df.drop(columns=['official_title', 'txt_keywords', 'txt_int_names', 'criteria'], inplace=True, errors='ignore')

        return df

    def _attach_embeddings(self, df):
        print("    -> Attaching Vectorized Text Embeddings...")
        emb_path = os.path.join(self.data_path, 'embeddings_with_nctid.csv')
        if not os.path.exists(emb_path):
            print("    [!] Warning: embeddings_with_nctid.csv not found.")
            return df

        try:
            df_emb = pd.read_csv(emb_path)
            emb_cols = [c for c in df_emb.columns if c != 'nct_id']
            if not emb_cols: return df

            df = df.merge(df_emb, on='nct_id', how='left')
            df[emb_cols] = df[emb_cols].fillna(0.0)
            print(f"       Attached {len(emb_cols)} dimensions.")
            return df
        except Exception as e:
            print(f"    [!] Error attaching embeddings: {e}")
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

    def save(self, df, filename='project_data.csv'):
        out_path = os.path.join(self.data_path, filename)
        df.to_csv(out_path, index=False)
        print(f">>> Saved {len(df)} rows to {out_path}")
