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

        # 0. Load Core Columns (Added safety columns: has_dmc, is_fda_regulated_drug)
        cols_studies = ['nct_id', 'overall_status', 'study_type', 'phase',
                        'start_date', 'number_of_arms',
                        'official_title', 'why_stopped',
                        'has_dmc', 'is_fda_regulated_drug']

        df = self._safe_load('studies.txt', cols=cols_studies)

        if df.empty:
            raise ValueError("Critical Error: 'studies.txt' failed to load.")

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

        # 1. Phase Grouping (New Step)
        df = self._engineer_phase_groups(df)

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

        # 8. Text Features (Needed for keywords/title)
        df = self._prepare_text(df)

        # 9. Agent Type (The Bulletproof Classifier)
        df = self._engineer_agent_type(df)

        #Now that we have agent_category AND therapeutic_subgroup_name, we can calculate competition
        df = self._calculate_competition(df)

        # 10. Smart Patterns (Rigor & Strictness)
        df = self._engineer_smart_patterns(df)

        # 11. Safe Features (DMC, Responsible Party)
        df = self._engineer_safe_features(df)

        # 12. Attach Embeddings (BioBERT)
        #df = self._attach_embeddings(df)

        # 13. Attach P-Values (Analysis Only)
        df = self._attach_p_values(df)

        # Cleanup
        # Handle Primary Endpoints (Renaming and Casting)
        if 'number_of_primary_outcomes_to_measure' in df.columns:
            df.rename(columns={'number_of_primary_outcomes_to_measure': 'num_primary_endpoints'}, inplace=True)

        # Ensure numeric type and fill missing with 1 (Every trial has at least one primary endpoint)
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
                'RNA_GENE_THERAPY': 1, 'CELL_THERAPY': 1, 'ANTIBODY_DRUG_CONJUGATE': 1,
                'BISPECIFIC_ANTIBODY': 1, 'MONOCLONAL_ANTIBODY': 1, 'GLP1_PEPTIDE': 1,
                'PI3K_INHIBITOR': 1, 'BTK_INHIBITOR': 1, 'JAK_INHIBITOR': 1,
                'PARP_INHIBITOR': 1, 'BCL2_INHIBITOR': 1, 'SGLT2_INHIBITOR': 1,
                'KINASE_INHIBITOR_TYROSINE': 1, 'TARGETED_KINASE_INHIBITOR': 1,
                'CHEMOTHERAPY': 2, 'HORMONAL_THERAPY': 2, 'STATIN_CHOLESTEROL': 2,
                'ENZYME_INHIBITOR': 2, 'BIOLOGIC_OTHER': 2, 'SMALL_MOLECULE_OTHER': 2,
                'PLACEBO_CTRL': 3  # <--- Absolute lowest priority
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
            df_int['priority'] = df_int['agent_category'].map(priority_map).fillna(2)

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
        print("    -> Attaching Medical Hierarchy (Preserving Tree Structure)...")

        # 1. Load the "Bridge" (nct_id -> mesh_term)
        cols_bridge = ['nct_id', 'mesh_term']
        df_bridge = self._safe_load('browse_conditions.txt', cols=cols_bridge)
        if df_bridge.empty:
            df_bridge = self._safe_load('conditions.txt', cols=['nct_id', 'name'])
            if not df_bridge.empty:
                df_bridge.rename(columns={'name': 'mesh_term'}, inplace=True)

        # 2. Load the "Dictionary" (mesh_term -> therapeutic_area)
        mesh_path = os.path.join(self.data_path, 'mesh_lookup.csv')
        df_dictionary = pd.DataFrame()
        if os.path.exists(mesh_path):
            try:
                df_dictionary = pd.read_csv(mesh_path, sep='|', on_bad_lines='skip')
                # Keep the name 'therapeutic_area' - DO NOT RENAME TO lookup_area
                if 'mesh_term' in df_dictionary.columns and 'therapeutic_area' in df_dictionary.columns:
                    df_dictionary = df_dictionary[['mesh_term', 'therapeutic_area']].drop_duplicates()
            except Exception as e:
                print(f"   [!] Warning: Could not load mesh_lookup.csv. Error: {e}")

        # 3. Perform Merges & Aggregation
        if not df_bridge.empty:
            if not df_dictionary.empty:
                df_full_mesh = df_bridge.merge(df_dictionary, on='mesh_term', how='left')
            else:
                df_full_mesh = df_bridge.copy()

            # --- CRITICAL SAFETY: Ensure the column exists for the .agg() call ---
            if 'therapeutic_area' not in df_full_mesh.columns:
                df_full_mesh['therapeutic_area'] = np.nan

            # Group by nct_id to handle trials with multiple conditions
            df_grouped = df_full_mesh.groupby('nct_id').agg({
                'mesh_term': 'first',
                'therapeutic_area': 'first' # This now matches the column name in df_full_mesh
            }).reset_index()
            df = df.merge(df_grouped, on='nct_id', how='left')
        else:
            df['mesh_term'], df['therapeutic_area'] = np.nan, np.nan

        # 4. Expanded Regex Fallback Dictionary
        fallbacks = {
            'Oncology': r'\b(cancer|tumor|carcinoma|lymphoma|leukemia|melanoma|neoplasm|oncology|solid|malignant|adenocarcinoma|sarcoma|myeloma|glioma|metastatic|advanced|recurrent|squamous|her2|kras)\b',
            'Cardiovascular': r'\b(heart|cardiac|vascular|stent|hypertension|myocardial|atrial|coronary|stroke|embolism|arrhythmia|cholesterol|angina|infarction|hfpef|hfrer|tachycardia|atherosclerosis)\b',
            'Metabolic': r'\b(diabetes|insulin|obesity|hyperlipidemia|metabolic|glucose|nash|steatohepatitis|dyslipidemia|t2dm|hypoglycemia|weight loss|fatty liver|endocrine|hypercholesterolemia)\b',
            'Neurology': r'\b(alzheimer|parkinson|brain|neurology|epilepsy|sclerosis|migraine|cns|neurodegenerative|dementia|als|huntington|seizure|neuropathic|pain|fibromyalgia)\b',
            'Infections': r'\b(infection|virus|hiv|bacterial|fungal|antibiotic|covid|hepatitis|influenza|pneumonia|sepsis|tuberculosis|vaccine|antiviral|hiv-1|sars-cov-2)\b',
            'Immunology': r'\b(arthritis|lupus|autoimmune|inflammation|crohn|psoriasis|rheumatoid|ulcerative colitis|dermatitis|eczema|asthma|atopic|sjogren|ankylosing)\b',
            'Gastrointestinal': r'\b(gastric|gi|bowel|stomach|liver|hepatic|cirrhosis|gerd|colitis|ibs|digestive|peptic|esophagitis)\b',
            'Renal/Urology': r'\b(kidney|renal|nephropathy|urology|bladder|ckd|dialysis|urinary|prostatitis|erectile)\b',
            'Psychiatry': r'\b(depression|anxiety|schizophrenia|bipolar|psychiatric|adhd|autism|ptsd|major depressive|mental|insomnia)\b',
            'Dermatology': r'\b(skin|dermatology|acne|urticaria|rosacea|alopecia|vitiligo|pruritus)\b'
        }

        def get_hierarchy(row):
            # --- STEP A: Determine Parent (therapeutic_area) ---
            parent = str(row.get('therapeutic_area', ''))
            if parent == 'nan' or parent == '' or parent == 'Other/Unclassified':
                title = str(row.get('official_title', '')).lower()
                parent = 'Unclassified'
                for area, pattern in fallbacks.items():
                    if re.search(pattern, title):
                        parent = area
                        break

            # --- STEP B: Determine Child (therapeutic_subgroup_name) ---
            child = row.get('mesh_term')
            if pd.isna(child) or str(child).lower() == 'nan':
                child = parent

            return pd.Series([parent, child])

        # Apply and create the two columns
        df[['therapeutic_area', 'therapeutic_subgroup_name']] = df.apply(get_hierarchy, axis=1)

        # Cleanup intermediate columns
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
