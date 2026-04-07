import pandas as pd
import numpy as np
import os
import sys
import csv
import re
from pathlib import Path
from dotenv import load_dotenv

#emb: Prepares text for embeddings
# from src.prep.text_cleaning import day_zero_reconstructor
from src.prep.text_cleaning_ui import ui_clean_text, ui_format_multiline, ui_truncate, ui_smart_title_case, ui_smart_sentence_case
# --- DYNAMIC PIPELINE IMPORT ---
from src.prep.pipeline import FEATURE_REGISTRY, UI_SCHEMA

# Load global variables
load_dotenv()

#emb: Custom exception for missing LLM or NLP enrichment data (not currently used)
# class EnrichmentCoverageError(Exception):
#     """Custom exception for missing LLM or NLP enrichment data."""
#     pass

class ClinicalTrialLoader:

    def __init__(self, data_path, processed_path=None):
        self.data_path = Path(data_path)
        self.processed_path = Path(processed_path) if processed_path else self.data_path / 'processed'
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

    def _load_llm_outputs(self):
        """Consolidates the 4 LLM enrichment runs into a single dataframe."""
        print(">>> Loading Consolidated LLM Enrichment Outputs...")
        def load_clean(filename, drop_cols):
            path = self.processed_path / filename
            if not path.exists():
                print(f"    [WARN] File not found: {path}")
                return pd.DataFrame()
            # [IRON GATE] Use quoting=csv.QUOTE_ALL for reading to handle shielded commas
            df = pd.read_csv(path, dtype={'nct_id': str}, quoting=csv.QUOTE_ALL).drop_duplicates('nct_id')
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
            return df

        df1 = load_clean('llm_out_01.csv', ['clinical_evidence', 'mapping_logic'])
        df2 = load_clean('llm_out_02.csv', ['pharmacology_logic'])
        df3 = load_clean('llm_out_03.csv', ['strategist_logic'])
        df4 = load_clean('llm_out_04.csv', ['structural_forensic_monologue', 'primary_duration_value', 'primary_duration_unit'])

        # Filter out empty dataframes before merging
        dfs = [df for df in [df1, df2, df3, df4] if not df.empty]
        if not dfs: return pd.DataFrame()

        df_llm = dfs[0]
        for df_next in dfs[1:]:
            df_llm = df_llm.merge(df_next, on='nct_id', how='left', validate='one_to_one')
        return df_llm

    def _safe_load(self, filename, cols=None):
        full_path = self.data_path / filename
        if not full_path.exists(): return pd.DataFrame()
        try:
            return pd.read_csv(full_path, usecols=cols, **self.params_perfect)
        except:
            try: return pd.read_csv(full_path, usecols=cols, **self.params_robust)
            except: return pd.DataFrame()
            

    def get_filtered_universe(self):
        """Applies raw clinical filters to identify the target cohort from AACT files."""
        print(">>> Building the Raw Trial Universe (AACT Source of Truth)...")

        # Get Dynamic Temporal Constraints from Environment
        import os
        start_year = int(os.getenv('GLOBAL_START_YEAR', 2009))
        end_year = int(os.getenv('GLOBAL_END_YEAR', 2025))

        cols_studies = ['nct_id', 'overall_status', 'study_type', 'phase', 'start_date', 'number_of_arms',
                        'official_title', 'brief_title', 'why_stopped', 'has_dmc', 'is_fda_regulated_drug',
                        'enrollment', 'enrollment_type', 'primary_completion_date', 'completion_date', 'has_expanded_access', 'acronym']

        df = self._safe_load('studies.txt', cols=cols_studies)
        if df.empty: raise ValueError("'studies.txt' failed to load.")
        print(f"    [Filter] Initial Load: {len(df)} trials from studies.txt.")

        # 1. Lead Sponsor Filter (Industry only)
        df_sponsors = self._safe_load('sponsors.txt', cols=['nct_id', 'lead_or_collaborator', 'agency_class', 'name'])
        if not df_sponsors.empty:
            leads = df_sponsors[(df_sponsors['lead_or_collaborator'].str.upper() == 'LEAD') & (df_sponsors['agency_class'].str.upper() == 'INDUSTRY')][['nct_id', 'name', 'agency_class']]
            leads = leads.rename(columns={'name': 'lead_sponsor'}).drop_duplicates('nct_id')
            df = df.merge(leads, on='nct_id', how='inner')
            print(f"    [Filter] Kept {len(df)} Industry-led trials.")

        # 2. Safety Valve: Filter out Compassionate Use (has_expanded_access)
        if 'has_expanded_access' in df.columns:
            df = df[df['has_expanded_access'].fillna('f').str.lower().isin(['f', 'false', '0', 'no'])]
            print(f"    [Filter] Kept {len(df)} Non-Expanded Access trials.")

        # 3. Interventional Only
        df = df[df['study_type'].str.upper() == 'INTERVENTIONAL'].copy()
        print(f"    [Filter] Kept {len(df)} Interventional trials.")

        # 4. Status Filter
        allowed_statuses = ['COMPLETED', 'TERMINATED', 'WITHDRAWN', 'RECRUITING', 'ACTIVE, NOT RECRUITING', 'NOT YET RECRUITING', 'ENROLLING BY INVITATION']
        df = df[df['overall_status'].str.upper().isin(allowed_statuses)]
        print(f"    [Filter] Kept {len(df)} trials with allowed statuses: {', '.join(allowed_statuses)}")

        # 5. Phase Filter (Phase 2 and 3 focus)
        excluded_phases = ['EARLY_PHASE1', 'PHASE1', 'PHASE4', 'NA']
        df = df[~df['phase'].fillna('NA').str.upper().isin(excluded_phases)].dropna(subset=['phase'])
        print(f"    [Filter] Kept {len(df)} Phase 2/3 focus trials (Excluded: {', '.join(excluded_phases)}).")

        # 6. Start Year Filter (Dynamic)
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['start_year'] = df['start_date'].dt.year
        df = df[df['start_year'].between(start_year, end_year)]
        print(f"    [Filter] Kept {len(df)} trials within dynamic range {start_year}-{end_year}.")

        # 7. Modality Filter (Drug/Biologic/Genetic)
        df_int = self._safe_load('interventions.txt', cols=['nct_id', 'intervention_type'])
        if not df_int.empty:
            drug_ids = df_int[df_int['intervention_type'].str.upper().isin(['DRUG', 'BIOLOGICAL', 'GENETIC'])]['nct_id'].unique()
            df = df[df['nct_id'].isin(drug_ids)]
            print(f"    [Filter] Kept {len(df)} Drug/Biologic/Genetic trials.")

        # 8. [Sanitizer] COVID Sanitizer (Remove Pandemic Failures)
        if 'why_stopped' in df.columns:
            covid_keywords = ['covid', 'pandemic', 'coronavirus', 'sars-cov-2', 'travel restrictions', 'quarantine', 'lockdown', 'sars-cov']
            mask_covid = df['why_stopped'].fillna('').astype(str).str.lower().apply(
                lambda x: any(k in x for k in covid_keywords)
            )
            if mask_covid.sum() > 0:
                print(f"    [Sanitizer] Dropping {mask_covid.sum()} trials terminated due to COVID/Logistics.")
                df = df[~mask_covid]
                print(f"    [Filter] Kept {len(df)} trials after COVID sanitization.")

        # 9. Create Target Column (Success=0, Failure=1, Ongoing=NaN)
        def determine_target(status):
            s = str(status).upper()
            if s == 'COMPLETED': return 0.0
            if s in ['TERMINATED', 'WITHDRAWN']: return 1.0
            return np.nan

        df['target'] = df['overall_status'].apply(determine_target)

        return df

    def load_base_data(self):
        """Phase 1 Loader: merges filtered raw data with ground-truth LLM enums."""
        df_llm = self._load_llm_outputs()
        df = self.get_filtered_universe()

        if not df_llm.empty:
            if not set(df['nct_id']).issubset(set(df_llm['nct_id'])):
                n_miss = len(set(df['nct_id']) - set(df_llm['nct_id']))
                print(f"    [WARN] Missing LLM data for {n_miss} trials in universe.")
            df = df.merge(df_llm, on='nct_id', how='inner')

        df['trial_segment'] = df['target'].apply(lambda x: 'HISTORICAL' if pd.notna(x) else 'ONGOING')

        # Fallback titles
        df['official_title'] = df['official_title'].fillna(df['brief_title'])
        df['brief_title'] = df['brief_title'].fillna(df['official_title'])

        # [REFINEMENT] Apply Smart Casing to titles for UI consistency
        df['official_title'] = df['official_title'].astype(str).apply(ui_smart_title_case)
        df['brief_title'] = df['brief_title'].astype(str).apply(ui_smart_title_case)

        return df

    def add_features(self, df):
        """Phase 2: Hybrid Engineering (Raw + LLM Ground Truth)."""
        df = df.copy()
        df = self._merge_file(df, 'designs.txt', ['nct_id', 'allocation', 'intervention_model', 'masking', 'primary_purpose'])

        # LOGIC-LOCK: Universal Unknown Alignment
        # Ensures all base fields are initialized before mapping
        base_cols = [c.replace('_ml', '') for c in FEATURE_REGISTRY.keys()]
        for col in base_cols:
            if col in df.columns:
                df[col] = df[col].fillna('UNKNOWN').astype(str).str.upper()

        df = self._engineer_eligibility_age(df)
        df = self._engineer_geography(df)
        df = self._engineer_facilities(df)
        df = self._engineer_safe_features(df)
        df = self._engineer_placebo(df)
        df = self._engineer_raw_scientific_evidence(df)
        df = self._add_ui_text_fields(df)
        df = self._apply_master_mapping(df)
        #emb: Prepares text for embeddings
        # df = self._engineer_nlp_pillars(df)
        #emb: Merges embedding repository
        # df = self._attach_embeddings(df)
        df = self._attach_p_values(df)

        print("    -> Finalizing Data Types...")

        # Robustly handle is_rare_disease if it's string-based 'True'/'False'
        if 'is_rare_disease' in df.columns:
            df['is_rare_disease'] = df['is_rare_disease'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}).fillna(0)

        # Fix: Only coerce fields that are meant to be numeric for XGBoost
        # This prevents text-based UI/Raw fields in the registry from being nullified
        registry_numeric_cols = [
            f for f, conf in FEATURE_REGISTRY.items() 
            if conf.get("encoding") in ["numeric", "ordinal", "target"]
        ]
        
        ui_numeric_cols = [
            'enrollment', 'number_of_facilities', 'min_p_value', 'scientific_success',
            'target', 'gbd_cause_id', 'gbd_hierarchy_level',
            'primary_duration_months', 'is_rare_disease',
            'daly_global', 'yld_global', 'yll_global', 'chronic_ratio_global',
            'daly_high_income', 'yld_high_income', 'yll_high_income', 'market_skew_index'
        ]
        
        for col in registry_numeric_cols + ui_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def _engineer_geography(self, df):
        df_countries = self._safe_load('countries.txt', cols=['nct_id', 'name'])
        if df_countries.empty:
            df['includes_us'] = 0; df['raw_geographic_footprint'] = "UNKNOWN"; return df
        us_trials = df_countries[df_countries['name'] == 'United States']['nct_id'].unique()
        df['includes_us'] = df['nct_id'].isin(us_trials).astype(int)
        footprint = df_countries.groupby('nct_id')['name'].apply(lambda x: ", ".join(x.dropna().unique())).reset_index(name='raw_geographic_footprint')
        return df.merge(footprint, on='nct_id', how='left')

    def _engineer_facilities(self, df):
        return self._merge_file(df, 'calculated_values.txt', ['nct_id', 'number_of_facilities'])

    def _engineer_raw_scientific_evidence(self, df):
        """Aggregates raw scientific evidence into pipe-separated structural strings."""
        print("    -> Merging Raw Scientific Evidence...")

        # 1. Conditions
        df_cond = self._safe_load('conditions.txt', cols=['nct_id', 'name'])
        if not df_cond.empty:
            conds = df_cond.groupby('nct_id')['name'].apply(lambda x: " || ".join(x.dropna().unique())).reset_index(name='raw_conditions')
            df = df.merge(conds, on='nct_id', how='left')

        # 2. Interventions (Enhanced with Type, Description, and Synonyms)
        df_int = self._safe_load('interventions.txt', cols=['nct_id', 'id', 'intervention_type', 'name', 'description'])
        df_others = self._safe_load('intervention_other_names.txt', cols=['nct_id', 'intervention_id', 'name'])

        if not df_int.empty:
            if not df_others.empty:
                others = df_others.groupby(['nct_id', 'intervention_id'])['name'].apply(lambda x: ", ".join(x.dropna().unique())).reset_index(name='synonyms')
                df_int = df_int.merge(others, left_on=['nct_id', 'id'], right_on=['nct_id', 'intervention_id'], how='left')
            else:
                df_int['synonyms'] = None

            def fmt_int(r):
                name = str(r['name'])
                syns = f" (Aliases: {r['synonyms']})" if pd.notna(r['synonyms']) else ""
                itype = f" [{r['intervention_type']}]" if pd.notna(r['intervention_type']) else ""
                desc = f"{r['description']}" if pd.notna(r['description']) else "No description"
                # [SEMANTIC UPGRADE] Structured with Newlines for easy simulation editing
                return f"NAME: {name}{syns}{itype}\nDESC: {desc}"

            df_int['fmt'] = df_int.apply(fmt_int, axis=1)
            ints = df_int.groupby('nct_id')['fmt'].apply(lambda x: "\n".join(x.dropna().unique())).reset_index(name='raw_interventions')
            df = df.merge(ints, on='nct_id', how='left')

        # 3. Primary Outcomes (Enhanced with Timeframe)
        df_outcomes = self._safe_load('design_outcomes.txt', cols=['nct_id', 'outcome_type', 'measure', 'time_frame'])
        if not df_outcomes.empty:
            primaries = df_outcomes[df_outcomes['outcome_type'].astype(str).str.upper().str.contains('PRIMARY', na=False)].copy()
            def fmt_out(r):
                m = str(r['measure'])[:500]
                # [FIX] Clean timeframe to prevent dangling line returns in UI
                tf_raw = str(r['time_frame']) if pd.notna(r['time_frame']) else "No timeframe"
                tf_clean = " ".join(tf_raw.replace("~", " ").split())
                return f"TITLE: {m}\nTIMEFRAME: {tf_clean}"
            primaries['fmt'] = primaries.apply(fmt_out, axis=1)
            outcomes = primaries.groupby('nct_id')['fmt'].apply(lambda x: "\n".join(x.dropna().unique())).reset_index(name='raw_primary_outcomes')
            df = df.merge(outcomes, on='nct_id', how='left')

        return df

    def _engineer_eligibility_age(self, df):
        df_elig = self._safe_load('eligibilities.txt', cols=['nct_id', 'gender', 'healthy_volunteers', 'minimum_age', 'maximum_age'])
        if df_elig.empty:
            df['gender'] = 'UNKNOWN'
            df['healthy_volunteers'] = 0
            df['child'] = np.nan
            df['adult'] = np.nan
            df['older_adult'] = np.nan
            return df

        df = df.merge(df_elig.drop_duplicates('nct_id'), on='nct_id', how='left')

        # Healthy Volunteers: Default to 0 (No) for Phase 2/3 focus
        df['healthy_volunteers'] = df['healthy_volunteers'].astype(str).str.lower().apply(
            lambda x: 1 if x in ['t', 'true', '1', 'yes'] else 0
        )

        df['gender'] = df['gender'].fillna('UNKNOWN').str.upper()

        def parse_age(val, default):
            if pd.isna(val) or str(val).lower() in ['nan', 'none', '']: return default
            try:
                match = re.search(r'(\d+(\.\d+)?)', str(val))
                if not match: return default
                num = float(match.group(1)); text = str(val).lower()
                if 'month' in text: num /= 12.0
                elif 'week' in text: num /= 52.0
                elif 'day' in text: num /= 365.0
                return num
            except: return default

        # Use NaN as sentinel for derivation logic
        df['min_age_years'] = df['minimum_age'].apply(lambda x: parse_age(x, np.nan))
        df['max_age_years'] = df['maximum_age'].apply(lambda x: parse_age(x, np.nan))

        # 1. CHILD: Floor < 18 OR Ceiling < 18
        df['child'] = 0
        df.loc[(df['min_age_years'] < 18) | (df['max_age_years'] < 18), 'child'] = 1

        # 2. ADULT: Exclude if Ceiling < 18 OR Floor >= 65. Default 1.
        df['adult'] = 1
        df.loc[(df['max_age_years'] < 18) | (df['min_age_years'] >= 65), 'adult'] = 0

        # 3. OLDER ADULT: Exclude if Ceiling <= 65. Default 1.
        df['older_adult'] = 1
        df.loc[df['max_age_years'] <= 65, 'older_adult'] = 0

        # FORENSIC OVERRIDE: If BOTH boundaries are missing/text-only, set to NaN to trigger "Not Specified" UI via UNKNOWN mapping
        mask_no_age = df['min_age_years'].isna() & df['max_age_years'].isna()
        df.loc[mask_no_age, ['adult', 'older_adult', 'child']] = np.nan

        return df.drop(columns=['min_age_years', 'max_age_years'])

    def _engineer_placebo(self, df):
        df_groups = self._safe_load('design_groups.txt', cols=['nct_id', 'group_type', 'title', 'description'])
        if df_groups.empty: df['has_placebo'] = 0; return df
        df_groups['text'] = (df_groups['title'].fillna('') + " " + df_groups['description'].fillna('')).str.lower()
        is_placebo = (df_groups['group_type'].fillna('').str.upper().str.contains('PLACEBO') | df_groups['text'].str.contains(r'\b(?:placebo|sham|vehicle)\b', regex=True))
        placebo_ids = df_groups[is_placebo]['nct_id'].unique()
        df['has_placebo'] = df['nct_id'].isin(placebo_ids).astype(int)
        if 'comparator_benchmark' in df.columns:
            df.loc[df['comparator_benchmark'] == 'PLACEBO', 'has_placebo'] = 1
        return df

    def _engineer_safe_features(self, df):
        df['has_dmc'] = df['has_dmc'].astype(str).apply(lambda x: 1 if x.lower() in ['true', 't', '1', 'yes'] else 0)

        # IRON LAW: FDA Regulated only if Start Year >= 2017
        if 'is_fda_regulated_drug' in df.columns:
            raw_fda = df['is_fda_regulated_drug'].astype(str).apply(lambda x: 1 if x.lower() in ['true', 't', '1', 'yes'] else 0)
            df['is_fda_regulated_drug'] = ((raw_fda == 1) & (df['start_year'] >= 2017)).astype(int)
        else:
            df['is_fda_regulated_drug'] = 0
        return df

    def _add_ui_text_fields(self, df):
        """Generates sanitized text for UI display."""
        df_summaries = self._safe_load("brief_summaries.txt", cols=["nct_id", "description"])
        if not df_summaries.empty:
            df = df.merge(df_summaries.rename(columns={"description": "ui_summary_raw"}), on="nct_id", how="left")

        df_elig = self._safe_load('eligibilities.txt', cols=['nct_id', 'criteria'])
        if not df_elig.empty:
            df = df.merge(df_elig.rename(columns={'criteria': 'ui_criteria_raw'}), on='nct_id', how='left')

        df["title"] = df["official_title"].fillna("").astype(str).apply(ui_clean_text).apply(ui_smart_title_case).apply(lambda x: ui_truncate(x, 1000))
        df["acronym_ui"] = df["acronym"].fillna("").astype(str).apply(ui_clean_text)
        # [LINGUISTIC DIET] Match LLM Context Caps (Run 1 & 3)
        df["summary_ui"] = df.get("ui_summary_raw", pd.Series([""]*len(df))).fillna("").astype(str).apply(ui_format_multiline).apply(ui_smart_sentence_case).apply(lambda x: ui_truncate(x, 8000))
        df["criteria_ui"] = df.get("ui_criteria_raw", pd.Series([""]*len(df))).fillna("").astype(str).apply(ui_format_multiline).apply(ui_smart_sentence_case).apply(lambda x: ui_truncate(x, 15000))

        # Raw Scientific Evidence Cleaning (Using Multiline Formatter)
        df["conditions_ui"] = df.get("raw_conditions", pd.Series([""]*len(df))).fillna("").astype(str).apply(ui_format_multiline).apply(ui_smart_sentence_case)
        df["interventions_ui"] = df.get("raw_interventions", pd.Series([""]*len(df))).fillna("").astype(str).apply(ui_format_multiline).apply(ui_smart_sentence_case)
        df["primary_outcomes_ui"] = df.get("raw_primary_outcomes", pd.Series([""]*len(df))).fillna("").astype(str).apply(ui_format_multiline).apply(ui_smart_sentence_case)

        # [SURGICAL PURGE] Drop raw source columns after UI engineering to maintain a clean final output
        redundant_raw = [
            "ui_summary_raw", "ui_criteria_raw", 
            "raw_conditions", "raw_interventions", "raw_primary_outcomes", "raw_geographic_footprint"
        ]
        return df.drop(columns=redundant_raw, errors="ignore")

    #emb: Restores concatenation of Title + Brief Summary for BioBERT.
    # def _engineer_nlp_pillars(self, df):
    #     """Restores concatenation of Title + Brief Summary for BioBERT."""
    #     # Note: We re-load descriptions to ensure BioBERT gets RAW text signal
    #     df_summaries = self._safe_load("brief_summaries.txt", cols=["nct_id", "description"])
    #     if not df_summaries.empty:
    #         df = df.merge(df_summaries.rename(columns={"description": "nlp_summary"}), on="nct_id", how="left")
    #     else:
    #         df["nlp_summary"] = ""
    #
    #     df['official_title'] = df['official_title'].fillna("").astype(str)
    #     df['nlp_summary'] = df['nlp_summary'].fillna("").astype(str)
    #
    #     df['txt_scientific_essence'] = (df['official_title'].str.strip() + " [SEP] " + df['nlp_summary'].str.strip())
    #     df['txt_criteria'] = df['ui_criteria'] # Criteria already loaded and sanitized
    #
    #     df_outcomes = self._safe_load('design_outcomes.txt', cols=['nct_id', 'measure', 'outcome_type'])
    #     if not df_outcomes.empty:
    #         primaries = df_outcomes[df_outcomes['outcome_type'].str.lower().str.contains('primary', na=False)]
    #         endpoints = primaries.groupby('nct_id')['measure'].apply(lambda x: " [SEP] ".join(x.dropna().astype(str))).reset_index(name='txt_primary_endpoints')
    #         df = df.merge(endpoints, on='nct_id', how='left')
    #
    #     df['txt_primary_endpoints'] = df.get('txt_primary_endpoints', pd.Series([""]*len(df))).fillna("")
    #     return df.drop(columns=['nlp_summary'], errors='ignore')

    #emb: Merges embedding repository
    # def _attach_embeddings(self, df):
    #     path_repo = self.processed_path / 'embeddings.parquet'
    #     if path_repo.exists(): df = df.merge(pd.read_parquet(path_repo), on='nct_id', how='left')
    #     return df

    def _attach_p_values(self, df):
        """Restores complete metadata extraction for p-values."""
        df_out = self._safe_load('outcomes.txt', cols=['nct_id', 'id', 'outcome_type'])
        df_ana = self._safe_load('outcome_analyses.txt', cols=['nct_id', 'outcome_id', 'p_value', 'p_value_modifier'])

        if df_out.empty or df_ana.empty:
            df['scientific_success'] = 0; df['min_p_value'] = np.nan; return df

        merged = df_ana.merge(df_out.rename(columns={'id': 'outcome_id'}), on=['nct_id', 'outcome_id'], how='inner')
        primary = merged[merged['outcome_type'].astype(str).str.lower().str.contains('primary', na=False)].copy()

        if primary.empty:
            df['scientific_success'] = 0; df['min_p_value'] = np.nan; return df

        primary['p_val_num'] = pd.to_numeric(primary['p_value'].astype(str).str.replace(',', '.'), errors='coerce')

        trial_stats = primary.sort_values(['nct_id', 'p_val_num']).groupby('nct_id').agg({
            'p_val_num': 'min',
            'p_value': 'first',
            'p_value_modifier': 'first'
        }).reset_index()

        trial_stats.rename(columns={'p_val_num': 'min_p_value'}, inplace=True)
        df = df.merge(trial_stats, on='nct_id', how='left')
        df['scientific_success'] = (df['min_p_value'] <= 0.05).astype(int).fillna(0)
        return df

    def _apply_master_mapping(self, df):
        """Dynamically applies FEATURE_REGISTRY to encode _ml and generate _ui fields."""
        print("    -> Applying Universal Registry Mappings...")
        df = df.copy()

        for field_name, config in FEATURE_REGISTRY.items():
            # Source detection: check for raw field or _ml field
            source_col = field_name if field_name in df.columns else field_name.replace('_ml', '')
            if source_col not in df.columns:
                continue

            # Determine target column names
            ml_col = field_name
            # Only create _ui if it's a mapped field (not numeric/target/identity)
            ui_col = field_name.replace('_ml', '') + "_ui"

            if "mapping" in config:
                mapping_dict = config["mapping"]

                def safe_lookup(val, dictionary, f_name):
                    # 1. Handle NaNs/Nulls
                    if pd.isna(val) or str(val).strip().upper() in ["NAN", "NONE", "", "NULL"]:
                        # Special Case for target: fallback to NaN to avoid labeling ongoing trials
                        default_val = np.nan if f_name == 'target' else 0
                        return dictionary.get("UNKNOWN", [default_val, "Not Specified"])

                    # 2. Exact match
                    if val in dictionary: return dictionary[val]

                    # 3. String normalization
                    s_val = str(val).strip().upper()
                    if s_val in dictionary: return dictionary[s_val]

                    # 4. Numeric normalization
                    try:
                        num_val = float(val)
                        if num_val == int(num_val): num_val = int(num_val)
                        if num_val in dictionary: return dictionary[num_val]
                    except: pass

                    # 5. Final fallback
                    return dictionary.get("UNKNOWN", [0, "Not Specified"])

                mapped_data = df[source_col].map(lambda x: safe_lookup(x, mapping_dict, field_name))
                df[ml_col] = mapped_data.map(lambda x: x[0])
                # Restore _ui for all mapped fields to ensure registry-aligned labels/sorting
                df[ui_col] = mapped_data.map(lambda x: x[1])

            elif config.get("encoding") in ["numeric", "target"]:
                df[ml_col] = pd.to_numeric(df[source_col], errors='coerce')

        # [FINAL CLEANUP] Remove ONLY specific redundant Identity/System _ui fields
        # requested to be listed only once.
        redundant = ["acronym_ui", "nct_id_ui", "target_ui", "is_duration_unknown_ui"]
        
        df = df.drop(columns=redundant, errors="ignore")
        return df

    def _merge_file(self, df, filename, cols):
        try:
            aux = self._safe_load(filename, cols=cols)
            if aux.empty: return df
            return df.merge(aux.drop_duplicates('nct_id'), on='nct_id', how='left')
        except: return df

    #emb: Checks embedding coverage and exports delta for Colab if needed.
    # def check_and_export_nlp(self, df):
    #     """Checks embedding coverage and exports delta for Colab if needed."""
    #     print(">>>  Checking NLP Embedding Coverage...")
    #     emb_cols = [c for c in df.columns if c.startswith('crit_')]
    #     n_missing = df[emb_cols[0]].isna().sum() if emb_cols else len(df)
    #     if n_missing == 0: self.save(df); return
    #
    #     mask_missing = df[emb_cols[0]].isna() if emb_cols else pd.Series([True]*len(df), index=df.index)
    #     df_delta = df[mask_missing][['nct_id', 'target', 'txt_scientific_essence', 'txt_criteria']].copy()
    #
    #     for col in ['txt_scientific_essence', 'txt_criteria']:
    #         if col in df_delta.columns:
    #             df_delta[col] = df_delta[col].astype(str).apply(lambda x: day_zero_reconstructor(x, col.replace('txt_', '')))
    #
    #     df_delta.to_csv(self.data_path / 'data_clinpred_emb_transform.csv', index=False)
    #     raise EnrichmentCoverageError(f"Missing embeddings for {n_missing} trials.")

    def save(self, df, filename='data_clinpred.csv'):
        out_path = self.data_path / filename
        # [IRON GATE] Apply strict quoting for master data integrity
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
        print(f">>> Saved {len(df)} rows to {out_path}")
