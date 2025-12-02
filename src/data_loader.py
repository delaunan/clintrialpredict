import pandas as pd
import numpy as np
import os
import csv

class ClinicalTrialLoader:
    """
    Handles the ETL process: Loading raw AACT data, filtering for scope,
    and engineering static features (Crowding, Text Tags).
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.load_params = {
            "sep": "|", "dtype": str, "header": 0, "quotechar": '"',
            "quoting": csv.QUOTE_MINIMAL, "low_memory": False, "on_bad_lines": "warn"
        }

    def load_and_clean(self):
        print(">>> 1. Loading Studies & Applying Filters...")

        # A. Load Core Study Data
        cols_studies = [
            'nct_id', 'overall_status', 'study_type', 'phase',
            'start_date', 'number_of_arms', 'official_title', 'why_stopped'
        ]
        df = pd.read_csv(os.path.join(self.data_path, 'studies.txt'), usecols=cols_studies, **self.load_params)

        # B. Filter: Interventional & Drugs Only
        df = df[df['study_type'] == 'INTERVENTIONAL'].copy()

        df_int = pd.read_csv(os.path.join(self.data_path, 'interventions.txt'), usecols=['nct_id', 'intervention_type'], **self.load_params)
        drug_ids = df_int[df_int['intervention_type'].str.upper().isin(['DRUG', 'BIOLOGICAL'])]['nct_id'].unique()
        df = df[df['nct_id'].isin(drug_ids)]

        # C. Filter: Status & Phase
        allowed_statuses = ['COMPLETED', 'TERMINATED', 'WITHDRAWN', 'SUSPENDED']
        excluded_phases = ['EARLY_PHASE1', 'PHASE4', 'NA']

        df = df[df['overall_status'].isin(allowed_statuses)]
        df = df[~df['phase'].isin(excluded_phases)]

        # D. Target & Dates
        df['target'] = df['overall_status'].apply(lambda x: 0 if x == 'COMPLETED' else 1)
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['start_year'] = df['start_date'].dt.year

        # Filter Years (2000 to Present)
        current_year = pd.Timestamp.now().year
        df = df[df['start_year'].between(2000, current_year - 1)]

        print(f"    Core Cohort: {len(df)} trials")
        return df

    def add_features(self, df):
        print(">>> 2. Engineering Features...")

        # A. Phase Ordinal (Numeric representation for models)
        phase_map = {'PHASE1': 1, 'PHASE1/PHASE2': 1.5, 'PHASE2': 2, 'PHASE2/PHASE3': 2.5, 'PHASE3': 3}
        df['phase_ordinal'] = df['phase'].map(phase_map).fillna(0)
        df = df[df['phase_ordinal'] > 0] # Drop unknown phases

        # B. Operational Flags
        df['covid_exposure'] = df['start_year'].between(2019, 2021).astype(int)

        # C. Geography (International/US)
        df_countries = pd.read_csv(os.path.join(self.data_path, 'countries.txt'), usecols=['nct_id', 'name'], **self.load_params)
        country_stats = df_countries.groupby('nct_id')['name'].agg(
            cnt='nunique',
            includes_us=lambda x: 1 if 'United States' in x.values else 0
        ).reset_index()

        df = df.merge(country_stats, on='nct_id', how='left')
        df['is_international'] = (df['cnt'] > 1).astype(int)
        df['includes_us'] = df['includes_us'].fillna(0).astype(int)
        df.drop(columns=['cnt'], inplace=True)

        # D. Merge Metadata (Sponsors, Design, Eligibility)
        # (Simplified for brevity - assumes files exist)
        self._merge_file(df, 'sponsors.txt', ['nct_id', 'agency_class'], filter_col='lead_or_collaborator', filter_val='lead')
        self._merge_file(df, 'designs.txt', ['nct_id', 'allocation', 'intervention_model', 'masking', 'primary_purpose'])
        self._merge_file(df, 'eligibilities.txt', ['nct_id', 'gender', 'healthy_volunteers', 'adult', 'child', 'older_adult'])
        self._merge_file(df, 'calculated_values.txt', ['nct_id', 'number_of_primary_outcomes_to_measure'])

        # Rename calculated column for clarity
        df.rename(columns={'number_of_primary_outcomes_to_measure': 'num_primary_endpoints'}, inplace=True)
        df['num_primary_endpoints'] = pd.to_numeric(df['num_primary_endpoints'], errors='coerce').fillna(1)

        return df

    def _merge_file(self, df, filename, cols, filter_col=None, filter_val=None):
        """Helper to safely merge auxiliary files"""
        try:
            aux = pd.read_csv(os.path.join(self.data_path, filename), usecols=cols + ([filter_col] if filter_col else []), **self.load_params)
            if filter_col:
                aux = aux[aux[filter_col] == filter_val]
                aux = aux.drop(columns=[filter_col])

            # Drop duplicates to prevent row explosion
            aux = aux.drop_duplicates('nct_id')

            # Merge in place
            df[cols[1:]] = df.merge(aux, on='nct_id', how='left')[cols[1:]]
        except Exception as e:
            print(f"Warning: Could not merge {filename}. Error: {e}")

    def save(self, df, filename='project_data.csv'):
        output_path = os.path.join(self.data_path, filename)
        df.to_csv(output_path, index=False)
        print(f">>> Saved {len(df)} rows to {output_path}")

# Allow running as a script

if __name__ == "__main__":
    # Update this path to your actual path
    PATH = "/home/delaunan/code/delaunan/clintrialpredict/data"
    loader = ClinicalTrialLoader(PATH)
    df = loader.load_and_clean()
    df = loader.add_features(df)
    loader.save(df)
