import pandas as pd
import numpy as np

def temporal_train_test_split(df, target_col='target', train_ratio=0.8, drop_cols=None):
    """
    Sorts by date, splits 80/20, separates X/y, and removes leakage.
    """
    # 1. Define Strict Leakage & Metadata columns to ALWAYS drop from X
    forbidden = [
        'overall_status',       # Text version of target (Leakage)
        'why_stopped', 'scientific_success', 'min_p_value',
        'start_date', 'start_year', 'nct_id', 'lead_sponsor', 'name',
        'txt_criteria', 'txt_tags', 'study_type'
    ]

    if drop_cols:
        forbidden.extend(drop_cols)

    # 2. Sort by Date
    if 'start_date' in df.columns:
        df = df.sort_values('start_date').reset_index(drop=True)
    elif 'start_year' in df.columns:
        df = df.sort_values('start_year').reset_index(drop=True)

    # 3. Split Dataframes
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    # 4. Separate X and y
    # We use the 'target_col' for y, and drop it (plus forbidden) from X

    # Train
    y_train = train[target_col]
    X_train = train.drop(columns=forbidden + [target_col], errors='ignore')

    # Test
    y_test = test[target_col]
    X_test = test.drop(columns=forbidden + [target_col], errors='ignore')

    return X_train, X_test, y_train, y_test
