import pandas as pd
import numpy as np

def temporal_train_test_split(df, train_ratio=0.8):
    """
    Splits the dataset into X and y for Train and Test sets,
    strictly respecting the temporal order.

    Returns:
        X_train, X_test, y_train, y_test
    """

    # 1. Sort by Date
    if 'start_date' in df.columns:
        df = df.sort_values('start_date').reset_index(drop=True)
    else:
        df = df.sort_values('start_year').reset_index(drop=True)

    # 2. Split DataFrames
    split_index = int(len(df) * train_ratio)
    df_train = df.iloc[:split_index].copy()
    df_test = df.iloc[split_index:].copy()

    # 3. Generate Audit (Before separating X and y)
    def get_stats(d):
        total = len(d)
        if total == 0: return 0, 0, 0, 0, 0, 0
        failures = d['target'].sum()
        successes = total - failures
        fail_rate = (failures / total) * 100
        min_year = d['start_year'].min() if 'start_year' in d.columns else 0
        max_year = d['start_year'].max() if 'start_year' in d.columns else 0
        return total, failures, successes, fail_rate, min_year, max_year

    tr_n, tr_f, tr_s, tr_rate, tr_min, tr_max = get_stats(df_train)
    te_n, te_f, te_s, te_rate, te_min, te_max = get_stats(df_test)
    tot_n, tot_f, tot_s, tot_rate, tot_min, tot_max = get_stats(df)

    print("\n" + "="*85)
    print(f" TEMPORAL SPLIT AUDIT (Train Ratio: {train_ratio:.0%})")
    print("="*85)
    print(f"{'METRIC':<20} | {'TRAIN SET':<20} | {'TEST SET':<20} | {'TOTAL':<20}")
    print("-" * 85)
    print(f"{'Date Range':<20} | {int(tr_min)}-{int(tr_max):<15} | {int(te_min)}-{int(te_max):<15} | {int(tot_min)}-{int(tot_max)}")
    print(f"{'Total Trials':<20} | {tr_n:<20} | {te_n:<20} | {tot_n}")
    print(f"{'Success (0)':<20} | {tr_s:<20} | {te_s:<20} | {tot_s}")
    print(f"{'Failures (1)':<20} | {tr_f:<20} | {te_f:<20} | {tot_f}")
    print(f"{'Failure Rate':<20} | {tr_rate:.2f}%{'':<14} | {te_rate:.2f}%{'':<14} | {tot_rate:.2f}%")
    print("="*85 + "\n")

    # 4. Separate X and y (The New Part)
    X_train = df_train.drop(columns=['target'])
    y_train = df_train['target']

    X_test = df_test.drop(columns=['target'])
    y_test = df_test['target']

    return X_train, X_test, y_train, y_test
