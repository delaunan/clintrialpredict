import pandas as pd

def temporal_train_test_split(df, train_end_year=2019, val_end_year=2022):
    """
    Splits the dataframe into Train, Validation, and Test sets based on year.
    Keeps all columns in X to allow for metadata analysis and late-stage filtering.
    """

    # Ensure start_year is numeric for filtering
    df['start_year'] = pd.to_numeric(df['start_year'])

    # 1. Define the temporal masks
    train_mask = df['start_year'] <= train_end_year
    val_mask = (df['start_year'] > train_end_year) & (df['start_year'] <= val_end_year)
    test_mask = df['start_year'] > val_end_year

    # 2. Partition the dataframes
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    # 3. Separate Target (y) from Features (X)
    # We ONLY drop 'target' from X. Everything else stays.
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']

    X_val = val_df.drop(columns=['target'])
    y_val = val_df['target']

    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    # Logging for verification
    print(f"Temporal Split Summary:")
    print(f"  - Train: {X_train.shape[0]} samples (Years <= {train_end_year})")
    print(f"  - Val:   {X_val.shape[0]} samples (Years {train_end_year+1}-{val_end_year})")
    print(f"  - Test:  {X_test.shape[0]} samples (Years > {val_end_year})")
    print(f"  - Features retained in X: {list(X_train.columns)}")

    return X_train, X_val, X_test, y_train, y_val, y_test
