
import pandas as pd
import numpy as np
import os

# 1. Load Data
DATA_PATH = "/home/delaunan/code/delaunan/clintrialpredict/data/data_clinpred.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

# 2. Filter for Historical Outcomes and Split
df_hist = df[df['target'].notna()].copy()
df_hist['start_year'] = pd.to_numeric(df_hist['start_year'], errors='coerce')

TRAIN_START_YEAR = 2009
TRAIN_END_YEAR = 2020
TEST_START_YEAR = 2021
TEST_END_YEAR = 2022

train_df = df_hist[df_hist['start_year'].between(TRAIN_START_YEAR, TRAIN_END_YEAR)].copy()
test_df = df_hist[df_hist['start_year'].between(TEST_START_YEAR, TEST_END_YEAR)].copy()

feature = 'gbd_cause_id_3_ml'
target = 'target'

# 1. How many unique values are there in the training set?
unique_values_train = train_df[feature].nunique()
print(f"1. Unique values in training set: {unique_values_train}")

# 2. What is the distribution of samples per category? (Top 10 and stats)
dist = train_df[feature].value_counts()
print("\n2. Distribution of samples per category (Top 10):")
print(dist.head(10))
print("\nDistribution Stats:")
print(dist.describe())

# 3. Calculate target encoding with smooth=100 and smooth=10
global_mean = train_df[target].mean()

def calculate_target_encoding(df, feature, target, smooth):
    stats = df.groupby(feature)[target].agg(['count', 'mean'])
    n_i = stats['count']
    y_i = stats['mean']
    encoded = (n_i * y_i + smooth * global_mean) / (n_i + smooth)
    return encoded

encoding_100 = calculate_target_encoding(train_df, feature, target, 100)
encoding_10 = calculate_target_encoding(train_df, feature, target, 10)

# Create a summary dataframe for the feature
summary = pd.DataFrame({
    'count': train_df[feature].value_counts(),
    'local_mean': train_df.groupby(feature)[target].mean(),
    'encoded_100': encoding_100,
    'encoded_10': encoding_10
})
summary['global_mean'] = global_mean
summary['diff_100_to_local'] = summary['encoded_100'] - summary['local_mean']
summary['diff_10_to_local'] = summary['encoded_10'] - summary['local_mean']
summary['pull_100'] = (summary['encoded_100'] - summary['local_mean']).abs()

print("\n3. Target Encoding Summary (Sample of 10):")
print(summary.head(10))

# 4. Check which categories have their encoded value significantly pulled towards the global mean by the high smoothing factor.
# "Significantly pulled" can be defined as pull > 0.05 or something similar.
summary['pull_strength'] = (summary['encoded_100'] - summary['local_mean']).abs()
significant_pull = summary[summary['count'] > 0].sort_values('pull_strength', ascending=False)
print("\n4. Categories with most significant pull towards global mean (smooth=100):")
print(significant_pull[['count', 'local_mean', 'encoded_100', 'pull_strength']].head(10))

# 5. Does this feature show signs of 'playing against' the test score?
# We map the training encodings to the test set and see how they perform.
test_df['encoded_100'] = test_df[feature].map(encoding_100).fillna(global_mean)
test_df['encoded_10'] = test_df[feature].map(encoding_10).fillna(global_mean)

# Analyze variance in test set outcomes for the same encoded value
# We can group by category in test set and see actual mean vs encoded
test_stats = test_df.groupby(feature).agg({
    target: ['count', 'mean', 'std'],
    'encoded_100': 'first'
})
test_stats.columns = ['test_count', 'test_mean', 'test_std', 'encoded_100_train']
test_stats = test_stats.merge(summary[['count', 'local_mean']], left_index=True, right_index=True, how='left')
test_stats.rename(columns={'count': 'train_count', 'local_mean': 'train_mean'}, inplace=True)

test_stats['error_100'] = (test_stats['test_mean'] - test_stats['encoded_100_train']).abs()

print("\n5. Feature Consistency Check (Train vs Test):")
# Categories that exist in both
common = test_stats[test_stats['train_count'].notna() & (test_stats['test_count'] >= 5)].sort_values('error_100', ascending=False)
print("Top 10 categories with highest discrepancy between train-encoded and test-actual (min 5 test samples):")
print(common[['train_count', 'train_mean', 'encoded_100_train', 'test_count', 'test_mean', 'error_100']].head(10))

# Correlation between encoded and test_mean
valid_test = test_stats[test_stats['test_count'] >= 5]
corr = valid_test['encoded_100_train'].corr(valid_test['test_mean'])
print(f"\nCorrelation between encoded_100 and test_mean (for categories with N>=5 in test): {corr:.4f}")

# Check variance
avg_test_std = valid_test['test_std'].mean()
print(f"Average standard deviation of target in test categories (N>=5): {avg_test_std:.4f}")
