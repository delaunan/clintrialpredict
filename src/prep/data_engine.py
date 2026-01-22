import pandas as pd
import numpy as np
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def get_temporal_masks(df):

    """
    Creates boolean masks based on the 'nct_id' index.
    Returns masks that can be used to slice any DataFrame indexed by nct_id.
    """
    # Ensure nct_id is the index for alignment safety
    if df.index.name != 'nct_id':
        df = df.set_index('nct_id')

    train_mask = df['start_year'] <= 2019
    val_mask = (df['start_year'] >= 2020) & (df['start_year'] <= 2022)
    test_mask = df['start_year'] > 2022
    return train_mask, val_mask, test_mask


def apply_temporal_pca(df_embeddings, train_mask, n_components, name, save_dir="models/"):
    """
    df_embeddings: DataFrame where index = nct_id and columns = 768 BERT dims.
    train_mask: Boolean Series indexed by nct_id.
    """
    os.makedirs(save_dir, exist_ok=True)

    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=42)

    # Alignment Check: Ensure the mask and embeddings have the same IDs in the same order
    # We reindex the mask to match the embeddings exactly
    current_train_mask = train_mask.reindex(df_embeddings.index)

    # 1. Scale based on Train set
    train_data = df_embeddings[current_train_mask]
    scaler.fit(train_data)

    # 2. Transform the whole set
    embeddings_scaled = scaler.transform(df_embeddings)

    # 3. Fit PCA on Train set (scaled)
    pca.fit(embeddings_scaled[current_train_mask])
    features_reduced = pca.transform(embeddings_scaled)

    # 4. Save transformers
    joblib.dump(scaler, f"{save_dir}scaler_{name}.joblib")
    joblib.dump(pca, f"{save_dir}pca_{name}.joblib")

    # 5. Return as DataFrame with the ORIGINAL nct_id index preserved
    cols = [f"{name}_pca_{i}" for i in range(n_components)]
    return pd.DataFrame(features_reduced, columns=cols, index=df_embeddings.index)
