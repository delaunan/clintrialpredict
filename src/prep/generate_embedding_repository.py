import os
import numpy as np
import pandas as pd
from pathlib import Path

def generate_central_repository():
    """
    Creates the first and only central repository for existing embeddings.
    Merges nct_id from data_clinpred_emb_transform.csv and the three .npy files
    (crit, sci, endp) into a single Parquet file for efficient retrieval.
    """
    data_path = Path("data")
    processed_path = Path("data/processed")
    repo_file = processed_path / "embeddings.parquet"

    # Define source paths
    path_key = data_path / "data_clinpred_emb_transform.csv"
    path_crit = data_path / "biobert_crit_raw.npy"
    path_sci = data_path / "biobert_sci_raw.npy"
    path_endp = data_path / "biobert_endp_raw.npy"

    print(f">>> Initializing Central Embedding Repository creation...")

    # 1. Load NCT IDs
    print(f"    - Loading NCT IDs from {path_key.name}...")
    df_key = pd.read_csv(path_key, usecols=["nct_id"], dtype={"nct_id": str})
    nct_ids = df_key["nct_id"].values
    n_samples = len(nct_ids)

    # 2. Load and Prepare Embeddings
    def load_embeddings(npy_path, prefix):
        print(f"    - Loading {prefix} embeddings from {npy_path.name}...")
        data = np.load(npy_path)
        if len(data) != n_samples:
            raise ValueError(f"Shape mismatch: {npy_path.name} has {len(data)} rows, expected {n_samples}.")
        
        # Create column names
        cols = [f"{prefix}_{i}" for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=cols, dtype="float32")

    df_crit = load_embeddings(path_crit, "crit")
    df_sci = load_embeddings(path_sci, "sci")
    df_endp = load_embeddings(path_endp, "endp")

    # 3. Concatenate all columns
    print("    - Merging all pillars into a single repository...")
    df_repo = pd.concat([df_key, df_crit, df_sci, df_endp], axis=1)

    # 4. Save to Parquet
    print(f"    - Saving repository to {repo_file}...")
    # Using 'brotli' or 'snappy' for compression. 
    # Index=False because we have nct_id as a column for easy merging.
    df_repo.to_parquet(repo_file, engine="pyarrow", index=False, compression="snappy")

    print("\n>>> Central Repository Created Successfully!")
    print(f"    - Final Shape: {df_repo.shape}")
    print(f"    - File Size: {repo_file.stat().st_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    generate_central_repository()
