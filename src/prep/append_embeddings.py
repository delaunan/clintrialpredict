import os
import numpy as np
import pandas as pd
from pathlib import Path

def append_new_embeddings(data_path=None):
    """
    Merges new embeddings (from Colab processing of data_clinpred_emb_transform.csv)
    into the central repository (embeddings.parquet).
    Handles deduplication and source cleanup.
    """
    if data_path is None:
        data_path = Path("data")
    else:
        data_path = Path(data_path)

    processed_path = data_path / "processed"
    repo_file = processed_path / "embeddings.parquet"

    # Define source paths for new data
    path_key = data_path / "data_clinpred_emb_transform.csv"
    path_crit = data_path / "biobert_crit_raw.npy"
    path_sci = data_path / "biobert_sci_raw.npy"
    path_endp = data_path / "biobert_endp_raw.npy"

    delta_files = [path_key, path_crit, path_sci, path_endp]
    exists = [p.exists() for p in delta_files]

    if not any(exists):
        # Silent return if nothing is there to merge
        return

    if not all(exists):
        print(">>> [Incomplete Delta] Some embedding files were found but not all four.")
        print(f"    Found: {[p.name for p in delta_files if p.exists()]}")
        print("    Waiting for all files to be present before merging.")
        return

    print(">>> New Embeddings Detected! Initializing Append process...")

    # 1. Load Delta NCT IDs
    df_key = pd.read_csv(path_key, usecols=["nct_id"], dtype={"nct_id": str})
    nct_ids = df_key["nct_id"].values
    n_samples = len(nct_ids)
    print(f"    - Loaded {n_samples} new NCT IDs.")

    # 2. Load and Prepare New Embeddings
    def load_embeddings(npy_path, prefix):
        data = np.load(npy_path)
        if len(data) != n_samples:
            raise ValueError(f"Shape mismatch: {npy_path.name} has {len(data)} rows, expected {n_samples}.")
        cols = [f"{prefix}_{i}" for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=cols, dtype="float32")

    print("    - Loading new embeddings...")
    df_crit = load_embeddings(path_crit, "crit")
    df_sci = load_embeddings(path_sci, "sci")
    df_endp = load_embeddings(path_endp, "endp")

    # 3. Concatenate new pillars
    df_new = pd.concat([df_key, df_crit, df_sci, df_endp], axis=1)

    # 4. Load Existing Repo
    if repo_file.exists():
        print(f"    - Loading existing repository from {repo_file.name}...")
        df_repo = pd.read_parquet(repo_file, engine="pyarrow")

        # 5. Append and Deduplicate
        print("    - Merging new embeddings and deduplicating...")
        df_updated = pd.concat([df_repo, df_new], ignore_index=True)
        df_updated.drop_duplicates("nct_id", keep="last", inplace=True)
    else:
        print("    - No existing repository found. Creating new one...")
        df_updated = df_new

    # 6. Save Updated Repo
    print(f"    - Saving updated repository to {repo_file}...")
    df_updated.to_parquet(repo_file, engine="pyarrow", index=False, compression="snappy")

    print("\n>>> Success! Repository updated.")
    print(f"    - New total records: {len(df_updated)}")

    # 7. Cleanup (Optional but requested)
    print("    - Cleaning up delta source files...")
    for p in [path_key, path_crit, path_sci, path_endp]:
        p.unlink()
    print("    - Cleanup complete.")

if __name__ == "__main__":
    append_new_embeddings()

