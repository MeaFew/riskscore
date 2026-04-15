"""Merge auxiliary aggregated features into train/test feature matrices.

Reads features_train.csv / features_test.csv and auxiliary_features.csv,
merges on SK_ID_CURR, fills missing auxiliary features with 0,
and overwrites the original files.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR


def merge_aux_features():
    train_path = PROCESSED_DATA_DIR / "features_train.csv"
    test_path = PROCESSED_DATA_DIR / "features_test.csv"
    aux_path = PROCESSED_DATA_DIR / "auxiliary_features.csv"

    print("Loading train features ...")
    train_df = pd.read_csv(train_path)
    print(f"  Train: {len(train_df):,} rows × {len(train_df.columns)} cols")

    print("Loading test features ...")
    test_df = pd.read_csv(test_path)
    print(f"  Test: {len(test_df):,} rows × {len(test_df.columns)} cols")

    print("Loading auxiliary features ...")
    aux_df = pd.read_csv(aux_path)
    print(f"  Aux: {len(aux_df):,} rows × {len(aux_df.columns)} cols")

    # Merge
    print("Merging auxiliary features into train ...")
    train_merged = train_df.merge(aux_df, on="SK_ID_CURR", how="left")
    train_merged = train_merged.fillna(0)

    print("Merging auxiliary features into test ...")
    test_merged = test_df.merge(aux_df, on="SK_ID_CURR", how="left")
    test_merged = test_merged.fillna(0)

    # Save atomically via temp files to avoid corruption on mid-write failure
    import os
    import tempfile
    for path, df in [(train_path, train_merged), (test_path, test_merged)]:
        fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="features_", dir=path.parent)
        os.close(fd)
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    print(f"\nDone. Train now: {len(train_merged):,} rows × {len(train_merged.columns)} cols")
    print(f"      Test now: {len(test_merged):,} rows × {len(test_merged.columns)} cols")
    print(f"      Added {len(aux_df.columns) - 1} auxiliary feature columns")


if __name__ == "__main__":
    merge_aux_features()
