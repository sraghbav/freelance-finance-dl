"""Loader for the Sparkov-simulated card transactions (Kaggle: kartik2112/fraud-detection)."""

import os

import pandas as pd

REQUIRED = ["trans_date_trans_time", "cc_num", "category", "amt", "is_fraud"]


def load_sparkov(paths):
    """Read one or more CSVs (e.g. fraudTrain.csv and fraudTest.csv) into the standard columns."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; the file has: {list(df.columns)}")
    return pd.DataFrame({
        "transaction_id": df["trans_num"].astype(str) if "trans_num" in df.columns else df.index.astype(str),
        "user_id": df["cc_num"].astype(str),
        "date": pd.to_datetime(df["trans_date_trans_time"], errors="coerce"),
        "transaction_type": "expense",
        "category": df["category"],
        "amount": pd.to_numeric(df["amt"], errors="coerce"),
        "is_fraud": df["is_fraud"].astype(int),
    })
