"""
Evaluate the detector on a dataset with REAL fraud labels (Sparkov), so no injected anomalies
are needed. A window is "fraud" if any of its 5 transactions is fraudulent ("fraud_last": if the
last one is). Users are split into train / held-out; the autoencoder is trained on a random
subset of training windows (the full data would take hours on a laptop CPU).

    python -m freelance_finance_dl.fraud_eval --csv data/sparkov/fraudTrain.csv data/sparkov/fraudTest.csv

Reports, on held-out users: AUROC, average precision (AP; chance = fraud base rate), and recall
when the top 5% highest-scoring windows are flagged, for the autoencoder and untrained rules.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from freelance_finance_dl.crossval import OUT_DIR, train_model
from freelance_finance_dl.dataloader import FinanceTransactionDataset
from freelance_finance_dl.evaluate import compute_reconstruction_errors


def metrics(name, score, labels, label_name):
    threshold = np.percentile(score, 95)
    return {"score": name, "label": label_name,
            "auroc": roc_auc_score(labels, score),
            "average_precision": average_precision_score(labels, score),
            "recall_at_5pct_flagged": float(np.mean(score[labels] > threshold)),
            "base_rate": float(labels.mean()), "n_windows": len(labels), "n_positive": int(labels.sum())}


def run(csv_files, latent=3, epochs=15, train_windows=150_000, val_frac=0.2, seed=42,
        max_users=None, tag="", features="logratio"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FinanceTransactionDataset(csv_files, clean=True, source="sparkov", features=features,
                                max_users=max_users)
    if ds.samples[0]["fraud"] is None:
        raise SystemExit("No fraud labels found in the data.")
    train_idx, val_idx = ds.get_user_split(val_frac, seed)
    rng = np.random.default_rng(seed)
    subset = rng.choice(train_idx, size=min(train_windows, len(train_idx)), replace=False).tolist()
    print(f"windows: {len(ds.samples):,} | train users' windows {len(train_idx):,} "
          f"(training on {len(subset):,}) | held-out windows {len(val_idx):,}", flush=True)

    model = train_model(ds, subset, latent, epochs, seed, device)
    loader = DataLoader(Subset(ds, val_idx), batch_size=2048)
    scores = {
        "autoencoder (mean error)": compute_reconstruction_errors(model, loader, device, "mean"),
        "autoencoder (max error)": compute_reconstruction_errors(model, loader, device, "max"),
    }
    X = np.stack([ds.featurize(ds.samples[i]["sequence"], ds.samples[i]["scale"], ds.samples[i]["spread"]) for i in val_idx])
    scores["rule: largest |log-ratio|"] = np.abs(X).max(1)
    scores["rule: largest log-ratio"] = X.max(1)
    scores["rule: most negative log-ratio"] = -X.min(1)

    rows = []
    for label_name, key in [("any fraud in window", "fraud"), ("last transaction fraud", "fraud_last")]:
        labels = np.array([ds.samples[i][key] for i in val_idx], dtype=bool)
        if labels.sum() == 0:
            continue
        for name, score in scores.items():
            rows.append(metrics(name, score, labels, label_name))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/fraud_eval_sparkov{tag}.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on real fraud labels")
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--latent", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train-windows", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", choices=["logratio", "robustz"], default="logratio")
    parser.add_argument("--tag", default="", help="Suffix for the output CSV, e.g. _l16_s42")
    parser.add_argument("--max-users", type=int, default=None,
                        help="Use a random subset of users if memory is tight")
    args = parser.parse_args()
    run(args.csv, args.latent, args.epochs, args.train_windows, seed=args.seed,
        max_users=args.max_users, tag=args.tag, features=args.features)
