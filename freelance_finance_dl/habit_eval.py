"""
Does structure help? Evaluate the autoencoder and untrained rules on SYNTHETIC data with habits
(habitsim.py), by category, with the same raw-amount injection used elsewhere.

One threshold is used for all categories (the 95th percentile of all held-out clean scores), as a
deployed detector would. With "logratio" features that threshold is dominated by volatile
categories (groceries, dining), so stable ones (rent, subscription) are under-detected;
"robustz" features divide by each history's own variability.

    python -m freelance_finance_dl.habit_eval --features robustz --latent 3 --tag _robustz_l3
"""

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from freelance_finance_dl.crossval import OUT_DIR, train_model
from freelance_finance_dl.dataloader import FinanceTransactionDataset
from freelance_finance_dl.evaluate import compute_reconstruction_errors, raw_injection_eval

FACTORS = [1.2, 1.5, 2, 3, 5, 10]
RULES = {
    "rule |z| (direction unknown)": lambda f: np.abs(f).max(1),
    "rule max z (spike known)": lambda f: f.max(1),
    "rule min z (drop known)": lambda f: -f.min(1),
}


def run(csv_file, features, latent, epochs, train_windows, seed=42, tag=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FinanceTransactionDataset(csv_file, clean=True, features=features)
    train_idx, val_idx = ds.get_user_split(0.2, seed)
    rng = np.random.default_rng(seed)
    subset = rng.choice(train_idx, size=min(train_windows, len(train_idx)), replace=False).tolist()
    print(f"{features} latent {latent}: {len(ds.samples):,} windows, training on {len(subset):,}, "
          f"{len(val_idx):,} held-out", flush=True)
    model = train_model(ds, subset, latent, epochs, seed, device)

    val_loader = DataLoader(Subset(ds, val_idx), batch_size=2048)
    threshold = float(np.percentile(compute_reconstruction_errors(model, val_loader, device), 95))
    categories = sorted({ds.samples[i]["category"] for i in val_idx})
    by_cat = {c: [i for i in val_idx if ds.samples[i]["category"] == c] for c in categories}

    rows = []
    for c, idx in by_cat.items():
        df = raw_injection_eval(model, ds, idx, threshold, device, factors=FACTORS, seed=seed)
        for r in df.itertuples():
            rows.append({"model": "autoencoder", "category": c, "kind": r.kind, "factor": r.factor,
                         "recall": r.recall, "false_positive_rate": r.false_positive_rate, "n": r.n})

    feat = lambda s, raw=None: ds.featurize(s["sequence"] if raw is None else raw, s["scale"], s["spread"])
    clean = {c: np.stack([feat(ds.samples[i]) for i in idx]) for c, idx in by_cat.items()}
    for name, score in RULES.items():
        thr = np.percentile(np.concatenate([score(x) for x in clean.values()]), 95)
        for c, idx in by_cat.items():
            samples = [ds.samples[i] for i in idx]
            positions = np.random.default_rng(seed).integers(0, ds.sequence_length, size=len(samples))
            fpr = float(np.mean(score(clean[c]) > thr))
            for factor in FACTORS:
                for kind in ("spike", "drop"):
                    X = []
                    for s, p in zip(samples, positions):
                        raw = s["sequence"].copy()
                        raw[p] = raw[p] * factor if kind == "spike" else raw[p] / factor
                        X.append(feat(s, raw))
                    rows.append({"model": name, "category": c, "kind": kind, "factor": factor,
                                 "recall": float(np.mean(score(np.stack(X)) > thr)),
                                 "false_positive_rate": fpr, "n": len(samples)})
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/habit_eval{tag}.csv", index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on synthetic data with habits")
    parser.add_argument("--csv", default="data/habitsim.csv")
    parser.add_argument("--features", choices=["logratio", "robustz"], default="robustz")
    parser.add_argument("--latent", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train-windows", type=int, default=30_000)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    run(args.csv, args.features, args.latent, args.epochs, args.train_windows, tag=args.tag)
