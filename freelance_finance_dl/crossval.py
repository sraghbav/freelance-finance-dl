"""
5-fold user-level cross-validation of the raw-space anomaly evaluation.

Every user appears in exactly one held-out fold. For each fold a fresh model is trained on the
other users for a fixed number of epochs (final weights, no checkpoint selection), the anomaly
threshold is the 95th percentile of the held-out fold's errors (same convention as the main
evaluation), and one amount per held-out sequence is spiked or dropped by each factor.

Usage (one process per configuration, then summarize):
    python -m freelance_finance_dl.crossval --config logratio_l3
    python -m freelance_finance_dl.crossval --config logratio_l16
    python -m freelance_finance_dl.crossval --config minmax_l3
    python -m freelance_finance_dl.crossval --summarize
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from freelance_finance_dl.dataloader import FinanceTransactionDataset
from freelance_finance_dl.evaluate import compute_reconstruction_errors, raw_injection_eval
from freelance_finance_dl.model import TransactionAutoencoder
from freelance_finance_dl.stats import wilson_interval

CONFIGS = {
    "logratio_l3": {"features": "logratio", "latent": 3},
    "logratio_l16": {"features": "logratio", "latent": 16},
    "minmax_l3": {"features": "minmax", "latent": 3},
}
FACTORS = [1.5, 2, 3, 5, 10]
OUT_DIR = "docs/sweeps"


def user_folds(dataset, n_folds, seed):
    users = sorted({s["user_id"] for s in dataset.samples})
    np.random.default_rng(seed).shuffle(users)
    return [set(users[k::n_folds]) for k in range(n_folds)]


def train_model(dataset, train_idx, latent, epochs, seed, device):
    torch.manual_seed(seed)
    model = TransactionAutoencoder(sequence_length=dataset.sequence_length, hidden_dim=64,
                                   latent_dim=latent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    return model


def run_config(name, csv_file, n_folds=5, epochs=50, seed=42):
    cfg = CONFIGS[name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FinanceTransactionDataset(csv_file, clean=True, robust=True,
                                        features=cfg["features"])
    frames = []
    for k, held in enumerate(user_folds(dataset, n_folds, seed)):
        train_idx = [i for i, s in enumerate(dataset.samples) if s["user_id"] not in held]
        val_idx = [i for i, s in enumerate(dataset.samples) if s["user_id"] in held]
        model = train_model(dataset, train_idx, cfg["latent"], epochs, seed + k, device)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64)
        threshold = float(np.percentile(
            compute_reconstruction_errors(model, val_loader, device), 95))
        df = raw_injection_eval(model, dataset, val_idx, threshold, device,
                                factors=FACTORS, seed=seed)
        df.insert(0, "fold", k)
        df.insert(0, "config", name)
        frames.append(df)
        spike10 = df[(df.kind == "spike") & (df.factor == 10)].recall.iloc[0]
        print(f"{name} fold {k}: {len(held)} held-out users, {len(val_idx)} sequences, "
              f"spike x10 recall {spike10:.2f}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.concat(frames).to_csv(f"{OUT_DIR}/crossval_{name}.csv", index=False)


def summarize():
    frames = [pd.read_csv(f"{OUT_DIR}/crossval_{name}.csv") for name in CONFIGS
              if os.path.exists(f"{OUT_DIR}/crossval_{name}.csv")]
    df = pd.concat(frames)
    df["hits"] = (df["recall"] * df["n"]).round().astype(int)
    rows = []
    for (config, kind, factor), g in df.groupby(["config", "kind", "factor"]):
        lo, hi = wilson_interval(int(g["hits"].sum()), int(g["n"].sum()))
        rows.append({"config": config, "kind": kind, "factor": factor,
                     "recall_mean_over_folds": g["recall"].mean(),
                     "recall_sd_over_folds": g["recall"].std(ddof=1),
                     "pooled_recall": g["hits"].sum() / g["n"].sum(),
                     "pooled_ci_low": lo, "pooled_ci_high": hi,
                     "false_positive_rate_mean": g["false_positive_rate"].mean(),
                     "n_total": int(g["n"].sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT_DIR}/crossval_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    labels = {"logratio_l3": "log-ratio, latent 3 (primary)", "logratio_l16": "log-ratio, latent 16",
              "minmax_l3": "min-max, latent 3 (class-project scaling)"}
    for ax, kind in zip(axes, ["spike", "drop"]):
        for config in summary["config"].unique():
            sub = summary[(summary.config == config) & (summary.kind == kind)].sort_values("factor")
            ax.errorbar(sub["factor"], sub["recall_mean_over_folds"],
                        yerr=sub["recall_sd_over_folds"], marker="o", capsize=3,
                        label=labels.get(config, config))
        fpr = summary["false_positive_rate_mean"].mean()
        ax.axhline(fpr, color="gray", linestyle=":", label="false-positive rate (chance)")
        ax.set(xlabel=f"{'Multiplied' if kind == 'spike' else 'Divided'} by factor",
               title=f"{kind.capitalize()} detection", xscale="log", ylim=(0, 1))
        ax.set_xticks(FACTORS)
        ax.set_xticklabels([str(f) for f in FACTORS])
        ax.minorticks_off()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recall (mean ± sd over 5 user folds)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/crossval_comparison.png", dpi=150)
    print(f"\nSaved {OUT_DIR}/crossval_summary.csv and crossval_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User-level cross-validation")
    parser.add_argument("--csv", default="data/budgetwise_finance_dataset.csv")
    parser.add_argument("--config", choices=list(CONFIGS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        run_config(args.config, args.csv, epochs=args.epochs)
