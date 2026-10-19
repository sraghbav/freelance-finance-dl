"""
Sensitivity sweep in RAW amount space: how large must a single charge change be before the
model flags the sequence? One amount is multiplied (spike) or divided (drop) by a factor, then
the sequence is featurized exactly like real data, so it stays in the training range.

Usage:
    python -m freelance_finance_dl.sweep --checkpoint checkpoints_logratio_l3/best_model.pt --tag primary
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from freelance_finance_dl.evaluate import (
    compute_reconstruction_errors,
    load_model_and_data,
    raw_injection_eval,
)

FACTORS = [1.5, 2, 3, 5, 10]


def run_sweep(csv_file, checkpoint_path, percentile=95.0, val_split=0.15, seed=42):
    model, dataset, _, val_indices, device = load_model_and_data(
        csv_file, checkpoint_path, val_split, seed
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=64)

    frames = []
    for reduce in ["mean", "max"]:
        val_errors = compute_reconstruction_errors(model, val_loader, device, reduce)
        threshold = float(np.percentile(val_errors, percentile))
        df = raw_injection_eval(model, dataset, val_indices, threshold, device,
                                factors=FACTORS, seed=seed, reduce=reduce)
        df.insert(0, "score", reduce)
        df.insert(1, "threshold", threshold)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_sweep(df, out_path, title=""):
    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {("spike", "mean"): "o-", ("spike", "max"): "o--",
              ("drop", "mean"): "s-", ("drop", "max"): "s--"}
    for (kind, score), style in styles.items():
        sub = df[(df["kind"] == kind) & (df["score"] == score)]
        ax.plot(sub["factor"], sub["recall"], style, label=f"{kind}, {score} score")
    ax.axhline(df["false_positive_rate"].iloc[0], color="gray", linestyle=":",
               label="false-positive rate (chance)")
    ax.set(xlabel="Change factor applied to one amount (x or /)", ylabel="Recall",
           title=title, ylim=(0, 1.05), xscale="log")
    ax.set_xticks(FACTORS)
    ax.set_xticklabels([str(f) for f in FACTORS])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raw-space injection sensitivity sweep")
    parser.add_argument("--csv", default="data/budgetwise_finance_dataset.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True, help="Suffix for output filenames")
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    args = parser.parse_args()

    df = run_sweep(args.csv, args.checkpoint, args.threshold_percentile)
    print(df.round(3).to_string(index=False))
    df.to_csv(f"docs/rawsweep_{args.tag}.csv", index=False)
    plot_sweep(df, f"docs/rawsweep_{args.tag}.png", title=args.tag)
    print(f"\nSaved docs/rawsweep_{args.tag}.csv and docs/rawsweep_{args.tag}.png")
