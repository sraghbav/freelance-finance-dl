"""
Sensitivity sweep: how large must a synthetic spike/drop be before the model catches it,
for mean-error scoring vs. max-error scoring?

Usage:
    python -m freelance_finance_dl.sweep --checkpoint checkpoints/best_model.pt --tag original
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
    synthetic_anomaly_eval,
)

SPIKES = [1.2, 1.5, 2.0, 3.0]
DROPS = [-0.1, -0.3, -0.5, -1.5]


def run_sweep(csv_file, checkpoint_path, percentile=95.0, val_split=0.15, seed=42):
    model, dataset, _, val_indices, device = load_model_and_data(
        csv_file, checkpoint_path, val_split, seed
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=64)

    rows = []
    for reduce in ["mean", "max"]:
        val_errors = compute_reconstruction_errors(model, val_loader, device, reduce)
        threshold = float(np.percentile(val_errors, percentile))
        for spike, drop in zip(SPIKES, DROPS):
            summary, _ = synthetic_anomaly_eval(
                model, dataset, val_indices, threshold, device, seed=seed,
                spike_val=spike, drop_val=drop, reduce=reduce, verbose=False,
            )
            rows.append({
                "score": reduce,
                "threshold": threshold,
                "spike_val": spike,
                "drop_val": drop,
                "spike_recall": summary["spike_recall"],
                "drop_recall": summary["drop_recall"],
                "false_positive_rate": summary["false_positive_rate"],
            })
    return pd.DataFrame(rows)


def plot_sweep(df, out_path, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for score, style in [("mean", "o-"), ("max", "s--")]:
        sub = df[df["score"] == score]
        axes[0].plot(sub["spike_val"], sub["spike_recall"], style, label=f"{score} error")
        axes[1].plot(sub["drop_val"], sub["drop_recall"], style, label=f"{score} error")
    axes[0].set(xlabel="Injected spike value (normalized units)", ylabel="Recall",
                title="Spike detection" + title_suffix, ylim=(0, 1.05))
    axes[1].set(xlabel="Injected drop value (normalized units)", ylabel="Recall",
                title="Drop detection" + title_suffix, ylim=(0, 1.05))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Injection magnitude sensitivity sweep")
    parser.add_argument("--csv", default="data/budgetwise_finance_dataset.csv")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--tag", default="original", help="Suffix for output filenames")
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    args = parser.parse_args()

    df = run_sweep(args.csv, args.checkpoint, args.threshold_percentile)
    print(df.round(3).to_string(index=False))
    df.to_csv(f"docs/sweep_{args.tag}.csv", index=False)
    plot_sweep(df, f"docs/sensitivity_{args.tag}.png", title_suffix=f" ({args.tag})")
    print(f"\nSaved docs/sweep_{args.tag}.csv and docs/sensitivity_{args.tag}.png")
