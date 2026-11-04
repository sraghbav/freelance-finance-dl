"""
Thesis figures.

    python -m freelance_finance_dl.plots
writes docs/figures/data_loss.png and docs/figures/primary_flagged_examples.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from freelance_finance_dl.dataloader import FinanceTransactionDataset
from freelance_finance_dl.explain import position_errors
from freelance_finance_dl.pipeline import score_and_flag

CSV = "data/budgetwise_finance_dataset.csv"
PRIMARY = "checkpoints/primary_logratio_l3/best_model.pt"
OUT_DIR = "docs/figures"


def data_loss_figure():
    raw_rows = len(pd.read_csv(CSV))
    counts = {
        "Original loader\n(ISO dates, plain numbers)": len(FinanceTransactionDataset(CSV, robust=False).data),
        "Robust parsing\n+ duplicates removed": len(FinanceTransactionDataset(CSV).data),
        "Robust parsing\n+ corrupt amounts dropped": len(FinanceTransactionDataset(CSV, clean=True).data),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(list(counts), list(counts.values()), color=["#c0504d", "#4f81bd", "#4f81bd"])
    ax.axhline(raw_rows, color="gray", linestyle="--")
    ax.text(len(counts) - 0.5, raw_rows * 1.01, f"{raw_rows:,} raw rows", ha="right", fontsize=9)
    for bar, value in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}\n({value / raw_rows:.0%})",
                ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Usable rows", ylim=(0, raw_rows * 1.15))
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/data_loss.png", dpi=150)
    plt.close(fig)


def flagged_examples_figure(top_n=6):
    model, dataset, device, errors, threshold, flagged = score_and_flag(CSV, PRIMARY)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, idx in zip(axes.ravel(), flagged[:top_n]):
        x, _ = dataset[idx]
        with torch.no_grad():
            recon = model(x.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        x = x.squeeze().numpy()
        pos = int(position_errors(model, dataset, idx, device).argmax())
        sample = dataset.samples[idx]
        positions = np.arange(1, len(x) + 1)
        ax.plot(positions, x, "o-", color="#4f81bd", label="actual")
        ax.plot(positions, recon, "s--", color="#e07b39", label="model's expectation")
        ax.fill_between(positions, x, recon, color="red", alpha=0.15)
        amount = sample["sequence"][pos]
        ax.set_title(f"{sample['user_id']}, {sample['category']}: {amount:,.0f} vs typical "
                     f"{sample['scale']:,.0f}", fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_xticks(positions)
    for ax in axes[:, 0]:
        ax.set_ylabel("log(amount / typical)")
    for ax in axes[-1]:
        ax.set_xlabel("Position in window")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Six highest-scoring flagged sequences (primary model)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/primary_flagged_examples.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    data_loss_figure()
    flagged_examples_figure()
    print(f"Saved figures to {OUT_DIR}/")
