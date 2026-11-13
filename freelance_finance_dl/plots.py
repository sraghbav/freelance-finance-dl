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


def detector_vs_rules_figure():
    """Cross-validated recall of the autoencoders against untrained one-line rules."""
    cv = pd.read_csv("docs/sweeps/crossval_summary.csv")
    cv = cv[cv.stream == "all"]
    rules = pd.read_csv("docs/sweeps/baseline_rule_summary.csv")
    rules = rules[rules.stream == "all"]
    factors = [1.5, 2, 3, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, kind, oracle in zip(axes, ["spike", "drop"], ["spike_rule", "drop_rule"]):
        for config, label, style in [("logratio_l3", "autoencoder, latent 3 (primary)", "o-"),
                                     ("logratio_l16", "autoencoder, latent 16", "o--"),
                                     ("minmax_l3", "autoencoder, min-max scaling", "o:")]:
            sub = cv[(cv.config == config) & (cv.kind == kind)].sort_values("factor")
            ax.errorbar(sub.factor, sub.recall_mean_over_folds, yerr=sub.recall_sd_over_folds,
                        fmt=style, capsize=3, label=label)
        for rule, label, style in [("two_sided", "rule: largest |log-ratio| (direction unknown)", "s-"),
                                   (oracle, f"rule: {'largest' if kind == 'spike' else 'most negative'} "
                                            "log-ratio (direction known)", "s--")]:
            sub = rules[(rules.rule == rule) & (rules.kind == kind)].sort_values("factor")
            ax.errorbar(sub.factor, sub.recall_mean, yerr=sub.recall_sd, fmt=style, capsize=3,
                        color="black" if rule == "two_sided" else "dimgray", label=label)
        ax.axhline(0.05, color="gray", linestyle=":")
        ax.set(xlabel=f"{'Multiplied' if kind == 'spike' else 'Divided'} by factor",
               title=f"{kind.capitalize()} detection at ~5% false positives", xscale="log", ylim=(0, 1))
        ax.set_xticks(factors)
        ax.set_xticklabels([str(f) for f in factors])
        ax.minorticks_off()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recall (mean ± sd over 5 user folds)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/detector_vs_rules.png", dpi=150)
    plt.close(fig)


def detector_by_stream_figure():
    """Expense vs income: autoencoders and untrained rules, cross-validated."""
    cv = pd.read_csv("docs/sweeps/crossval_summary.csv")
    rules = pd.read_csv("docs/sweeps/baseline_rule_summary.csv")
    factors = [1.5, 2, 3, 5, 10]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    for r, stream in enumerate(["expense", "income"]):
        for c, (kind, oracle) in enumerate([("spike", "spike_rule"), ("drop", "drop_rule")]):
            ax = axes[r, c]
            for config, label, style in [("logratio_l3", "autoencoder, latent 3", "o-"),
                                         ("logratio_l16", "autoencoder, latent 16", "o--")]:
                sub = cv[(cv.config == config) & (cv.stream == stream) & (cv.kind == kind)].sort_values("factor")
                ax.errorbar(sub.factor, sub.recall_mean_over_folds, yerr=sub.recall_sd_over_folds,
                            fmt=style, capsize=3, label=label)
            for rule, label, style, color in [("two_sided", "rule: largest |log-ratio|", "s-", "black"),
                                              (oracle, f"rule: {'largest' if kind == 'spike' else 'most negative'} log-ratio (direction known)", "s--", "dimgray")]:
                sub = rules[(rules.rule == rule) & (rules.stream == stream) & (rules.kind == kind)].sort_values("factor")
                ax.errorbar(sub.factor, sub.recall_mean, yerr=sub.recall_sd, fmt=style, capsize=3,
                            color=color, label=label)
            ax.axhline(0.05, color="gray", linestyle=":")
            ax.set(title=f"{stream.capitalize()}: {kind} detection", xscale="log", ylim=(0, 1),
                   xlabel=f"{'Multiplied' if kind == 'spike' else 'Divided'} by factor")
            ax.set_xticks(factors)
            ax.set_xticklabels([str(f) for f in factors])
            ax.minorticks_off()
            ax.grid(alpha=0.3)
        axes[r, 0].set_ylabel("Recall (mean ± sd, 5 user folds)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/detector_by_stream.png", dpi=150)
    plt.close(fig)


def berka_figure():
    """Berka: injected-anomaly recall by stream, autoencoders vs rules vs Isolation Forest."""
    va = pd.read_csv("docs/sweeps/cv_berka_robustz_l16_summary.csv")
    lr = pd.read_csv("docs/sweeps/cv_berka_logratio_l16_summary.csv")
    factors = [1.5, 2, 3, 5, 10]
    lines = [(va, "autoencoder", "autoencoder, variability-aware", "o-", "tab:blue"),
             (lr, "autoencoder", "autoencoder, log-ratio", "o--", "tab:cyan"),
             (va, "isolation forest", "isolation forest, variability-aware", "^-", "tab:green"),
             (va, "rule |z| (direction unknown)", "rule |z|, variability-aware", "s-", "black"),
             (va, "SPIKE_OR_DROP", "rule (direction known), variability-aware", "s--", "dimgray"),
             (lr, "SPIKE_OR_DROP", "rule (direction known), log-ratio", "s:", "tab:red")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    for r, stream in enumerate(["expense", "income"]):
        for c, kind in enumerate(["spike", "drop"]):
            ax = axes[r, c]
            for df, model, label, style, color in lines:
                m = model if model != "SPIKE_OR_DROP" else (
                    "rule max z (spike known)" if kind == "spike" else "rule min z (drop known)")
                sub = df[(df.stream == stream) & (df.model == m) & (df.kind == kind)].sort_values("factor")
                ax.errorbar(sub.factor, sub.recall_mean, yerr=sub.recall_sd, fmt=style, capsize=2,
                            color=color, label=label, markersize=4)
            ax.axhline(0.05, color="gray", linestyle=":")
            ax.set(title=f"{stream.capitalize()}: {kind} detection", xscale="log", ylim=(0, 1),
                   xlabel=f"{'Multiplied' if kind == 'spike' else 'Divided'} by factor")
            ax.set_xticks(factors)
            ax.set_xticklabels([str(f) for f in factors])
            ax.minorticks_off()
            ax.grid(alpha=0.3)
        axes[r, 0].set_ylabel("Recall (mean ± sd, 5 account folds)")
    axes[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/berka_detection.png", dpi=150)
    plt.close(fig)


def simulation_figure():
    """Simulation study: detection quality by habit level (3 seeds)."""
    d = pd.read_csv("docs/sweeps/sim_summary.csv")
    d = d[d.group == "overall"]
    levels = ["strong", "default", "weak", "none"]
    names = [("autoencoder (robustz, latent 16)", "autoencoder, variability-aware", "o-"),
             ("autoencoder (logratio, latent 16)", "autoencoder, log-ratio", "o--"),
             ("isolation forest (robustz)", "isolation forest", "^-"),
             ("rule |z| (robustz)", "rule |z|, variability-aware", "s-"),
             ("rule |z| (logratio)", "rule |z|, log-ratio", "s--")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (col, title) in zip(axes, [("average_precision", "Average precision"), ("auroc", "AUROC")]):
        for key, label, style in names:
            sub = d[d.score == key].set_index("level").loc[levels]
            ax.errorbar(range(4), sub[f"{col}_mean"], yerr=sub[f"{col}_std"], fmt=style, capsize=3, label=label)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["strong habits", "default", "weak", "no habits"])
        ax.set(title=f"{title} on labeled anomalies (simulated data)", ylim=(0, 1))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean ± sd over 3 data seeds")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/simulation_levels.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    data_loss_figure()
    flagged_examples_figure()
    detector_vs_rules_figure()
    detector_by_stream_figure()
    berka_figure()
    simulation_figure()
    print(f"Saved figures to {OUT_DIR}/")
