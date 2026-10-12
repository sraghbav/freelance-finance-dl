"""
End-to-end pipeline: transactions -> anomaly flags -> plain-language explanations -> tax note.

Usage:
    python -m freelance_finance_dl.pipeline --checkpoint checkpoints/best_model.pt --top-n 10
    python -m freelance_finance_dl.pipeline --llm ollama      # free, local
    python -m freelance_finance_dl.pipeline --llm anthropic   # needs ANTHROPIC_API_KEY
"""

import argparse
import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from freelance_finance_dl.evaluate import compute_reconstruction_errors, load_model_and_data
from freelance_finance_dl.explain import (
    build_facts, llm_explanation, ollama_explanation, template_explanation)
from freelance_finance_dl.tax import tax_note


def run(csv_file, checkpoint_path, top_n=10, backend=None, score="mean",
        percentile=95.0, seed=42):
    model, dataset, _, val_indices, device = load_model_and_data(
        csv_file, checkpoint_path, seed=seed
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=64)
    full_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    threshold = float(np.percentile(
        compute_reconstruction_errors(model, val_loader, device, score), percentile))
    errors = compute_reconstruction_errors(model, full_loader, device, score)

    flagged = [int(i) for i in np.argsort(-errors) if errors[i] > threshold][:top_n]

    client = None
    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()

    rows = []
    for idx in flagged:
        facts = build_facts(dataset, model, device, idx, errors[idx], threshold)
        row = {"facts": facts, "template": template_explanation(facts)}
        if backend == "anthropic":
            row["llm"] = llm_explanation(facts, client)
        elif backend == "ollama":
            row["llm"] = ollama_explanation(facts)
        rows.append(row)
    return rows, dataset, threshold, int((errors > threshold).sum()), len(errors)


def to_markdown(rows, dataset, threshold, n_flagged, n_total):
    lines = [
        "# Anomaly report",
        f"\nThreshold {threshold:.4f}; {n_flagged}/{n_total} sequences flagged; "
        f"showing top {len(rows)}.\n",
    ]
    seen_users = []
    for i, r in enumerate(rows, 1):
        f = r["facts"]
        lines.append(f"## {i}. {f['user_id']} - {f['category']}")
        lines.append(f"Amounts: {f['amounts_in_window']}\n")
        lines.append(f"**Template:** {r['template']}\n")
        if "llm" in r:
            lines.append(f"**LLM:** {r['llm']}\n")
        if f["user_id"] not in seen_users:
            seen_users.append(f["user_id"])
    lines.append("## Tax set-aside notes (educational only)")
    for user in seen_users:
        lines.append(f"- {tax_note(dataset.data, user)}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly -> explanation -> tax note pipeline")
    parser.add_argument("--csv", default="data/budgetwise_finance_dataset.csv")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--score", choices=["mean", "max"], default="mean")
    parser.add_argument("--llm", choices=["anthropic", "ollama"], default=None,
                        help="Also generate LLM explanations (ollama is free and local)")
    parser.add_argument("--output", default="results/report.md")
    args = parser.parse_args()

    rows, dataset, threshold, n_flagged, n_total = run(
        args.csv, args.checkpoint, args.top_n, args.llm, args.score)
    report = to_markdown(rows, dataset, threshold, n_flagged, n_total)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        fh.write(report)
    print(report)
    print(f"Saved to {args.output}")
