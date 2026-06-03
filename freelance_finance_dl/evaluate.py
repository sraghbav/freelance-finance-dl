"""
Evaluate a trained autoencoder: compute per-sequence reconstruction errors,
set an anomaly threshold from the validation error distribution, and flag anomalies.

Usage:
    python -m freelance_finance_dl.evaluate \
        --csv data/budgetwise_finance_dataset.csv \
        --checkpoint checkpoints/best_model.pt \
        --output results/anomalies.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Subset

from freelance_finance_dl.dataloader import FinanceTransactionDataset
from freelance_finance_dl.model import TransactionAutoencoder


def compute_reconstruction_errors(model, loader, device):
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    errors = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            recon = model(x)
            # Mean MSE over the sequence dimension for each sample
            per_sample = criterion(recon, y).mean(dim=(1, 2))
            errors.extend(per_sample.cpu().numpy().tolist())
    return np.array(errors)


def evaluate(
    csv_file: str,
    checkpoint_path: str,
    output_path: str = "results/anomalies.csv",
    threshold_percentile: float = 95.0,
    sequence_length: int = 5,
    batch_size: int = 64,
    val_split: float = 0.15,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and rebuild model with the same hyperparams
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hp = checkpoint.get("hyperparams", {})
    model = TransactionAutoencoder(
        sequence_length=hp.get("sequence_length", sequence_length),
        hidden_dim=hp.get("hidden_dim", 64),
        latent_dim=hp.get("latent_dim", 16),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')} "
          f"(val_mse={checkpoint.get('val_loss', float('nan')):.6f})")

    seq_len = hp.get("sequence_length", sequence_length)
    dataset = FinanceTransactionDataset(csv_file, sequence_length=seq_len)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_indices, val_indices = random_split(
        range(len(dataset)), [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    # Use validation split to set the threshold, full dataset to score anomalies
    val_loader = DataLoader(Subset(dataset, list(val_indices)), batch_size=batch_size)
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Computing reconstruction errors on validation set for threshold calibration...")
    val_errors = compute_reconstruction_errors(model, val_loader, device)
    threshold = float(np.percentile(val_errors, threshold_percentile))
    print(f"Anomaly threshold ({threshold_percentile}th percentile of val errors): {threshold:.6f}")
    print(f"  Val error — mean: {val_errors.mean():.6f}  std: {val_errors.std():.6f}  "
          f"max: {val_errors.max():.6f}")

    print("Computing reconstruction errors on full dataset...")
    all_errors = compute_reconstruction_errors(model, full_loader, device)

    # Build results dataframe with metadata
    rows = []
    for i in range(len(dataset)):
        meta = dataset.get_sample_metadata(i)
        rows.append({
            "user_id": meta["user_id"],
            "category": meta["category"],
            "sequence": list(meta["sequence"]),
            "reconstruction_mse": all_errors[i],
            "is_anomaly": bool(all_errors[i] > threshold),
        })

    results = pd.DataFrame(rows)
    anomalies = results[results["is_anomaly"]]

    print(f"\nTotal sequences evaluated: {len(results)}")
    print(f"Flagged as anomalies:      {len(anomalies)} "
          f"({100 * len(anomalies) / len(results):.1f}%)")

    print("\nTop 10 highest reconstruction errors:")
    top10 = results.nlargest(10, "reconstruction_mse")[
        ["user_id", "category", "reconstruction_mse", "sequence"]
    ]
    print(top10.to_string(index=False))

    print("\nAnomaly counts by category:")
    print(anomalies.groupby("category").size().sort_values(ascending=False).to_string())

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

    return results, threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate autoencoder for anomaly detection")
    parser.add_argument("--csv", default="data/budgetwise_finance_dataset.csv")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output", default="results/anomalies.csv")
    parser.add_argument("--threshold-percentile", type=float, default=95.0,
                        help="Percentile of val errors to use as anomaly threshold")
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    evaluate(
        csv_file=args.csv,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        threshold_percentile=args.threshold_percentile,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
