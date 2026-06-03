import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from freelance_finance_dl.dataloader import FinanceTransactionDataset
from freelance_finance_dl.model import TransactionAutoencoder


def train(
    csv_file: str,
    output_dir: str = "checkpoints",
    sequence_length: int = 5,
    hidden_dim: int = 64,
    latent_dim: int = 16,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.15,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = FinanceTransactionDataset(csv_file, sequence_length=sequence_length)
    print(f"Total sequences: {len(dataset)}")

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TransactionAutoencoder(
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                recon = model(x)
                val_loss += criterion(recon, y).item() * len(x)
        val_loss /= n_val

        print(f"Epoch {epoch:3d}/{epochs}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "hyperparams": {
                        "sequence_length": sequence_length,
                        "hidden_dim": hidden_dim,
                        "latent_dim": latent_dim,
                    },
                },
                checkpoint_path,
            )
            print(f"  -> Saved best model (val_mse={best_val_loss:.6f})")

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.6f}")
    print(f"Best model saved to: {os.path.join(output_dir, 'best_model.pt')}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transaction autoencoder")
    parser.add_argument("--csv", default="data/budgetwise_finance_dataset.csv")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    train(
        csv_file=args.csv,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
    )
