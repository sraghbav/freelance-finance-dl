import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class FinanceTransactionDataset(Dataset):
    def __init__(self, csv_file, sequence_length=5):
        self.sequence_length = sequence_length
        self.data = pd.read_csv(csv_file)

        # Clean date column
        self.data["date"] = pd.to_datetime(self.data["date"], errors="coerce")

        # Clean amount column
        self.data["amount"] = (
            self.data["amount"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )

        self.data["amount"] = pd.to_numeric(self.data["amount"], errors="coerce")

        # Keep only usable rows
        self.data = self.data.dropna(
            subset=["user_id", "date", "category", "amount"]
        )

        # Sort by user, category, and time
        self.data = self.data.sort_values(
            ["user_id", "category", "date"]
        ).reset_index(drop=True)

        # Create sequences
        self.samples = self._create_sequences()

    def _create_sequences(self):
        samples = []

        grouped = self.data.groupby(["user_id", "category"])

        for (user_id, category), group_df in grouped:
            group_df = group_df.sort_values("date").reset_index(drop=True)

            if len(group_df) < self.sequence_length:
                continue

            amounts = group_df["amount"].values.astype(float)

            for i in range(len(amounts) - self.sequence_length + 1):
                sequence = amounts[i : i + self.sequence_length]
                samples.append(
                    {
                        "sequence": sequence,
                        "user_id": user_id,
                        "category": category,
                    }
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        sequence = torch.tensor(
            sample["sequence"],
            dtype=torch.float32,
        ).unsqueeze(-1)

        # Autoencoder input and target are the same
        return sequence, sequence

    def get_sample_metadata(self, idx):
        sample = self.samples[idx]

        return {
            "user_id": sample["user_id"],
            "category": sample["category"],
            "sequence": sample["sequence"],
        }


def get_data_loader(csv_file, batch_size=32, sequence_length=5, shuffle=True):
    dataset = FinanceTransactionDataset(
        csv_file=csv_file,
        sequence_length=sequence_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return loader
