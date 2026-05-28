import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class FinanceTransactionDataset(Dataset):
    def __init__(self, csv_file, sequence_length=5):
        self.sequence_length = sequence_length
        self.data = pd.read_csv(csv_file)

        self.data["date"] = pd.to_datetime(self.data["date"], errors="coerce")

        self.data["amount"] = (
            self.data["amount"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )

        self.data["amount"] = pd.to_numeric(self.data["amount"], errors="coerce")

        self.data = self.data.dropna(
            subset=["user_id", "date", "transaction_type", "category", "amount"]
        )

        self.data = self.data.sort_values(["user_id", "date"]).reset_index(drop=True)

        self.data["transaction_type_id"] = (
            self.data["transaction_type"].astype("category").cat.codes
        )

        self.data["category_id"] = self.data["category"].astype("category").cat.codes

        if "payment_mode" in self.data.columns:
            self.data["payment_mode_id"] = (
                self.data["payment_mode"].astype("category").cat.codes
            )
        else:
            self.data["payment_mode_id"] = 0

        self.feature_cols = [
            "transaction_type_id",
            "category_id",
            "payment_mode_id",
            "amount",
        ]

        self.samples = self._create_sequences()

    def _create_sequences(self):
        samples = []

        for user_id, user_df in self.data.groupby("user_id"):
            user_df = user_df.sort_values("date").reset_index(drop=True)

            if len(user_df) <= self.sequence_length:
                continue

            features = user_df[self.feature_cols].values.astype(float)
            targets = user_df["amount"].values.astype(float)

            for i in range(len(user_df) - self.sequence_length):
                x = features[i : i + self.sequence_length]
                y = targets[i + self.sequence_length]
                samples.append((x, y))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, target = self.samples[idx]

        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return features, target


def get_data_loader(csv_file, batch_size=32, sequence_length=5, shuffle=True):
    dataset = FinanceTransactionDataset(
        csv_file=csv_file,
        sequence_length=sequence_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
