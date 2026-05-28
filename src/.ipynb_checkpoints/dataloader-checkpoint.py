import pandas as pd
import torch
from torch.utils.data import Dataset


class FinanceTransactionDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

        # Basic cleaning
        self.data["date"] = pd.to_datetime(self.data["date"], errors="coerce")

        self.data["amount"] = (
            self.data["amount"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )

        self.data["amount"] = pd.to_numeric(self.data["amount"], errors="coerce")

        self.data = self.data.dropna(
            subset=[
                "user_id",
                "date",
                "transaction_type",
                "category",
                "amount",
            ]
        )

        # Sort by user and time
        self.data = self.data.sort_values(["user_id", "date"]).reset_index(drop=True)

        # Encode categorical columns
        self.data["transaction_type_id"] = (
            self.data["transaction_type"].astype("category").cat.codes
        )

        self.data["category_id"] = (
            self.data["category"].astype("category").cat.codes
        )

        if "payment_mode" in self.data.columns:
            self.data["payment_mode_id"] = (
                self.data["payment_mode"].astype("category").cat.codes
            )
        else:
            self.data["payment_mode_id"] = 0

        # Create target: next transaction amount for same user
        self.data["target_next_amount"] = (
            self.data.groupby("user_id")["amount"].shift(-1)
        )

        self.data = self.data.dropna(subset=["target_next_amount"]).reset_index(drop=True)

        self.feature_cols = [
            "transaction_type_id",
            "category_id",
            "payment_mode_id",
            "amount",
        ]

        self.target_col = "target_next_amount"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        features = torch.tensor(
            row[self.feature_cols].values.astype(float),
            dtype=torch.float32,
        )

        target = torch.tensor(
            row[self.target_col],
            dtype=torch.float32,
        )

        return features, target

    def get_feature_columns(self):
        return self.feature_cols

    def get_target_column(self):
        return self.target_col
