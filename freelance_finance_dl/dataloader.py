import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Maps every observed spelling/casing variant to a canonical category name.
CATEGORY_MAP = {
    # Food
    "food": "food", "foods": "food", "fod": "food", "foodd": "food", "FOOD": "food",
    "Food": "food",
    # Rent
    "rent": "rent", "RENT": "rent", "Rent": "rent", "rentt": "rent", "rnt": "rent",
    "Rentt": "rent", "Rnt": "rent",
    # Travel
    "travel": "travel", "TRAVEL": "travel", "Travel": "travel",
    "traval": "travel", "travl": "travel", "Traval": "travel", "Travl": "travel",
    # Entertainment
    "entertainment": "entertainment", "Entertainment": "entertainment",
    "Entertain": "entertainment", "Entrtnmnt": "entertainment",
    # Utilities
    "utilities": "utilities", "Utilities": "utilities", "Utility": "utilities",
    "Utilties": "utilities", "Utlities": "utilities", "utility": "utilities",
    "utilties": "utilities", "utlities": "utilities",
    # Education
    "education": "education", "Education": "education",
    "Educaton": "education", "EDU": "education",
    # Health
    "health": "health", "Health": "health", "HEALTH": "health", "Helth": "health",
    # Savings
    "savings": "savings", "Savings": "savings", "SAVINGS": "savings", "Saving": "savings",
    # Income categories
    "Freelance": "freelance", "freelance": "freelance",
    "Salary": "salary", "salary": "salary",
    "Bonus": "bonus", "bonus": "bonus",
    "Investment": "investment", "investment": "investment",
    # Miscellaneous
    "Others": "other", "others": "other", "OTHERS": "other",
    "Other": "other", "other": "other", "Misc": "other", "misc": "other",
}


class FinanceTransactionDataset(Dataset):
    def __init__(self, csv_file, sequence_length=5):
        self.sequence_length = sequence_length
        self.data = pd.read_csv(csv_file)

        # Normalize category spellings/casing to canonical names.
        # Lowercase first so the map only needs one entry per variant.
        lowered = self.data["category"].str.strip().str.lower()
        lowercase_map = {k.lower(): v for k, v in CATEGORY_MAP.items()}
        self.data["category"] = lowered.map(lowercase_map).fillna(lowered)

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

        seq = sample["sequence"].astype("float32")

        # Min-max normalize each sequence independently so all inputs are in [0, 1].
        # Avoids the model being dominated by high-value categories (e.g. rent vs coffee).
        seq_min, seq_max = seq.min(), seq.max()
        if seq_max - seq_min > 0:
            seq = (seq - seq_min) / (seq_max - seq_min)

        sequence = torch.tensor(seq, dtype=torch.float32).unsqueeze(-1)

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
