import numpy as np
import pandas as pd
import torch
import random
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

# The raw CSV contains sentinel/corrupt amounts (999999, 999999999, and values <= 0).
INVALID_AMOUNT_CUTOFF = 999_999

# Ratios are clipped before the log so zero/placeholder amounts stay finite.
LOG_RATIO_CLIP = (1e-3, 1e3)

# The raw dates use several formats. Each is internally consistent: slash dates are always
# month/day/year (no first field above 12), dash dates are always day-month.
DATE_FORMATS = [
    (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
    (r"\d{1,2}/\d{1,2}/\d{4}", "%m/%d/%Y"),
    (r"\d{1,2}-\d{1,2}-\d{4}", "%d-%m-%Y"),
    (r"\d{1,2}-\d{1,2}-\d{2}", "%d-%m-%y"),
]


def parse_dates(series):
    text = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for pattern, fmt in DATE_FORMATS:
        mask = text.str.fullmatch(pattern).fillna(False).astype(bool)
        parsed[mask] = pd.to_datetime(text[mask], format=fmt, errors="coerce")
    return parsed


def parse_amounts(series):
    """Strip currency text ('Rs.828', '5070 INR', the rupee sign, '$', commas) before parsing."""
    text = series.astype(str).str.strip().str.replace(
        r"(?i)rs\.?|inr|\u20b9|\$|,|\s", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def normalize_sequence(seq):
    """Per-sequence min-max to [0, 1]; a constant sequence is returned unchanged."""
    seq = np.asarray(seq, dtype="float32")
    low, high = seq.min(), seq.max()
    return (seq - low) / (high - low) if high - low > 0 else seq


class FinanceTransactionDataset(Dataset):
    def __init__(self, csv_file, sequence_length=5, clean=False, robust=True,
                 features="logratio"):
        self.sequence_length = sequence_length
        self.features = features
        self.clean = clean
        self.robust = robust
        self.data = pd.read_csv(csv_file)

        # Normalize category spellings/casing to canonical names.
        # Lowercase first so the map only needs one entry per variant.
        lowered = self.data["category"].str.strip().str.lower()
        lowercase_map = {k.lower(): v for k, v in CATEGORY_MAP.items()}
        self.data["category"] = lowered.map(lowercase_map).fillna(lowered)

        if robust:
            self.data["date"] = parse_dates(self.data["date"])
            self.data["amount"] = parse_amounts(self.data["amount"])
        else:
            # Legacy parsing (the original class project): ISO dates and plain numbers only.
            self.data["date"] = pd.to_datetime(self.data["date"], errors="coerce")
            self.data["amount"] = pd.to_numeric(
                self.data["amount"].astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False),
                errors="coerce",
            )

        # Keep only usable rows
        self.data = self.data.dropna(
            subset=["user_id", "date", "category", "amount"]
        )

        if robust:
            self.data = self.data.drop_duplicates(
                subset=["transaction_id", "user_id", "date", "transaction_type",
                        "category", "amount"]
            )

        if clean:
            self.data = self.data[
                (self.data["amount"] > 0) & (self.data["amount"] < INVALID_AMOUNT_CUTOFF)
            ]

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
            valid = amounts[(amounts > 0) & (amounts < INVALID_AMOUNT_CUTOFF)]
            if len(valid) == 0:
                continue
            # The user's own typical amount for this category; "logratio" features are
            # measured relative to it so absolute size is preserved.
            scale = float(np.median(valid))

            for i in range(len(amounts) - self.sequence_length + 1):
                sequence = amounts[i : i + self.sequence_length]
                samples.append(
                    {
                        "sequence": sequence,
                        "user_id": user_id,
                        "category": category,
                        "scale": scale,
                    }
                )

        return samples

    def get_user_split(self, val_frac=0.15, seed=42):
        """
        Split by user_id (not by sequence) so no user's sequences appear in
        both train and val — avoids the leakage caused by overlapping windows.

        Returns:
            train_indices, val_indices: lists of integer indices into self.samples
        """
        # 1. get unique user_ids in a deterministic order (sorted set)
        unique_user_ids = sorted({s['user_id'] for s in self.samples})
        # 2. shuffle them with the given seed
        rng = random.Random(seed)
        rng.shuffle(unique_user_ids)
        # 3. split into train/val groups by val_frac
        n_val = int(len(unique_user_ids) * val_frac)
        val_users = set(unique_user_ids[:n_val])
        # 4. walk self.samples and bucket each index by which group its user_id landed in
        train_indices, val_indices = [], []
        for idx, sample in enumerate(self.samples):
            if sample['user_id'] in val_users:
                val_indices.append(idx)
            else:
                train_indices.append(idx)

        return train_indices, val_indices

    def __len__(self):
        return len(self.samples)

    def featurize(self, raw_sequence, scale):
        """Raw amounts -> model input. "logratio": log(amount / user's typical amount);
        "minmax": per-sequence min-max to [0, 1] (original method, discards absolute size)."""
        if self.features == "minmax":
            return normalize_sequence(raw_sequence)
        ratio = np.asarray(raw_sequence, dtype="float32") / scale
        return np.log(np.clip(ratio, *LOG_RATIO_CLIP)).astype("float32")

    def __getitem__(self, idx):
        sample = self.samples[idx]

        seq = self.featurize(sample["sequence"], sample["scale"])

        sequence = torch.tensor(seq, dtype=torch.float32).unsqueeze(-1)

        # Autoencoder input and target are the same
        return sequence, sequence

    def get_sample_metadata(self, idx):
        sample = self.samples[idx]

        return {
            "user_id": sample["user_id"],
            "category": sample["category"],
            "sequence": sample["sequence"],
            "scale": sample["scale"],
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
