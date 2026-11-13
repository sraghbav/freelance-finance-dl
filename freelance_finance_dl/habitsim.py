"""
Synthetic personal-finance data WITH habits, to test whether a sequence model helps when the data
have structure. Not real data. Output uses the same columns as the BudgetWise file.

Per user (seeded): a salaried or freelance income stream; rent that is nearly constant; a fixed
subscription; utilities with a yearly seasonal cycle; volatile groceries and dining.

    python -m freelance_finance_dl.habitsim --users 100 --out data/habitsim.csv
"""

import argparse

import numpy as np
import pandas as pd


def simulate(n_users=100, years=3, seed=7):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2021-01-01")
    months = pd.date_range(start, periods=12 * years, freq="MS")
    rows = []

    def add(user, date, kind, category, amount):
        rows.append((user, date, kind, category, round(float(amount), 2)))

    for u in range(n_users):
        user = f"H{u:03d}"
        freelancer = rng.random() < 0.5
        income = rng.lognormal(np.log(45_000), 0.4)
        rent = 0.3 * income * rng.lognormal(0, 0.15)
        subscription = rng.lognormal(np.log(500), 0.3)
        utilities = rng.lognormal(np.log(4_000), 0.3)
        groceries = rng.lognormal(np.log(3_000), 0.3)
        dining = rng.lognormal(np.log(1_500), 0.3)
        pay_day = int(rng.integers(1, 6))
        rent_day = int(rng.integers(1, 6))
        for m in months:
            season = 1 + 0.3 * np.cos(2 * np.pi * (m.month - 1) / 12)
            day = lambda d: m + pd.Timedelta(days=int(d) - 1)
            if freelancer:
                for _ in range(int(rng.integers(1, 4))):
                    add(user, day(rng.integers(1, 28)), "Income", "income",
                        income / 2 * rng.lognormal(0, 0.35))
            else:
                add(user, day(pay_day), "Income", "income", income * (1 + rng.normal(0, 0.02)))
            add(user, day(rent_day), "Expense", "rent", rent * (1 + rng.normal(0, 0.01)))
            add(user, day(10), "Expense", "subscription", subscription)
            add(user, day(15), "Expense", "utilities", utilities * season * (1 + rng.normal(0, 0.08)))
            for _ in range(int(rng.integers(4, 6))):
                add(user, day(rng.integers(1, 28)), "Expense", "groceries", groceries * rng.lognormal(0, 0.35))
            for _ in range(int(rng.integers(5, 8))):
                add(user, day(rng.integers(1, 28)), "Expense", "dining", dining * rng.lognormal(0, 0.5))
    df = pd.DataFrame(rows, columns=["user_id", "date", "transaction_type", "category", "amount"])
    df.insert(0, "transaction_id", [f"S{i}" for i in range(len(df))])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data with habits")
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/habitsim.csv")
    args = parser.parse_args()
    out = simulate(args.users, args.years, args.seed)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out):,} rows for {args.users} users to {args.out}")
