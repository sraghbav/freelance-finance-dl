"""
Synthetic personal-finance data WITH habits and LABELED anomalies, for a controlled simulation
study of when a detector helps. Not real data. Output uses the BudgetWise columns plus
`is_anomaly` and `anomaly_type`.

Habit levels (how much structure the data has):
  strong   habits are very regular (noise x0.5), yearly utility cycle +/-30%
  default  noise x1, utility cycle +/-30%
  weak     noise x2, utility cycle +/-15%
  none     no habits: every category is an independent lognormal draw, same for every user

Per user (levels other than "none"): salaried (steady pay) or freelance (irregular) income; rent
that is nearly constant; a fixed subscription; utilities with a yearly cycle; volatile groceries
and dining. With probability `anomaly_rate` per transaction (4x for utilities, so seasonal anomalies are frequent enough to evaluate) an anomaly replaces the normal amount:
  spike (x U(2,10)), drop (/ U(2,10)), seasonal (utilities: the opposite season's amount),
  duplicate (groceries, dining: the previous amount repeated).

    python -m freelance_finance_dl.habitsim --level default --seed 7 --out data/sim/default_s7.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

LEVELS = {
    "strong": dict(noise=0.5, season=0.30, habits=True),
    "default": dict(noise=1.0, season=0.30, habits=True),
    "weak": dict(noise=2.0, season=0.15, habits=True),
    "none": dict(noise=1.0, season=0.0, habits=False),
}


def simulate(n_users=100, years=3, seed=7, level="default", anomaly_rate=0.01):
    cfg = LEVELS[level]
    rng = np.random.default_rng(seed)
    months = pd.date_range("2021-01-01", periods=12 * years, freq="MS")
    rows, last = [], {}
    n = cfg["noise"]

    def draw(sigma):
        return rng.lognormal(0, sigma)

    for u in range(n_users):
        user = f"H{u:03d}"
        if cfg["habits"]:
            freelancer = rng.random() < 0.5
            income = rng.lognormal(np.log(45_000), 0.4)
            rent = 0.3 * income * draw(0.15)
            subscription = rng.lognormal(np.log(500), 0.3)
            utilities, groceries, dining = (rng.lognormal(np.log(v), 0.3) for v in (4_000, 3_000, 1_500))
            pay_day, rent_day = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        for m in months:
            amp = cfg["season"]
            phase = 2 * np.pi * (m.month - 1) / 12
            day = lambda d: m + pd.Timedelta(days=int(d) - 1)

            def add(date, kind, category, amount, can_dup=False):
                atype = ""
                if rng.random() < anomaly_rate * (4 if category == "utilities" else 1):
                    options = ["spike", "drop"] + (["seasonal"] if category == "utilities" and amp > 0 else []) \
                        + (["duplicate"] if can_dup and (user, category) in last else [])
                    atype = options[int(rng.integers(len(options)))]
                    if atype == "spike":
                        amount *= rng.uniform(2, 10)
                    elif atype == "drop":
                        amount /= rng.uniform(2, 10)
                    elif atype == "seasonal":
                        amount *= (1 + amp * np.cos(phase + np.pi)) / (1 + amp * np.cos(phase))
                    else:
                        amount = last[(user, category)]
                last[(user, category)] = amount
                rows.append((user, date, kind, category, round(float(amount), 2), int(bool(atype)), atype))

            if not cfg["habits"]:
                add(day(pay_day if False else rng.integers(1, 28)), "Income", "income", 45_000 * draw(0.5))
                add(day(rng.integers(1, 28)), "Expense", "rent", 4_000 * draw(0.6 * n))
                add(day(rng.integers(1, 28)), "Expense", "subscription", 4_000 * draw(0.6 * n))
                add(day(rng.integers(1, 28)), "Expense", "utilities", 4_000 * draw(0.6 * n))
                for _ in range(int(rng.integers(4, 6))):
                    add(day(rng.integers(1, 28)), "Expense", "groceries", 4_000 * draw(0.6 * n), True)
                for _ in range(int(rng.integers(5, 8))):
                    add(day(rng.integers(1, 28)), "Expense", "dining", 4_000 * draw(0.6 * n), True)
                continue
            season = 1 + amp * np.cos(phase)
            if freelancer:
                for _ in range(int(rng.integers(1, 4))):
                    add(day(rng.integers(1, 28)), "Income", "income", income / 2 * draw(0.35 * n))
            else:
                add(day(pay_day), "Income", "income", income * (1 + rng.normal(0, 0.02 * n)))
            add(day(rent_day), "Expense", "rent", rent * (1 + rng.normal(0, 0.01 * n)))
            add(day(10), "Expense", "subscription", subscription)
            add(day(15), "Expense", "utilities", utilities * season * (1 + rng.normal(0, 0.08 * n)))
            for _ in range(int(rng.integers(4, 6))):
                add(day(rng.integers(1, 28)), "Expense", "groceries", groceries * draw(0.35 * n), True)
            for _ in range(int(rng.integers(5, 8))):
                add(day(rng.integers(1, 28)), "Expense", "dining", dining * draw(0.5 * n), True)
    df = pd.DataFrame(rows, columns=["user_id", "date", "transaction_type", "category", "amount",
                                     "is_anomaly", "anomaly_type"])
    df.insert(0, "transaction_id", [f"S{i}" for i in range(len(df))])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data with habits and labeled anomalies")
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--level", choices=list(LEVELS), default="default")
    parser.add_argument("--anomaly-rate", type=float, default=0.01)
    parser.add_argument("--out", default="data/sim/default_s7.csv")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = simulate(args.users, args.years, args.seed, args.level, args.anomaly_rate)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out):,} rows ({int(out.is_anomaly.sum())} anomalies) to {args.out}")
