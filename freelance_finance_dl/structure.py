"""
Does the data have structure a sequence model could use?

  persistent user level  intraclass correlation of log-amount by user within a category
                         (near 0: users do not differ in how much they typically spend)
  time structure         mean lag-1 autocorrelation of log-amount within a (user, category)
                         history (near 0: one transaction does not predict the next)

    python -m freelance_finance_dl.structure
    python -m freelance_finance_dl.structure --source sparkov --csv data/sparkov/fraudTrain.csv data/sparkov/fraudTest.csv
"""

import argparse

import numpy as np

from freelance_finance_dl.dataloader import FinanceTransactionDataset


def lag1_autocorr(data, min_len=10):
    values = []
    for _, g in data.groupby(["user_id", "category"]):
        a = np.log(g.sort_values("date")["amount"].values)
        if len(a) >= min_len and a.std() > 0:
            values.append(np.corrcoef(a[:-1], a[1:])[0, 1])
    return len(values), float(np.mean(values)), float(np.std(values))


def icc_by_user(data, category):
    g = data[data["category"] == category]
    groups = [np.log(v["amount"].values) for _, v in g.groupby("user_id") if len(v) >= 2]
    k = len(groups)
    n_i = np.array([len(v) for v in groups])
    n = n_i.sum()
    grand = np.concatenate(groups).mean()
    ssb = sum(len(v) * (v.mean() - grand) ** 2 for v in groups)
    ssw = sum(((v - v.mean()) ** 2).sum() for v in groups)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - (n_i ** 2).sum() / n) / (k - 1)
    return k, float((msb - msw) / (msb + (n0 - 1) * msw))


def report(data, top_categories=3):
    n_groups, mean_ac, sd_ac = lag1_autocorr(data)
    print(f"rows: {len(data):,} | users: {data['user_id'].nunique():,}")
    print(f"lag-1 autocorrelation of log-amount: mean {mean_ac:.3f} (sd {sd_ac:.3f}) over {n_groups} histories")
    for category in data["category"].value_counts().index[:top_categories]:
        k, icc = icc_by_user(data, category)
        print(f"intraclass correlation by user, {category}: {icc:.3f} ({k} users)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structure checks")
    parser.add_argument("--source", choices=["budgetwise", "sparkov"], default="budgetwise")
    parser.add_argument("--csv", nargs="+", default=["data/budgetwise_finance_dataset.csv"])
    args = parser.parse_args()
    ds = FinanceTransactionDataset(args.csv if args.source == "sparkov" else args.csv[0],
                                   clean=True, source=args.source)
    expense = ds.data[ds.data["transaction_type"].str.lower() == "expense"]
    report(expense)
