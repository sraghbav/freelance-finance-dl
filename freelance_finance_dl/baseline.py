"""
Non-neural baseline for the raw-space anomaly evaluation, on the same 5 user folds and the same
injected anomalies as crossval.py.

Three untrained scores per window, all from log(amount / the user's typical amount):
  two_sided   largest |value| in the window
  spike_rule  largest value            (knows an anomaly would be a spike)
  drop_rule   largest -value, i.e. the most negative entry (knows it would be a drop)
Each threshold is the 95th percentile of the clean scores of the held-out fold's stream
("all", "expense", "income"), the same convention as crossval.py.
The direction-aware rules use knowledge a real detector would not have; they are a ceiling for
what a one-line rule can do.

    python -m freelance_finance_dl.baseline
"""

import numpy as np
import pandas as pd

from freelance_finance_dl.crossval import FACTORS, OUT_DIR, STREAMS, user_folds
from freelance_finance_dl.dataloader import FinanceTransactionDataset

SCORERS = {
    "two_sided": lambda f: float(np.max(np.abs(f))),
    "spike_rule": lambda f: float(np.max(f)),
    "drop_rule": lambda f: float(-np.min(f)),
}


def run(csv_file, n_folds=5, seed=42, features="logratio", tag=""):
    dataset = FinanceTransactionDataset(csv_file, clean=True, robust=True, features=features)
    rows = []
    for k, held in enumerate(user_folds(dataset, n_folds, seed)):
        for stream in STREAMS:
            samples = [s for s in dataset.samples if s["user_id"] in held
                       and (stream == "all" or s["transaction_type"] == stream)]
            positions = np.random.default_rng(seed).integers(0, dataset.sequence_length, size=len(samples))
            for name, score in SCORERS.items():
                clean = np.array([score(dataset.featurize(s["sequence"], s["scale"], s["spread"])) for s in samples])
                threshold = float(np.percentile(clean, 95))
                fpr = float(np.mean(clean > threshold))
                for factor in FACTORS:
                    for kind in ("spike", "drop"):
                        hits = []
                        for s, pos in zip(samples, positions):
                            raw = s["sequence"].copy()
                            raw[pos] = raw[pos] * factor if kind == "spike" else raw[pos] / factor
                            hits.append(score(dataset.featurize(raw, s["scale"], s["spread"])) > threshold)
                        rows.append({"rule": name, "stream": stream, "fold": k, "kind": kind,
                                     "factor": factor, "recall": float(np.mean(hits)),
                                     "false_positive_rate": fpr, "n": len(samples)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/baseline_rule_folds{tag}.csv", index=False)
    summary = (df.groupby(["rule", "stream", "kind", "factor"])["recall"].agg(["mean", "std"]).round(3)
               .rename(columns={"mean": "recall_mean", "std": "recall_sd"}).reset_index())
    summary.to_csv(f"{OUT_DIR}/baseline_rule_summary{tag}.csv", index=False)
    return summary


if __name__ == "__main__":
    import sys
    feats = sys.argv[1] if len(sys.argv) > 1 else "logratio"
    out = run("data/budgetwise_finance_dataset.csv", features=feats,
              tag="" if feats == "logratio" else f"_{feats}")
    print(out[out.stream == "all"].to_string(index=False))
