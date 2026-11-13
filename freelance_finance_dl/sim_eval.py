"""
Simulation study: detectors on synthetic data with labeled anomalies, across habit levels.

For each (level, seed): generate data, split users 80/20, train the autoencoder (variability-aware
and log-ratio features, latent 16), score held-out windows with it, two rules, and an Isolation
Forest. A window is positive if it contains a labeled anomaly. Reports AUROC, average precision, and
the share of positives caught when the top 5% of windows are flagged, overall, by stream, and by
anomaly type.

    python -m freelance_finance_dl.sim_eval --seed 1
    python -m freelance_finance_dl.sim_eval --summarize
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from freelance_finance_dl.crossval import OUT_DIR, train_model
from freelance_finance_dl.dataloader import FinanceTransactionDataset
from freelance_finance_dl.evaluate import compute_reconstruction_errors
from freelance_finance_dl.habitsim import LEVELS, simulate

ANOMALY_TYPES = ["spike", "drop", "seasonal", "duplicate"]


def metric_rows(level, seed, name, score, y, types, streams):
    rows = []
    thr = np.percentile(score, 95)
    groups = {"overall": np.ones_like(y, bool), "expense": streams == "expense", "income": streams == "income"}
    for t in ANOMALY_TYPES:
        groups[f"type:{t}"] = (types == t) | ~y
    for g, mask in groups.items():
        yy, ss = y[mask], score[mask]
        if g.startswith("type:"):
            yy = (types[mask] == g.split(":")[1])
        if yy.sum() < 5 or (~yy).sum() < 5:
            continue
        rows.append(dict(level=level, seed=seed, score=name, group=g, auroc=roc_auc_score(yy, ss),
                         average_precision=average_precision_score(yy, ss),
                         recall_at_5pct=float(np.mean(ss[yy] > thr)), n_pos=int(yy.sum()), n_neg=int((~yy).sum())))
    return rows


def run_one(level, seed, latent=16, epochs=15, max_train=30_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("data/sim", exist_ok=True)
    path = f"data/sim/{level}_s{seed}.csv"
    simulate(100, 3, seed, level).to_csv(path, index=False)
    rows, X = [], {}
    for feat in ("robustz", "logratio"):
        ds = FinanceTransactionDataset(path, clean=True, features=feat)
        train_idx, val_idx = ds.get_user_split(0.2, seed)
        sub = np.random.default_rng(seed).choice(train_idx, size=min(max_train, len(train_idx)),
                                                 replace=False).tolist()
        model = train_model(ds, sub, latent, epochs, seed, device)
        ae = compute_reconstruction_errors(model, DataLoader(Subset(ds, val_idx), batch_size=4096), device)
        feats = lambda idx: np.stack([ds.featurize(ds.samples[i]["sequence"], ds.samples[i]["scale"],
                                                    ds.samples[i]["spread"]) for i in idx])
        Xv = feats(val_idx)
        y = np.array([ds.samples[i]["anomaly"] for i in val_idx], dtype=bool)
        types = np.array([ds.samples[i]["anomaly_type"] for i in val_idx])
        streams = np.array([ds.samples[i]["transaction_type"] for i in val_idx])
        scores = {f"autoencoder ({feat}, latent {latent})": ae, f"rule |z| ({feat})": np.abs(Xv).max(1)}
        if feat == "robustz":
            forest = IsolationForest(n_estimators=100, random_state=seed).fit(feats(sub))
            scores["isolation forest (robustz)"] = -forest.score_samples(Xv)
        for name, sc in scores.items():
            rows += metric_rows(level, seed, name, sc, y, types, streams)
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/sim_{level}_s{seed}.csv", index=False)
    return rows


def summarize():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT_DIR}/sim_*_s*.csv")])
    g = d.groupby(["level", "score", "group"])[["auroc", "average_precision", "recall_at_5pct"]].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    g["n_seeds"] = d.groupby(["level", "score", "group"]).seed.nunique().values
    g.to_csv(f"{OUT_DIR}/sim_summary.csv", index=False)
    print(g[g.group == "overall"].round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation study")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--levels", nargs="+", default=list(LEVELS))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        for lv in args.levels:
            run_one(lv, args.seed)
            print(f"{lv} seed {args.seed} done", flush=True)
