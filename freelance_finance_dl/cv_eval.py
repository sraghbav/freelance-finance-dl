"""
5-fold user-level cross-validation with raw-amount injection for any data source (Berka, the
synthetic habit data, ...), separately for the expense and income streams. Compares the
autoencoder with three untrained rules and an Isolation Forest, all on the same injected anomalies.

    python -m freelance_finance_dl.cv_eval --source berka --csv data/berka/trans.csv \
        --features robustz --latent 16 --tag _berka_robustz_l16
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, Subset

from freelance_finance_dl.crossval import FACTORS, OUT_DIR, train_model, user_folds
from freelance_finance_dl.dataloader import FinanceTransactionDataset
from freelance_finance_dl.evaluate import compute_reconstruction_errors, raw_injection_eval

RULES = {
    "rule |z| (direction unknown)": lambda X: np.abs(X).max(1),
    "rule max z (spike known)": lambda X: X.max(1),
    "rule min z (drop known)": lambda X: -X.min(1),
}


def features(ds, samples, factor=None, kind=None, positions=None):
    X = []
    for i, s in enumerate(samples):
        raw = s["sequence"]
        if factor is not None:
            raw = raw.copy()
            raw[positions[i]] = raw[positions[i]] * factor if kind == "spike" else raw[positions[i]] / factor
        X.append(ds.featurize(raw, s["scale"], s["spread"]))
    return np.stack(X)


def run(source, csv_file, feature_type, latent, epochs, max_train, max_eval, seed=42, tag="", n_folds=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FinanceTransactionDataset(csv_file, clean=True, source=source, features=feature_type)
    n_windows = {t: sum(s["transaction_type"] == t for s in ds.samples) for t in ("expense", "income")}
    print(f"{source} {feature_type} latent {latent}: {len(ds.samples):,} windows {n_windows}", flush=True)
    rows = []
    for k, held in enumerate(user_folds(ds, n_folds, seed)):
        train_idx = [i for i, s in enumerate(ds.samples) if s["user_id"] not in held]
        val_idx = [i for i, s in enumerate(ds.samples) if s["user_id"] in held]
        rng = np.random.default_rng(seed + k)
        subset = rng.choice(train_idx, size=min(max_train, len(train_idx)), replace=False).tolist()
        model = train_model(ds, subset, latent, epochs, seed + k, device)
        forest = IsolationForest(n_estimators=100, random_state=seed + k).fit(
            features(ds, [ds.samples[i] for i in subset]))
        scorers = dict(RULES)
        scorers["isolation forest"] = lambda X: -forest.score_samples(X)
        for stream in ("all", "expense", "income"):
            idx = [i for i in val_idx if stream == "all" or ds.samples[i]["transaction_type"] == stream]
            if len(idx) < 50:
                continue
            ae_thr = float(np.percentile(compute_reconstruction_errors(
                model, DataLoader(Subset(ds, idx), batch_size=4096), device), 95))
            sample = sorted(rng.choice(idx, size=min(max_eval, len(idx)), replace=False).tolist())
            df = raw_injection_eval(model, ds, sample, ae_thr, device, factors=FACTORS, seed=seed)
            for r in df.itertuples():
                rows.append(dict(fold=k, stream=stream, model="autoencoder", kind=r.kind, factor=r.factor,
                                 recall=r.recall, fpr=r.false_positive_rate, n=r.n))
            clean_all = features(ds, [ds.samples[i] for i in idx])
            samples = [ds.samples[i] for i in sample]
            positions = np.random.default_rng(seed).integers(0, ds.sequence_length, size=len(samples))
            for name, score in scorers.items():
                thr = np.percentile(score(clean_all), 95)
                fpr = float(np.mean(score(features(ds, samples)) > thr))
                for factor in FACTORS:
                    for kind in ("spike", "drop"):
                        X = features(ds, samples, factor, kind, positions)
                        rows.append(dict(fold=k, stream=stream, model=name, kind=kind, factor=factor,
                                         recall=float(np.mean(score(X) > thr)), fpr=fpr, n=len(samples)))
        print(f"fold {k} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/cv{tag}.csv", index=False)
    summary = (out.groupby(["stream", "model", "kind", "factor"])["recall"].agg(["mean", "std"])
               .reset_index().rename(columns={"mean": "recall_mean", "std": "recall_sd"}))
    summary.to_csv(f"{OUT_DIR}/cv{tag}_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User-level CV with injected anomalies")
    parser.add_argument("--source", default="berka")
    parser.add_argument("--csv", default="data/berka/trans.csv")
    parser.add_argument("--features", choices=["logratio", "robustz"], default="robustz")
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-train", type=int, default=40_000)
    parser.add_argument("--max-eval", type=int, default=3_000)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    run(args.source, args.csv, args.features, args.latent, args.epochs, args.max_train, args.max_eval,
        tag=args.tag)
