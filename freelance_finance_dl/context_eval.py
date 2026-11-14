"""
Evaluation of the context models (context_model.py) against the untrained rules and the original
window-of-5 autoencoder, on the simulation (labeled anomalies) and on Berka (injected anomalies).

Scores compared (all per transaction; "signed" scores are used for the matching direction only):
  rule |z|                    the variability-aware amount rule from the main study
  rule timing |gap z|         how unusual the time since the previous transaction is, vs this stream
  autoencoder (window 5)      the original model, error at the transaction's position
  forecast |r| / masked |r|   standardized residual of the amount from the context models
  forecast timing |r_gap|     standardized residual of the gap (forecast model only)
  rule amount+timing          max of the two rules (each calibrated to a tail probability)
  forecast amount+timing      max of the forecast model's two residuals (calibrated)
  hybrid rule+forecast        max of all four (calibrated)

Simulation:  python -m freelance_finance_dl.context_eval sim --seed 1 [--timing]
             python -m freelance_finance_dl.context_eval sim-summary
Berka:       python -m freelance_finance_dl.context_eval berka
"""

import argparse
import glob
import random

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from freelance_finance_dl.context_model import (Streams, autoencoder_scores, calibrate, predict,
                                                train_autoencoder, train_context, windows)
from freelance_finance_dl.crossval import FACTORS, OUT_DIR
from freelance_finance_dl.habitsim import LEVELS, simulate

TYPES = ["spike", "drop", "seasonal", "duplicate", "late", "double_charge"]
DELAYS = [3, 7, 14, 30]


def fit_all(S, train_idx, seed, max_train, ctx_epochs, ae_epochs, ae_max):
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(train_idx, size=min(max_train, len(train_idx)), replace=False))
    models = {m: train_context(S, sub, m, epochs=ctx_epochs, seed=seed) for m in ("forecast", "masked")}
    starts = windows(S, train_idx)
    starts = rng.choice(starts, size=min(ae_max, len(starts)), replace=False)
    models["ae"] = train_autoencoder(S, starts, latent=16, epochs=ae_epochs, seed=seed)
    return models, sub


def base_scores(S, models, idx, z=None, lgap=None, time=None):
    """Raw per-transaction scores. z / lgap / time override the targets' amount feature, log gap,
    and time features (for injected anomalies)."""
    z = S.z[idx] if z is None else z
    lgap = S.lgap[idx] if lgap is None else lgap
    hg = S.has_gap[idx]
    gz = np.where(hg > 0, (lgap - S.gap_med[idx]) / S.gap_spread[idx], 0.0)
    mu, sd, mg, sg = predict(models["forecast"], S, idx, time=time)
    r = (z - mu) / sd
    rg = np.where(hg > 0, (lgap - mg) / sg, 0.0)
    mm, ms, _, _ = predict(models["masked"], S, idx, time=time)
    rm = (z - mm) / ms
    ae = autoencoder_scores(models["ae"], S, idx, z_override=z)
    return {"z": z, "gz": gz, "r": r, "rg": rg, "rm": rm, "ae": ae, "stype": S.stream_type[idx]}


def named_scores(b, cal):
    s = {
        "rule |z|": np.abs(b["z"]),
        "rule z (spike known)": b["z"],
        "rule -z (drop known)": -b["z"],
        "rule timing |gap z|": np.abs(b["gz"]),
        "rule timing gap z (late known)": b["gz"],
        "autoencoder (window 5, latent 16)": np.nan_to_num(b["ae"], nan=0.0),
        "forecast |r|": np.abs(b["r"]),
        "forecast r (spike known)": b["r"],
        "forecast -r (drop known)": -b["r"],
        "forecast timing |r_gap|": np.abs(b["rg"]),
        "forecast timing r_gap (late known)": b["rg"],
        "masked |r|": np.abs(b["rm"]),
        "masked r (spike known)": b["rm"],
        "masked -r (drop known)": -b["rm"],
    }
    if cal is not None:
        # Calibrated separately for expense and income, whose score distributions differ.
        c = {}
        for k in cal:
            c[k] = np.zeros(len(s[k]))
            for t, f in cal[k].items():
                m = b["stype"] == t
                c[k][m] = f(s[k][m])
        s["rule amount+timing"] = np.maximum(c["rule |z|"], c["rule timing |gap z|"])
        s["forecast amount+timing"] = np.maximum(c["forecast |r|"], c["forecast timing |r_gap|"])
        s["hybrid rule+forecast"] = np.max([c[k] for k in cal], 0)
    return s


CALIBRATED = ["rule |z|", "rule timing |gap z|", "forecast |r|", "forecast timing |r_gap|"]


def make_calibration(S, models, idx):
    b = base_scores(S, models, idx)
    s = named_scores(b, None)
    return {k: {t: calibrate(s[k][b["stype"] == t]) for t in ("expense", "income") if (b["stype"] == t).any()}
            for k in CALIBRATED}


# ---------------------------------------------------------------- simulation

def metric_rows(meta, name, score, y, types, streams):
    rows = []
    thr = np.percentile(score, 95)
    groups = {"overall": np.ones_like(y, bool), "expense": streams == "expense", "income": streams == "income"}
    for t in TYPES:
        groups[f"type:{t}"] = (types == t) | ~y
    for g, mask in groups.items():
        yy, ss = y[mask], score[mask]
        if g.startswith("type:"):
            yy = types[mask] == g.split(":")[1]
        if yy.sum() < 5 or (~yy).sum() < 5:
            continue
        rows.append(dict(**meta, score=name, group=g, auroc=roc_auc_score(yy, ss),
                         average_precision=average_precision_score(yy, ss),
                         recall_at_5pct=float(np.mean(ss[yy] > thr)), n_pos=int(yy.sum()), n_neg=int((~yy).sum())))
    return rows


def run_sim(level, seed, timing, ctx_epochs=10):
    df = simulate(100, 3, seed, level, timing=timing)
    S = Streams(df)
    users = sorted(set(S.user))
    random.Random(seed).shuffle(users)               # same 80/20 user split as sim_eval
    val_users = set(users[:int(len(users) * 0.2)])
    is_val = np.array([u in val_users for u in S.user])
    train_idx, val_idx = np.where(~is_val)[0], np.where(is_val)[0]
    models, sub = fit_all(S, train_idx, seed, 60_000, ctx_epochs, 15, 30_000)
    cal = make_calibration(S, models, sub)
    sc = named_scores(base_scores(S, models, val_idx), cal)
    y, types, streams = S.label[val_idx], S.atype[val_idx], S.stream_type[val_idx]
    tag = "timing" if timing else "amount"
    rows = []
    for name, s in sc.items():
        rows += metric_rows(dict(data=tag, level=level, seed=seed, unit="transaction"), name, s, y, types, streams)
    # Window level (5 consecutive transactions in a stream; positive if any is labeled), as in the
    # main simulation study: a window's score is the maximum over its transactions, except the
    # autoencoder, which uses its original window score (mean reconstruction error).
    starts_local = np.where(S.pos[val_idx] + 5 <= S.stream_len[val_idx])[0]
    wl = np.stack([starts_local + k for k in range(5)], 1)           # local positions in val_idx
    wy = y[wl].any(1)
    wt = np.array([next((t for t in types[w] if t), "") for w in wl])
    ws = streams[wl[:, 0]]
    for name, s in sc.items():
        if name.startswith("autoencoder"):
            continue
        rows += metric_rows(dict(data=tag, level=level, seed=seed, unit="window"), name, s[wl].max(1), wy, wt, ws)
    import torch
    X = torch.tensor(np.stack([S.z[val_idx[wl[:, k]]] for k in range(5)], 1)).unsqueeze(-1)
    with torch.no_grad():
        models["ae"].eval()
        aew = ((models["ae"](X) - X) ** 2).mean((1, 2)).numpy()
    rows += metric_rows(dict(data=tag, level=level, seed=seed, unit="window"),
                        "autoencoder (window 5, latent 16)", aew, wy, wt, ws)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/ctx_sim_{tag}_{level}_s{seed}.csv", index=False)
    return out


def sim_summary():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT_DIR}/ctx_sim_*_s*.csv")])
    keys = ["data", "unit", "level", "score", "group"]
    g = d.groupby(keys)[["auroc", "average_precision", "recall_at_5pct"]].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    g["n_seeds"] = d.groupby(keys).seed.nunique().values
    g["n_pos_total"] = d.groupby(keys).n_pos.sum().values
    g.to_csv(f"{OUT_DIR}/ctx_sim_summary.csv", index=False)
    return g


# ---------------------------------------------------------------- Berka (injected anomalies)

def run_berka(csv_file, n_folds=5, seed=42, max_train=150_000, max_eval=3_000, ctx_epochs=6, ae_epochs=20):
    from freelance_finance_dl.berka import load_berka
    S = Streams(load_berka(csv_file))
    print(f"berka: {len(S):,} transactions, {S.stream.max() + 1:,} streams", flush=True)
    users = sorted(set(S.user))
    np.random.default_rng(seed).shuffle(users)
    folds = [set(users[k::n_folds]) for k in range(n_folds)]
    rows = []
    for k, held in enumerate(folds):
        is_val = np.array([u in held for u in S.user])
        # Score only transactions with at least one earlier transaction in a stream of 5 or more,
        # so every method (including the window-of-5 autoencoder) has history to use.
        eligible = (S.stream_len >= 5) & (S.pos >= 1)
        train_idx, val_idx = np.where(~is_val)[0], np.where(is_val & eligible)[0]
        models, sub = fit_all(S, train_idx, seed + k, max_train, ctx_epochs, ae_epochs, 40_000)
        cal = make_calibration(S, models, sub)
        rng = np.random.default_rng(seed + k)
        for stream in ("expense", "income"):
            idx = val_idx[S.stream_type[val_idx] == stream]
            clean = named_scores(base_scores(S, models, idx), cal)
            thr = {n: np.percentile(s, 95) for n, s in clean.items()}
            pick = np.sort(rng.choice(len(idx), size=min(max_eval, len(idx)), replace=False))
            tgt = idx[pick]
            fpr = {n: float(np.mean(clean[n][pick] > thr[n])) for n in clean}

            def record(kind, factor, scores, n):
                for name, s in scores.items():
                    rows.append(dict(fold=k, stream=stream, kind=kind, factor=factor, score=name,
                                     recall=float(np.mean(s > thr[name])), fpr=fpr[name], n=n))
            for f in FACTORS:
                for kind in ("spike", "drop"):
                    amt = S.amount[tgt] * f if kind == "spike" else S.amount[tgt] / f
                    z = Streams.featurize(amt, S.scale[tgt], S.spread[tgt])
                    record(kind, f, named_scores(base_scores(S, models, tgt, z=z), cal), len(tgt))
            # Late arrival: the transaction is moved d days later (needs an earlier transaction).
            late = tgt[S.has_gap[tgt] > 0]
            gap_days = np.expm1(S.lgap[late])
            for dly in DELAYS:
                lg = np.log1p(gap_days + dly).astype("float32")
                date = pd.DatetimeIndex(S.date[late]) + pd.Timedelta(days=dly)
                time = S.time_features(late, lgap=lg, date=date)
                b = base_scores(S, models, late, lgap=lg, time=time)
                record("late", dly, named_scores(b, cal), len(late))
        print(f"fold {k} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/ctx_berka.csv", index=False)
    summary = (out.groupby(["stream", "kind", "factor", "score"])
               .agg(recall_mean=("recall", "mean"), recall_sd=("recall", "std"), fpr_mean=("fpr", "mean"))
               .reset_index())
    summary.to_csv(f"{OUT_DIR}/ctx_berka_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Context-model evaluation")
    p.add_argument("what", choices=["sim", "sim-summary", "berka"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--levels", nargs="+", default=list(LEVELS))
    p.add_argument("--timing", action="store_true")
    p.add_argument("--csv", default="data/berka/trans.csv")
    a = p.parse_args()
    if a.what == "sim":
        for lv in a.levels:
            run_sim(lv, a.seed, a.timing)
            print(f"{lv} seed {a.seed} timing={a.timing} done", flush=True)
    elif a.what == "sim-summary":
        print(sim_summary().query("group == 'overall'").round(3).to_string(index=False))
    else:
        print(run_berka(a.csv).round(3).to_string(index=False))
