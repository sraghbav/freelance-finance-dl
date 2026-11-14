"""
Context models: an improved detector that scores each TRANSACTION from its history, with timing,
category, and longer context. Built to test the improvements listed after the main study:

  1. timing inputs   days since the previous transaction in the stream, day of month, month of year
  2. longer context  the previous 12 transactions (instead of a window of 5)
  3. a prediction objective instead of reconstruction:
       "forecast"  reads the previous 12 transactions and predicts the next amount AND how uncertain
                   it is (Gaussian mean and standard deviation), plus the gap until it arrives
       "masked"    reads 6 transactions before and 6 after, with the middle amount hidden, and
                   predicts it (a bidirectional fill-in model; it cannot copy the hidden amount)
  4. category and stream (expense or income) as inputs, and SIGNED scores, so a spike, a drop, and a
     late payment can be scored separately
  5. a hybrid that combines the untrained rule with the model (Section in context_eval)

Amounts use the same variability-aware feature as the main study:
z = arcsinh(log(amount / typical) / spread), typical and spread taken from the stream's own history.
A transaction's anomaly score is its standardized residual r = (z - predicted mean) / predicted sd.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from freelance_finance_dl.dataloader import INVALID_AMOUNT_CUTOFF, LOG_RATIO_CLIP, MIN_SPREAD

CONTEXT = 12
GAP_NORM = np.log1p(30.0)
STEP_FEATURES = 9   # z, gap, has_gap, dom sin/cos, month sin/cos, present, is_target
TIME_FEATURES = 6   # gap, has_gap, dom sin/cos, month sin/cos (of the transaction being scored)


def _robust_spread(x):
    return max(float(1.4826 * np.median(np.abs(x - np.median(x)))), MIN_SPREAD)


class Streams:
    """All transactions as flat arrays, sorted so each (user, type, category) stream is contiguous
    and in date order. Income is pooled into one stream per user, as in the main study."""

    def __init__(self, df):
        d = df.copy()
        d["_row"] = np.arange(len(d))
        d["transaction_type"] = d["transaction_type"].astype(str).str.strip().str.lower()
        d["category"] = d["category"].astype(str).str.strip().str.lower()
        d.loc[d["transaction_type"] == "income", "category"] = "income"
        d["date"] = pd.to_datetime(d["date"])
        d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
        d = d[(d["amount"] > 0) & (d["amount"] < INVALID_AMOUNT_CUTOFF)].dropna(subset=["date", "user_id"])
        if "is_anomaly" not in d:
            d["is_anomaly"] = 0
        if "anomaly_type" not in d:
            d["anomaly_type"] = ""
        d["anomaly_type"] = d["anomaly_type"].fillna("").astype(str)
        d = d.sort_values(["user_id", "transaction_type", "category", "date", "_row"], kind="stable")
        d = d.reset_index(drop=True)
        keys = ["user_id", "transaction_type", "category"]
        g = d.groupby(keys, sort=False)
        d["stream"] = g.ngroup()
        d["pos"] = g.cumcount()
        d["stream_len"] = g["amount"].transform("size")
        d["scale"] = g["amount"].transform("median")
        logs = np.log(d["amount"] / d["scale"])
        d["_log"] = logs
        d["spread"] = d.groupby("stream")["_log"].transform(lambda x: _robust_spread(x.values))
        gap = d.groupby("stream")["date"].diff().dt.days
        d["has_gap"] = gap.notna().astype("float32")
        d["lgap"] = np.log1p(gap.fillna(0).clip(lower=0))
        self.df = d
        self.categories = sorted(d["category"].unique())
        self.cat_id = d["category"].map({c: i for i, c in enumerate(self.categories)}).values
        self.type_id = (d["transaction_type"] == "income").astype(int).values
        self.stream = d["stream"].values
        self.pos = d["pos"].values
        self.stream_len = d["stream_len"].values
        self.scale = d["scale"].values
        self.spread = d["spread"].values
        self.amount = d["amount"].values
        self.date = d["date"].values
        self.lgap = d["lgap"].values.astype("float32")
        self.has_gap = d["has_gap"].values
        self.z = self.featurize(self.amount, self.scale, self.spread)
        self.label = d["is_anomaly"].values.astype(bool)
        self.atype = d["anomaly_type"].values
        self.user = d["user_id"].astype(str).values
        self.stream_type = d["transaction_type"].values
        # Timing rule: how unusual the gap is relative to this stream's own gaps.
        gl = d.loc[d["has_gap"] == 1].groupby("stream")["lgap"]
        med = gl.median().reindex(range(d["stream"].max() + 1)).fillna(0).values
        spr = gl.agg(lambda x: _robust_spread(x.values)).reindex(range(d["stream"].max() + 1)).fillna(MIN_SPREAD).values
        self.gap_med, self.gap_spread = med[self.stream], spr[self.stream]

    @staticmethod
    def featurize(amount, scale, spread):
        ratio = np.clip(np.asarray(amount, float) / scale, *LOG_RATIO_CLIP)
        return np.arcsinh(np.log(ratio) / spread).astype("float32")

    def gap_z(self, lgap=None):
        lgap = self.lgap if lgap is None else lgap
        return ((lgap - self.gap_med) / self.gap_spread).astype("float32")

    def __len__(self):
        return len(self.amount)

    def time_features(self, idx, lgap=None, date=None):
        lgap = self.lgap[idx] if lgap is None else lgap
        date = pd.DatetimeIndex(self.date[idx] if date is None else date)
        dom = 2 * np.pi * (date.day.values - 1) / 31
        moy = 2 * np.pi * (date.month.values - 1) / 12
        return np.stack([lgap / GAP_NORM, self.has_gap[idx], np.sin(dom), np.cos(dom),
                         np.sin(moy), np.cos(moy)], 1).astype("float32")

    def steps(self, idx, offsets, mask_center=False):
        """(N, len(offsets), STEP_FEATURES) for each target index and relative offsets; positions
        outside the target's stream are all-zero (present = 0)."""
        idx = np.asarray(idx)
        out = np.zeros((len(idx), len(offsets), STEP_FEATURES), dtype="float32")
        for j, off in enumerate(offsets):
            k = idx + off
            ok = (self.pos[idx] + off >= 0) & (self.pos[idx] + off < self.stream_len[idx])
            kk = k[ok]
            out[ok, j, 0] = self.z[kk]
            out[ok, j, 1:7] = self.time_features(kk)[:, [0, 1, 2, 3, 4, 5]]
            out[ok, j, 7] = 1.0
            if off == 0 and mask_center:
                out[ok, j, 0] = 0.0
                out[ok, j, 8] = 1.0
        return out


def forecast_offsets(context=CONTEXT):
    return list(range(-context, 0))


def masked_offsets(context=CONTEXT):
    half = context // 2
    return list(range(-half, half + 1))


class ContextModel(nn.Module):
    """LSTM over the context; heads predict the amount feature (mean, log sd) and, for the forecast
    model, the log gap until the transaction (mean, log sd)."""

    def __init__(self, n_categories, mode="forecast", hidden=64, emb=8):
        super().__init__()
        self.mode = mode
        bi = mode == "masked"
        self.rnn = nn.LSTM(STEP_FEATURES, hidden, batch_first=True, bidirectional=bi)
        h = hidden * (2 if bi else 1)
        self.cat = nn.Embedding(n_categories, emb)
        self.typ = nn.Embedding(2, 4)
        self.amount_head = nn.Sequential(nn.Linear(h + emb + 4 + TIME_FEATURES, 64), nn.ReLU(), nn.Linear(64, 2))
        self.gap_head = nn.Sequential(nn.Linear(h + emb + 4, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, steps, time, cat, typ):
        out, _ = self.rnn(steps)
        h = out[:, -1] if self.mode == "forecast" else out[:, out.shape[1] // 2]
        e = torch.cat([h, self.cat(cat), self.typ(typ)], 1)
        a = self.amount_head(torch.cat([e, time], 1))
        g = self.gap_head(e)
        return a[:, 0], a[:, 1].clamp(-5, 3), g[:, 0], g[:, 1].clamp(-5, 3)


def _nll(y, mu, logsd):
    return 0.5 * ((y - mu) / logsd.exp()) ** 2 + logsd


def _inputs(S, idx, mode):
    offs = forecast_offsets() if mode == "forecast" else masked_offsets()
    return (S.steps(idx, offs, mask_center=mode == "masked"), S.time_features(idx),
            S.cat_id[idx], S.type_id[idx])


def train_context(S, idx, mode="forecast", epochs=8, seed=0, batch=512, lr=2e-3):
    torch.manual_seed(seed)
    model = ContextModel(len(S.categories), mode)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps, time, cat, typ = _inputs(S, idx, mode)
    steps, time = torch.tensor(steps), torch.tensor(time)
    cat, typ = torch.tensor(cat), torch.tensor(typ)
    z = torch.tensor(S.z[idx])
    lg, hg = torch.tensor(S.lgap[idx]), torch.tensor(S.has_gap[idx], dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(idx), generator=gen).split(batch):
            mu, ls, mg, lsg = model(steps[b], time[b], cat[b], typ[b])
            loss = _nll(z[b], mu, ls).mean()
            if mode == "forecast":
                loss = loss + (_nll(lg[b], mg, lsg) * hg[b]).sum() / hg[b].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, S, idx, time=None, batch=8192):
    """Predicted (mean, sd) of the amount feature and of the log gap for each target index.
    `time` overrides the targets' own time features (used when a date is shifted)."""
    model.eval()
    steps, t, cat, typ = _inputs(S, idx, model.mode)
    t = t if time is None else time
    outs = []
    for s in range(0, len(idx), batch):
        sl = slice(s, s + batch)
        mu, ls, mg, lsg = model(torch.tensor(steps[sl]), torch.tensor(t[sl]), torch.tensor(cat[sl]),
                                torch.tensor(typ[sl]))
        outs.append(torch.stack([mu, ls.exp(), mg, lsg.exp()], 1).numpy())
    o = np.concatenate(outs)
    return o[:, 0], o[:, 1], o[:, 2], o[:, 3]


# ----- the original window-of-5 autoencoder, scored per transaction for comparison -----

def windows(S, idx, length=5):
    """Window start indices (global) of all length-5 windows inside a stream, for training."""
    idx = np.asarray(idx)
    return idx[S.pos[idx] + length <= S.stream_len[idx]]


def train_autoencoder(S, starts, latent=16, epochs=15, seed=0, batch=64, length=5):
    from freelance_finance_dl.model import TransactionAutoencoder
    torch.manual_seed(seed)
    model = TransactionAutoencoder(sequence_length=length, hidden_dim=64, latent_dim=latent)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.tensor(np.stack([S.z[starts + k] for k in range(length)], 1)).unsqueeze(-1)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(X), generator=gen).split(batch):
            opt.zero_grad()
            nn.functional.mse_loss(model(X[b]), X[b]).backward()
            opt.step()
    return model


@torch.no_grad()
def autoencoder_scores(model, S, idx, z_override=None, length=5):
    """Per-transaction score: squared reconstruction error at the transaction's position, averaged
    over every length-5 window of its stream that contains it. `z_override` replaces the target's
    own feature (for injected anomalies)."""
    model.eval()
    idx = np.asarray(idx)
    total = np.zeros(len(idx))
    count = np.zeros(len(idx))
    for k in range(length):                       # target sits at position k of the window
        start = idx - k
        ok = (S.pos[idx] - k >= 0) & (S.pos[idx] - k + length <= S.stream_len[idx])
        if not ok.any():
            continue
        st = start[ok]
        X = np.stack([S.z[st + j] for j in range(length)], 1).copy()
        if z_override is not None:
            X[:, k] = z_override[ok]
        Xt = torch.tensor(X).unsqueeze(-1)
        err = ((model(Xt) - Xt) ** 2).squeeze(-1)[:, k].numpy()
        total[ok] += err
        count[ok] += 1
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def calibrate(train_scores):
    """Return a function mapping scores to -log(tail probability) under the training scores, so
    scores on different scales can be combined by taking the maximum."""
    ref = np.sort(np.asarray(train_scores)[np.isfinite(train_scores)])
    n = len(ref)

    def f(s):
        s = np.nan_to_num(np.asarray(s, float), nan=-np.inf)
        above = n - np.searchsorted(ref, s, side="left")
        return -np.log((above + 1) / (n + 1))
    return f
