"""Cross sectional MLP with a penalised tangency minus minimum variance book.

The signal is unchanged: a network scores each stock from its characteristics,
mixed with a pooled cross sectional context, and is fitted under the maximum
Sharpe ratio regression objective on a rolling window.

Earlier work applied this same construction to a coupled factor model, whose
score is already a tangency combination of 32 factors under a 32 by 32 factor
covariance. Stacking a second risk aware step on top of a score that has already
been shaped by one risk model gave a mixed result, better than the raw construction
but still short of the incumbent. This applies the construction to a raw per stock
score instead, with no risk adjustment already built into it, which is a closer
match to how the technique was used on the leaderboard entry that motivated it.

The book is the fully invested tangency portfolio minus the fully invested minimum
variance portfolio under a heavily ridged covariance of the trailing daily window,
so it holds none of the view free component and sums to zero:

    a is A inverse applied to the score, b is A inverse applied to the vector of
    ones, w is a minus b scaled by the ratio of their sums, normalised to unit
    gross.

Shrinkage is toward a scaled identity with a closed form intensity, and the ridge
is a multiple of the mean eigenvalue, so nothing about the covariance needs to be
chosen beyond how strongly to penalise it.

Market beta is projected out of the book without an intercept, so the net exposure
of the book survives and only the component lying along beta is removed. That
matters because demeaning a book on this panel has cost roughly half a Sharpe point
in earlier work, since the net tilt is itself rewarded while the residual beta is
not.

use_mvgmv of False recovers the plain cross sectional MLP with unit gross weights
and no risk construction."""

import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# architecture
d_model = 128
d_hidden = 256
lr = 5e-5
weight_decay = 1e-4
grad_clip = 1.0

# training
window = 60
n_epochs_cold = 50
n_epochs_warm = 10
n_seeds = 3
min_obs = 60

# data preprocessing
min_coverage = 0.90
pre_test_date = "1990-01-01"
min_stocks = 30
max_miss_frac = 1.0 / 3.0

# penalised tangency minus minimum variance book
use_mvgmv = True
mvgmv_window_days = 756
mvgmv_kappa = 3.0
mvgmv_min_obs = 250
mvgmv_min_names = 50

# volatility overlay, act on the finished book the same way as elsewhere
use_vol_target = True
target_vol_annual = 0.10
gross_cap = 12.0
cov_lookback_days = 126
min_day_frac = 0.5
min_universe = 30
trading_days_per_month = 21.0

# beta neutralisation, None disables it
beta_col = "beta_60m"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def lw_identity_covariance(x):
    """Ledoit and Wolf shrinkage toward a scaled identity, closed form intensity."""
    t, n = x.shape
    xc = x - x.mean(axis=0, keepdims=True)
    sm = (xc.T @ xc) / t
    tr_s = float(np.trace(sm))
    mu_lw = tr_s / n
    norm2 = float(np.square(sm).sum())
    d2 = norm2 - 2.0 * mu_lw * tr_s + n * mu_lw ** 2
    row_sq = np.square(xc).sum(axis=1)
    b2 = float(np.square(row_sq).sum()) / (t ** 2) - norm2 / t
    delta = min(max(b2, 0.0), d2) / d2 if d2 > 1e-300 else 1.0
    sm = sm * (1.0 - delta)
    sm[np.diag_indices(n)] += delta * mu_lw
    return sm


def tangency_minus_gmv(mu, sigma, kappa):
    """Fully invested tangency minus fully invested minimum variance."""
    n = mu.shape[0]
    lam = kappa * float(np.trace(sigma)) / n
    a_mat = sigma + lam * np.eye(n)
    one = np.ones(n)
    try:
        sol = np.linalg.solve(a_mat, np.column_stack([mu, one]))
    except np.linalg.LinAlgError:
        return None
    a, b = sol[:, 0], sol[:, 1]
    sum_b = float(one @ b)
    if not np.isfinite(sum_b) or abs(sum_b) < 1e-18:
        return None
    w = a - (float(one @ a) / sum_b) * b
    d = np.abs(w).sum()
    return w / d if d > 1e-18 else None


def daily_window(d_date, d_id, d_r, eom, ids, window_days, min_obs, min_names):
    """Trailing daily returns for the traded universe, as a dense matrix."""
    hi = np.datetime64(pd.Timestamp(eom), "ns")
    lo = hi - np.timedelta64(int(window_days * 1.6), "D")
    a = int(np.searchsorted(d_date, lo, side="right"))
    b = int(np.searchsorted(d_date, hi, side="right"))
    if b - a < min_obs:
        return None, None
    sub_id, sub_r, sub_d = d_id[a:b], d_r[a:b], d_date[a:b]
    pos_all = pd.Index(ids).get_indexer(sub_id)
    sel = pos_all >= 0
    if sel.sum() < min_obs:
        return None, None
    dates_u, day_ix = np.unique(sub_d[sel], return_inverse=True)
    if len(dates_u) < min_obs:
        return None, None
    if len(dates_u) > window_days:
        cut = len(dates_u) - window_days
        keep = day_ix >= cut
        day_ix = day_ix[keep] - cut
        stock_ix = pos_all[sel][keep]
        vals = sub_r[sel][keep]
        n_days = window_days
    else:
        stock_ix = pos_all[sel]
        vals = sub_r[sel]
        n_days = len(dates_u)
    mat = np.full((n_days, len(ids)), np.nan)
    mat[day_ix, stock_ix] = vals
    ok = np.flatnonzero(np.isfinite(mat).sum(axis=0) >= min_obs)
    if len(ok) < min_names:
        return None, None
    x = mat[:, ok]
    cm = np.nanmean(x, axis=0)
    bad = np.where(np.isnan(x))
    x[bad] = np.take(cm, bad[1])
    return x, ok


def project_out_beta(w, beta):
    """Remove the component of the book along beta, keeping net exposure.

    No intercept is included, so demeaning is not forced. Removing the net tilt as
    well as the beta exposure has cost performance in earlier work on this panel,
    since the net tilt is itself rewarded.
    """
    b = beta.reshape(-1, 1)
    coef, _, _, _ = np.linalg.lstsq(b, w, rcond=None)
    return w - b @ coef


class xmlp_net(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))

    def score(self, x):
        z = self.encoder(x)
        ctx = z.mean(dim=0, keepdim=True).expand_as(z)
        return self.head(torch.cat([z, ctx], dim=1)).squeeze(-1)


def train_model(model, x_list, r_list, n_epochs, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    xg = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in x_list]
    rg = [torch.as_tensor(r, dtype=torch.float32, device=device) for r in r_list]
    t = len(xg)
    for _ in range(n_epochs):
        for j in np.random.permutation(t):
            w = model.score(xg[j])
            loss = (1.0 - (w * rg[j]).sum()) ** 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
    del xg, rg
    return model


def rolling_backtest(train_months, month_x, month_ids, month_mask, month_r, n_features, device, d_date=None, d_id=None, d_r=None, month_beta=None):
    all_months = sorted(month_x.keys())
    seed_states = [None] * n_seeds
    results = []
    n_skip = 0
    n_done = 0
    t0 = time.time()

    for m_idx, eom in enumerate(all_months):
        cutoff = eom - pd.DateOffset(months=1)
        avail = [m for m in train_months if m <= cutoff]
        if len(avail) < min_obs:
            n_skip += 1
            continue

        win = avail[-window:] if len(avail) > window else avail
        x_list = [month_x[m][month_mask[m]] for m in win]
        r_list = [month_r[m] for m in win]

        x_oos = month_x[eom]
        ids_oos = month_ids[eom]
        if len(ids_oos) < min_stocks:
            n_skip += 1
            continue

        w_sum = np.zeros(len(ids_oos), dtype=np.float64)
        n_valid = 0
        for s in range(n_seeds):
            seed_all(s * 10000 + 42)
            model = xmlp_net(n_features).to(device)
            if seed_states[s] is not None:
                model.load_state_dict(seed_states[s])
                n_ep = n_epochs_warm
            else:
                n_ep = n_epochs_cold
            model = train_model(model, x_list, r_list, n_ep, device)
            seed_states[s] = {k: v.cpu() for k, v in model.state_dict().items()}
            with torch.no_grad():
                w = model.score(torch.as_tensor(x_oos, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)
            denom = np.abs(w).sum()
            if denom > 1e-10:
                w_sum += w / denom
                n_valid += 1
            del model

        if n_valid == 0:
            n_skip += 1
            continue

        mu = w_sum / n_valid
        ids_out = ids_oos

        if use_mvgmv:
            xw, ok = daily_window(d_date, d_id, d_r, eom, ids_out,
                                  mvgmv_window_days, mvgmv_min_obs, mvgmv_min_names)
            if xw is None:
                n_skip += 1
                continue
            sigma = lw_identity_covariance(xw)
            wm = tangency_minus_gmv(mu[ok], sigma, mvgmv_kappa)
            if wm is None:
                n_skip += 1
                continue
            mu = wm
            ids_out = ids_out[ok]
            del xw, sigma

        if beta_col is not None and month_beta is not None:
            bt = month_beta.get(eom)
            if bt is not None:
                b_ids, b_vals = bt
                pos = pd.Index(b_ids).get_indexer(ids_out)
                sel = pos >= 0
                if sel.all():
                    r_beta = pd.Series(b_vals[pos]).rank(pct=True).to_numpy()
                    mu = project_out_beta(mu, r_beta - 0.5)
                elif sel.sum() >= min_stocks:
                    # some names in the book lack a beta observation; those are left
                    # unadjusted rather than dropped from the book
                    r_beta = np.full(len(ids_out), 0.5)
                    r_beta[sel] = pd.Series(b_vals[pos[sel]]).rank(pct=True).to_numpy()
                    mu = project_out_beta(mu, r_beta - 0.5)

        if use_vol_target:
            xw, ok = daily_window(d_date, d_id, d_r, eom, ids_out,
                                  cov_lookback_days, int(cov_lookback_days * min_day_frac), min_universe)
            if xw is None:
                n_skip += 1
                continue
            sigma = lw_identity_covariance(xw) * trading_days_per_month
            wl = mu[ok]
            d_gross = np.abs(wl).sum()
            if d_gross < 1e-12:
                n_skip += 1
                continue
            wl = wl / d_gross
            vol = float(np.sqrt(max(wl @ (sigma @ wl), 0.0)))
            if vol < 1e-12:
                n_skip += 1
                continue
            mu = wl * min((target_vol_annual / np.sqrt(12.0)) / vol, gross_cap)
            ids_out = ids_out[ok]
        else:
            d_gross = np.abs(mu).sum()
            mu = mu / d_gross if d_gross > 1e-12 else mu

        w_avg = mu.astype(np.float32)
        keep = np.abs(w_avg) > 1e-15
        if keep.any():
            results.append(pd.DataFrame({"id": ids_out[keep], "eom": eom, "w": w_avg[keep]}))
        n_done += 1
        if n_done % 50 == 0 or (m_idx + 1) == len(all_months):
            el = time.time() - t0
            print("progress", m_idx + 1, "of", len(all_months), "done", n_done, "skip", n_skip, "elapsed_min", round(el / 60.0, 1), flush=True)

    el = time.time() - t0
    print("backtest complete oos_months", len(results), "skipped", n_skip, "total_min", round(el / 60.0, 1), flush=True)
    if not results:
        return pd.DataFrame(columns=["id", "eom", "w"])
    return pd.concat(results, ignore_index=True)


def main(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
    print("torch", torch.__version__, "device", device, "mvgmv", use_mvgmv, "kappa", mvgmv_kappa, "vol_target", use_vol_target, flush=True)
    seed_all(42)

    feature_names = features["features"].tolist()
    feat_cols = sorted([f for f in feature_names if f in chars.columns])
    print("candidate features", len(feat_cols), flush=True)

    chars = chars.copy()
    chars["eom"] = pd.to_datetime(chars["eom"])

    pre_test = chars[chars["eom"] < pre_test_date]
    coverage = pre_test[feat_cols].notna().mean()
    valid_features = sorted([f for f in feat_cols if coverage[f] >= min_coverage])
    n_features = len(valid_features)
    print("features at coverage", min_coverage, "pre", pre_test_date[:4], "n", n_features, flush=True)

    month_x = {}
    month_ids = {}
    month_mask = {}
    month_r = {}

    month_beta = {}
    n_months = chars["eom"].nunique()
    print("processing months", n_months, flush=True)
    t0 = time.time()
    for i, (eom, md) in enumerate(chars.groupby("eom", sort=True)):
        n_missing = md[valid_features].isna().sum(axis=1)
        keep_stock = (n_missing <= n_features * max_miss_frac).values
        md = md.loc[keep_stock]
        if len(md) < min_stocks:
            continue
        x = md[valid_features].rank(pct=True).values.astype(np.float32)
        x = np.nan_to_num(x, nan=0.5) - 0.5
        ids = md["id"].values
        rets = md["ret_exc_lead1m"].values
        has_ret = np.isfinite(rets)
        month_x[eom] = x
        month_ids[eom] = ids
        if has_ret.sum() >= min_stocks:
            month_mask[eom] = has_ret
            month_r[eom] = rets[has_ret].astype(np.float32)
        if beta_col is not None and beta_col in md.columns:
            bt = md[beta_col].to_numpy(dtype=np.float64)
            ok = np.isfinite(bt)
            if ok.sum() >= min_stocks:
                month_beta[eom] = (ids[ok], bt[ok])
        if (i + 1) % 200 == 0:
            print("  processed", i + 1, "of", n_months, "sec", round(time.time() - t0), flush=True)

    train_months = sorted(month_r.keys())
    print("train months", len(train_months), "total months", len(month_x), "n_features", n_features, flush=True)

    d_date = d_id = d_r = None
    if use_mvgmv or use_vol_target:
        cols_d = list(daily_ret.columns)
        idc = "id" if "id" in cols_d else next(c for c in cols_d if "id" in c.lower())
        dtc = next(c for c in ("date", "datadate", "day") if c in cols_d)
        rtc = next(c for c in ("ret_exc", "ret", "ret_local") if c in cols_d)
        dd = daily_ret[[idc, dtc, rtc]].copy()
        dd.columns = ["id", "date", "r"]
        dd["date"] = pd.to_datetime(dd["date"], errors="coerce")
        dd["id"] = pd.to_numeric(dd["id"], errors="coerce")
        dd["r"] = pd.to_numeric(dd["r"], errors="coerce")
        dd = dd.dropna(subset=["id", "date", "r"])
        dd["id"] = dd["id"].astype("int64")
        dd = dd.sort_values("date", kind="mergesort")
        d_date = dd["date"].to_numpy(dtype="datetime64[ns]")
        d_id = dd["id"].to_numpy()
        d_r = dd["r"].to_numpy(dtype=np.float64)
        del dd
        print("daily rows", len(d_date), flush=True)

    output = rolling_backtest(train_months, month_x, month_ids, month_mask, month_r, n_features, device, d_date, d_id, d_r, month_beta)
    print("output rows", len(output), "months", output["eom"].nunique() if len(output) else 0, flush=True)

    if len(output):
        output["eom"] = pd.to_datetime(output["eom"]).dt.strftime("%Y-%m-%d")
        output["id"] = output["id"].astype(int)
        output["w"] = output["w"].astype(float)
    return output[["id", "eom", "w"]]


if __name__ == "__main__":
    chars = pd.read_parquet("jkp-data/chars.parquet")
    features = pd.read_parquet("jkp-data/features.parquet")
    daily_ret = pd.read_parquet("jkp-data/daily_ret.parquet")
    pf = main(chars, features, daily_ret)
    pf.to_csv("output.csv", index=False)
