"""Time series low rank factor portfolio.

Characteristics do not predict returns directly. They determine a stock's exposure
to a small number of common return drivers, so the predictable part of the cross
section is low rank. Every model in this project has imposed that structure through
cross sectional regressions, one month at a time, recovering factor returns per
month and fitting loadings to explain them. This imposes it in the other direction.

Each characteristic is turned into a long short portfolio by ranking within the
month, demeaning and dividing by gross exposure. The daily returns of those
characteristic portfolios are then formed by applying the weights of month t to the
daily returns realised during month t plus one. That gives a panel of managed
portfolio returns at daily frequency, from which the covariance of a few hundred
portfolios can be estimated from thousands of observations rather than a few
hundred monthly ones.

The low rank structure is imposed on that panel. Its leading eigenvectors are
linear combinations of characteristic portfolios, and are themselves tradeable. A
network maps characteristics to weights over those components, conditioned on a
pooled cross sectional context, so a stock's exposure to each component depends on
the composition of its month. The network is fitted on the realised Sharpe ratio of
the resulting book.

The traded book combines the components using exponentially weighted moments, with
market beta projected out without an intercept so the net exposure survives, and is
then scaled so that its forecast volatility meets a target. That overlay was worth
far more than any change to the signal in earlier work on this panel, since the
Sharpe ratio is invariant to a constant scale and the gain comes entirely from
varying leverage with conditions the covariance can foresee.

The components are also not selected by variance alone. Taking the leading
eigenvectors of the managed portfolio covariance ranks directions by how much they
move, which is not the same as how much they earn: a combination with a high Sharpe
ratio but modest variance is discarded in favour of a large but unrewarded one.
Following the risk premium formulation of principal components, the selection
criterion adds the squared mean return of each direction to its variance, so
directions are kept for their return content as well as their movement. Unlike the
instrumented construction, whose book is invariant to any rotation of its loadings,
the components here are individually tradeable, so which basis is chosen changes the
portfolio and the criterion is not inert.

Temporal integrity: the eigenbasis is estimated from months at or before each
year's cutoff, the network is refitted yearly on the same window, and the coverage
filter is recomputed yearly.
"""

import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# low rank structure
n_components = 24
cov_lookback_months = 120
rp_weight = 1.0

# volatility overlay
use_vol_target = True
target_vol_annual = 0.10
gross_cap = 12.0
cov_lookback_days = 126
min_day_frac = 0.5
min_universe = 30
trading_days_per_month = 21.0
moment_halflife = 120.0
cov_shrink = 0.2
factor_ridge = 1e-1

# network
d_model = 128
d_hidden = 256
dropout = 0.1

# training
lr = 3e-4
weight_decay = 1e-4
grad_clip = 1.0
epochs_cold = 40
epochs_warm = 12
batch_months = 24
n_seeds = 4
base_seed = 24
min_train_months = 120

# beta neutralisation, None disables it
beta_col = "beta_60m"

# data preprocessing
min_coverage = 0.90
pre_test_date = "1990-01-01"
min_stocks = 30
max_miss_frac = 1.0 / 3.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
	random.seed(s)
	np.random.seed(s)
	torch.manual_seed(s)
	torch.cuda.manual_seed_all(s)


def resolve_daily(daily_ret):
	cols = list(daily_ret.columns)
	id_col = "id" if "id" in cols else next(c for c in cols if "id" in c.lower())
	date_col = next((c for c in ("date", "datadate", "day") if c in cols), None)
	if date_col is None:
		date_col = next(c for c in cols if np.issubdtype(daily_ret[c].dtype, np.datetime64))
	ret_col = next((c for c in ("ret_exc", "ret", "ret_local") if c in cols), None)
	if ret_col is None:
		ret_col = next(c for c in cols
		               if c not in (id_col, date_col) and np.issubdtype(daily_ret[c].dtype, np.number))
	return id_col, date_col, ret_col


def characteristic_portfolios(md, cols):
	"""Each characteristic as a long short portfolio of unit gross exposure."""
	r = md[cols].rank(pct=True).to_numpy(dtype=np.float64)
	r = np.nan_to_num(r, nan=0.5) - 0.5
	r = r - r.mean(axis=0, keepdims=True)
	return r / (np.abs(r).sum(axis=0, keepdims=True) + 1e-12)


def managed_daily_returns(by_month, cols, daily_ret):
	"""Daily returns of the characteristic portfolios.

	Weights formed at the end of month t are applied to daily returns during month
	t plus one, so nothing from the future enters.
	"""
	id_col, date_col, ret_col = resolve_daily(daily_ret)
	print("daily columns resolved to", id_col, date_col, ret_col, flush=True)
	d = pd.DataFrame({
		"id": pd.to_numeric(daily_ret[id_col], errors="coerce").astype("int64"),
		"date": pd.to_datetime(daily_ret[date_col]),
		"r": daily_ret[ret_col].astype("float64").fillna(0.0).to_numpy(),
	})
	d["eom"] = d["date"] + pd.offsets.MonthEnd(0)
	daily_by_month = {m: g for m, g in d.groupby("eom", sort=True)}

	rows, stamps = [], []
	months = sorted(by_month.keys())
	t0 = time.time()
	for i, m in enumerate(months):
		nxt = pd.Timestamp(m) + pd.offsets.MonthEnd(1)
		if nxt not in daily_by_month:
			continue
		md = by_month[m]
		md = md.loc[(md[cols].isna().sum(axis=1) <= len(cols) * max_miss_frac).to_numpy()]
		if len(md) < min_stocks:
			continue
		ids = md["id"].to_numpy()
		dm = daily_by_month[nxt]
		common = np.intersect1d(ids, dm["id"].unique())
		if len(common) < min_stocks:
			continue
		keep = np.isin(ids, common)
		md = md.loc[keep]
		order = np.argsort(md["id"].to_numpy())
		md = md.iloc[order]
		w = characteristic_portfolios(md, cols)

		dm = dm[dm["id"].isin(common)]
		pos = {v: k for k, v in enumerate(md["id"].to_numpy())}
		dates = np.sort(dm["date"].unique())
		dpos = {v: k for k, v in enumerate(dates)}
		mat = np.zeros((len(pos), len(dates)))
		mat[dm["id"].map(pos).to_numpy(), dm["date"].map(dpos).to_numpy()] = dm["r"].to_numpy()

		rows.append((w.T @ mat).T)
		stamps.append(np.full(len(dates), nxt))
		if (i + 1) % 200 == 0:
			print("  managed returns", i + 1, "of", len(months), "sec", round(time.time() - t0), flush=True)
	print("managed daily returns built in", round(time.time() - t0), "sec", flush=True)
	return np.vstack(rows), np.concatenate(stamps)


def eigenbasis(block):
	"""Directions of the managed portfolio panel, ranked by movement and return.

	The second moment about zero is the covariance plus the outer product of the
	means, so its leading eigenvectors rank a direction by variance and squared mean
	together. rp_weight scales the contribution of the mean, with zero recovering the
	usual variance ranking and larger values favouring directions that earn rather
	than merely move.
	"""
	cov = np.atleast_2d(np.cov(block, rowvar=False))
	k = cov.shape[0]
	if rp_weight > 0.0:
		mu = block.mean(axis=0)
		cov = cov + rp_weight * np.outer(mu, mu) * block.shape[0]
	cov = cov + factor_ridge * (np.trace(cov) / k) * np.eye(k)
	vals, vecs = np.linalg.eigh(cov)
	idx = np.argsort(vals)[::-1][:n_components]
	return vecs[:, idx]


def shrink_covariance(x):
	"""Ledoit and Wolf shrinkage toward constant correlation."""
	t, n = x.shape
	xc = x - x.mean(axis=0, keepdims=True)
	sm = (xc.T @ xc) / t
	var = np.maximum(np.diag(sm).copy(), 1e-16)
	sd = np.sqrt(var)
	outer = np.outer(sd, sd)
	off = ~np.eye(n, dtype=bool)
	r_bar = float((sm / outer)[off].mean()) if n > 1 else 0.0
	target = r_bar * outer
	np.fill_diagonal(target, var)
	x2 = xc * xc
	pi_mat = (x2.T @ x2) / t - sm * sm
	pi_hat = float(pi_mat.sum())
	a = ((x2 * xc).T @ xc) / t
	ratio = np.outer(1.0 / sd, sd)
	rho_off = (r_bar / 2.0) * (ratio * (a - var[:, None] * sm) + ratio.T * (a.T - var[None, :] * sm))
	rho_hat = float(np.diag(pi_mat).sum() + rho_off[off].sum())
	diff = target - sm
	gamma = float((diff * diff).sum())
	delta = 0.0 if gamma <= 1e-30 else float(np.clip((pi_hat - rho_hat) / gamma / t, 0.0, 1.0))
	cov = delta * target + (1.0 - delta) * sm
	cov.flat[:: n + 1] += 1e-12
	return cov


def book_risk(d_date, d_id, d_r, eom, ids):
	"""Forecast covariance of the traded universe from the trailing daily window.

	Sliced by position on a sorted date array rather than by boolean mask, so no
	comparison can propagate a missing value and only the relevant window is touched.
	"""
	hi = np.datetime64(pd.Timestamp(eom), "ns")
	lo = hi - np.timedelta64(int(cov_lookback_days * 1.6), "D")
	a = int(np.searchsorted(d_date, lo, side="right"))
	b = int(np.searchsorted(d_date, hi, side="right"))
	if b - a < 40:
		return None, None
	sub_id, sub_r, sub_d = d_id[a:b], d_r[a:b], d_date[a:b]
	pos_all = pd.Index(ids).get_indexer(sub_id)
	sel = pos_all >= 0
	if sel.sum() < 40:
		return None, None
	dates_u, day_ix = np.unique(sub_d[sel], return_inverse=True)
	if len(dates_u) < 40:
		return None, None
	if len(dates_u) > cov_lookback_days:
		cut = len(dates_u) - cov_lookback_days
		keep_rows = day_ix >= cut
		day_ix = day_ix[keep_rows] - cut
		stock_ix = pos_all[sel][keep_rows]
		vals = sub_r[sel][keep_rows]
		n_days = cov_lookback_days
	else:
		stock_ix = pos_all[sel]
		vals = sub_r[sel]
		n_days = len(dates_u)
	mat = np.full((n_days, len(ids)), np.nan)
	mat[day_ix, stock_ix] = vals
	frac = np.mean(np.isfinite(mat), axis=0)
	ok = np.flatnonzero(frac >= min_day_frac)
	if len(ok) < min_universe:
		return None, None
	x = mat[:, ok]
	cm = np.nanmean(x, axis=0)
	bad = np.where(np.isnan(x))
	x[bad] = np.take(cm, bad[1])
	return shrink_covariance(x) * trading_days_per_month, ok


class exposure_net(nn.Module):
	"""Characteristics to weights over the low rank components."""

	def __init__(self, n_features):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(n_features, d_hidden),
			nn.LayerNorm(d_hidden),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(d_hidden, d_model),
			nn.LayerNorm(d_model),
		)
		self.head = nn.Sequential(
			nn.Linear(2 * d_model, d_hidden),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(d_hidden, n_components),
		)

	def forward(self, x):
		z = self.encoder(x)
		ctx = z.mean(dim=0, keepdim=True).expand_as(z)
		h = self.head(torch.cat([z, ctx], dim=1))
		return h - h.mean(dim=0, keepdim=True)


def component_weights(h):
	"""Exposures to component portfolios, unit gross each."""
	h = h - h.mean(dim=0, keepdim=True)
	return h / (h.abs().sum(dim=0, keepdim=True) + 1e-8)


def train_exposures(model, x_list, r_list, n_epochs, device):
	"""Fit on the realised Sharpe of the equally combined components."""
	opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	model.train()
	xg = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in x_list]
	rg = [torch.as_tensor(r, dtype=torch.float32, device=device) for r in r_list]
	t = len(xg)
	for _ in range(n_epochs):
		order = np.random.permutation(t)
		for start in range(0, t, batch_months):
			idx = order[start:start + batch_months]
			if len(idx) < 6:
				continue
			rets = []
			for j in idx:
				w = component_weights(model(xg[j]))
				rets.append((w.transpose(0, 1) @ rg[j]).mean())
			rets = torch.stack(rets)
			sharpe = rets.mean() / (rets.std(unbiased=False) + 1e-8)
			opt.zero_grad(set_to_none=True)
			(-sharpe).backward()
			nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			opt.step()
	del xg, rg
	return model


@torch.no_grad()
def component_history(model, x_list, r_list, device):
	model.train(False)
	rows = []
	for x, r in zip(x_list, r_list):
		w = component_weights(model(torch.as_tensor(x, dtype=torch.float32, device=device)))
		rows.append((w.transpose(0, 1) @ torch.as_tensor(r, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64))
	return np.stack(rows, axis=0)


def combination_weights(f_hist):
	"""Combination from exponentially weighted moments."""
	n, k = f_hist.shape
	age = np.arange(n - 1, -1, -1, dtype=np.float64)
	w = np.exp(-np.log(2.0) * age / max(moment_halflife, 1.0))
	w = w / w.sum()
	mu = (f_hist * w[:, None]).sum(axis=0)
	d = f_hist - mu
	cov = np.atleast_2d((d * w[:, None]).T @ d / max(1.0 - (w ** 2).sum(), 1e-8))
	target = (np.trace(cov) / k) * np.eye(k)
	cov = (1.0 - cov_shrink) * cov + cov_shrink * target
	cov = cov + factor_ridge * (np.trace(cov) / k) * np.eye(k)
	try:
		return np.linalg.solve(cov, mu)
	except np.linalg.LinAlgError:
		return mu


@torch.no_grad()
def book_weights(model, lam, x, device):
	model.train(False)
	w = component_weights(model(torch.as_tensor(x, dtype=torch.float32, device=device)))
	raw = (w @ torch.as_tensor(lam, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)
	d = np.abs(raw).sum()
	return raw / d if d > 1e-12 else raw


def project_out_beta(w, beta):
	"""Remove the component of the book along beta, keeping net exposure."""
	if beta is None:
		return w
	b = beta.reshape(-1, 1)
	coef, _, _, _ = np.linalg.lstsq(b, w, rcond=None)
	return w - b @ coef


def sharpe_of(series):
	if len(series) < 12:
		return float("nan")
	sd = series.std(ddof=1)
	return float(series.mean() / sd * np.sqrt(12.0)) if sd > 1e-12 else float("nan")


def report_sharpe(out, work):
	ret = work[["id", "eom", "ret_exc_lead1m"]].rename(columns={"ret_exc_lead1m": "r"}).dropna(subset=["r"])
	j = out.merge(ret, on=["id", "eom"], how="inner")
	j["c"] = j["w"] * j["r"]
	s = j.groupby("eom")["c"].sum().sort_index()
	pre = s[s.index < pre_test_date]
	post = s[s.index >= pre_test_date]
	print()
	print("months total", len(s), "pre", len(pre), "test", len(post), flush=True)
	print("TEST WINDOW   sharpe", round(sharpe_of(post), 3), flush=True)
	print("PRE TEST      sharpe", round(sharpe_of(pre), 3), flush=True)
	if len(post) > 1:
		print("annualised return", round(float(post.mean()) * 12.0, 4),
		      "volatility", round(float(post.std(ddof=1)) * np.sqrt(12.0), 4), flush=True)


def main(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
	t_start = time.time()
	print("torch", torch.__version__, "device", device, "components", n_components, "rp", rp_weight, "vol_target", use_vol_target,
	      "seeds", n_seeds, "beta", beta_col, flush=True)
	seed_all(base_seed)

	candidates = [f for f in features["features"].tolist() if f in chars.columns]
	work = chars.copy()
	work["eom"] = pd.to_datetime(work["eom"]) + pd.offsets.MonthEnd(0)
	work["id"] = pd.to_numeric(work["id"], errors="coerce").astype("int64")

	pre = work[work["eom"] < pre_test_date]
	cols = [c for c in candidates if pre[c].notna().mean() >= min_coverage] if len(pre) else candidates
	n_in = len(cols)
	print("characteristics", n_in, flush=True)

	by_month = {eom: md for eom, md in work.groupby("eom", sort=True)}
	all_months = sorted(by_month.keys())
	target_months = [m for m in all_months
	                 if np.isfinite(by_month[m]["ret_exc_lead1m"].to_numpy(dtype=np.float64)).sum() >= min_stocks]

	print("building managed portfolio returns", flush=True)
	mret, mstamp = managed_daily_returns(by_month, cols, daily_ret)
	print("managed panel", mret.shape, flush=True)

	d_date = d_id = d_r = None
	if use_vol_target:
		idc, dtc, rtc = resolve_daily(daily_ret)
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
		print("daily rows for the overlay", len(d_date), flush=True)

	years = sorted({pd.Timestamp(m).year for m in all_months})
	results = []
	nets, lams, basis = [], [], None
	n_done = n_skip = n_scaled = 0

	for yi, yr in enumerate(years, 1):
		eoms = [m for m in all_months if pd.Timestamp(m).year == yr]
		cutoff = max([m for m in target_months if m < min(eoms)], default=None)
		if cutoff is None:
			continue
		train_months = [m for m in target_months if m <= cutoff]
		if len(train_months) < min_train_months:
			continue

		t0 = time.time()
		lo = pd.Timestamp(cutoff) - pd.offsets.MonthEnd(cov_lookback_months)
		sel = (mstamp > lo) & (mstamp <= cutoff)
		if sel.sum() < 200:
			continue
		basis = eigenbasis(mret[sel])

		x_list, r_list = [], []
		for mm in train_months:
			md = by_month[mm]
			md = md.loc[(md[cols].isna().sum(axis=1) <= n_in * max_miss_frac).to_numpy()]
			rv = md["ret_exc_lead1m"].to_numpy(dtype=np.float64)
			fin = np.isfinite(rv)
			if fin.sum() < min_stocks:
				continue
			md = md.loc[fin]
			x = md[cols].rank(pct=True).to_numpy(dtype=np.float32)
			x_list.append(np.nan_to_num(x, nan=0.5) - 0.5)
			r_list.append(rv[fin].astype(np.float32))
		if len(x_list) < min_train_months // 2:
			continue

		warm = len(nets) == n_seeds
		new_n, new_lam = [], []
		for s in range(n_seeds):
			seed_all(base_seed + s)
			net = exposure_net(n_in).to(device)
			n_ep = epochs_cold
			if warm:
				try:
					net.load_state_dict(nets[s].state_dict())
					n_ep = epochs_warm
				except Exception:
					n_ep = epochs_cold
			net = train_exposures(net, x_list, r_list, n_ep, device)
			new_n.append(net)
			new_lam.append(combination_weights(component_history(net, x_list, r_list, device)))
		nets, lams = new_n, new_lam
		del x_list, r_list
		print("year", yr, "(", yi, "of", len(years), ") train months", len(train_months),
		      "fit_sec", round(time.time() - t0),
		      "elapsed_min", round((time.time() - t_start) / 60.0, 1), flush=True)

		for eom in eoms:
			md = by_month[eom]
			md = md.loc[(md[cols].isna().sum(axis=1) <= n_in * max_miss_frac).to_numpy()]
			if len(md) < min_stocks:
				n_skip += 1
				continue
			x = md[cols].rank(pct=True).to_numpy(dtype=np.float32)
			x = np.nan_to_num(x, nan=0.5) - 0.5
			acc = np.zeros(len(md), dtype=np.float64)
			for net, lam in zip(nets, lams):
				acc += book_weights(net, lam, x, device)
			w = acc / len(nets)
			if beta_col is not None and beta_col in md.columns:
				bt = md[beta_col].rank(pct=True).to_numpy(dtype=np.float64)
				w = project_out_beta(w, np.nan_to_num(bt, nan=0.5) - 0.5)
			d = np.abs(w).sum()
			if d < 1e-12:
				n_skip += 1
				continue
			w = w / d
			ids_out = md["id"].to_numpy()

			if use_vol_target:
				sigma, ok = book_risk(d_date, d_id, d_r, eom, ids_out)
				if sigma is None:
					n_skip += 1
					continue
				w = w[ok]
				ids_out = ids_out[ok]
				d = np.abs(w).sum()
				if d < 1e-12:
					n_skip += 1
					continue
				w = w / d
				vol = float(np.sqrt(max(w @ (sigma @ w), 0.0)))
				if vol < 1e-12:
					n_skip += 1
					continue
				w = w * min((target_vol_annual / np.sqrt(12.0)) / vol, gross_cap)
				n_scaled += 1

			results.append(pd.DataFrame({"id": ids_out, "eom": eom, "w": w}))
			n_done += 1

	if not results:
		raise ValueError("no weights produced")

	out = pd.concat(results, ignore_index=True)
	print("months solved", n_done, "skipped", n_skip, "volatility scaled", n_scaled,
	      "total_min", round((time.time() - t_start) / 60.0, 1), flush=True)
	print("output rows", len(out), "months", out["eom"].nunique(), flush=True)

	report_sharpe(out, work)

	out["eom"] = pd.to_datetime(out["eom"]).dt.strftime("%Y-%m-%d")
	out["id"] = out["id"].astype(int)
	out["w"] = out["w"].astype(float)
	return out[["id", "eom", "w"]]


if __name__ == "__main__":
	chars = pd.read_parquet("jkp-data/chars.parquet")
	features = pd.read_parquet("jkp-data/features.parquet")
	daily_ret = pd.read_parquet("jkp-data/daily_ret.parquet")
	pf = main(chars, features, daily_ret)
	pf.to_csv("output2.csv", index=False)
	print("wrote output.csv rows", len(pf), "months", pf["eom"].nunique(), flush=True)
