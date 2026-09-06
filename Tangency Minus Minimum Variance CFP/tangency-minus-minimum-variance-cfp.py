"""Coupled factor signal with a penalised tangency minus minimum variance book.

The signal is unchanged: a network maps characteristics to k latent factor
loadings conditioned on a pooled cross sectional context, each loading column is
normalised into a long short factor portfolio, a second network supplies
idiosyncratic precision, and the factors are combined under exponentially weighted
moments. That combined score is then treated as an expected return and the book is
solved in the full stock space.

An earlier attempt at this construction lost heavily, and the diagnosis was that
the covariance was too poorly conditioned to invert: realised volatility overshot
its forecast in every month. Two settings were wrong rather than the construction.
The window was 126 daily observations on a cross section of well over a thousand
names, and the ridge was a thousandth of the mean eigenvalue. Here the window is
756 days and the ridge equals the mean eigenvalue, so the inverse is dominated by
the prior rather than by the sample. Shrinkage is toward a scaled identity, whose
intensity is closed form and needs no choosing.

The book is the tangency portfolio minus the minimum variance portfolio, both
fully invested under the ridged covariance, so their difference holds no view free
component and sums to zero by construction:

	a is A inverse applied to mu, b is A inverse applied to the vector of ones,
	w is a minus b scaled by the ratio of their sums, normalised to unit gross.

The minimum variance leg depends only on the covariance and expresses no view, so
removing it leaves the part the signal actually informs.

use_vol_target scales the finished book to a volatility target as before, setting
it to False gives the plain unit gross book."""

import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# latent factors
k_factors = 32
moment_halflife = 120.0
cov_shrink = 0.2
factor_ridge = 1e-4

# precision weighting
use_precision = True
precision_clip = (-4.0, 4.0)

# networks
d_model = 128
d_hidden = 256
dropout = 0.1

# training
lr = 3e-4
weight_decay = 1e-4
grad_clip = 1.0
epochs_cold = 40
epochs_warm = 12
epochs_var = 20
batch_months = 24
n_seeds = 4
base_seed = 42
min_train_months = 120

# beta neutralisation
beta_col = "beta_60m"

# penalised tangency minus minimum variance book
use_mvgmv = True
mvgmv_window_days = 756
mvgmv_kappa = 1.0
mvgmv_min_obs = 250
mvgmv_min_names = 50

# volatility targeting. Each month the book is rescaled so its ex ante volatility,
# measured against a shrinkage covariance from the trailing year of daily returns,
# meets the target, subject to a gross exposure cap.
use_vol_target = True
target_vol_annual = 0.10
gross_cap = 12.0
cov_lookback_days = 126
min_day_frac = 0.5
min_universe = 30
trading_days_per_month = 21.0

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


class loading_net(nn.Module):
	"""Characteristics to k loadings, conditioned on the cross section."""

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
			nn.Linear(d_hidden, k_factors),
		)

	def forward(self, x):
		z = self.encoder(x)
		ctx = z.mean(dim=0, keepdim=True).expand_as(z)
		h = self.head(torch.cat([z, ctx], dim=1))
		return h - h.mean(dim=0, keepdim=True)


class precision_net(nn.Module):
	"""Characteristics to a log idiosyncratic variance."""

	def __init__(self, n_features):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_features, d_hidden),
			nn.LayerNorm(d_hidden),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(d_hidden, d_hidden),
			nn.GELU(),
			nn.Linear(d_hidden, 1),
		)

	def forward(self, x):
		return self.net(x).clamp(precision_clip[0], precision_clip[1]).squeeze(-1)


def resolve(df, candidates, label):
	for c in candidates:
		if c in df.columns:
			return c
	raise KeyError(label + " not resolvable")


def build_daily(daily_ret):
	idc = resolve(daily_ret, ["id", "permno"], "daily id")
	datec = resolve(daily_ret, ["date", "day"], "daily date")
	retc = resolve(daily_ret, ["ret_exc", "ret", "ret_local"], "daily return")
	d = daily_ret[[idc, datec, retc]].copy()
	d.columns = ["id", "date", "r"]
	d["date"] = pd.to_datetime(d["date"])
	d = d.dropna(subset=["id", "date", "r"])
	return d.sort_values("date", kind="mergesort").reset_index(drop=True)


def shrink_covariance(x):
	"""Shrinkage covariance with a constant correlation target. x is T by N."""
	t, n = x.shape
	xc = x - x.mean(axis=0, keepdims=True)
	s = (xc.T @ xc) / t
	var = np.maximum(np.diag(s).copy(), 1e-16)
	sd = np.sqrt(var)
	outer_sd = np.outer(sd, sd)
	off = ~np.eye(n, dtype=bool)
	r_bar = float((s / outer_sd)[off].mean()) if n > 1 else 0.0
	target = r_bar * outer_sd
	np.fill_diagonal(target, var)

	x2 = xc * xc
	pi_mat = (x2.T @ x2) / t - s * s
	a = ((x2 * xc).T @ xc) / t
	ratio = np.outer(1.0 / sd, sd)
	rho_off = (r_bar / 2.0) * (ratio * (a - var[:, None] * s) + ratio.T * (a.T - var[None, :] * s))
	rho_hat = float(np.diag(pi_mat).sum() + rho_off[off].sum())
	diff = target - s
	gamma_hat = float((diff * diff).sum())
	delta = 0.0 if gamma_hat <= 1e-30 else float(np.clip((float(pi_mat.sum()) - rho_hat) / gamma_hat / t, 0.0, 1.0))

	sigma = delta * target + (1.0 - delta) * s
	sigma.flat[:: n + 1] += 1e-12
	return sigma


def covariance_for(daily, eom, ids):
	"""Monthly covariance from the trailing year of daily returns."""
	start = eom - pd.DateOffset(days=int(cov_lookback_days * 1.6))
	win = daily[(daily["date"] > start) & (daily["date"] <= eom)]
	if not len(win):
		return None, None
	win = win[win["id"].isin(set(ids.tolist()))]
	if not len(win):
		return None, None
	piv = win.pivot_table(index="date", columns="id", values="r", aggfunc="last").tail(cov_lookback_days)
	if piv.shape[0] < 40:
		return None, None
	frac = piv.notna().mean(axis=0)
	keep = frac[frac >= min_day_frac].index
	if len(keep) < min_universe:
		return None, None
	piv = piv[keep]
	x = np.array(piv.to_numpy(dtype=np.float64), copy=True)
	col_mean = np.nanmean(x, axis=0)
	inds = np.where(np.isnan(x))
	x[inds] = np.take(col_mean, inds[1])
	return shrink_covariance(x) * trading_days_per_month, np.asarray(keep)


def factor_weights(h, omega=None):
	"""Loadings to factor portfolios, one unit of gross exposure each."""
	if omega is not None:
		h = h * omega.unsqueeze(1)
	h = h - h.mean(dim=0, keepdim=True)
	return h / (h.abs().sum(dim=0, keepdim=True) + 1e-8)


def train_loadings(model, prec, x_list, r_list, n_epochs, device):
	"""Fit loadings on the realised Sharpe of the equally combined factors."""
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
				om = None
				if prec is not None:
					with torch.no_grad():
						om = torch.exp(-prec(xg[j]))
						om = om / (om.mean() + 1e-8)
				w = factor_weights(model(xg[j]), om)
				rets.append((w.transpose(0, 1) @ rg[j]).mean())
			rets = torch.stack(rets)
			sharpe = rets.mean() / (rets.std(unbiased=False) + 1e-8)
			opt.zero_grad(set_to_none=True)
			(-sharpe).backward()
			nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			opt.step()
	del xg, rg
	return model


def train_precision(model, loadings, x_list, r_list, n_epochs, device):
	"""Gaussian negative log likelihood on the fitted residuals."""
	loadings.train(False)
	net = model
	opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
	blocks = []
	with torch.no_grad():
		for x, r in zip(x_list, r_list):
			xg = torch.as_tensor(x, dtype=torch.float32, device=device)
			rg = torch.as_tensor(r, dtype=torch.float32, device=device)
			h = loadings(xg)
			eye = factor_ridge * torch.eye(h.shape[1], device=device)
			f = torch.linalg.solve(h.transpose(0, 1) @ h + eye, h.transpose(0, 1) @ rg)
			blocks.append((xg, ((h @ f) - rg) ** 2))
	net.train()
	order = list(range(len(blocks)))
	for _ in range(n_epochs):
		random.shuffle(order)
		for j in order:
			xg, res_sq = blocks[j]
			log_var = net(xg)
			loss = (log_var + res_sq * torch.exp(-log_var)).mean()
			opt.zero_grad(set_to_none=True)
			loss.backward()
			nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
			opt.step()
	del blocks
	net.train(False)
	return net


@torch.no_grad()
def factor_history(loadings, prec, x_list, r_list, device):
	loadings.train(False)
	rows = []
	for x, r in zip(x_list, r_list):
		xg = torch.as_tensor(x, dtype=torch.float32, device=device)
		om = None
		if prec is not None:
			om = torch.exp(-prec(xg))
			om = om / (om.mean() + 1e-8)
		w = factor_weights(loadings(xg), om)
		f = w.transpose(0, 1) @ torch.as_tensor(r, dtype=torch.float32, device=device)
		rows.append(f.cpu().numpy().astype(np.float64))
	return np.stack(rows, axis=0)


def combination_weights(f_hist):
	"""Factor combination from exponentially weighted moments."""
	n, k = f_hist.shape
	age = np.arange(n - 1, -1, -1, dtype=np.float64)
	w = np.exp(-np.log(2.0) * age / max(moment_halflife, 1.0))
	w = w / w.sum()

	mu = (f_hist * w[:, None]).sum(axis=0)
	d = f_hist - mu
	cov = (d * w[:, None]).T @ d / max(1.0 - (w ** 2).sum(), 1e-8)
	cov = np.atleast_2d(cov)
	target = (np.trace(cov) / k) * np.eye(k)
	cov = (1.0 - cov_shrink) * cov + cov_shrink * target
	cov = cov + factor_ridge * (np.trace(cov) / k) * np.eye(k)
	try:
		return np.linalg.solve(cov, mu)
	except np.linalg.LinAlgError:
		return mu


@torch.no_grad()
def book_weights(loadings, prec, lam, x, device):
	loadings.train(False)
	xg = torch.as_tensor(x, dtype=torch.float32, device=device)
	om = None
	if prec is not None:
		om = torch.exp(-prec(xg))
		om = om / (om.mean() + 1e-8)
	w = factor_weights(loadings(xg), om)
	raw = (w @ torch.as_tensor(lam, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)
	d = np.abs(raw).sum()
	return raw / d if d > 1e-12 else raw


def lw_identity_covariance(x):
	"""Ledoit and Wolf shrinkage toward a scaled identity.

	The intensity is closed form, so there is nothing to select. The target is the
	mean eigenvalue on the diagonal, which is the right prior when the cross section
	is far wider than the window and the sample correlations carry little
	information.
	"""
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
	"""Fully invested tangency minus fully invested minimum variance.

	Both legs sum to one under the ridged covariance, so their difference sums to
	zero and holds none of the minimum variance component. The ridge is a multiple
	of the mean eigenvalue, which makes it scale free in the units of the returns
	and in the width of the cross section.
	"""
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


def mvgmv_window(d_date, d_id, d_r, eom, ids):
	"""Trailing daily returns for the traded universe, as a dense matrix.

	A stock is held only if it has at least mvgmv_min_obs observations in the
	window, which is what keeps the covariance from being driven by names that
	barely traded.
	"""
	hi = np.datetime64(pd.Timestamp(eom), "ns")
	lo = hi - np.timedelta64(int(mvgmv_window_days * 1.6), "D")
	a = int(np.searchsorted(d_date, lo, side="right"))
	b = int(np.searchsorted(d_date, hi, side="right"))
	if b - a < mvgmv_min_obs:
		return None, None
	sub_id, sub_r, sub_d = d_id[a:b], d_r[a:b], d_date[a:b]
	pos_all = pd.Index(ids).get_indexer(sub_id)
	sel = pos_all >= 0
	if sel.sum() < mvgmv_min_obs:
		return None, None
	dates_u, day_ix = np.unique(sub_d[sel], return_inverse=True)
	if len(dates_u) < mvgmv_min_obs:
		return None, None
	if len(dates_u) > mvgmv_window_days:
		cut = len(dates_u) - mvgmv_window_days
		keep = day_ix >= cut
		day_ix = day_ix[keep] - cut
		stock_ix = pos_all[sel][keep]
		vals = sub_r[sel][keep]
		n_days = mvgmv_window_days
	else:
		stock_ix = pos_all[sel]
		vals = sub_r[sel]
		n_days = len(dates_u)
	mat = np.full((n_days, len(ids)), np.nan)
	mat[day_ix, stock_ix] = vals
	ok = np.flatnonzero(np.isfinite(mat).sum(axis=0) >= mvgmv_min_obs)
	if len(ok) < mvgmv_min_names:
		return None, None
	x = mat[:, ok]
	cm = np.nanmean(x, axis=0)
	bad = np.where(np.isnan(x))
	x[bad] = np.take(cm, bad[1])
	return x, ok


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
	print("torch", torch.__version__, "device", device, "seeds", n_seeds, "k", k_factors,
	      "vol_target", use_vol_target, "mvgmv", use_mvgmv, "window", mvgmv_window_days, "kappa", mvgmv_kappa, flush=True)
	seed_all(base_seed)

	candidates = [f for f in features["features"].tolist() if f in chars.columns]
	work = chars.copy()
	work["eom"] = pd.to_datetime(work["eom"]) + pd.offsets.MonthEnd(0)
	work["id"] = pd.to_numeric(work["id"], errors="coerce").astype("int64")

	daily = None
	if use_vol_target:
		if daily_ret is None or not len(daily_ret):
			raise ValueError("daily returns are required when use_vol_target is on")
		daily = build_daily(daily_ret)

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

	by_month = {eom: md for eom, md in work.groupby("eom", sort=True)}
	all_months = sorted(by_month.keys())
	target_months = [m for m in all_months
	                 if np.isfinite(by_month[m]["ret_exc_lead1m"].to_numpy(dtype=np.float64)).sum() >= min_stocks]
	print("months", len(all_months), "with targets", len(target_months), flush=True)

	years = sorted({pd.Timestamp(m).year for m in all_months})
	target_vol = target_vol_annual / np.sqrt(12.0)
	n_scaled = 0
	results = []
	loaders, precs, lams, cols = [], [], [], None
	n_done = 0
	n_skip = 0

	for yi, yr in enumerate(years, 1):
		eoms = [m for m in all_months if pd.Timestamp(m).year == yr]
		cutoff = max([m for m in target_months if m < min(eoms)], default=None)
		if cutoff is None:
			continue
		train_months = [m for m in target_months if m <= cutoff]
		if len(train_months) < min_train_months:
			continue

		t0 = time.time()
		panel = work[work["eom"].isin(train_months)]
		frac = panel[candidates].notna().mean()
		cols = [c for c in candidates if frac[c] >= min_coverage]
		n_in = len(cols)

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

		warm = len(loaders) == n_seeds
		new_l, new_p, new_lam = [], [], []
		for s in range(n_seeds):
			seed_all(base_seed + s)
			ln = loading_net(n_in).to(device)
			pn = precision_net(n_in).to(device) if use_precision else None
			n_ep = epochs_cold
			if warm:
				try:
					ln.load_state_dict(loaders[s].state_dict())
					if pn is not None and precs[s] is not None:
						pn.load_state_dict(precs[s].state_dict())
					n_ep = epochs_warm
				except Exception:
					n_ep = epochs_cold
			ln = train_loadings(ln, None, x_list, r_list, n_ep, device)
			if pn is not None:
				pn = train_precision(pn, ln, x_list, r_list, epochs_var, device)
				ln = train_loadings(ln, pn, x_list, r_list, max(n_ep // 2, 4), device)
			f_hist = factor_history(ln, pn, x_list, r_list, device)
			new_l.append(ln)
			new_p.append(pn)
			new_lam.append(combination_weights(f_hist))
		loaders, precs, lams = new_l, new_p, new_lam
		del x_list, r_list
		print("year", yr, "(", yi, "of", len(years), ") train months", len(train_months),
		      "inputs", n_in, "fit_sec", round(time.time() - t0),
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
			for ln, pn, lam in zip(loaders, precs, lams):
				acc += book_weights(ln, pn, lam, x, device)
			w = acc / len(loaders)
			if beta_col is not None and beta_col in md.columns:
				bt = md[beta_col].rank(pct=True).to_numpy(dtype=np.float64)
				w = project_out_beta(w, np.nan_to_num(bt, nan=0.5) - 0.5)
			d = np.abs(w).sum()
			if d < 1e-12:
				n_skip += 1
				continue
			w = w / d
			ids_out = md["id"].to_numpy()

			if use_mvgmv:
				# the combined factor score is the expected return, and the book is
				# the tangency minus minimum variance solution under a heavily
				# ridged covariance of the trailing daily window
				xw, ok = mvgmv_window(d_date, d_id, d_r, eom, ids_out)
				if xw is None:
					n_skip += 1
					continue
				sigma = lw_identity_covariance(xw)
				wm = tangency_minus_gmv(w[ok], sigma, mvgmv_kappa)
				if wm is None:
					n_skip += 1
					continue
				w = wm
				ids_out = ids_out[ok]
				del xw, sigma

			if use_vol_target:
				sigma, ok2 = covariance_for(daily, eom, ids_out)
				if sigma is None:
					n_skip += 1
					continue
				pos = pd.Index(ids_out).get_indexer(ok2)
				sel = pos >= 0
				if sel.sum() < min_universe:
					n_skip += 1
					continue
				sigma = sigma[np.ix_(sel, sel)]
				w = w[pos[sel]]
				ids_out = ids_out[pos[sel]]
				d = np.abs(w).sum()
				if d < 1e-12:
					n_skip += 1
					continue
				w = w / d
				vol = float(np.sqrt(max(w @ (sigma @ w), 0.0)))
				if vol < 1e-12:
					n_skip += 1
					continue
				w = w * min(target_vol / vol, gross_cap)
				n_scaled += 1

			results.append(pd.DataFrame({"id": ids_out, "eom": eom, "w": w}))
			n_done += 1

	if not results:
		raise ValueError("no weights produced")

	out = pd.concat(results, ignore_index=True)
	print("months solved", n_done, "skipped", n_skip, "volatility scaled", n_scaled,
	      "rows", len(out), "total_min", round((time.time() - t_start) / 60.0, 1), flush=True)

	report_sharpe(out, work)

	out["eom"] = pd.to_datetime(out["eom"]).dt.strftime("%Y-%m-%d")
	out["id"] = out["id"].astype(int)
	out["w"] = out["w"].astype(float)
	return out[["id", "eom", "w"]]


def _find_data_dir():
	"""Locate jkp-data/: $JKP_DATA, then CWD, then next to this file, then one level up."""
	import os
	here = os.path.dirname(os.path.abspath(__file__))
	candidates = [os.environ.get("JKP_DATA"), "jkp-data",
	              os.path.join(here, "jkp-data"), os.path.join(here, "..", "jkp-data")]
	for c in candidates:
		if c and os.path.isfile(os.path.join(c, "chars.parquet")):
			return c
	raise FileNotFoundError("jkp-data/chars.parquet not found; tried " + ", ".join(str(c) for c in candidates))


if __name__ == "__main__":
	import os
	data_dir = _find_data_dir()
	chars = pd.read_parquet(os.path.join(data_dir, "chars.parquet"))
	features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
	daily_ret = pd.read_parquet(os.path.join(data_dir, "daily_ret.parquet"))
	pf = main(chars, features, daily_ret)
	pf.to_csv("output.csv", index=False)