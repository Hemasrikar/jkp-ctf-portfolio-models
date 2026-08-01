"""
Multilayer perceptron forecast with a Ledoit and Wolf shrinkage covariance, combined
in a volatility targeted mean variance portfolio.

Uses the shrinkage estimator of Ledoit and Wolf and mean variance selection in the
sense of Markowitz.

In month t let x be the cross sectionally ranked characteristics of a security, r its
excess return over the following month, and w the traded weights.

For forecast, a network with a pooled contextual term estimates mu = E[r - r_bar | x]
under squared error, r_bar being the cross sectional mean.

covariance. Let S be the sample covariance of the trailing year of daily returns. We
shrink toward a constant correlation target F, which retains the sample variances and
replaces each pairwise correlation by the cross sectional average:

delta = max(0, min(1, (pi - rho) / gamma / T)),   Sigma = delta F + (1 - delta) S,

with pi the summed asymptotic variance of the entries of S, rho their summed
asymptotic covariance with those of F, and gamma the squared distance between F and S.
The daily estimate is scaled to a monthly horizon.

For portfolio, we solve maximise mu'w   subject to   w'Sigma w <= v^2,  |w_i| <= c,  |w|_1 <= G

The inner problem at a multiplier lam is solved by accelerated projected gradient with
a box projection, lam is set by bisection on the risk constraint, and G is imposed
afterwards by a uniform scaling. Weights are not renormalised to constant gross
exposure.

temporal integrity. The network is refitted yearly on months up to the last month end
preceding that year, the covariance uses only the preceding year of daily returns, and
the coverage filter is recomputed yearly from months at or before the cutoff.
"""

import random
import time
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# portfolio constraints
target_vol_annual = 0.10
gross_max = 4.0
min_breadth = 0.05
trading_days_per_month = 21.0

# optimiser
lam_lo = 1e-4
lam_hi = 1e8
n_bisect = 40
inner_iters = 2000
inner_tol = 1e-9

# covariance
cov_lookback_days = 252
min_day_frac = 0.5
min_universe = 30

# forecasting network
wide_dim = 128
n_layers = 4
dropout = 0.1
lr = 3e-4
weight_decay = 1e-4
epochs_fit = 40
accum_steps = 12
min_train_months = 120
retrain_every_years = 1
n_seeds = 3
base_seed = 20260730

# preprocessing
min_coverage = 0.90
pre_test_date = "1990-01-01"
min_stocks = 30
max_miss_frac = 1.0 / 3.0

# set to the pre test boundary for a cheap selection pass
max_eval_date = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
	random.seed(s)
	np.random.seed(s)
	torch.manual_seed(s)
	torch.cuda.manual_seed_all(s)


def resolve(df, candidates, label):
	for c in candidates:
		if c in df.columns:
			return c
	raise KeyError(label + " not resolvable among " + str(list(df.columns)[:12]))


def ledoit_wolf_constant_correlation(x):
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
	pi_hat = float(pi_mat.sum())

	a = ((x2 * xc).T @ xc) / t
	theta_ii = a - var[:, None] * s
	theta_jj = a.T - var[None, :] * s
	ratio = np.outer(1.0 / sd, sd)
	rho_off = (r_bar / 2.0) * (ratio * theta_ii + ratio.T * theta_jj)
	rho_hat = float(np.diag(pi_mat).sum() + rho_off[off].sum())

	diff = target - s
	gamma_hat = float((diff * diff).sum())
	delta = 0.0 if gamma_hat <= 1e-30 else float(np.clip((pi_hat - rho_hat) / gamma_hat / t, 0.0, 1.0))

	sigma = delta * target + (1.0 - delta) * s
	sigma.flat[:: n + 1] += 1e-12
	return sigma, delta


def max_eigenvalue(sigma, iters=60, seed=0):
	"""Largest eigenvalue by power iteration, used for the gradient step length."""
	rng = np.random.default_rng(seed)
	v = rng.standard_normal(sigma.shape[0])
	v /= np.linalg.norm(v) + 1e-300
	lam = 1.0
	for _ in range(iters):
		u = sigma @ v
		nrm = np.linalg.norm(u)
		if nrm < 1e-300:
			return 1.0
		v = u / nrm
		lam = nrm
	return float(max(lam, 1e-12))


def solve_box_quadratic(mu, sigma, lam, cap, eig_max, w0=None):
	"""Maximise mu'w - (lam/2) w'Sigma w subject to |w_i| <= cap.
	Momentum is restarted whenever it opposes progress, since plain acceleration
	oscillates here and leaves the iterate short of the precision the outer
	volatility constraint requires.
	"""
	step = 1.0 / (lam * eig_max)
	w = np.zeros_like(mu) if w0 is None else np.clip(w0, -cap, cap)
	y = w.copy()
	t_k = 1.0
	scale = max(float(np.abs(mu).max()), 1e-12)
	for it in range(inner_iters):
		grad = mu - lam * (sigma @ y)
		w_new = np.clip(y + step * grad, -cap, cap)
		if float((y - w_new) @ (w_new - w)) > 0.0:
			w = w_new
			y = w_new.copy()
			t_k = 1.0
			continue
		t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
		y = w_new + ((t_k - 1.0) / t_next) * (w_new - w)
		w = w_new
		t_k = t_next
		if (it + 1) % 25 == 0:
			g = mu - lam * (sigma @ w)
			resid = float(np.abs(np.clip(w + step * g, -cap, cap) - w).max()) / step
			if resid < inner_tol * scale:
				break
	return w


def optimise_month(mu, sigma, target_vol, cap, gross, eig_max):
	"""Smallest multiplier meeting the risk limit, then the gross limit by scaling."""
	def attained(lam, w0):
		w = solve_box_quadratic(mu, sigma, lam, cap, eig_max, w0)
		return w, float(np.sqrt(max(w @ (sigma @ w), 0.0)))

	def apply_gross(w):
		g = float(np.abs(w).sum())
		return w * (gross / g) if g > gross and g > 1e-18 else w

	w_hi, risk_hi = attained(lam_hi, None)
	if risk_hi > target_vol:
		# even maximal damping breaches the risk limit, so scale down to comply
		w = w_hi * (target_vol / risk_hi) if risk_hi > 1e-18 else w_hi
		return apply_gross(w)

	lo, hi = np.log(lam_lo), np.log(lam_hi)
	w_best = w_hi
	w_warm = w_hi
	for _ in range(n_bisect):
		mid = 0.5 * (lo + hi)
		w_mid, risk_mid = attained(np.exp(mid), w_warm)
		w_warm = w_mid
		if risk_mid <= target_vol:
			# feasible, so a smaller multiplier may also be feasible and is preferred
			w_best = w_mid
			hi = mid
		else:
			lo = mid
	return apply_gross(w_best)


def build_daily(daily_ret):
	idc = resolve(daily_ret, ["id", "permno"], "daily id")
	datec = resolve(daily_ret, ["date", "day"], "daily date")
	retc = resolve(daily_ret, ["ret_exc", "ret", "ret_local"], "daily return")
	d = daily_ret[[idc, datec, retc]].copy()
	d.columns = ["id", "date", "r"]
	d["date"] = pd.to_datetime(d["date"])
	d = d.dropna(subset=["id", "date", "r"])
	return d.sort_values("date", kind="mergesort").reset_index(drop=True)


def covariance_for(daily, eom, ids):
	"""Shrunk covariance from the trailing year of daily returns, scaled to a month.

	Securities without sufficient daily history are excluded and receive no weight.
	Remaining gaps are filled at the column mean, which slightly understates variance
	for sparse names; the coverage requirement bounds that understatement.
	"""
	start = eom - pd.DateOffset(days=int(cov_lookback_days * 1.6))
	win = daily[(daily["date"] > start) & (daily["date"] <= eom)]
	if not len(win):
		return None, None, None
	win = win[win["id"].isin(set(ids.tolist()))]
	if not len(win):
		return None, None, None

	piv = win.pivot_table(index="date", columns="id", values="r", aggfunc="last").tail(cov_lookback_days)
	if piv.shape[0] < 40:
		return None, None, None

	frac = piv.notna().mean(axis=0)
	keep = frac[frac >= min_day_frac].index
	if len(keep) < min_universe:
		return None, None, None
	piv = piv[keep]

	x = np.array(piv.to_numpy(dtype=np.float64), copy=True)
	col_mean = np.nanmean(x, axis=0)
	inds = np.where(np.isnan(x))
	x[inds] = np.take(col_mean, inds[1])

	sigma_d, delta = ledoit_wolf_constant_correlation(x)
	return sigma_d * trading_days_per_month, np.asarray(keep), delta


def build_body(in_dim):
	layers = [nn.Linear(in_dim, wide_dim), nn.LayerNorm(wide_dim), nn.GELU(), nn.Dropout(dropout)]
	for _ in range(max(n_layers - 2, 0)):
		layers += [nn.Linear(wide_dim, wide_dim), nn.LayerNorm(wide_dim), nn.GELU(), nn.Dropout(dropout)]
	return nn.Sequential(*layers)


class forecast_net(nn.Module):
	def __init__(self, in_dim):
		super().__init__()
		self.body = build_body(in_dim)
		self.head = nn.Sequential(nn.Linear(2 * wide_dim, wide_dim), nn.GELU(), nn.Linear(wide_dim, 1))

	def forward(self, x):
		z = self.body(x)
		# contemporaneous context, so no information crosses a month boundary
		ctx = z.mean(dim=0, keepdim=True).expand_as(z)
		return self.head(torch.cat([z, ctx], dim=1)).squeeze(-1)


def train_forecast(in_dim, samples, dev):
	net = forecast_net(in_dim).to(dev)
	opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
	sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs_fit, 1))
	net.train(True)
	for _ in range(epochs_fit):
		order = list(range(len(samples)))
		random.shuffle(order)
		opt.zero_grad(set_to_none=True)
		for step, i in enumerate(order):
			x, r = samples[i]
			loss = ((net(x) - r) ** 2).mean()
			(loss / accum_steps).backward()
			if (step + 1) % accum_steps == 0 or (step + 1) == len(order):
				opt.step()
				opt.zero_grad(set_to_none=True)
		sched.step()
	net.train(False)
	return net


def sharpe_of(series):
	if len(series) < 12:
		return float("nan")
	sd = series.std(ddof=1)
	return float(series.mean() / sd * np.sqrt(12.0)) if sd > 1e-12 else float("nan")


def report_sharpe(weights, chars):
	if not len(weights):
		print("no weights to score", flush=True)
		return
	ret = chars[["id", "eom", "ret_exc_lead1m"]].copy()
	ret["eom"] = pd.to_datetime(ret["eom"]) + pd.offsets.MonthEnd(0)
	ret = ret.rename(columns={"ret_exc_lead1m": "r"}).dropna(subset=["r"])
	w = weights.copy()
	w["eom"] = pd.to_datetime(w["eom"]) + pd.offsets.MonthEnd(0)
	j = w.merge(ret, on=["id", "eom"], how="inner")
	j["c"] = j["w"] * j["r"]
	series = j.groupby("eom")["c"].sum().sort_index()
	pre = series[series.index < pd.Timestamp(pre_test_date)]
	post = series[series.index >= pd.Timestamp(pre_test_date)]
	print("months total", len(series), "pre", len(pre), "test", len(post), flush=True)
	if len(pre) >= 12:
		print("Pre-Test Window", flush=True)
		print("  sharpe", round(sharpe_of(pre), 3), flush=True)
	if len(post) >= 12:
		print("Test Window", flush=True)
		print("  sharpe", round(sharpe_of(post), 3), flush=True)
	if len(series) >= 2:
		print("annualised return", round(float(series.mean()) * 12.0, 4),
		      "annualised volatility", round(float(series.std(ddof=1)) * np.sqrt(12.0), 4), flush=True)


def select_features(work, candidates, months):
	"""Coverage filter over the supplied months only, which keeps it causal."""
	panel = work[work["eom"].isin(months)]
	if not len(panel):
		return list(candidates)
	frac = panel[candidates].notna().mean()
	return [f for f in candidates if frac[f] >= min_coverage]


def month_matrix(md, cols):
	x = md[cols].rank(pct=True).to_numpy(dtype=np.float32)
	return np.nan_to_num(x, nan=0.5) - 0.5


def main(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
	t_start = time.time()
	print("torch", torch.__version__, "device", device, flush=True)
	print("target_vol_annual", target_vol_annual, "gross_max", gross_max,
	      "min_breadth", min_breadth, "seeds", n_seeds, flush=True)
	seed_all(base_seed)

	target_vol = target_vol_annual / np.sqrt(12.0)
	candidates = [f for f in features["features"].tolist() if f in chars.columns]
	work = chars.copy()
	work["eom"] = pd.to_datetime(work["eom"]) + pd.offsets.MonthEnd(0)

	if daily_ret is None or not len(daily_ret):
		raise ValueError("daily returns are required for the covariance estimate")
	daily = build_daily(daily_ret)

	by_month = {eom: md for eom, md in work.groupby("eom", sort=True)}
	all_months = sorted(by_month.keys())
	target_months = [m for m in all_months if np.isfinite(by_month[m]["ret_exc_lead1m"].to_numpy(dtype=np.float64)).sum() >= min_stocks]
	print("months", len(all_months), "with targets", len(target_months), flush=True)

	# the validation dataset is small, so the history requirement adapts downward
	eff_min_train = min(min_train_months, max(24, len(target_months) - 6))
	if eff_min_train != min_train_months:
		print("history requirement reduced to", eff_min_train, "for a short panel", flush=True)

	eval_months = all_months if max_eval_date is None else [m for m in all_months if m <= pd.Timestamp(max_eval_date)]
	years = sorted({pd.Timestamp(m).year for m in eval_months})
	results = []
	last_fit = None
	nets = []
	cols = None
	n_solved = 0
	n_skip = 0
	n_gross_bound = 0
	deltas = []

	for yi, yr in enumerate(years, 1):
		eoms = [m for m in eval_months if pd.Timestamp(m).year == yr]
		if not eoms:
			continue
		# every training month must be observable at the first decision of the year
		cutoff = max([m for m in target_months if m < min(eoms)], default=None)
		if cutoff is None:
			continue
		train_months = [m for m in target_months if m <= cutoff]
		if len(train_months) < eff_min_train:
			continue

		if last_fit is None or (yr - last_fit) >= retrain_every_years:
			t0 = time.time()
			cols = select_features(work, candidates, train_months)
			if not cols:
				continue
			n_in = len(cols)
			samples = []
			for mm in train_months:
				md = by_month[mm]
				md = md.loc[(md[cols].isna().sum(axis=1) <= n_in * max_miss_frac).to_numpy()]
				rv = md["ret_exc_lead1m"].to_numpy(dtype=np.float64)
				fin = np.isfinite(rv)
				if fin.sum() < min_stocks:
					continue
				rv = rv[fin]
				# demeaned within the month, so the forecast is a relative mean
				rv = rv - rv.mean()
				samples.append((torch.as_tensor(month_matrix(md, cols)[fin], dtype=torch.float32, device=device),
				                torch.as_tensor(rv, dtype=torch.float32, device=device)))
			if len(samples) < eff_min_train:
				continue
			nets = []
			for s in range(n_seeds):
				seed_all(base_seed + s)
				nets.append(train_forecast(n_in, samples, device))
			del samples
			last_fit = yr
			print("year", yr, "(", yi, "of", len(years), ") train months", len(train_months),
			      "inputs", n_in, "fit_sec", round(time.time() - t0),
			      "elapsed_min", round((time.time() - t_start) / 60.0, 1), flush=True)
		if not nets or cols is None:
			continue

		n_in = len(cols)
		for eom in eoms:
			md = by_month[eom]
			md = md.loc[(md[cols].isna().sum(axis=1) <= n_in * max_miss_frac).to_numpy()]
			if len(md) < min_stocks:
				n_skip += 1
				continue
			ids = md["id"].to_numpy()
			xt = torch.as_tensor(month_matrix(md, cols), dtype=torch.float32, device=device)
			with torch.no_grad():
				mu_all = np.mean([net(xt).cpu().numpy().astype(np.float64) for net in nets], axis=0)

			sigma, keep_ids, delta = covariance_for(daily, eom, ids)
			if sigma is None:
				n_skip += 1
				continue
			deltas.append(delta)
			pos = pd.Index(ids).get_indexer(keep_ids)
			ok = pos >= 0
			if ok.sum() < min_universe:
				n_skip += 1
				continue
			if not ok.all():
				sigma = sigma[np.ix_(ok, ok)]
				keep_ids = keep_ids[ok]
				pos = pos[ok]
			mu = mu_all[pos]

			cap = 1.0 / max(len(mu) * min_breadth, 1.0)
			eig = max_eigenvalue(sigma, seed=base_seed)
			w = optimise_month(mu, sigma, target_vol, cap, gross_max, eig)
			if w is None or not np.isfinite(w).all():
				n_skip += 1
				continue
			sel = np.abs(w) > 1e-10
			if not sel.any():
				n_skip += 1
				continue
			if float(np.abs(w).sum()) >= gross_max - 1e-9:
				n_gross_bound += 1
			n_solved += 1
			results.append(pd.DataFrame({"id": keep_ids[sel], "eom": eom, "w": w[sel]}))
			if n_solved % 100 == 0:
				print("  solved", n_solved, "skipped", n_skip, "universe", len(mu),
				      "gross bound", n_gross_bound, "shrinkage", round(float(np.mean(deltas[-100:])), 3),
				      "elapsed_min", round((time.time() - t_start) / 60.0, 1), flush=True)

	if not results:
		raise ValueError("no weights produced, check the daily coverage and history requirement")

	out = pd.concat(results, ignore_index=True)
	print("months solved", n_solved, "skipped", n_skip,
	      "gross limit binding in", n_gross_bound, "of", n_solved, "months",
	      "mean shrinkage", round(float(np.mean(deltas)), 3) if deltas else None,
	      "total_min", round((time.time() - t_start) / 60.0, 1), flush=True)
	print("output rows", len(out), "months", out["eom"].nunique(), flush=True)

	report_sharpe(out, chars)

	out["eom"] = pd.to_datetime(out["eom"]).dt.strftime("%Y-%m-%d")
	out["id"] = out["id"].astype(int)
	out["w"] = out["w"].astype(float)
	return out[["id", "eom", "w"]]

if __name__ == "__main__":

	chars = pd.read_parquet("jkp-data/chars.parquet")
	features = pd.read_parquet("jkp-data/features.parquet")
	daily_ret = pd.read_parquet("jkp-data/daily_ret.parquet")
	pf = main(chars, features, daily_ret)
	pf.to_csv("output.csv", index=False)