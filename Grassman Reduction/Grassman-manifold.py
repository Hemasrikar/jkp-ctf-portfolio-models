"""Subspace portfolio estimated on the Grassmann manifold.

The factor construction used previously produces a loading matrix whose columns
become long short portfolios. That matrix is not identified. Replacing it by any
invertible transform, with the inverse applied to the combination, leaves the
traded book essentially unchanged, so the object the objective can actually
distinguish is the subspace the columns span, not the columns themselves. A network
parameterising the matrix therefore spends capacity choosing a basis that the
objective is nearly blind to.

Here the subspace is estimated directly. The parameter is a point on the Grassmann
manifold of k dimensional subspaces of the characteristic space, represented by an
orthonormal frame and updated by Riemannian gradient steps: the Euclidean gradient
is projected onto the horizontal space, which discards exactly the directions that
rotate the frame without moving the subspace, and the iterate is returned to the
manifold by polar retraction. The redundancy is quotiented out rather than learned
around.

Stocks are projected onto the subspace, the projections are ranked within the month
and read by a small network which supplies the nonlinearity, and the resulting
scores are combined into a book. Separating the subspace from the nonlinearity in
this way is the point: the manifold handles which directions matter and the network
handles what to do with them, whereas a single unconstrained matrix conflates the
two."""

import multiprocessing as mp
import os
import random
import time

# all but two logical cores for the numeric libraries, set before they load
n_threads = max(1, (os.cpu_count() or 2) - 2)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
	os.environ.setdefault(_v, str(n_threads))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import save_file

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
parallel_seeds = True
min_train_months = 120

# beta neutralisation
beta_col = "beta_60m"

# grassmann subspace, zero recovers the factor construction
n_subspace = 48
subspace_steps = 300
subspace_lr = 0.05
subspace_slices = 10
subspace_save = 0.5

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
torch.set_num_threads(n_threads)
if os.name == "nt":
	# below normal priority: full machine when idle, yields to the desktop otherwise
	import ctypes
	ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)


def seed_all(s):
	random.seed(s)
	np.random.seed(s)
	torch.manual_seed(s)
	torch.cuda.manual_seed_all(s)


def stiefel_project(y):
	"""Nearest orthonormal frame, by polar retraction."""
	u, _, vt = np.linalg.svd(y, full_matrices=False)
	return u @ vt


def horizontal(egrad, y):
	"""Component of the gradient that actually moves the subspace.

	Directions lying in the current span only rotate the frame and leave the
	subspace fixed, so they are removed. What remains is the Grassmann gradient.
	"""
	return egrad - y @ (y.T @ egrad)


def fit_subspace(x_all, r_all):
	"""Subspace capturing the response conditional structure, by Riemannian ascent.

	Two criteria are combined. The variance of the predictor means across slices of
	the response detects directions on which the response depends monotonically. It
	is blind to symmetric dependence, since a predictor entering through an even
	function has the same mean in high and low slices, so the deviation of each
	slice covariance from the pooled covariance is added, which detects exactly
	those directions. Slicing on the within month rank makes both invariant to the
	level of returns in a month.
	"""
	p = x_all[0].shape[1]
	between = np.zeros((p, p))
	within = np.zeros((p, p))
	save_acc = [np.zeros((p, p))]
	n_used = 0
	for x, r in zip(x_all, r_all):
		if len(r) < subspace_slices * 5:
			continue
		xc = x.astype(np.float64)
		within += xc.T @ xc
		n_used += len(r)
		order = np.argsort(r)
		mus = []
		for part in np.array_split(order, subspace_slices):
			if len(part) >= 5:
				mus.append(xc[part].mean(axis=0))
		if len(mus) < 2:
			continue
		m = np.stack(mus, axis=0)
		m = m - m.mean(axis=0, keepdims=True)
		between += (m.T @ m) / len(m)
		if subspace_save > 0.0:
			for part in np.array_split(order, subspace_slices):
				if len(part) < 5:
					continue
				bl = xc[part]
				bc = bl - bl.mean(axis=0)
				cs = (bc.T @ bc) / len(part)
				dv = cs - (xc - xc.mean(axis=0)).T @ (xc - xc.mean(axis=0)) / len(xc)
				save_acc[0] += (dv @ dv) / subspace_slices
	if n_used == 0:
		return None
	within = within / n_used
	within = within + 1e-3 * (np.trace(within) / p) * np.eye(p)
	if subspace_save > 0.0:
		sc = save_acc[0]
		nb, ns = np.trace(between), np.trace(sc)
		if ns > 1e-18 and nb > 1e-18:
			sc = sc * (nb / ns)
		between = (1.0 - subspace_save) * between + subspace_save * sc
		between = (between + between.T) / 2.0

	rng = np.random.default_rng(base_seed)
	y = stiefel_project(rng.normal(size=(p, n_subspace)))
	for _ in range(subspace_steps):
		# ratio of between slice to within variance inside the subspace, whose
		# gradient is the generalised Rayleigh quotient gradient
		b = y.T @ between @ y
		w = y.T @ within @ y
		try:
			w_inv = np.linalg.inv(w + 1e-8 * np.eye(n_subspace))
		except np.linalg.LinAlgError:
			break
		egrad = 2.0 * (between @ y @ w_inv - within @ y @ w_inv @ b @ w_inv)
		d = horizontal(egrad, y)
		nrm = np.linalg.norm(d)
		if not np.isfinite(nrm) or nrm < 1e-12:
			break
		y = stiefel_project(y + subspace_lr * d / nrm)
	return y


def apply_subspace(x, basis):
	"""Project a month onto the subspace and rank within the month."""
	if basis is None:
		return x
	z = x.astype(np.float64) @ basis
	order = np.argsort(np.argsort(z, axis=0), axis=0).astype(np.float64)
	return ((order + 1.0) / (z.shape[0] + 1.0) - 0.5).astype(np.float32)


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
	omg = [None] * t
	if prec is not None:
		with torch.no_grad():
			for j in range(t):
				om = torch.exp(-prec(xg[j]))
				omg[j] = om / (om.mean() + 1e-8)
	for _ in range(n_epochs):
		order = np.random.permutation(t)
		for start in range(0, t, batch_months):
			idx = order[start:start + batch_months]
			if len(idx) < 6:
				continue
			rets = []
			for j in idx:
				w = factor_weights(model(xg[j]), omg[j])
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


def project_out_beta(w, beta):
	"""Remove the component of the book along beta, keeping net exposure."""
	if beta is None:
		return w
	b = beta.reshape(-1, 1)
	coef, _, _, _ = np.linalg.lstsq(b, w, rcond=None)
	return w - b @ coef


def state_cpu(net):
	return None if net is None else {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def fit_seed(s, prev, x_list, r_list, score_x):
	"""Fit one seed for one year, warm started from that seed's previous networks.

	Returns the networks, the combination vector and the raw book for every month
	in score_x. prev is (loading_net, precision_net) from the previous year or None.
	"""
	n_width = x_list[0].shape[1]
	seed_all(base_seed + s)
	ln = loading_net(n_width).to(device)
	pn = precision_net(n_width).to(device) if use_precision else None
	n_ep = epochs_cold
	if prev is not None:
		try:
			ln.load_state_dict(prev[0].state_dict())
			if pn is not None and prev[1] is not None:
				pn.load_state_dict(prev[1].state_dict())
			n_ep = epochs_warm
		except Exception:
			n_ep = epochs_cold
	ln = train_loadings(ln, None, x_list, r_list, n_ep, device)
	if pn is not None:
		pn = train_precision(pn, ln, x_list, r_list, epochs_var, device)
		ln = train_loadings(ln, pn, x_list, r_list, max(n_ep // 2, 4), device)
	lam = combination_weights(factor_history(ln, pn, x_list, r_list, device))
	books = [book_weights(ln, pn, lam, x, device) for x in score_x]
	return ln, pn, lam, books


def seed_worker(s, conn):
	"""Process owning one seed's chain of networks across the years."""
	torch.set_num_threads(max(1, n_threads // n_seeds))
	prev = None
	while True:
		job = conn.recv()
		if job is None:
			break
		ln, pn, lam, books = fit_seed(s, prev, *job)
		prev = (ln, pn)
		conn.send((state_cpu(ln), state_cpu(pn), lam, books))
	conn.close()


def snapshot_models(saved, yr, loaders, precs, lams, basis):
	"""Add one year's fitted networks, combination vectors and basis to a flat dict."""
	for s, (ln, pn, lam) in enumerate(zip(loaders, precs, lams)):
		pre = f"year_{yr}/seed_{s}/"
		for k, v in ln.items():
			saved[pre + "loading/" + k] = v
		if pn is not None:
			for k, v in pn.items():
				saved[pre + "precision/" + k] = v
		saved[pre + "lam"] = torch.as_tensor(np.asarray(lam, dtype=np.float64))
	if basis is not None:
		saved[f"year_{yr}/basis"] = torch.as_tensor(np.asarray(basis, dtype=np.float64))
	return saved


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
	      "vol_target", use_vol_target, "subspace", n_subspace, flush=True)
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

	by_month = {eom: md for eom, md in work.groupby("eom", sort=True)}
	all_months = sorted(by_month.keys())
	target_months = [m for m in all_months
	                 if np.isfinite(by_month[m]["ret_exc_lead1m"].to_numpy(dtype=np.float64)).sum() >= min_stocks]
	print("months", len(all_months), "with targets", len(target_months), flush=True)

	years = sorted({pd.Timestamp(m).year for m in all_months})
	target_vol = target_vol_annual / np.sqrt(12.0)
	n_scaled = 0
	results = []
	cols = None
	saved, saved_meta = {}, {}
	prev = [None] * n_seeds
	conns = []
	if parallel_seeds:
		# the seeds are independent chains, so each trains in its own process and
		# they share the device concurrently
		ctx = mp.get_context("spawn")
		for s in range(n_seeds):
			parent, child = ctx.Pipe()
			ctx.Process(target=seed_worker, args=(s, child), daemon=True).start()
			conns.append(parent)
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

		# the subspace is fitted on the training window only, then applied to both
		# the training months and the months being scored
		basis = fit_subspace(x_list, r_list) if n_subspace > 0 else None
		if basis is not None:
			x_list = [apply_subspace(x, basis) for x in x_list]
		n_width = x_list[0].shape[1]

		# months to score, prepared once so every seed reads the same inputs
		score = []
		for eom in eoms:
			md = by_month[eom]
			md = md.loc[(md[cols].isna().sum(axis=1) <= n_in * max_miss_frac).to_numpy()]
			if len(md) < min_stocks:
				n_skip += 1
				continue
			x = md[cols].rank(pct=True).to_numpy(dtype=np.float32)
			score.append((eom, md, apply_subspace(np.nan_to_num(x, nan=0.5) - 0.5, basis)))
		score_x = [x for _, _, x in score]

		if conns:
			for conn in conns:
				conn.send((x_list, r_list, score_x))
		# the covariances are cpu work, done while the workers train
		covs = {eom: covariance_for(daily, eom, md["id"].to_numpy()) for eom, md, _ in score} if use_vol_target else {}
		if conns:
			outs = [conn.recv() for conn in conns]
		else:
			outs = []
			for s in range(n_seeds):
				ln, pn, lam, books = fit_seed(s, prev[s], x_list, r_list, score_x)
				prev[s] = (ln, pn)
				outs.append((state_cpu(ln), state_cpu(pn), lam, books))
		snapshot_models(saved, yr, [o[0] for o in outs], [o[1] for o in outs], [o[2] for o in outs], basis)
		saved_meta[f"year_{yr}/cols"] = ",".join(cols)
		save_file(saved, "model.safetensors", metadata=saved_meta)
		del x_list, r_list
		print("year", yr, "(", yi, "of", len(years), ") train months", len(train_months),
		      "inputs", n_width, "fit_sec", round(time.time() - t0),
		      "elapsed_min", round((time.time() - t_start) / 60.0, 1), flush=True)

		for j, (eom, md, x) in enumerate(score):
			w = np.mean([o[3][j] for o in outs], axis=0)
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
				sigma, keep_ids = covs[eom]
				if sigma is None:
					n_skip += 1
					continue
				pos = pd.Index(ids_out).get_indexer(keep_ids)
				ok = pos >= 0
				if ok.sum() < min_universe:
					n_skip += 1
					continue
				if not ok.all():
					sigma = sigma[np.ix_(ok, ok)]
					pos = pos[ok]
				# both must be reordered to the covariance row order, unconditionally
				w = w[pos]
				ids_out = ids_out[pos]
				d = np.abs(w).sum()
				if d < 1e-12:
					n_skip += 1
					continue
				w = w / d
				vol = float(np.sqrt(max(w @ (sigma @ w), 0.0)))
				if vol < 1e-12:
					n_skip += 1
					continue
				scale = min(target_vol / vol, gross_cap)
				w = w * scale
				n_scaled += 1

			results.append(pd.DataFrame({"id": ids_out, "eom": eom, "w": w}))
			n_done += 1

	for conn in conns:
		conn.send(None)

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


if __name__ == "__main__":
	chars = pd.read_parquet("jkp-data/chars.parquet")
	features = pd.read_parquet("jkp-data/features.parquet")
	daily_ret = pd.read_parquet("jkp-data/daily_ret.parquet")
	pf = main(chars, features, daily_ret)
	pf.to_csv("output.csv", index=False)
