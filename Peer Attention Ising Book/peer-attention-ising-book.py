"""Peer attention alpha.

In this model we start from the observation that most of the predictable part of a stock
return is relative: what matters is how a stock differs from the firms that resemble it. In
our analysis of the pre 1990 data the industry level component of the signal has a negative
Sharpe ratio, while the component that is idiosyncratic relative to industry and
characteristic peers carries most of the alpha, and one month reversal measured against
characteristic peers earns far more than own reversal. A forecaster that looks at one stock
at a time can only learn such comparisons indirectly. Hence, the central learner of the
model looks at every stock together with its peers and learns which differences matter.

Peer attention network: For every stock we take its forty nearest neighbours in
characteristic space, measured by the cosine similarity of the characteristic ranks. The
network first embeds the inputs of every stock. For each peer it then forms a comparison
from the embedding of the stock, the embedding of the peer and their difference, and an
attention score decides how much each comparison counts. The weighted comparisons are
summarised into a peer context, and the prediction is read from the stock embedding
together with that context. The network is refit every January on all earlier months, two
seeds are averaged, and each fit is early stopped on the information coefficient of the
last three years of its training window.

Signal: All learners read the same inputs, the cross sectional ranks of the characteristics
together with fourteen short horizon features that we build from the daily return file.
These describe how the last month was travelled: the returns over the last day, week,
fortnight and month, the separate legs of that path, the five day extremes, the daily
volatility, and the industry and peer averages of the weekly and monthly returns. Next to
the peer attention network we use a ridge regression with size interactions and a gradient
boosted tree, both refit every January, the tree treating missing characteristics as a
separate branch. All three are trained on the forward return measured relative to its three
digit industry, passed through the arctangent of its cross sectional z score, so that the
learners focus on differences between peers and are not driven by extreme returns. The
three predictions are standardised and combined with weights 0.25 for the ridge, 0.25 for
the network and 0.5 for the tree.

Spectral filter: Every month we decompose the signal as a Laplacian pyramid over a
hierarchy of peer graphs. The levels are the two digit industry, the three digit industry
within it, the four digit industry within that, the average over the twenty nearest
neighbours in characteristic space, and finally the idiosyncratic remainder. By
construction the five bands add up to the original signal. Each band can also be held as a
portfolio, so we set the gain on each band equal to the tangency weight of the band
portfolios, estimated only on their past realised returns. This works as a Wiener type
filter whose frequency response is learned from the data rather than fixed in advance.

Book: In place of a mean variance optimiser we use a mean field spin system. Each stock
carries a spin between minus one and one. The filtered signal acts as an external field on
the spins, and the return correlation from the Ledoit and Wolf estimate over the last three
years of daily returns couples every pair of spins, so that holding two correlated stocks
on the same side raises the energy. We take the spins that minimise the free energy of this
system. At the minimum each spin equals the hyperbolic tangent of the field it feels after
the pull of all the other spins, so the positions are bounded by construction and risk is
spread without inverting a covariance matrix. Formally, with h the standardised signal, C
the correlation matrix, J the coupling and beta the inverse temperature, the spins minimise

	F(m) = J/2 m'Cm - h'm + (1/beta) sum over i of the binary entropy of (1 + m_i)/2,

whose stationary point is m = tanh(beta (h - J C m)). We use beta = 0.5 and J = 0.3, and we
solve the problem with a bounded quasi Newton method. The spins are scaled by the inverse
volatility of each stock so that a spin is a unit of risk, made neutral to market beta and
log size, and sized against the exponentially weighted volatility v of the trailing profit
and loss of the book (halflife of 21 days). The leverage is (target / v)(v_ref / v)^0.5,
where v_ref is the median of the volatilities estimated in earlier months, so the book is
sized in proportion to v to the power minus 1.5 rather than minus 1. Because the book is
positioned against recent moves, its back cast volatility mostly measures how large the
current reversal setup is, and the steeper sizing rule leans harder into the months where
that setup is quiet. The target is 10 percent a year with a gross cap of 12.

Implementation: Characteristics are converted to cross sectional percentile ranks centred
on zero, with missing values set to zero. For the tree the exact zero is restored to
missing, so that the tree can route missing characteristics to their own branch. In the
spectral filter the characteristic peers are the twenty stocks with the highest cosine similarity of their rank
vectors. The covariance is the Ledoit and Wolf shrinkage estimate of the last 756 daily
returns, and a stock enters the book only if it has at least 250 daily observations in that
window. When a GPU is available it carries the network, the peer similarity and the
covariance, and otherwise everything runs on the CPU.

Every quantity used to form or size the portfolio of month t is available at month t. The
learners are refit only on months that are already realised, and the band gains use band
returns that have already been observed. With a fixed seed the whole run is deterministic.
"""

from __future__ import annotations

import os
import time
from typing import Any

# a fixed cuBLAS workspace is required for deterministic matrix products on the GPU
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

ridge = 1e3
pan_peers = 40
pan_dim = 64
pan_seeds = 2
pan_epochs = 25
pan_patience = 4
pan_lr = 1e-3
pan_weight_decay = 1e-4
pan_dropout = 0.1
pan_valid_years = 3
pan_min_names = 60
w_lin = 0.25
w_pan = 0.25
w_tree = 0.5
tree_params: dict[str, Any] = dict(
	n_estimators=1000, learning_rate=0.01, num_leaves=255, min_child_samples=2000, colsample_bytree=0.5,
	random_state=42, n_jobs=8, deterministic=True, force_row_wise=True, verbose=-1,
)

knn_peers = 20
band_window = 180
band_shrink = 0.5
min_band_history = 60

sample_window_days = 756
min_daily_obs = 250
days_per_month = 21
spin_beta = 0.5
spin_j = 0.3
min_names = 50

target_vol_annual = 0.10
gross_cap = 12.0
realised_lookback_days = 252
realised_halflife_days = 21
vol_power = 1.5

start_date = "1960-01-01"
target_col = "ret_exc_lead1m"
feature_list_col = "features"
pre_test_date = "1990-01-01"
seed = 42
n_threads = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_linalg = device.type == "cuda"


def group_mean(v, lab):
	u, inv = np.unique(lab, return_inverse=True)
	return (np.bincount(inv, v) / np.bincount(inv))[inv]


def peer_label(y, sic3):
	y = y - y.mean()
	y = y - group_mean(y, sic3)
	sd = y.std()
	return np.arctan(y / sd) if sd > 1e-12 else y * 0.0


def to_dev(a):
	return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device)


def knn(rk, k):
	if torch_linalg:
		x = to_dev(rk)
		xn = x / (x.norm(dim=1, keepdim=True) + 1e-12)
		sim = xn @ xn.T
		sim.fill_diagonal_(float("-inf"))
		return torch.topk(sim, k, dim=1).indices.cpu().numpy()
	rn = rk / (np.linalg.norm(rk, axis=1, keepdims=True) + 1e-12)
	sim = rn @ rn.T
	np.fill_diagonal(sim, -np.inf)
	return np.argpartition(-sim, k, axis=1)[:, :k]


def ledoit_wolf(x):
	if torch_linalg:
		xt = to_dev(x)
		t, n = xt.shape
		s = xt.T @ xt / t
		tr = torch.trace(s)
		mu = tr / n
		ss = (s ** 2).sum()
		d2 = ss - 2 * mu * tr + n * mu * mu
		rs = (xt ** 2).sum(1)
		b2 = (rs ** 2).sum() / t ** 2 - ss / t
		d2v, b2v = float(d2), float(b2)
		delta = min(max(b2v, 0.0), d2v) / d2v if d2v > 0 else 1.0
		s = s * (1 - delta)
		s.diagonal().add_(delta * float(mu))
		return s
	t, n = x.shape
	s = x.T @ x / t
	mu = np.trace(s) / n
	d2 = (s ** 2).sum() - 2 * mu * np.trace(s) + n * mu * mu
	rs = (x ** 2).sum(1)
	b2 = (rs ** 2).sum() / t ** 2 - (s ** 2).sum() / t
	delta = min(max(b2, 0.0), d2) / d2 if d2 > 0 else 1.0
	s *= 1 - delta
	s[np.diag_indices(n)] += delta * mu
	return s


def spin_book(h, sigma):
	if torch.is_tensor(sigma):
		sigma = sigma.cpu().numpy()
	sd = np.sqrt(np.maximum(np.diag(sigma), 1e-18))
	corr = sigma / np.outer(sd, sd)
	h = (h - h.mean()) / (h.std() + 1e-12)
	eps = 1e-6

	def free_energy(m):
		cm = corr @ m
		mc = np.clip(m, -1 + eps, 1 - eps)
		ent = ((1 + mc) / 2 * np.log((1 + mc) / 2) + (1 - mc) / 2 * np.log((1 - mc) / 2)).sum() / spin_beta
		return 0.5 * spin_j * m @ cm - h @ m + ent, spin_j * cm - h + np.arctanh(mc) / spin_beta

	res = minimize(free_energy, 0.5 * np.tanh(spin_beta * h), jac=True, method="L-BFGS-B",
	               bounds=[(-1 + eps, 1 - eps)] * len(h), options={"maxiter": 200})
	return res.x / sd


def rank_panel(x, starts, ends):
	out = np.zeros(x.shape, dtype=np.float32)
	for a, b in zip(starts, ends):
		rk = pd.DataFrame(x[a:b]).rank(pct=True).to_numpy(dtype=np.float32) - 0.5
		rk[np.isnan(rk)] = 0.0
		out[a:b] = rk
	return out


def build_daily_matrix(daily_ret, panel_ids):
	t0 = time.perf_counter()
	rid = daily_ret["id"].to_numpy()
	rid_int = rid if np.issubdtype(rid.dtype, np.integer) else rid.astype(np.int64)
	pid = np.asarray(panel_ids).astype(np.int64)
	keep = np.isin(rid_int.astype(np.int64), pid, kind="table")
	rid = rid_int[keep].astype(np.int64)
	rdate = daily_ret["date"].to_numpy()[keep].astype("datetime64[D]")
	rret = daily_ret["ret_exc"].to_numpy()[keep].astype(np.float32)
	date_ix, dates = pd.factorize(rdate, sort=True)
	id_ix, ids = pd.factorize(rid, sort=True)
	m = np.full((len(dates), len(ids)), np.nan, dtype=np.float32)
	m[date_ix, id_ix] = rret
	print(f"daily matrix {m.shape[0]:,} days x {m.shape[1]:,} ids in {time.perf_counter() - t0:.0f}s", flush=True)
	return np.asarray(dates, dtype="datetime64[D]"), np.asarray(ids), m


def daily_feature_ranks(r, starts, ends, months, dates, dmat, col_all, has_d, sic, n_base):
	nf = 14
	out = np.zeros((r.shape[0], nf), dtype=np.float32)
	for t in range(len(months)):
		a, b = starts[t], ends[t]
		rows = np.arange(a, b)
		hd = has_d[rows]
		end = int(np.searchsorted(dates, np.datetime64(months[t], "D"), side="right"))
		if end < 25 or hd.sum() < 25:
			continue
		rw = rows[hd]
		w = dmat[end - 21:end][:, col_all[rw]].astype(np.float64)
		ok = np.isfinite(w).sum(0) >= 10
		w = np.nan_to_num(w)
		r5, r21 = w[-5:].sum(0), w.sum(0)
		s3 = sic[rw] // 10
		nn = knn(r[rw, :n_base].astype(np.float64), knn_peers)
		f = np.column_stack([
			r5, w[-10:].sum(0), r21, w[-1], w[-5:].max(0), w[-5:].min(0), w.std(0), w[-5:-1].sum(0),
			w[-10:-5].sum(0), w[:-10].sum(0), group_mean(r5, s3), group_mean(r21, s3), r5[nn].mean(1), r21[nn].mean(1),
		])
		f[~ok] = np.nan
		rk = pd.DataFrame(f).rank(pct=True).to_numpy(dtype=np.float32) - 0.5
		rk[np.isnan(rk)] = 0.0
		out[rw] = rk
	return out


class peer_attention(torch.nn.Module):
	def __init__(self, n_in):
		super().__init__()
		d = pan_dim
		self.emb = torch.nn.Sequential(torch.nn.Linear(n_in, 2 * d), torch.nn.GELU(), torch.nn.Dropout(pan_dropout), torch.nn.Linear(2 * d, d))
		self.score = torch.nn.Sequential(torch.nn.Linear(3 * d, d), torch.nn.Tanh(), torch.nn.Linear(d, 1))
		self.val = torch.nn.Linear(2 * d, d)
		self.head = torch.nn.Sequential(torch.nn.Linear(2 * d, d), torch.nn.GELU(), torch.nn.Linear(d, 1))

	def forward(self, x, nn):
		e = self.emb(x)
		ep = e[nn]
		ei = e.unsqueeze(1).expand_as(ep)
		att = torch.softmax(self.score(torch.cat([ei, ep, ei - ep], 2)).squeeze(2), 1)
		v = self.val(torch.cat([ei - ep, ep], 2))
		h = (att.unsqueeze(2) * v).sum(1)
		out = self.head(torch.cat([e, h], 1)).squeeze(1)
		return out - out.mean()


def month_ic(p, lab):
	p = p - p.mean()
	lab = lab - lab.mean()
	return float((p * lab).sum() / (p.norm() * lab.norm() + 1e-12))


def fit_pan(month_data, fit_idx, valid_idx, n_in):
	models = []
	for s in range(pan_seeds):
		torch.manual_seed(seed + s)
		net = peer_attention(n_in).to(device)
		opt = torch.optim.AdamW(net.parameters(), lr=pan_lr, weight_decay=pan_weight_decay)
		gen = np.random.default_rng(seed + s)
		best, best_state, bad = -np.inf, None, 0
		for _ in range(pan_epochs):
			net.train()
			for i in gen.permutation(fit_idx):
				x, nn, lab, fin = (u.to(device) for u in month_data[i])
				p = net(x, nn)
				loss = ((p[fin] - lab[fin]) ** 2).mean()
				opt.zero_grad()
				loss.backward()
				opt.step()
			net.eval()
			with torch.no_grad():
				ics = []
				for i in valid_idx:
					x, nn, lab, fin = (u.to(device) for u in month_data[i])
					ics.append(month_ic(net(x, nn)[fin], lab[fin]))
			v = float(np.mean(ics)) if ics else -np.inf
			if v > best:
				best, best_state, bad = v, {k: w.clone() for k, w in net.state_dict().items()}, 0
			else:
				bad += 1
				if bad >= pan_patience:
					break
		if best_state is not None:
			net.load_state_dict(best_state)
		net.eval()
		models.append(net)
	if device.type == "cuda":
		torch.cuda.empty_cache()
	return models


def predict_pan(models, data):
	x, nn = data[0].to(device), data[1].to(device)
	out = np.zeros(len(data[0]))
	with torch.no_grad():
		for m in models:
			out += zs(m(x, nn).cpu().numpy().astype(np.float64))
	return out


def tree_inputs(x):
	x = np.array(x, dtype=np.float32, copy=True)
	x[x == 0] = np.nan
	return x


def predict_tree(tree, x):
	return np.asarray(tree.booster_.predict(tree_inputs(x)), dtype=np.float64)


def zs(v):
	v = v - v.mean()
	sd = v.std()
	return v / sd if sd > 1e-12 else v


def build_weights(chars, features, daily_ret):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.set_num_threads(n_threads)
	torch.use_deterministic_algorithms(True, warn_only=True)
	if device.type == "cuda":
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.benchmark = False
		torch.backends.cuda.matmul.allow_tf32 = False
	print(f"device {device}, gpu linear algebra {torch_linalg}", flush=True)
	t0 = time.perf_counter()
	feats = [str(f) for f in features[feature_list_col].tolist() if str(f) in chars.columns]
	p = len(feats)
	panel = chars[["id", "eom", target_col, "sic"] + feats].copy()
	panel["eom"] = pd.to_datetime(panel["eom"])
	panel = panel.sort_values(["eom", "id"], kind="stable").reset_index(drop=True)
	ids = np.asarray(pd.to_numeric(panel["id"]), dtype=np.int64)
	eom_out = np.asarray(panel["eom"], dtype="datetime64[ns]")
	y_all = panel[target_col].to_numpy(dtype=np.float64)
	sic = np.nan_to_num(np.asarray(pd.to_numeric(panel["sic"], errors="coerce"), dtype=np.float64), nan=9999.0)
	me = panel["market_equity"].to_numpy(dtype=np.float64) if "market_equity" in panel.columns else np.ones(len(panel))
	bet = panel["beta_60m"].to_numpy(dtype=np.float64) if "beta_60m" in panel.columns else np.zeros(len(panel))
	size_ix = feats.index("market_equity") if "market_equity" in feats else None

	months, mix = np.unique(eom_out.astype("datetime64[D]"), return_inverse=True)
	mdates = [m.astype("datetime64[D]").item() for m in months]
	starts = np.searchsorted(mix, np.arange(len(months)))
	ends = np.searchsorted(mix, np.arange(len(months)), side="right")
	x = panel[feats].to_numpy(dtype=np.float32)
	del panel
	r = rank_panel(x, starts, ends)
	del x
	print(f"{len(ids):,} rows, {p} characteristics, {len(months)} months, ranked in "
	      f"{time.perf_counter() - t0:.0f}s", flush=True)

	dates, daily_ids, dmat = build_daily_matrix(daily_ret, np.unique(ids))
	col_all = np.clip(np.searchsorted(daily_ids, ids), 0, len(daily_ids) - 1)
	has_d = daily_ids[col_all] == ids
	t1 = time.perf_counter()
	r = np.concatenate([r, daily_feature_ranks(r, starts, ends, months, dates, dmat, col_all, has_d, sic, p)], axis=1)
	print(f"daily features added, {r.shape[1]} inputs, in {time.perf_counter() - t1:.0f}s", flush=True)
	n_chars = p
	p = r.shape[1]

	def expand(rk):
		if size_ix is None:
			return rk
		return np.column_stack([rk, rk * rk[:, [size_ix]]])

	month_rows, month_lab = [], []
	for tt in range(len(months)):
		rw = np.arange(starts[tt], ends[tt])
		rw = rw[np.isfinite(y_all[rw])]
		month_rows.append(rw)
		month_lab.append(peer_label(y_all[rw], sic[rw] // 10) if len(rw) > 2 else np.zeros(len(rw)))

	t1 = time.perf_counter()
	month_data: list[Any] = []
	for tt in range(len(months)):
		a, b = starts[tt], ends[tt]
		if b - a < pan_min_names:
			month_data.append(None)
			continue
		yv = y_all[a:b]
		fin = np.isfinite(yv)
		lab = np.zeros(b - a, dtype=np.float32)
		if fin.sum() > 2:
			lab[fin] = peer_label(yv[fin], sic[a:b][fin] // 10)
		nn = knn(r[a:b, :n_chars].astype(np.float64), pan_peers)
		month_data.append((torch.from_numpy(r[a:b]), torch.from_numpy(np.ascontiguousarray(nn, dtype=np.int64)),
		                   torch.from_numpy(lab), torch.from_numpy(fin)))
	print(f"peer sets built in {time.perf_counter() - t1:.0f}s", flush=True)

	q = expand(np.zeros((1, p))).shape[1]
	xtx = np.zeros((q, q))
	xty = np.zeros(q)
	coef = None
	models = None
	tree = None
	band_hist = []
	vol_hist = []
	frames = []
	t_start = int(np.searchsorted(months, np.datetime64(start_date, "D")))
	n_emit = 0
	for t in range(len(months)):
		a, b = starts[t], ends[t]
		rows = np.arange(a, b)
		fin = np.isfinite(y_all[rows])
		if t >= t_start:
			if coef is None or mdates[t].month == 1:
				coef = np.linalg.solve(xtx + ridge * np.eye(q), xty)
			if tree is None or mdates[t].month == 1:
				tr_rows, tr_y = [], []
				for tt in range(t):
					rw = month_rows[tt]
					if len(rw) < min_names:
						continue
					tr_rows.append(rw)
					tr_y.append(month_lab[tt].astype(np.float32))
				tr_rows = np.concatenate(tr_rows)
				tr_y = np.concatenate(tr_y)
				x_tr = r[tr_rows]
				tree = lgb.LGBMRegressor(**tree_params).fit(tree_inputs(x_tr), tr_y)
				del x_tr
				cut = mdates[t].replace(year=mdates[t].year - pan_valid_years, day=min(mdates[t].day, 28))
				usable = [tt for tt in range(t) if month_data[tt] is not None and month_data[tt][3].float().mean() > 0.5]
				valid_idx = [tt for tt in usable if mdates[tt] >= cut]
				fit_idx = np.array([tt for tt in usable if mdates[tt] < cut])
				models = fit_pan(month_data, fit_idx, valid_idx, p)
				print(f"refit tree and attention network {mdates[t].year} on {len(tr_rows):,} rows, elapsed "
				      f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)

			# the traded universe never depends on whether a forward return exists
			rr = rows[has_d[rows]]
			end = int(np.searchsorted(dates, np.datetime64(months[t], "D"), side="right"))
			lo = max(0, end - sample_window_days)
			xd = dmat[lo:end][:, col_all[rr]].astype(np.float64)
			obs = np.isfinite(xd)
			enough = obs.sum(0) >= min_daily_obs
			rr = rr[enough]
			xd = np.where(obs[:, enough], xd[:, enough], 0.0)
			xd -= xd.mean(0)
			if len(rr) >= min_names and xd.shape[0] >= 40:
				assert coef is not None and models is not None and tree is not None
				rk = r[rr].astype(np.float64)
				s_lin = expand(rk) @ coef
				assert month_data[t] is not None
				s_pan = predict_pan(models, month_data[t])[rr - a]
				s_tree = predict_tree(tree, r[rr])
				s = w_lin * zs(s_lin) + w_pan * zs(s_pan) + w_tree * zs(s_tree)
				s3 = sic[rr]
				c1 = group_mean(s, s3 // 100)
				c2 = group_mean(s, s3 // 10) - c1
				c3 = group_mean(s, s3) - c1 - c2
				r3 = s - c1 - c2 - c3
				nn = knn(rk, knn_peers)
				c4 = r3[nn].mean(1)
				c5 = r3 - c4
				bands = np.column_stack([c1, c2, c3, c4, c5])

				if len(band_hist) >= min_band_history:
					past = np.array(band_hist[-band_window:])
					mu = past.mean(0)
					cv = np.cov(past.T)
					cv = (1 - band_shrink) * cv + band_shrink * np.diag(np.diag(cv))
					theta = np.linalg.solve(cv, mu)
					scale = np.maximum(np.abs(bands).mean(0), 1e-12)
					sig = bands @ (theta / scale)
					sigma = ledoit_wolf(xd * np.sqrt(days_per_month))
					w = spin_book(sig, sigma)
					bt = np.nan_to_num(bet[rr])
					lm = np.log(np.maximum(np.nan_to_num(me[rr], nan=1.0), 1e-3))
					basis = np.column_stack([np.ones(len(rr)), bt - bt.mean(), lm - lm.mean()])
					cf, *_ = np.linalg.lstsq(basis, w, rcond=None)
					w = w - basis @ cf
					w = w - w.mean()
					g = np.abs(w).sum()
					if g > 1e-12:
						w = w / g
						pnl = xd[-realised_lookback_days:] @ w
						decay = 0.5 ** (np.arange(len(pnl))[::-1] / realised_halflife_days)
						vol = np.sqrt((decay * pnl ** 2).sum() / decay.sum() * days_per_month)
						if vol > 1e-12:
							vref = float(np.median(vol_hist)) if len(vol_hist) >= 12 else vol
							lev = target_vol_annual / np.sqrt(12.0) / vol * (vref / vol) ** (vol_power - 1.0)
							w = w * min(lev, gross_cap)
							vol_hist.append(vol)
						wt = np.zeros(b - a)
						wt[rr - a] = w
						frames.append(pd.DataFrame({"id": ids[a:b], "eom": eom_out[a:b], "w": wt}))
						n_emit += 1

				# band returns of this month are used only from next month onward
				yy = y_all[rr]
				ok = np.isfinite(yy)
				if ok.sum() >= min_names:
					br = []
					for k in range(bands.shape[1]):
						v = bands[ok, k] - bands[ok, k].mean()
						gk = np.abs(v).sum()
						br.append(float((v / gk) @ yy[ok]) if gk > 0 else 0.0)
					band_hist.append(br)
			if mdates[t].month == 12:
				print(f"{mdates[t].year} emitted {n_emit} months, elapsed "
				      f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)

		# this month enters the ridge only after its return is realised
		if fin.sum() > 2:
			rw = month_rows[t]
			xr = expand(r[rw].astype(np.float64))
			xtx += xr.T @ xr
			xty += xr.T @ month_lab[t]

	out = pd.concat(frames, ignore_index=True)
	out["id"] = out["id"].astype(np.int64)
	print(f"done: {len(out):,} rows, {n_emit} months in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
	return out


def sharpe_of(series):
	if len(series) < 12:
		return float("nan")
	sd = series.std(ddof=1)
	return float(series.mean() / sd * np.sqrt(12.0)) if sd > 1e-12 else float("nan")


def report_sharpe(weights, chars):
	ret = chars[["id", "eom", target_col]].copy()
	ret["eom"] = pd.to_datetime(ret["eom"])
	ret["id"] = np.asarray(pd.to_numeric(ret["id"]), dtype=np.int64)
	ret = ret.rename(columns={target_col: "r"}).dropna(subset=["r"])
	w = weights.copy()
	w["eom"] = pd.to_datetime(w["eom"])
	j = w.merge(ret, on=["id", "eom"], how="inner")
	j["c"] = j["w"] * j["r"]
	s = j.groupby("eom")["c"].sum().sort_index()
	pre = s[s.index < pre_test_date]
	post = s[s.index >= pre_test_date]
	print()
	print("months total", len(s), "pre", len(pre), "test", len(post), flush=True)
	print("TEST WINDOW sharpe", round(sharpe_of(post), 3), flush=True)
	print("PRE TEST sharpe", round(sharpe_of(pre), 3), flush=True)
	if len(post) > 1:
		print("annualised return", round(float(post.mean()) * 12.0, 4),
		      "volatility", round(float(post.std(ddof=1)) * np.sqrt(12.0), 4), flush=True)


def is_test_row(flag):
	if pd.api.types.is_bool_dtype(flag):
		return flag.fillna(False)
	if pd.api.types.is_numeric_dtype(flag):
		return flag.fillna(0).ne(0)
	return flag.astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "t"])


def restrict_to_test(chars, weights):
	if "ctff_test" not in chars.columns:
		return weights
	mask = is_test_row(chars["ctff_test"])
	if not mask.any():
		return weights
	keys = chars.loc[mask, ["id", "eom"]].copy()
	keys["id"] = np.asarray(pd.to_numeric(keys["id"]), dtype=np.int64)
	keys["eom"] = pd.to_datetime(keys["eom"])
	keys = keys.drop_duplicates()
	w = weights.copy()
	w["eom"] = pd.to_datetime(w["eom"])
	out = w.merge(keys, on=["id", "eom"], how="inner")
	assert len(out), "ctff_test is set but no weights match those rows"
	return out


def main(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
	weights = build_weights(chars, features, daily_ret)
	report_sharpe(weights, chars)
	out = restrict_to_test(chars, weights)
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
	print("wrote output.csv rows", len(pf), "months", pf["eom"].nunique(), flush=True)
