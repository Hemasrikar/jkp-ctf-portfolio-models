"""Coupled latent factor portfolio.

A network maps each stock's characteristics to k latent factor loadings. Each
stock's embedding is mixed with a pooled context over the whole cross section
before the loadings are read off, so a loading depends on the composition of the
month rather than on the stock alone. Loadings are demeaned across stocks and each
column is normalised to unit gross exposure, giving k long short factor
portfolios.

A second network predicts each stock's log idiosyncratic variance, fitted by
Gaussian likelihood on the residuals of the factor model. Its inverse weights each
stock's contribution to the factor portfolios, so noisy names carry less.

The loading network is fitted on the realised Sharpe ratio of the equally combined
factors across a batch of months. The combination actually traded is solved in
closed form from the mean and covariance of the factor returns, exponentially
weighted toward recent months and shrunk toward a scaled identity.

The book is the combination of factor portfolios, averaged over seeds, with the
component lying along market beta projected out. That projection excludes an
intercept, so the net exposure of the book is preserved.

Networks are refitted yearly on months whose forward returns precede that year,
moments use only training window factor returns, and the coverage filter is
recomputed yearly from months at or before the cutoff."""

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
n_seeds = 8
base_seed = 42
min_train_months = 120

# beta neutralisation
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
	print("torch", torch.__version__, "device", device, "seeds", n_seeds, "beta", beta_col, "k", k_factors,
	      "halflife", moment_halflife, "precision", use_precision, flush=True)
	seed_all(base_seed)

	candidates = [f for f in features["features"].tolist() if f in chars.columns]
	work = chars.copy()
	work["eom"] = pd.to_datetime(work["eom"]) + pd.offsets.MonthEnd(0)
	work["id"] = pd.to_numeric(work["id"], errors="coerce").astype("int64")

	by_month = {eom: md for eom, md in work.groupby("eom", sort=True)}
	all_months = sorted(by_month.keys())
	target_months = [m for m in all_months
	                 if np.isfinite(by_month[m]["ret_exc_lead1m"].to_numpy(dtype=np.float64)).sum() >= min_stocks]
	print("months", len(all_months), "with targets", len(target_months), flush=True)

	years = sorted({pd.Timestamp(m).year for m in all_months})
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
			# loadings first without precision, then precision on their residuals,
			# then loadings refined with precision in place
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
			results.append(pd.DataFrame({"id": md["id"].to_numpy(), "eom": eom, "w": w / d}))
			n_done += 1

	if not results:
		raise ValueError("no weights produced")

	out = pd.concat(results, ignore_index=True)
	print("months solved", n_done, "skipped", n_skip,
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
	pf.to_csv("output.csv", index=False)
	print("wrote output.csv rows", len(pf), "months", pf["eom"].nunique(), flush=True)
