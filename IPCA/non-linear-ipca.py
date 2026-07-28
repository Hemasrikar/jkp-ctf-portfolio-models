"""Nonlinear IPCA with precision weighting.

Ported from the original with two changes only:

	The per year loop scores every year with sufficient history, not just test
	years. The reference produces weights from 1990 onward.

Sharpe is reported separately for the test window and the pre test window, since
those are different samples and conflating them has caused confusion before.
"""

import gc
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data_dir = os.environ.get("jkp_data_dir", "jkp-data")

# architecture, as the reference
h_dim = 32
wide_dim = 256
n_layers = 7
var_wide_dim = 256
var_n_layers = 7
dropout = 0.1

# training, as the reference
lr = 3e-4
epochs_load = 50
epochs_var = 50
accum_steps = 12
weight_decay = 1e-4
min_train_months = 120
seed = 42

# scoring
min_stocks = 30
test_start = "1990-01-01"
first_year = None      # None means every year with enough history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
	random.seed(s)
	np.random.seed(s)
	torch.manual_seed(s)
	torch.cuda.manual_seed_all(s)


def prepare_data(chars, feat_cols):
	out = chars.copy()
	for feat in feat_cols:
		zeros = out[feat] == 0
		out[feat] = out.groupby("eom")[feat].transform(lambda x: x.rank(method="max", pct=True))
		out.loc[zeros, feat] = 0.5
		out[feat] = out[feat].fillna(0.5)
	out[feat_cols] = out[feat_cols] - 0.5
	return out


def proj(in_dim, width, p):
	layers = [nn.Linear(in_dim, width), nn.SiLU()]
	if p > 0:
		layers.append(nn.Dropout(p))
	return nn.Sequential(*layers)


def hidden_block(dim, p):
	layers = [nn.Linear(dim, dim), nn.SiLU()]
	if p > 0:
		layers.append(nn.Dropout(p))
	return nn.Sequential(*layers)


class LoadingNet(nn.Module):
	def __init__(self, in_dim, hd, nl, width, p=dropout):
		super().__init__()
		self.input_proj = proj(in_dim, width, p)
		self.res_blocks = nn.ModuleList([hidden_block(width, p) for _ in range(max(0, nl - 2))])
		self.output = nn.Linear(width, hd)

	def forward(self, x):
		x = self.input_proj(x)
		for b in self.res_blocks:
			x = b(x) + x
		h = self.output(x)
		return h - h.mean(dim=0, keepdim=True)


class VarianceNet(nn.Module):
	def __init__(self, in_dim, nl, width, p=dropout):
		super().__init__()
		self.input_proj = proj(in_dim, width, p)
		self.res_blocks = nn.ModuleList([hidden_block(width, p) for _ in range(max(0, nl - 2))])
		self.output = nn.Linear(width, 1)

	def forward(self, x):
		x = self.input_proj(x)
		for b in self.res_blocks:
			x = b(x) + x
		return self.output(x).clamp(-10, 4)


def loading_loss(h, r_next):
	reg = 1e-4 * torch.eye(h.shape[1], device=h.device)
	f = torch.linalg.solve(h.T @ h + reg, h.T @ r_next)
	residuals = h @ f - r_next
	return (residuals ** 2).mean(), residuals


def variance_loss(log_var, residuals_sq):
	inv_var = (-log_var.squeeze()).exp()
	return (log_var.squeeze() + residuals_sq * inv_var).mean()


def build_samples(chars_train, feat_cols, device):
	samples = []
	for eom in sorted(chars_train["eom"].unique()):
		ct = chars_train[chars_train["eom"] == eom]
		if len(ct) < 20:
			continue
		x_full = ct[feat_cols].values.astype(float)
		r_next = ct["ret_exc_lead1m"].values.astype(float)
		valid = ~np.isnan(r_next)
		if valid.sum() < 10:
			continue
		samples.append((
			torch.tensor(x_full, dtype=torch.float32, device=device),
			torch.tensor(r_next, dtype=torch.float32, device=device),
			torch.tensor(valid, device=device),
		))
	if not samples:
		raise ValueError("no valid training months")
	return samples


def train_loading_net(samples, in_dim, device):
	net = LoadingNet(in_dim, h_dim, n_layers, wide_dim).to(device)
	opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
	sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_load)
	net.train()
	for _ in range(epochs_load):
		random.shuffle(samples)
		opt.zero_grad()
		for i, (xg, rg, vg) in enumerate(samples):
			h = net(xg)
			if vg.sum() >= 10:
				l_pred, _ = loading_loss(h[vg], rg[vg])
			else:
				l_pred = h.new_zeros(())
			(l_pred / accum_steps).backward()
			if (i + 1) % accum_steps == 0 or (i + 1) == len(samples):
				opt.step()
				opt.zero_grad()
		sched.step()
	return net


def train_variance_net(loading_net, samples, in_dim, device):
	loading_net.train(False)
	samples_v2, resid_sq = [], []
	with torch.no_grad():
		for xg, rg, vg in samples:
			if vg.sum() >= 10:
				h = loading_net(xg)
				_, res = loading_loss(h[vg], rg[vg])
				samples_v2.append((xg, vg))
				resid_sq.append(res.detach() ** 2)

	net = VarianceNet(in_dim, var_n_layers, var_wide_dim).to(device)
	opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
	sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_var)
	net.train()
	for _ in range(epochs_var):
		order = list(range(len(samples_v2)))
		random.shuffle(order)
		opt.zero_grad()
		for step, i in enumerate(order):
			xg, vg = samples_v2[i]
			log_var = net(xg)[vg]
			l_var = variance_loss(log_var, resid_sq[i])
			(l_var / accum_steps).backward()
			if (step + 1) % accum_steps == 0 or (step + 1) == len(order):
				opt.step()
				opt.zero_grad()
		sched.step()
	return net


@torch.no_grad()
def factor_stats(loading_net, samples, device):
	loading_net.train(False)
	reg = 1e-4 * torch.eye(h_dim, device=device)
	fs = []
	for xg, rg, vg in samples:
		hv = loading_net(xg)[vg]
		rv = rg[vg]
		f = torch.linalg.solve(hv.T @ hv + reg, hv.T @ rv)
		fs.append(f.cpu().numpy())
	fmat = np.stack(fs, axis=0)
	return fmat.mean(axis=0), np.cov(fmat.T) + 1e-6 * np.eye(h_dim), fmat


@torch.no_grad()
def max_sharpe_weights(loading_net, var_net, chars_t, feat_cols, f_bar, f_cov, device):
	ids = chars_t["id"].values
	x = torch.tensor(chars_t[feat_cols].values.astype(float), dtype=torch.float32, device=device)
	loading_net.train(False)
	var_net.train(False)
	h = loading_net(x).cpu().numpy()
	omega = np.exp(-var_net(x).squeeze().cpu().numpy())
	omega = omega / (omega.mean() + 1e-8)
	h_omega = h * omega[:, None]
	k = f_cov.shape[0]
	wf = np.linalg.solve(f_cov + 1e-6 * np.eye(k), f_bar)
	m = h_omega.T @ h + 1e-6 * np.eye(k)
	w_raw = h_omega @ np.linalg.solve(m, wf)
	denom = np.abs(w_raw).sum()
	w = w_raw / denom if denom > 1e-10 else np.ones(len(ids)) / len(ids)
	return pd.DataFrame({"id": ids, "w": w})


def sharpe_of(series):
	if len(series) < 12:
		return float("nan")
	sd = series.std(ddof=1)
	return float(series.mean() / sd * np.sqrt(12.0)) if sd > 1e-12 else float("nan")


def scaled_sharpe(series, target_annual=0.10, lookback=36):
	vals = series.to_numpy()
	tgt = target_annual / np.sqrt(12.0)
	out = np.zeros_like(vals)
	hist = []
	for i, v in enumerate(vals):
		lev = tgt / (float(np.std(hist[-lookback:], ddof=1)) + 1e-8) if len(hist) >= lookback else 1.0
		out[i] = float(np.clip(lev, 0.0, 10.0)) * v
		hist.append(v)
	sc = out[lookback:] if len(out) > lookback else out
	sd = np.std(sc, ddof=1)
	return float(np.mean(sc) / sd * np.sqrt(12.0)) if sd > 1e-12 else float("nan")


def main():
	t_start = time.time()
	chars = pd.read_parquet(data_dir + "/chars.parquet")
	features = pd.read_parquet(data_dir + "/features.parquet")
	print("torch", torch.__version__, "device", device, flush=True)

	feat_cols = [f for f in features["features"].tolist() if f in chars.columns]
	chars = chars.copy()
	chars["eom"] = pd.to_datetime(chars["eom"])
	print("features", len(feat_cols), flush=True)

	print("preparing characteristics", flush=True)
	chars = prepare_data(chars, feat_cols)

	years = sorted(chars["eom"].dt.year.unique())
	if first_year is not None:
		years = [y for y in years if y >= first_year]
	print("candidate years", years[0], "to", years[-1], flush=True)

	results = []
	for yi, yr in enumerate(years, 1):
		eval_eoms = sorted(chars.loc[chars["eom"].dt.year == yr, "eom"].unique())
		if not eval_eoms:
			continue
		cutoff = chars.loc[chars["eom"] < min(eval_eoms), "eom"].max()
		if pd.isna(cutoff):
			continue
		train_eoms = sorted(chars.loc[chars["eom"] <= cutoff, "eom"].unique())
		if len(train_eoms) < min_train_months:
			continue

		seed_all(seed)
		chars_train = chars[chars["eom"].isin(train_eoms)]
		t0 = time.time()
		samples = build_samples(chars_train, feat_cols, device)
		ln = train_loading_net(samples, len(feat_cols), device)
		vn = train_variance_net(ln, samples, len(feat_cols), device)
		f_bar, f_cov, _ = factor_stats(ln, samples, device)

		for eom in eval_eoms:
			ct = chars[chars["eom"] == eom]
			if len(ct) < min_stocks:
				continue
			wdf = max_sharpe_weights(ln, vn, ct, feat_cols, f_bar, f_cov, device)
			wdf["eom"] = eom
			results.append(wdf)

		del samples, ln, vn
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		print("year", yr, "(", yi, "of", len(years), ")", "train months", len(train_eoms),
		      "sec", round(time.time() - t0), "elapsed_min", round((time.time() - t_start) / 60.0, 1), flush=True)

	out = pd.concat(results, ignore_index=True)[["id", "eom", "w"]]
	out["id"] = out["id"].astype(int)
	out["w"] = out["w"].astype(float)

	ret = chars[["id", "eom", "ret_exc_lead1m"]].rename(columns={"ret_exc_lead1m": "r"}).dropna(subset=["r"])
	j = out.merge(ret, on=["id", "eom"], how="inner")
	j["c"] = j["w"] * j["r"]
	series = j.groupby("eom")["c"].sum().sort_index()

	pre = series[series.index < test_start]
	post = series[series.index >= test_start]

	print("months total", len(series), "pre", len(pre), "test", len(post), flush=True)
	print("TEST WINDOW (the leaderboard basis)")
	print("sharpe unscaled", round(sharpe_of(post), 3))
	print("sharpe scaled  ", round(scaled_sharpe(post), 3))
	print("PRE TEST WINDOW (the honest selection basis)")
	print("sharpe unscaled", round(sharpe_of(pre), 3))
	print("sharpe scaled  ", round(scaled_sharpe(pre), 3))
	print("FULL HISTORY")
	print("sharpe unscaled", round(sharpe_of(series), 3))
	out["eom"] = pd.to_datetime(out["eom"]).dt.strftime("%Y-%m-%d")
	out.to_csv("ipca_weights.csv", index=False)


if __name__ == "__main__":
	main()
