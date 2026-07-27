"""JEPA regularised cross sectional MLP with corrected masking.

The base network is the cross sectional MLP that scores 3.322 on this panel: each
stock is encoded, mixed with a pooled cross sectional context so stocks inform
each other, and scored under the MSRR objective with L1 normalised weights.

A joint embedding predictive branch regularises the encoder. Two defects in the
earlier attempt are corrected here, and both mattered.

	Masking is visible. Previously masked cells were multiplied by zero, but the
	inputs are rank standardised about zero, so a masked cell became the median
	rank and the encoder could not distinguish hidden from median. Masked entries
	are now replaced by a learned per characteristic mask embedding, so the model
	knows what it is being asked to infer.

	Masking is structured. Previously individual cells were masked independently.
	The characteristics are heavily redundant, with many momentum, growth and
	valuation variants, so a masked cell was recoverable by interpolation from its
	correlated siblings and the pretext task taught little. Whole characteristics
	are now masked across every stock in a month, which forces genuine inference
	and mirrors the way characteristics are actually absent in the panel.

lambda_jepa switches the branch off at zero, recovering the base model exactly,
so the same script produces both arms of the comparison.
"""

import copy
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# architecture
d_model = 128
d_hidden = 256

# jepa branch
lambda_jepa = 0.5
mask_ratio = 0.25
ema_decay = 0.996
lambda_var = 1.0
lambda_cov = 0.04

# training
lr = 5e-5
weight_decay = 1e-4
grad_clip = 1.0
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

vol_target_annual = 0.10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
	random.seed(s)
	np.random.seed(s)
	torch.manual_seed(s)
	torch.cuda.manual_seed_all(s)


class jepa_mlp(nn.Module):
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
		# learned replacement value per characteristic, so masking is visible
		self.mask_emb = nn.Parameter(torch.zeros(n_features))
		self.target_encoder = copy.deepcopy(self.encoder)
		for p in self.target_encoder.parameters():
			p.requires_grad = False
		self.predictor = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_model))

	def score(self, x):
		z = self.encoder(x)
		ctx = z.mean(dim=0, keepdim=True).expand_as(z)
		return self.head(torch.cat([z, ctx], dim=1)).squeeze(-1)

	@torch.no_grad()
	def update_target(self):
		for pt, ps in zip(self.target_encoder.parameters(), self.encoder.parameters()):
			pt.mul_(ema_decay).add_(ps.detach(), alpha=1.0 - ema_decay)


def variance_reg(z):
	std = torch.sqrt(z.var(dim=0) + 1e-4)
	return torch.mean(F.relu(1.0 - std))


def covariance_reg(z):
	zc = z - z.mean(dim=0)
	cov = (zc.t() @ zc) / max(zc.shape[0] - 1, 1)
	off = cov - torch.diag(torch.diag(cov))
	return off.pow(2).sum() / z.shape[1]


def jepa_objective(model, x):
	# structured masking: whole characteristics are hidden for every stock
	n_features = x.shape[1]
	col_mask = (torch.rand(n_features, device=x.device) < mask_ratio).float()
	x_masked = x * (1.0 - col_mask) + model.mask_emb * col_mask
	zc = model.encoder(x_masked)
	with torch.no_grad():
		zt = model.target_encoder(x)
	pred = model.predictor(zc)
	inv = F.mse_loss(pred, zt)
	reg = lambda_var * (variance_reg(zc) + variance_reg(pred))
	reg = reg + lambda_cov * (covariance_reg(zc) + covariance_reg(pred))
	return inv + reg


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
			if lambda_jepa > 0:
				loss = loss + lambda_jepa * jepa_objective(model, xg[j])
			opt.zero_grad(set_to_none=True)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			opt.step()
			if lambda_jepa > 0:
				model.update_target()
	del xg, rg
	return model


def rolling_backtest(train_months, month_x, month_ids, month_mask, month_r, n_features, device):
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
			model = jepa_mlp(n_features).to(device)
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

		w_avg = (w_sum / n_valid).astype(np.float32)
		keep = np.abs(w_avg) > 1e-15
		if keep.any():
			results.append(pd.DataFrame({"id": ids_oos[keep], "eom": eom, "w": w_avg[keep]}))
		n_done += 1
		if n_done % 50 == 0 or (m_idx + 1) == len(all_months):
			print("progress", m_idx + 1, "of", len(all_months), "done", n_done, "skip", n_skip, "elapsed_min", round((time.time() - t0) / 60.0, 1), flush=True)

	print("backtest complete oos_months", len(results), "skipped", n_skip, "total_min", round((time.time() - t0) / 60.0, 1), flush=True)
	if not results:
		return pd.DataFrame(columns=["id", "eom", "w"])
	return pd.concat(results, ignore_index=True)


def report_sharpe(weights, chars):
	ret = chars[["id", "eom", "ret_exc_lead1m"]].copy()
	ret["eom"] = pd.to_datetime(ret["eom"]) + pd.offsets.MonthEnd(0)
	ret = ret.rename(columns={"ret_exc_lead1m": "r"}).dropna(subset=["r"])
	w = weights.copy()
	w["eom"] = pd.to_datetime(w["eom"]) + pd.offsets.MonthEnd(0)
	j = w.merge(ret, on=["id", "eom"], how="inner")
	j["c"] = j["w"] * j["r"]
	series = j.groupby("eom")["c"].sum().sort_index()
	if len(series) < 12:
		print("sharpe not reported, too few observed months", flush=True)
		return
	sd = series.std(ddof=1)
	sharpe = float(series.mean() / sd * np.sqrt(12.0)) if sd > 1e-12 else float("nan")
	target = vol_target_annual / np.sqrt(12.0)
	vals = series.to_numpy()
	scaled = np.zeros_like(vals)
	hist = []
	for i, v in enumerate(vals):
		lev = target / (float(np.std(hist[-36:], ddof=1)) + 1e-8) if len(hist) >= 36 else 1.0
		scaled[i] = float(np.clip(lev, 0.0, 10.0)) * v
		hist.append(v)
	sc = scaled[36:] if len(scaled) > 36 else scaled
	sdsc = np.std(sc, ddof=1)
	sh_sc = float(np.mean(sc) / sdsc * np.sqrt(12.0)) if sdsc > 1e-12 else float("nan")
	print("months scored", len(series), flush=True)
	print("sharpe unscaled", round(sharpe, 3), flush=True)
	print("sharpe scaled to 10 percent vol", round(sh_sc, 3), flush=True)


def main(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
	print("torch", torch.__version__, "device", device, "lambda_jepa", lambda_jepa, "mask_ratio", mask_ratio, flush=True)
	seed_all(42)

	feature_names = [f for f in features["features"].tolist() if f in chars.columns]

	work = chars.copy()
	work["eom"] = pd.to_datetime(work["eom"]) + pd.offsets.MonthEnd(0)

	pre = work[work["eom"] < pre_test_date]
	if len(pre):
		frac = pre[feature_names].notna().mean()
		feature_names = [f for f in feature_names if frac[f] >= min_coverage]
	n_features = len(feature_names)
	print("features", n_features, flush=True)

	month_x = {}
	month_ids = {}
	month_mask = {}
	month_r = {}

	n_months = work["eom"].nunique()
	print("processing months", n_months, flush=True)
	t0 = time.time()
	for i, (eom, md) in enumerate(work.groupby("eom", sort=True)):
		n_missing = md[feature_names].isna().sum(axis=1)
		md = md.loc[(n_missing <= n_features * max_miss_frac).values]
		if len(md) < min_stocks:
			continue
		x = md[feature_names].rank(pct=True).values.astype(np.float32)
		x = np.nan_to_num(x, nan=0.5) - 0.5
		ids = md["id"].values
		rets = md["ret_exc_lead1m"].values
		has_ret = np.isfinite(rets)
		month_x[eom] = x
		month_ids[eom] = ids
		if has_ret.sum() >= min_stocks:
			month_mask[eom] = has_ret
			month_r[eom] = rets[has_ret].astype(np.float32)
		if (i + 1) % 200 == 0:
			print("  processed", i + 1, "of", n_months, "sec", round(time.time() - t0), flush=True)

	train_months = sorted(month_r.keys())
	print("train months", len(train_months), "total months", len(month_x), flush=True)

	output = rolling_backtest(train_months, month_x, month_ids, month_mask, month_r, n_features, device)
	print("output rows", len(output), "months", output["eom"].nunique() if len(output) else 0, flush=True)

	report_sharpe(output, chars)

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
