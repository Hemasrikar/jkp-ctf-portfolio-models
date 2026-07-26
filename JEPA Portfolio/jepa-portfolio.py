"""Joint Embedding Predictive Architecture for CTF portfolio construction.

The data pipeline and training protocol follow the nonlinear portfolio
transformer of Kelly et al. (2025). Features are selected from pre-test
coverage so the input dimension is fixed, returns come directly from
ret_exc_lead1m, and training uses a rolling window with per seed warm
starting and per seed L1 normalisation before averaging. The model itself
is a JEPA with an EMA target encoder, a latent predictor, VICReg style
regularisers, and a supervised weight head trained under the MSRR objective.
"""

import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# architecture
d_model = 128
d_hidden = 256
mask_ratio = 0.3
ema_decay = 0.996
lambda_jepa = 0.5
lambda_var = 1.0
lambda_cov = 0.04

# training
window = 60
n_epochs_cold = 50
n_epochs_warm = 10
n_seeds = 3
lr = 1e-4
grad_clip = 1.0
min_obs = 60

# data preprocessing
min_coverage = 0.90
pre_test_date = "1990-01-01"
min_stocks = 30
max_miss_frac = 1.0 / 3.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class jepa(nn.Module):
	def __init__(self, n_features):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(n_features, d_hidden),
			nn.LayerNorm(d_hidden),
			nn.GELU(),
			nn.Linear(d_hidden, d_model),
			nn.LayerNorm(d_model),
		)
		self.target_encoder = deepcopy(self.encoder)
		for p in self.target_encoder.parameters():
			p.requires_grad = False
		self.predictor = nn.Sequential(
			nn.Linear(d_model, d_hidden),
			nn.GELU(),
			nn.Linear(d_hidden, d_model),
		)
		self.head = nn.Sequential(
			nn.Linear(d_model, d_hidden),
			nn.GELU(),
			nn.Linear(d_hidden, 1),
		)

	def score(self, x):
		return self.head(self.encoder(x)).squeeze(-1)

	@torch.no_grad()
	def update_target(self):
		for pt, ps in zip(self.target_encoder.parameters(), self.encoder.parameters()):
			pt.mul_(ema_decay).add_(ps.detach(), alpha=1.0 - ema_decay)


def variance_loss(z):
	std = torch.sqrt(z.var(dim=0) + 1e-4)
	return torch.mean(F.relu(1.0 - std))


def covariance_loss(z):
	zc = z - z.mean(dim=0)
	cov = (zc.t() @ zc) / max(zc.shape[0] - 1, 1)
	off = cov - torch.diag(torch.diag(cov))
	return off.pow(2).sum() / z.shape[1]


def jepa_objective(model, x):
	mask = (torch.rand(x.shape, device=x.device) > mask_ratio).float()
	zc = model.encoder(x * mask)
	with torch.no_grad():
		zt = model.target_encoder(x)
	pred = model.predictor(zc)
	inv = F.mse_loss(pred, zt)
	reg = lambda_var * (variance_loss(zc) + variance_loss(pred))
	reg = reg + lambda_cov * (covariance_loss(zc) + covariance_loss(pred))
	return inv + reg


def train_model(model, x_list, r_list, n_epochs, device):
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	model.train()
	t = len(x_list)
	x_gpu = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in x_list]
	r_gpu = [torch.as_tensor(r, dtype=torch.float32, device=device) for r in r_list]
	for _ in range(n_epochs):
		order = np.random.permutation(t)
		for j in order:
			w = model.score(x_gpu[j])
			msrr = (1.0 - (w * r_gpu[j]).sum()) ** 2
			aux = jepa_objective(model, x_gpu[j])
			loss = msrr + lambda_jepa * aux
			opt.zero_grad(set_to_none=True)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			opt.step()
			model.update_target()
	del x_gpu, r_gpu
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
			torch.manual_seed(s * 10000 + 42)
			np.random.seed(s * 10000 + 42)
			model = jepa(n_features).to(device)
			if seed_states[s] is not None:
				model.load_state_dict(seed_states[s])
				n_ep = n_epochs_warm
			else:
				n_ep = n_epochs_cold

			model = train_model(model, x_list, r_list, n_ep, device)
			seed_states[s] = {k: v.cpu() for k, v in model.state_dict().items()}

			with torch.no_grad():
				x_t = torch.as_tensor(x_oos, dtype=torch.float32, device=device)
				w = model.score(x_t).cpu().numpy().astype(np.float64)

			abs_sum = np.abs(w).sum()
			if abs_sum > 1e-10:
				w /= abs_sum
				w_sum += w
				n_valid += 1

			del model, x_t

		if n_valid == 0:
			n_skip += 1
			continue

		w_avg = (w_sum / n_valid).astype(np.float32)
		keep = np.abs(w_avg) > 1e-15
		if keep.any():
			results.append(pd.DataFrame({
				"id": ids_oos[keep],
				"eom": eom,
				"w": w_avg[keep],
			}))
		n_done += 1

		if n_done % 50 == 0 or (m_idx + 1) == len(all_months):
			el = time.time() - t0
			print("progress", m_idx + 1, "of", len(all_months), "done", n_done, "skip", n_skip, "elapsed_min", round(el / 60.0, 1))

	el = time.time() - t0
	print("backtest complete oos_months", len(results), "skipped", n_skip, "total_min", round(el / 60.0, 1))
	if not results:
		return pd.DataFrame(columns=["id", "eom", "w"])
	return pd.concat(results, ignore_index=True)


def main(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
	print("torch", torch.__version__, "device", device)

	feature_names = features["features"].tolist()
	feat_cols = sorted([f for f in feature_names if f in chars.columns])
	print("candidate features", len(feat_cols))

	chars = chars.copy()
	chars["eom"] = pd.to_datetime(chars["eom"])

	pre_test = chars[chars["eom"] < pre_test_date]
	coverage = pre_test[feat_cols].notna().mean()
	valid_features = sorted([f for f in feat_cols if coverage[f] >= min_coverage])
	n_features = len(valid_features)
	print("features at coverage", min_coverage, "pre", pre_test_date[:4], "n", n_features)

	month_x = {}
	month_ids = {}
	month_mask = {}
	month_r = {}

	n_months = chars["eom"].nunique()
	print("processing months", n_months)
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

		if (i + 1) % 200 == 0:
			print("  processed", i + 1, "of", n_months, "sec", round(time.time() - t0))

	train_months = sorted(month_r.keys())
	print("train months", len(train_months), "total months", len(month_x), "n_features", n_features)

	output = rolling_backtest(train_months, month_x, month_ids, month_mask, month_r, n_features, device)
	print("output rows", len(output), "months", output["eom"].nunique() if len(output) else 0)

	if len(output):
		output["eom"] = pd.to_datetime(output["eom"]).dt.strftime("%Y-%m-%d")
	return output[["id", "eom", "w"]]


if __name__ == "__main__":
	chars = pd.read_parquet("jkp-data/chars.parquet")
	features = pd.read_parquet("jkp-data/features.parquet")
	daily_ret = pd.read_parquet("jkp-data/daily_ret.parquet")
	pf = main(chars, features, daily_ret)
	pf.to_csv("output.csv", index=False)
