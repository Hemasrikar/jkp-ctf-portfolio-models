"""Ensemble of the cross sectional MLP and the StockMixer model for CTF.

Both models are trained on the same rolling window with per seed warm starting
and seed averaging, their per month weights are blended equally and renormalised
to unit gross exposure, and the blended weights are returned for scoring. The
data pipeline follows the settled protocol. The end of run prints the realised
Sharpe on the observed return months, unscaled and scaled to a ten percent annual
volatility target.
"""

import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# architecture
d_model = 128
d_hidden = 256
d_token = 64
d_market = 32
mix_hidden = 128
dropout = 0.1

# training, full budget
lr = 1e-4
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

# reporting
vol_target_annual = 0.10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
	random.seed(s)
	np.random.seed(s)
	torch.manual_seed(s)
	torch.cuda.manual_seed_all(s)


class mlp_net(nn.Module):
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


def small_mlp(in_dim, hidden, out_dim):
	return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))


class stockmixer_net(nn.Module):
	def __init__(self, n_features):
		super().__init__()
		self.indicator_norm = nn.LayerNorm(n_features)
		self.indicator_mix = small_mlp(n_features, mix_hidden, n_features)
		self.tokeniser = nn.Sequential(nn.Linear(n_features, d_token), nn.LayerNorm(d_token), nn.GELU())
		self.to_market = small_mlp(d_token, mix_hidden, d_market)
		self.from_market = small_mlp(d_token + d_market, mix_hidden, d_token)
		self.token_norm = nn.LayerNorm(d_token)
		self.head = nn.Sequential(nn.Linear(d_token, mix_hidden), nn.GELU(), nn.Linear(mix_hidden, 1))

	def score(self, x):
		x = x + self.indicator_mix(self.indicator_norm(x))
		z = self.tokeniser(x)
		market = self.to_market(z).mean(dim=0, keepdim=True).expand(z.shape[0], -1)
		z = z + self.from_market(torch.cat([z, market], dim=1))
		z = self.token_norm(z)
		return self.head(z).squeeze(-1)


def train_score_model(model, x_list, r_list, n_epochs, device):
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


factories = {"mlp": mlp_net, "stockmixer": stockmixer_net}


def run_model(name, train_months, month_x, month_ids, month_mask, month_r, n_features, device):
	net_factory = factories[name]
	all_months = sorted(month_x.keys())
	seed_states = [None] * n_seeds
	rows = []
	t0 = time.time()
	done = 0
	for eom in all_months:
		cutoff = eom - pd.DateOffset(months=1)
		avail = [m for m in train_months if m <= cutoff]
		if len(avail) < min_obs:
			continue
		win = avail[-window:] if len(avail) > window else avail
		x_list = [month_x[m][month_mask[m]] for m in win]
		r_list = [month_r[m] for m in win]
		x_oos = month_x[eom]
		ids_oos = month_ids[eom]
		if len(ids_oos) < min_stocks:
			continue

		w_sum = np.zeros(len(ids_oos), dtype=np.float64)
		n_valid = 0
		for s in range(n_seeds):
			seed_all(s * 10000 + 42)
			model = net_factory(n_features).to(device)
			if seed_states[s] is not None:
				model.load_state_dict(seed_states[s])
				n_ep = n_epochs_warm
			else:
				n_ep = n_epochs_cold
			model = train_score_model(model, x_list, r_list, n_ep, device)
			seed_states[s] = {k: v.cpu() for k, v in model.state_dict().items()}
			model.train(False)
			with torch.no_grad():
				w = model.score(torch.as_tensor(x_oos, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)
			denom = np.abs(w).sum()
			if denom > 1e-10:
				w_sum += w / denom
				n_valid += 1
			del model
		if n_valid == 0:
			continue
		rows.append(pd.DataFrame({"id": ids_oos, "eom": eom, "w": w_sum / n_valid}))
		done += 1
		if done % 100 == 0:
			print(" ", name, "done", done, "elapsed_min", round((time.time() - t0) / 60.0, 1), flush=True)
	print(name, "complete months", len(rows), "min", round((time.time() - t0) / 60.0, 1), flush=True)
	if not rows:
		return pd.DataFrame(columns=["id", "eom", "w"])
	return pd.concat(rows, ignore_index=True)


def ensemble_weights(a, b):
	merged = a.merge(b, on=["id", "eom"], how="outer", suffixes=("_a", "_b")).fillna(0.0)
	merged["w"] = 0.5 * merged["w_a"] + 0.5 * merged["w_b"]
	merged["w"] = merged.groupby("eom")["w"].transform(lambda x: x / (x.abs().sum() + 1e-12))
	return merged[["id", "eom", "w"]]


def report_sharpe(weights, chars):
	ret = chars[["id", "eom", "ret_exc_lead1m"]].copy()
	ret = ret.rename(columns={"ret_exc_lead1m": "r"}).dropna(subset=["r"])
	w = weights.merge(ret, on=["id", "eom"], how="inner")
	w["c"] = w["w"] * w["r"]
	series = w.groupby("eom")["c"].sum().sort_index()
	if len(series) < 12:
		print("sharpe not reported, too few observed months", flush=True)
		return
	mu = series.mean()
	sd = series.std(ddof=1)
	sharpe = float(mu / sd * np.sqrt(12.0)) if sd > 1e-12 else float("nan")
	target_monthly = vol_target_annual / np.sqrt(12.0)
	values = series.to_numpy()
	scaled = np.zeros_like(values)
	realised = []
	for i, v in enumerate(values):
		if len(realised) >= 36:
			trailing = float(np.std(realised[-36:], ddof=1))
			lev = target_monthly / (trailing + 1e-8)
		else:
			lev = 1.0
		scaled[i] = float(np.clip(lev, 0.0, 10.0)) * v
		realised.append(v)
	sc = scaled[36:] if len(scaled) > 36 else scaled
	sd_sc = np.std(sc, ddof=1)
	sharpe_scaled = float(np.mean(sc) / sd_sc * np.sqrt(12.0)) if sd_sc > 1e-12 else float("nan")
	print("months scored", len(series), flush=True)
	print("sharpe unscaled", round(sharpe, 3), flush=True)
	print("sharpe scaled to", int(vol_target_annual * 100), "percent vol", round(sharpe_scaled, 3), flush=True)


def main(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
	print("torch", torch.__version__, "device", device, flush=True)
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
		if (i + 1) % 200 == 0:
			print("  processed", i + 1, "of", n_months, "sec", round(time.time() - t0), flush=True)

	train_months = sorted(month_r.keys())
	print("train months", len(train_months), "total months", len(month_x), "n_features", n_features, flush=True)

	print("running mlp", flush=True)
	mlp_w = run_model("mlp", train_months, month_x, month_ids, month_mask, month_r, n_features, device)
	print("running stockmixer", flush=True)
	sm_w = run_model("stockmixer", train_months, month_x, month_ids, month_mask, month_r, n_features, device)

	output = ensemble_weights(mlp_w, sm_w)
	print("output rows", len(output), "months", output["eom"].nunique() if len(output) else 0, flush=True)

	report_sharpe(output, chars)

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