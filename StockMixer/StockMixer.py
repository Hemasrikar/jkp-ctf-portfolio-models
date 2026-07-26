"""StockMixer style cross sectional model for CTF portfolio construction.

The model follows the mixing idea of StockMixer (Fan and Shen, AAAI 2024).
Indicator mixing exchanges information across characteristics, then stock mixing
exchanges information across the cross section through explicit stock to market
and market to stock steps, which generalises the single pooled context that
lifted the plain per stock model. Time mixing is omitted, since the monthly
panel carries no per stock lookback. Weights are the L1 normalised scores under
the MSRR objective, with the settled data pipeline, rolling window, per seed warm
starting, and seed averaging. After producing the weights the script reports the
realised Sharpe on the observed return months, unscaled and scaled to a ten
percent annual volatility target.
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
d_token = 64
d_market = 32
mix_hidden = 128
dropout = 0.1

# training
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


def mlp(in_dim, hidden, out_dim):
	return nn.Sequential(
		nn.Linear(in_dim, hidden),
		nn.GELU(),
		nn.Dropout(dropout),
		nn.Linear(hidden, out_dim),
	)


class stockmixer_net(nn.Module):
	def __init__(self, n_features):
		super().__init__()
		# indicator mixing: exchange information across characteristics per stock
		self.indicator_norm = nn.LayerNorm(n_features)
		self.indicator_mix = mlp(n_features, mix_hidden, n_features)
		# per stock token
		self.tokeniser = nn.Sequential(nn.Linear(n_features, d_token), nn.LayerNorm(d_token), nn.GELU())
		# stock to market: pool tokens into a market state
		self.to_market = mlp(d_token, mix_hidden, d_market)
		# market to stock: broadcast the market state back to each stock
		self.from_market = mlp(d_token + d_market, mix_hidden, d_token)
		self.token_norm = nn.LayerNorm(d_token)
		# score head
		self.head = nn.Sequential(nn.Linear(d_token, mix_hidden), nn.GELU(), nn.Linear(mix_hidden, 1))

	def score(self, x):
		x = x + self.indicator_mix(self.indicator_norm(x))
		z = self.tokeniser(x)
		market = self.to_market(z).mean(dim=0, keepdim=True).expand(z.shape[0], -1)
		z = z + self.from_market(torch.cat([z, market], dim=1))
		z = self.token_norm(z)
		return self.head(z).squeeze(-1)


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
			opt.zero_grad(set_to_none=True)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			opt.step()
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
			model = stockmixer_net(n_features).to(device)
			if seed_states[s] is not None:
				model.load_state_dict(seed_states[s])
				n_ep = n_epochs_warm
			else:
				n_ep = n_epochs_cold
			model = train_model(model, x_list, r_list, n_ep, device)
			seed_states[s] = {k: v.cpu() for k, v in model.state_dict().items()}
			model.eval()
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
			el = time.time() - t0
			print("progress", m_idx + 1, "of", len(all_months), "done", n_done, "skip", n_skip, "elapsed_min", round(el / 60.0, 1), flush=True)

	el = time.time() - t0
	print("backtest complete oos_months", len(results), "skipped", n_skip, "total_min", round(el / 60.0, 1), flush=True)
	if not results:
		return pd.DataFrame(columns=["id", "eom", "w"])
	return pd.concat(results, ignore_index=True)


def report_sharpe(weights, chars):
	ret = chars[["id", "eom", "ret_exc_lead1m"]].copy()
	ret["eom"] = pd.to_datetime(ret["eom"]) + pd.offsets.MonthEnd(0)
	ret = ret.rename(columns={"ret_exc_lead1m": "r"}).dropna(subset=["r"])

	w = weights.copy()
	w["eom"] = pd.to_datetime(w["eom"]) + pd.offsets.MonthEnd(0)

	joined = w.merge(ret, on=["id", "eom"], how="inner")
	joined["c"] = joined["w"] * joined["r"]
	series = joined.groupby("eom")["c"].sum().sort_index()
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
