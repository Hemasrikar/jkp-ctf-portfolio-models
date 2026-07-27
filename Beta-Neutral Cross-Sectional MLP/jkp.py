"""Cross sectional MLP with market beta neutralisation.

Each stock is encoded from its characteristics, mixed with a pooled cross
sectional context so that stocks inform each other before scoring, and trained
under a maximum Sharpe ratio regression objective on a rolling sixty month
window with per seed warm starting and seed averaging.

The traded weights are the L1 normalised scores with the component lying along
market beta projected out. The projection deliberately excludes an intercept, so
the portfolio's net exposure is preserved: removing it costs roughly half a Sharpe
point on this panel, while the residual market beta it leaves behind is
unrewarded risk. Beta is taken from beta_60m at the formation date, so the
projection uses no information beyond that month.

The realised Sharpe on the months whose forward returns are observed is printed
at the end, both unscaled and scaled to a ten percent annual volatility target.
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

# training
lr = 5e-5
weight_decay = 1e-4
grad_clip = 1.0
window = 60
n_epochs_cold = 50
n_epochs_warm = 10
n_seeds = 3
min_obs = 60

# neutralisation, the intercept is excluded so net exposure is preserved
beta_col = "beta_60m"

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


class xmlp_net(nn.Module):
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


def neutralise_beta(w, beta):
	# remove the component of w along beta, no intercept so net exposure survives
	if beta is None:
		return w
	b = beta.reshape(-1, 1)
	coef, _, _, _ = np.linalg.lstsq(b, w, rcond=None)
	return w - b @ coef


def rolling_backtest(train_months, month_x, month_ids, month_mask, month_r, month_beta, n_features, device):
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
			model = xmlp_net(n_features).to(device)
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

		w_avg = w_sum / n_valid
		w_avg = neutralise_beta(w_avg, month_beta.get(eom))
		denom = np.abs(w_avg).sum()
		if denom < 1e-12:
			n_skip += 1
			continue
		w_avg = (w_avg / denom).astype(np.float32)

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
	print("torch", torch.__version__, "device", device, flush=True)
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

	has_beta = beta_col in work.columns
	print("beta column", beta_col, "found" if has_beta else "NOT FOUND, neutralisation disabled", flush=True)

	month_x = {}
	month_ids = {}
	month_mask = {}
	month_r = {}
	month_beta = {}

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
		if has_beta:
			bt = md[beta_col].rank(pct=True).values.astype(np.float64)
			month_beta[eom] = np.nan_to_num(bt, nan=0.5) - 0.5
		if has_ret.sum() >= min_stocks:
			month_mask[eom] = has_ret
			month_r[eom] = rets[has_ret].astype(np.float32)
		if (i + 1) % 200 == 0:
			print("  processed", i + 1, "of", n_months, "sec", round(time.time() - t0), flush=True)

	train_months = sorted(month_r.keys())
	print("train months", len(train_months), "total months", len(month_x), flush=True)

	output = rolling_backtest(train_months, month_x, month_ids, month_mask, month_r, month_beta, n_features, device)
	print("output rows", len(output), "months", output["eom"].nunique() if len(output) else 0, flush=True)

	report_sharpe(output, chars)

	if len(output):
		output["eom"] = pd.to_datetime(output["eom"]).dt.strftime("%Y-%m-%d")
		output["id"] = output["id"].astype(int)
		output["w"] = output["w"].astype(float)
	return output[["id", "eom", "w"]]


if __name__ == "__main__":
	# local testing only, the pipeline does not run this block and loads the data
	# itself. the competition filenames are tried first, then the local layout, so
	# the same file runs locally and is submittable without edits.
	try:
		chars = pd.read_parquet("ctff_chars.parquet")
		features = pd.read_parquet("ctff_features.parquet")
		daily_ret = pd.read_parquet("ctff_daily_ret.parquet")
	except FileNotFoundError:
		chars = pd.read_parquet("jkp-data/chars.parquet")
		features = pd.read_parquet("jkp-data/features.parquet")
		daily_ret = pd.read_parquet("jkp-data/daily_ret.parquet")
	pf = main(chars, features, daily_ret)
	pf.to_csv("output.csv", index=False)
	print("wrote output.csv rows", len(pf), "months", pf["eom"].nunique(), flush=True)
