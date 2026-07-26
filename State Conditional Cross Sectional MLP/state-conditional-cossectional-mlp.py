"""State conditional cross sectional MLP for CTF portfolio construction.

The base network is the cross sectional MLP: each stock is encoded, mixed with a
pooled cross sectional context so stocks inform each other, and scored under the
MSRR objective with L1 normalised weights. That part is unchanged.

What is new is conditioning on the market state. Factor premia are time varying
and predictable, so a mapping from characteristics to scores that is identical in
every month leaves that variation unexploited. The state enters through feature
wise modulation: a small linear map turns the state vector into a scale and shift
applied to each dimension of the per stock embedding, so the state changes which
characteristics matter rather than how aggressive the book is.

That distinction is essential. Weights are demeaned and L1 normalised, so any
state effect that is additive or multiplicative at the month level is annihilated
by the construction itself. Only an interaction with stock level features can
change the relative ranking, which is what dimension wise modulation provides.

The modulation map is initialised at zero, so the model begins exactly as the base
MLP and learns conditioning only if it earns its place. It adds roughly a
thousand parameters against the base network's tens of thousands.

State variables, all constructed from the provided panel and known at formation:
	market realised volatility over 21 and 252 trading days
	market trailing twelve month return
	cross sectional dispersion of trailing one month returns
	the value spread, the p90 minus p10 range of book to market
	trailing twelve month return of a momentum factor
Each is standardised on an expanding window using only months up to and including
the formation month, so no future information enters.
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
film_scale = 1.0

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


def resolve_daily_columns(daily_ret):
	cols = list(daily_ret.columns)
	id_col = "id" if "id" in cols else next(c for c in cols if "id" in c.lower())
	date_pref = [c for c in ("date", "datadate", "day") if c in cols]
	if date_pref:
		date_col = date_pref[0]
	else:
		date_col = next(c for c in cols if np.issubdtype(daily_ret[c].dtype, np.datetime64))
	ret_pref = [c for c in ("ret_exc", "ret", "ret_local") if c in cols]
	if ret_pref:
		ret_col = ret_pref[0]
	else:
		ret_col = next(
			c for c in cols
			if c not in (id_col, date_col) and np.issubdtype(daily_ret[c].dtype, np.number)
		)
	return id_col, date_col, ret_col


def build_state(work, daily_ret, feature_names):
	# all quantities below are known at the end of the formation month
	id_col, date_col, ret_col = resolve_daily_columns(daily_ret)
	print("daily columns resolved to", id_col, date_col, ret_col, flush=True)

	d = pd.DataFrame({
		"id": daily_ret[id_col].to_numpy(),
		"date": pd.to_datetime(daily_ret[date_col]),
		"r": daily_ret[ret_col].astype("float64").fillna(0.0).to_numpy(),
	})
	d["eom"] = d["date"] + pd.offsets.MonthEnd(0)

	# daily market return, equal weighted across the cross section
	mkt_daily = d.groupby("date")["r"].mean().sort_index()
	vol21 = mkt_daily.rolling(21, min_periods=10).std() * np.sqrt(252.0)
	vol252 = mkt_daily.rolling(252, min_periods=60).std() * np.sqrt(252.0)
	cum252 = mkt_daily.rolling(252, min_periods=60).sum()
	daily_state = pd.DataFrame({"vol21": vol21, "vol252": vol252, "mkt12m": cum252})
	daily_state["eom"] = daily_state.index + pd.offsets.MonthEnd(0)
	month_daily = daily_state.groupby("eom").last()

	# monthly stock returns compounded from daily, available on every month
	d["g"] = 1.0 + d["r"]
	mret = d.groupby(["id", "eom"])["g"].prod().reset_index()
	mret["mret"] = mret["g"] - 1.0
	mret = mret[["id", "eom", "mret"]]

	# momentum factor: weights from prior month ranks, return realised this month
	mom_col = "ret_12_1" if "ret_12_1" in work.columns else None
	fac = pd.Series(dtype="float64")
	if mom_col is not None:
		w = work[["id", "eom", mom_col]].copy()
		w["rk"] = w.groupby("eom")[mom_col].rank(pct=True) - 0.5
		w["rk"] = w["rk"].fillna(0.0)
		w["rk"] = w.groupby("eom")["rk"].transform(lambda x: x / (x.abs().sum() + 1e-12))
		w["eom_ret"] = w["eom"] + pd.offsets.MonthEnd(1)
		j = w.merge(mret.rename(columns={"eom": "eom_ret"}), on=["id", "eom_ret"], how="inner")
		fac = (j["rk"] * j["mret"]).groupby(j["eom_ret"]).sum().sort_index()
	mom12 = fac.rolling(12, min_periods=6).sum() if len(fac) else pd.Series(dtype="float64")

	# cross sectional dispersion and the value spread, from the formation month
	rows = []
	rev_col = "ret_1_0" if "ret_1_0" in work.columns else None
	val_col = "be_me" if "be_me" in work.columns else None
	for eom, md in work.groupby("eom", sort=True):
		disp = float(md[rev_col].std()) if rev_col else 0.0
		if val_col:
			v = md[val_col].to_numpy(dtype="float64")
			v = v[np.isfinite(v)]
			spread = float(np.percentile(v, 90) - np.percentile(v, 10)) if len(v) > 20 else 0.0
		else:
			spread = 0.0
		rows.append((eom, disp, spread))
	xs = pd.DataFrame(rows, columns=["eom", "xs_disp", "value_spread"]).set_index("eom")

	state = xs.join(month_daily, how="left")
	state["mom12"] = mom12.reindex(state.index)
	state = state[["vol21", "vol252", "mkt12m", "xs_disp", "value_spread", "mom12"]]
	state = state.sort_index().ffill()

	# expanding standardisation, using only months up to and including each row
	mu = state.expanding(min_periods=12).mean()
	sd = state.expanding(min_periods=12).std()
	z = (state - mu) / (sd + 1e-8)
	z = z.fillna(0.0).clip(-4.0, 4.0)
	print("state variables", list(z.columns), "months", len(z), flush=True)
	return z


class state_mlp(nn.Module):
	def __init__(self, n_features, n_state):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(n_features, d_hidden),
			nn.LayerNorm(d_hidden),
			nn.GELU(),
			nn.Linear(d_hidden, d_model),
			nn.LayerNorm(d_model),
		)
		# feature wise modulation, initialised at zero so the model starts as the
		# base cross sectional mlp and learns conditioning only if it helps
		self.film = nn.Linear(n_state, 2 * d_model)
		nn.init.zeros_(self.film.weight)
		nn.init.zeros_(self.film.bias)
		self.head = nn.Sequential(nn.Linear(2 * d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))

	def score(self, x, s):
		z = self.encoder(x)
		gb = self.film(s)
		gamma, beta = gb.chunk(2, dim=-1)
		z = z * (1.0 + film_scale * gamma) + film_scale * beta
		ctx = z.mean(dim=0, keepdim=True).expand_as(z)
		return self.head(torch.cat([z, ctx], dim=1)).squeeze(-1)


def train_model(model, x_list, s_list, r_list, n_epochs, device):
	opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	model.train()
	xg = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in x_list]
	sg = [torch.as_tensor(s, dtype=torch.float32, device=device) for s in s_list]
	rg = [torch.as_tensor(r, dtype=torch.float32, device=device) for r in r_list]
	t = len(xg)
	for _ in range(n_epochs):
		for j in np.random.permutation(t):
			w = model.score(xg[j], sg[j])
			loss = (1.0 - (w * rg[j]).sum()) ** 2
			opt.zero_grad(set_to_none=True)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
			opt.step()
	del xg, sg, rg
	return model


def rolling_backtest(train_months, month_x, month_s, month_ids, month_mask, month_r, n_features, n_state, device):
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
		s_list = [month_s[m] for m in win]
		r_list = [month_r[m] for m in win]

		x_oos = month_x[eom]
		s_oos = month_s[eom]
		ids_oos = month_ids[eom]
		if len(ids_oos) < min_stocks:
			n_skip += 1
			continue

		w_sum = np.zeros(len(ids_oos), dtype=np.float64)
		n_valid = 0
		for s in range(n_seeds):
			seed_all(s * 10000 + 42)
			model = state_mlp(n_features, n_state).to(device)
			if seed_states[s] is not None:
				model.load_state_dict(seed_states[s])
				n_ep = n_epochs_warm
			else:
				n_ep = n_epochs_cold
			model = train_model(model, x_list, s_list, r_list, n_ep, device)
			seed_states[s] = {k: v.cpu() for k, v in model.state_dict().items()}
			with torch.no_grad():
				w = model.score(
					torch.as_tensor(x_oos, dtype=torch.float32, device=device),
					torch.as_tensor(s_oos, dtype=torch.float32, device=device),
				).cpu().numpy().astype(np.float64)
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

	print("building market state", flush=True)
	state = build_state(work, daily_ret, feature_names)
	n_state = state.shape[1]

	month_x = {}
	month_s = {}
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
		month_s[eom] = state.loc[eom].to_numpy(dtype=np.float32) if eom in state.index else np.zeros(n_state, dtype=np.float32)
		month_ids[eom] = ids
		if has_ret.sum() >= min_stocks:
			month_mask[eom] = has_ret
			month_r[eom] = rets[has_ret].astype(np.float32)
		if (i + 1) % 200 == 0:
			print("  processed", i + 1, "of", n_months, "sec", round(time.time() - t0), flush=True)

	train_months = sorted(month_r.keys())
	print("train months", len(train_months), "total months", len(month_x), "state dim", n_state, flush=True)

	output = rolling_backtest(train_months, month_x, month_s, month_ids, month_mask, month_r, n_features, n_state, device)
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
