import numpy as np
import pandas as pd

target_dir = "StockMixer"

chars = pd.read_parquet("jkp-data/chars.parquet")
w = pd.read_csv(f"{target_dir}/output.csv")

flag = chars["ctff_test"]
print("ctff_test dtype:", flag.dtype)
print("ctff_test values:", flag.value_counts(dropna=False).to_dict())

if pd.api.types.is_bool_dtype(flag):
	mask = flag.fillna(False).to_numpy(dtype=bool)
elif pd.api.types.is_numeric_dtype(flag):
	mask = flag.fillna(0).to_numpy() > 0.5
else:
	mask = flag.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"]).to_numpy()

test = chars.loc[mask, ["id", "eom"]].copy()
test["eom"] = (pd.to_datetime(test["eom"]) + pd.offsets.MonthEnd(0)).dt.strftime("%Y-%m-%d")
test["id"] = test["id"].astype("int64")
w["id"] = w["id"].astype("int64")

print("test rows:", len(test), "test months:", test["eom"].nunique())
print("test eom range:", test["eom"].min(), "->", test["eom"].max())
print("weights eom range:", w["eom"].min(), "->", w["eom"].max())

out = test.merge(w, on=["id", "eom"], how="inner")
out["w"] = out["w"].round(8)
out.to_csv(f"{target_dir}/weights.csv", index=False)

print("matched rows:", len(out), "months:", out["eom"].nunique())
print("size MB:", round(len(out.to_csv(index=False).encode()) / 1e6, 1))


def sharpe_report(label, weights, chars, vol_target_annual=0.10, lookback=36, max_lev=10.0):
	ret = chars[["id", "eom", "ret_exc_lead1m"]].copy()
	ret["eom"] = (pd.to_datetime(ret["eom"]) + pd.offsets.MonthEnd(0)).dt.strftime("%Y-%m-%d")
	ret["id"] = ret["id"].astype("int64")
	ret = ret.rename(columns={"ret_exc_lead1m": "r"}).dropna(subset=["r"])

	j = weights.merge(ret, on=["id", "eom"], how="inner")
	if len(j) == 0:
		print(label, "no overlap with observed returns")
		return
	j["c"] = j["w"] * j["r"]
	series = j.groupby("eom")["c"].sum().sort_index()
	if len(series) < 12:
		print(label, "too few months to score:", len(series))
		return

	ann = np.sqrt(12.0)
	sd = series.std(ddof=1)
	sharpe = float(series.mean() / sd * ann) if sd > 1e-12 else float("nan")

	# ex ante volatility target, leverage from a trailing estimate lagged one month
	target_monthly = vol_target_annual / ann
	vals = series.to_numpy()
	scaled = np.zeros_like(vals)
	hist = []
	for i, v in enumerate(vals):
		if len(hist) >= lookback:
			trailing = float(np.std(hist[-lookback:], ddof=1))
			lev = float(np.clip(target_monthly / (trailing + 1e-8), 0.0, max_lev))
		else:
			lev = 1.0
		scaled[i] = lev * v
		hist.append(v)
	sc = scaled[lookback:] if len(scaled) > lookback else scaled
	sd_sc = np.std(sc, ddof=1)
	sharpe_scaled = float(np.mean(sc) / sd_sc * ann) if sd_sc > 1e-12 else float("nan")

	print()
	print(label)
	print("  months scored:", len(series), "range", series.index.min(), "->", series.index.max())
	print("  annualised return:", round(float(series.mean()) * 12.0, 4))
	print("  annualised volatility:", round(float(sd) * ann, 4))
	print("  sharpe unscaled:", round(sharpe, 3))
	print("  sharpe scaled to", int(vol_target_annual * 100), "percent vol:", round(sharpe_scaled, 3))


# the test rows are what the competition scores, the full file is context only
sharpe_report("submitted weights, test rows only", out, chars)
sharpe_report("all months in the weights file", w, chars)
