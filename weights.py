"""Filter output.csv to the CTF test rows, write weights.csv, and print the Sharpe report.

Run from the main folder (jkp-ctf-portfolio-models/) after copying a model's output.csv here:

    python weights.py
    python weights.py --model "Coupled Factor Portfolio/coupled-factor-portfolio.py"
    python weights.py --model coupled-factor-portfolio

Outputs:
    weights.csv                                   submission file, always written here
    weights_history/weights_<date>_<time>_<model>.csv   timestamped copy of the same file

The <model> label is the .py filename of the model that generated output.csv, without
the extension. Pass it with --model (a path or just the name; only the stem is used).
If --model is omitted, the script looks for a subfolder whose output.csv is byte-identical
to ./output.csv and uses that folder's model script name. If no match is found the copy
is labelled "unknown-model", so pass --model explicitly when output.csv was edited or
came from somewhere else.
"""

import argparse
import hashlib
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

target_dir = ""

parser = argparse.ArgumentParser(description="filter output.csv to test rows, write weights.csv, print the Sharpe report")
parser.add_argument("--model", default=None,
	help="model script (e.g. 'Coupled Factor Portfolio/coupled-factor-portfolio.py' or just 'coupled-factor-portfolio') "
	     "used to label the timestamped copy; auto-detected from output.csv if omitted")
args = parser.parse_args()

chars = pd.read_parquet("jkp-data/chars.parquet")
w = pd.read_csv(f"output.csv")
weights_file = f"weights.csv"


def _file_hash(path):
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


def _model_script_in(folder):
	"""Pick the .py in a model folder that actually writes output.csv."""
	cands = sorted(p for p in Path(folder).glob("*.py"))
	writers = []
	for p in cands:
		try:
			if "output.csv" in p.read_text(encoding="utf-8", errors="ignore"):
				writers.append(p)
		except OSError:
			pass
	pick = writers or cands
	return pick[0].stem if pick else None


def detect_model_name(explicit=None, output_path="output.csv"):
	"""Name of the model .py that produced output.csv (stem only, no extension)."""
	if explicit:
		return Path(explicit).stem
	here = Path(".")
	target_size = os.path.getsize(output_path)
	target_hash = None
	for cand in sorted(here.glob("*/output.csv")):
		if cand.resolve() == Path(output_path).resolve():
			continue
		if os.path.getsize(cand) != target_size:
			continue
		target_hash = target_hash or _file_hash(output_path)
		if _file_hash(cand) == target_hash:
			name = _model_script_in(cand.parent)
			if name:
				return name
	return "unknown-model"


model_name = detect_model_name(args.model)
run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("weights_history", exist_ok=True)
weights_copy = os.path.join("weights_history", f"weights_{run_stamp}_{model_name}.csv")
print("model:", model_name, "| timestamped copy:", weights_copy)

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
out.to_csv(weights_file, index=False)
out.to_csv(weights_copy, index=False)
print("wrote", weights_file, "and", weights_copy)

print("matched rows:", len(out), "months:", out["eom"].nunique())
print("size MB:", round(len(out.to_csv(index=False).encode()) / 1e6, 1))


def max_drawdown(returns):
	cum = np.cumprod(1.0 + returns)
	peak = np.maximum.accumulate(cum)
	dd = cum / peak - 1.0
	return float(dd.min())


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

	mdd = max_drawdown(vals)
	mdd_scaled = max_drawdown(sc)

	print()
	print(label)
	print("months scored:", len(series), "range", series.index.min(), "->", series.index.max())
	print("annualised return:", round(float(series.mean()) * 12.0, 4))
	print("annualised volatility:", round(float(sd) * ann, 4))
	print("sharpe unscaled:", round(sharpe, 3))
	print("max drawdown unscaled:", round(mdd, 4))
	print("sharpe scaled to", int(vol_target_annual * 100), "percent vol:", round(sharpe_scaled, 3))
	print("max drawdown scaled:", round(mdd_scaled, 4))


sharpe_report("submitted weights, test rows only", out, chars)
sharpe_report("all months in the weights file", w, chars)
