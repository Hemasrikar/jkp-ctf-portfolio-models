import numpy as np
import pandas as pd

chars = pd.read_parquet("jkp-data/chars.parquet")
w = pd.read_csv("Beta-Neutral Cross-Sectional MLP/output.csv")

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
out.to_csv("weights.csv", index=False)

print("matched rows:", len(out), "months:", out["eom"].nunique())
print("size MB:", round(len(out.to_csv(index=False).encode()) / 1e6, 1))