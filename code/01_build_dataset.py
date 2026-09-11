"""
01_build_dataset.py — build the analysis table, run integrity checks, save processed data.
"""
import json

import numpy as np
import pandas as pd

from hea_common import OUT, ROOT, build, feature_blocks, save_json

df = build()
fb = feature_blocks(df)
print("analysis rows:", len(df))
print("feature blocks:", {k: len(v) for k, v in fb.items()})

checks = {}
checks["n_rows"] = int(len(df))
checks["n_element_features"] = len(fb["comp"])
checks["s_phase_counts"] = df["s_phase"].value_counts().to_dict()
checks["y_ss_counts"] = df["y_ss"].value_counts().to_dict()
checks["y_lattice_counts"] = df["y_lattice"].value_counts().to_dict()
checks["n_elem_dist"] = df["n_elem"].value_counts().sort_index().to_dict()

# mole fraction integrity
X = df[fb["comp"]].to_numpy()
checks["frac_rowsum_min_max"] = [float(X.sum(1).min()), float(X.sum(1).max())]

# duplication structure
vc = df["elem_set_key"].value_counts()
checks["n_unique_elem_sets"] = int(vc.size)
checks["n_alloys_in_shared_elem_sets"] = int(vc[vc > 1].sum())
checks["pct_alloys_in_shared_elem_sets"] = round(100 * vc[vc > 1].sum() / len(df), 2)
checks["n_exact_duplicate_compositions"] = int(len(df) - df["comp_key"].nunique())
checks["largest_elem_set_group"] = int(vc.iloc[0])
checks["median_group_size_where_shared"] = float(vc[vc > 1].median())

# how many alloys share a *composition* with another (near-duplicate, 1% L1 tolerance)
Xr = np.round(X, 3)
keys = pd.Series([",".join(map(str, r)) for r in Xr])
checks["n_unique_rounded_compositions"] = int(keys.nunique())

# element usage
used = (df[fb["comp"]] > 0).sum().sort_values(ascending=False)
checks["n_elements_ever_used"] = int((used > 0).sum())
checks["top20_elements"] = {c.replace("x_", ""): int(v) for c, v in used.head(20).items()}
checks["rare_elements_lt5"] = {c.replace("x_", ""): int(v) for c, v in used[used < 5].items()}

# label / descriptor integrity
checks["descriptor_missing_pct"] = {
    c: round(100 * float(df[c].isna().mean()), 2) for c in fb["desc"]}
checks["omega_range"] = [float(df["omega"].min()), float(df["omega"].max())]
checks["omega_median"] = float(df["omega"].median())

# consistency between S_Phase==SS and the raw Phase string
ss = df[df["s_phase"] == "SS"]
checks["SS_rows_with_no_lattice_token"] = int((~(ss["has_fcc"] | ss["has_bcc"] | ss["has_hcp"])).sum())
non_ss = df[df["s_phase"] != "SS"]
checks["nonSS_rows_with_lattice_only"] = int(
    ((non_ss["has_fcc"] | non_ss["has_bcc"]) & ~non_ss["has_im"]).sum())

store = df.copy()
store["elem_set"] = store["elem_set"].map(lambda s: "+".join(sorted(s)))
store.to_parquet(ROOT / "data" / "hea_processed.parquet", index=False)
store.drop(columns=["comp_key"]).to_csv(ROOT / "data" / "hea_processed.csv", index=False)
save_json(checks, OUT / "01_integrity.json")

print(json.dumps({k: v for k, v in checks.items() if k not in ("top20_elements", "rare_elements_lt5")},
                 indent=2, ensure_ascii=False, default=str))
print("\ntop elements:", json.dumps(checks["top20_elements"], ensure_ascii=False))
print("\nrare (<5 alloys):", json.dumps(checks["rare_elements_lt5"], ensure_ascii=False))
print("\ndescriptor summary:")
print(df[fb["desc"]].describe().T.round(4).to_string())
