"""
00_profile.py — Profile the raw HEA dataset before any modelling.
Source: Zenodo 10.5281/zenodo.6403257 (CC-BY-4.0), file 2022-03-31-HEAs_dataset_v2.xlsx
Run:  python 00_profile.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "heas_v2.xlsx"
OUT = ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(RAW, sheet_name="Sheet1")

meta_cols = ['Alloy', 'No', 'S_Phase', 'Phase', 'a (Å)', 'δ', 'T_m (K)', 'σT_m (K)',
             'ΔH_mix (kJ/mol)', 'σH_mix (kJ/mol)', 'S_id (R)', 'χ', 'Δχ', 'VEC',
             'σVEC', 'Bulk Modulus (GPa)', 'σBulk (GPa)', "Young's modulus E (GPa)",
             "Shear modulus G (GPa)"]
elem_cols = [c for c in df.columns if c not in meta_cols]

report = {}
report["n_rows"] = int(len(df))
report["n_cols"] = int(df.shape[1])
report["n_element_cols"] = len(elem_cols)
report["element_cols"] = elem_cols

# ---- label distributions -------------------------------------------------
report["S_Phase_counts"] = df["S_Phase"].value_counts(dropna=False).to_dict()
report["Phase_counts"] = df["Phase"].value_counts(dropna=False).head(40).to_dict()
report["No_counts"] = df["No"].value_counts(dropna=False).sort_index().to_dict()

# ---- composition sanity --------------------------------------------------
X = df[elem_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
report["fraction_sum_stats"] = X.sum(axis=1).describe().round(4).to_dict()
report["n_nonzero_elements_stats"] = (X > 0).sum(axis=1).describe().round(3).to_dict()

# ---- element-set duplication (the leakage mechanism) ---------------------
elem_sets = X.gt(0).apply(lambda r: frozenset(r.index[r.values]), axis=1)
report["n_unique_element_sets"] = int(elem_sets.nunique())
report["n_alloys"] = int(len(elem_sets))
vc = elem_sets.value_counts()
report["top_element_sets"] = {"+".join(sorted(k)): int(v) for k, v in vc.head(15).items()}
report["pct_alloys_in_multi_alloy_sets"] = round(
    100.0 * vc[vc > 1].sum() / len(elem_sets), 2)

# exact duplicate compositions (rounded to 3 decimals)
comp_key = X.round(3).astype(str).agg("|".join, axis=1)
report["n_exact_duplicate_compositions"] = int(len(comp_key) - comp_key.nunique())

# ---- missingness of descriptors -----------------------------------------
desc = ['δ', 'ΔH_mix (kJ/mol)', 'S_id (R)', 'χ', 'Δχ', 'VEC', 'σVEC', 'T_m (K)']
report["descriptor_missing_pct"] = {
    c: round(100.0 * df[c].isna().mean(), 2) for c in desc}

# ---- label consistency: S_Phase vs Phase ---------------------------------
ss = df[df["S_Phase"] == "SS"]
report["SS_Phase_value_counts"] = ss["Phase"].value_counts(dropna=False).head(20).to_dict()
report["n_SS"] = int(len(ss))

# ---- parse element counts out of the Alloy string (sanity check) ---------
def parse_formula(s):
    return dict((el, float(v)) for el, v in re.findall(r"([A-Z][a-z]?)(\d+\.?\d*)", str(s)))

mismatch = 0
checked = 0
for i, row in df.head(400).iterrows():
    parsed = parse_formula(row["Alloy"])
    if not parsed:
        continue
    checked += 1
    for el, val in parsed.items():
        if el in elem_cols:
            if abs(float(row[el]) - val) > 0.02:
                mismatch += 1
                break
report["formula_parse_checked"] = checked
report["formula_parse_mismatch"] = mismatch

print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
(OUT / "00_profile.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))

print("\n--- head ---")
print(df[['Alloy', 'No', 'S_Phase', 'Phase', 'VEC', 'δ', 'ΔH_mix (kJ/mol)', 'Δχ']].head(8).to_string())
