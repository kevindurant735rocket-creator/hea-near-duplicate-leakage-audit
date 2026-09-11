"""
hea_common.py — shared loading / featurisation utilities.

Data source (unmodified, downloaded once):
  Materials for Design Open Repository. High Entropy Alloys
  Zenodo record 6403257, file 2022-03-31-HEAs_dataset_v2.xlsx
  DOI 10.5281/zenodo.6403257, licence CC-BY-4.0

Raw file lives at data/heas_v2.xlsx and is never edited in place.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "heas_v2.xlsx"
OUT = ROOT / "out"
FIG = ROOT / "figures"
for _p in (OUT, FIG):
    if not _p.exists():
        _p.mkdir(parents=True)

R_GAS = 8.314462618  # J mol^-1 K^-1

DESCRIPTOR_COLS = [
    "delta", "dH_mix", "S_id", "omega", "VEC", "sigma_VEC", "delta_chi", "Tm", "chi",
]

_META = ['Alloy', 'No', 'S_Phase', 'Phase', 'a (Å)', 'δ', 'T_m (K)', 'σT_m (K)',
         'ΔH_mix (kJ/mol)', 'σH_mix (kJ/mol)', 'S_id (R)', 'χ', 'Δχ', 'VEC',
         'σVEC', 'Bulk Modulus (GPa)', 'σBulk (GPa)', "Young's modulus E (GPa)",
         "Shear modulus G (GPa)"]


def load_raw() -> pd.DataFrame:
    return pd.read_excel(RAW, sheet_name="Sheet1")


def element_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META]


def parse_formula(s: str) -> dict[str, float]:
    """Parse a formula string such as 'AlCoCrFeNi' or 'Ag5Cd8' into {el: count}."""
    out: dict[str, float] = {}
    for el, val in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", str(s)):
        if not el:
            continue
        v = float(val) if val not in ("", ".") else 1.0
        out[el] = out.get(el, 0.0) + v
    return out


def build() -> pd.DataFrame:
    """Return the cleaned analysis table with mole fractions, descriptors and targets."""
    df = load_raw()
    ec = element_columns(df)

    counts = df[ec].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = counts.sum(axis=1)
    keep = total > 0
    df, counts, total = df[keep].copy(), counts[keep], total[keep]
    frac = counts.div(total, axis=0)
    frac.columns = [f"x_{c}" for c in ec]

    out = pd.DataFrame(index=df.index)
    out["alloy"] = df["Alloy"].astype(str)
    out["n_elem"] = counts.gt(0).sum(axis=1).astype(int)
    out["n_elem_reported"] = pd.to_numeric(df["No"], errors="coerce")

    # ---- descriptors ------------------------------------------------------
    out["delta"] = pd.to_numeric(df["δ"], errors="coerce")
    out["dH_mix"] = pd.to_numeric(df["ΔH_mix (kJ/mol)"], errors="coerce")
    out["S_id"] = pd.to_numeric(df["S_id (R)"], errors="coerce")          # units of R
    out["VEC"] = pd.to_numeric(df["VEC"], errors="coerce")
    out["sigma_VEC"] = pd.to_numeric(df["σVEC"], errors="coerce")
    out["delta_chi"] = pd.to_numeric(df["Δχ"], errors="coerce")
    out["chi"] = pd.to_numeric(df["χ"], errors="coerce")
    out["Tm"] = pd.to_numeric(df["T_m (K)"], errors="coerce")
    # Omega = Tm * dS_mix / |dH_mix|   (Yang & Zhang 2012)
    dS = out["S_id"] * R_GAS                      # J mol^-1 K^-1
    out["omega"] = out["Tm"] * dS / (out["dH_mix"].abs() * 1000.0)
    out["omega"] = out["omega"].replace([np.inf, -np.inf], np.nan)

    # ---- labels -----------------------------------------------------------
    out["s_phase"] = df["S_Phase"].astype("string").fillna("")
    out["phase_raw"] = df["Phase"].astype("string").fillna("")

    ph = out["phase_raw"].fillna("").str.upper()
    out["has_fcc"] = ph.str.contains("FCC")
    out["has_bcc"] = ph.str.contains("BCC")
    out["has_im"] = ph.str.contains("IM|LAVES|SIGMA|Ïƒ|B2|L12|L10|C16|D85|D02|A5|CO2TI|CO7MO6")
    out["has_hcp"] = ph.str.contains("HCP")

    out["y_ss"] = np.where(out["s_phase"].eq("SS"), 1, 0)
    # clean single-lattice subset for the FCC-vs-BCC task
    out["y_lattice"] = np.where(ph.eq("FCC"), "FCC", np.where(ph.eq("BCC"), "BCC", None))

    # ---- grouping keys ----------------------------------------------------
    out["elem_set"] = [frozenset(c[2:] for c, v in zip(frac.columns, row) if v > 0)
                       for row in frac.to_numpy()]
    out["elem_set_key"] = out["elem_set"].map(lambda s: "+".join(sorted(s)))
    out["comp_key"] = frac.round(4).astype(str).agg("|".join, axis=1)

    res = pd.concat([out, frac], axis=1)
    res = res[res["s_phase"].ne("")].reset_index(drop=True)
    return res


def feature_blocks(df: pd.DataFrame) -> dict[str, list[str]]:
    xcols = [c for c in df.columns if c.startswith("x_")]
    return {"comp": xcols, "desc": DESCRIPTOR_COLS, "both": xcols + DESCRIPTOR_COLS}


def save_json(obj, path: Path) -> None:
    import json
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
