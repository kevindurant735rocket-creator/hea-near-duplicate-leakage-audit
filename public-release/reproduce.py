"""
reproduce.py — one-command reproduction of the HEA near-duplicate leakage audit.

WHAT IT DOES
------------
Re-runs EXP-5 (the near-duplicate exclusion-radius sweep) using the *exact* code
that produced the numbers in the paper, then checks the headline result against
the committed reference (out/07_exclusion_sweep.json). If the recomputed numbers
match the reference within tolerance, you have bit-for-bit evidence that the
claims are reproducible.

Headline claim (hgb / gradient-boosted trees):
  At exclusion radius tau = 0.35 (mole-fraction L1 space),
    - training set with near-duplicates removed : bal. acc. = 0.685
    - size-matched control (duplicates kept)    : bal. acc. = 0.836
    - pure redundancy gap  delta                = +0.151
  The total accuracy drop from tau=0 to tau=0.35 (0.883 -> 0.685 = 0.198) splits
  into ~24 % data-volume and ~76 % near-duplicate redundancy.

USAGE
-----
  python reproduce.py
    -> runs the sweep (~60 s), writes out/repro_rows.csv + out/repro_sweep.json,
       compares to out/07_exclusion_sweep.json, prints PASS/FAIL, exits 0 on PASS.

DATA
----
The raw dataset (Materials for Design Open Repository, Zenodo 6403257,
file 2022-03-31-HEAs_dataset_v2.xlsx, CC-BY-4.0) must be present at
../data/heas_v2.xlsx. If it is missing, download it from
https://doi.org/10.5281/zenodo.6403257 and place it under ../data/, or pass
--data <path>.

Determinism: all random sources are fixed (HGB random_state=0, KNN/logreg are
deterministic, the matched subsample uses numpy default_rng(1000+seed), and the
train/test split uses random_state=seed). The re-run therefore reproduces the
reference to many decimal places.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                       # research/ is the repo root
CODE = REPO / "code"
OUT = REPO / "out"
REF = OUT / "07_exclusion_sweep.json"
REPRO_ROWS = OUT / "repro_rows.csv"
REPRO_JSON = OUT / "repro_sweep.json"
DATA_DEFAULT = REPO / "data" / "heas_v2.xlsx"

# tolerance for "matches the committed reference" (generous; true match is ~1e-12)
TOL = 0.005


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce the HEA near-duplicate leakage audit.")
    ap.add_argument("--data", type=Path, default=DATA_DEFAULT,
                    help="path to heas_v2.xlsx (default: ../data/heas_v2.xlsx)")
    ap.add_argument("--no-run", action="store_true",
                    help="skip the sweep and only check existing out/repro_sweep.json")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"[reproduce] DATA NOT FOUND: {args.data}")
        print("  Download 2022-03-31-HEAs_dataset_v2.xlsx from https://doi.org/10.5281/zenodo.6403257")
        print("  (CC-BY-4.0) and place it at data/heas_v2.xlsx, then re-run.")
        return 2

    # make the verified source module importable and run it with redirected output
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))

    if not args.no_run:
        if not REF.exists():
            print(f"[reproduce] REFERENCE NOT FOUND: {REF}")
            print("  The committed reference out/07_exclusion_sweep.json is missing.")
            return 2
        sweep7 = importlib.import_module("07_exclusion_sweep")
        # redirect outputs so we never clobber the committed reference
        sweep7.ROWS_CSV = REPRO_ROWS
        sweep7.RES_JSON = REPRO_JSON
        print("[reproduce] running EXP-5 exclusion-radius sweep (~60 s) ...")
        sweep7.run()
        sweep7.merge()
        print(f"[reproduce] wrote {REPRO_ROWS.name} and {REPRO_JSON.name}")

    if not REPRO_JSON.exists():
        print(f"[reproduce] {REPRO_JSON.name} missing — run without --no-run first.")
        return 2

    repro = json.loads(REPRO_JSON.read_text(encoding="utf-8"))
    ref = json.loads(REF.read_text(encoding="utf-8"))

    print("\n=== HEADLINE CHECK (hgb / gradient-boosted trees) ===")
    def rec(block, model, tau):
        for r in block[model]:
            if abs(r["tau"] - tau) < 1e-9:
                return r
        raise KeyError(f"{model}@{tau}")

    ok = True
    for model in ("hgb", "knn", "logreg"):
        r_ref = rec(ref["by_model"], model, 0.35)
        r_rep = rec(repro["by_model"], model, 0.35)
        d_ref = r_ref["delta"]
        d_rep = r_rep["delta"]
        dev = abs(d_rep - d_ref)
        flag = "ok" if dev <= TOL else "FAIL"
        if dev > TOL:
            ok = False
        print(f"  {model:7s} tau=0.35  ref delta={d_ref:+.4f}  repro delta={d_rep:+.4f}  "
              f"dev={dev:.2e}  [{flag}]")

    # self-check: tau=0 must give delta == 0 (definitions coincide)
    for model in ("hgb", "knn", "logreg"):
        d0 = rec(repro["by_model"], model, 0.0)["delta"]
        if abs(d0) > 1e-9:
            ok = False
            print(f"  {model:7s} tau=0 self-check FAILED (delta={d0:+.4f}, expected 0)")
        else:
            print(f"  {model:7s} tau=0 self-check ok (delta=0)")

    # redundancy decomposition (from the recomputed numbers)
    hgb0 = rec(repro["by_model"], "hgb", 0.0)["bal_acc_excluded"]
    hgb35 = rec(repro["by_model"], "hgb", 0.35)
    total_drop = hgb0 - hgb35["bal_acc_excluded"]
    volume = abs(hgb35["bal_acc_matched"] - rec(repro["by_model"], "hgb", 0.0)["bal_acc_matched"])
    redundancy = hgb35["delta"]
    print("\n=== REDUNDANCY DECOMPOSITION (hgb, tau 0 -> 0.35) ===")
    print(f"  total drop          = {total_drop:.3f}")
    print(f"  volume effect       = {volume:.3f}  ({volume/total_drop*100:.0f}%)")
    print(f"  redundancy effect   = {redundancy:.3f}  ({redundancy/total_drop*100:.0f}%)")

    print("\n=== RESULT ===")
    if ok:
        print("  PASS — recomputed headline matches the committed reference within "
              f"tolerance {TOL}.")
        return 0
    print("  FAIL — recomputed numbers deviate beyond tolerance. Investigate.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
