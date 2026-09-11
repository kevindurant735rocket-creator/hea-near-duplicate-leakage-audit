"""
07_exclusion_sweep.py — near-duplicate exclusion-radius sweep (EXP-5).

Motivation
----------
EXP-1 showed that a system-disjoint training set costs the boosted model ~0.13
balanced accuracy. That number is a lower bound, and a critic can still say the
drop is just "less training data". EXP-5 removes that ambiguity.

Protocol
--------
Fix the same stratified 20 % test set T as in EXP-1. For a radius tau:

  excluded(tau)     : training alloys whose L1 distance (mole-fraction space) to
                      the nearest alloy in T is >= tau. Near-duplicates of the
                      test set are physically removed.
  matched(tau)      : a random subsample of the FULL training pool drawn to the
                      same size as excluded(tau). Identical training volume, but
                      the near-duplicates are still inside.

  delta(tau) = balAcc(matched) - balAcc(excluded)

delta(tau) isolates the near-duplicate contribution from the data-volume
contribution. As tau grows, matched(tau) loses only volume while excluded(tau)
loses volume AND redundancy, so the two curves separate by exactly the amount
that redundant test-adjacent alloys were buying.

Usage (staged; the sandbox reaps long-lived children):
    python 07_exclusion_sweep.py run     # the sweep  (~60 s)
    python 07_exclusion_sweep.py merge   # summarise into out/07_exclusion_sweep.json
"""
import gc
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hea_common import OUT, build, feature_blocks, save_json

warnings.filterwarnings("ignore")

TAUS = [0.00, 0.02, 0.05, 0.08, 0.12, 0.20, 0.35]
SPLITS = [0, 1, 2]
TEST_FRAC = 0.20
ROWS_CSV = OUT / "07_exclusion_rows.csv"
RES_JSON = OUT / "07_exclusion_sweep.json"


def make_models():
    return {
        "logreg": Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=3000, C=1.0,
                                                       class_weight="balanced"))]),
        "hgb": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
            min_samples_leaf=5, l2_regularization=1.0, random_state=0),
        "knn": Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance",
                                                      metric="cityblock"))]),
    }


def load():
    df = build()
    fb = feature_blocks(df)
    y = df["y_ss"].to_numpy()
    X = df[fb["both"]].to_numpy(float)
    C = df[fb["comp"]].to_numpy(float)
    return df, y, X, C


def run():
    df, y, X, C = load()
    rows = []

    for seed in SPLITS:
        idx = np.arange(len(y))
        tr_pool, te = train_test_split(idx, test_size=TEST_FRAC, random_state=seed,
                                       stratify=y)
        # distance from every training-pool alloy to the nearest test alloy
        d = cdist(C[tr_pool], C[te], metric="cityblock").min(axis=1)
        rng = np.random.default_rng(1000 + seed)

        for tau in TAUS:
            keep = d >= tau
            tr_excl = tr_pool[keep]
            n = len(tr_excl)
            if n < 60:
                continue
            # size-matched control: same n, drawn from the whole pool
            tr_match = rng.choice(tr_pool, size=n, replace=False)

            for mname in ("logreg", "hgb", "knn"):
                for cond, tr in (("excluded", tr_excl), ("matched", tr_match)):
                    m = make_models()[mname]
                    m.fit(X[tr], y[tr])
                    prob = m.predict_proba(X[te])[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    rows.append({
                        "seed": seed, "tau": tau, "model": mname, "condition": cond,
                        "n_train": int(len(tr)),
                        "n_removed": int(len(tr_pool) - n) if cond == "excluded" else int(len(tr_pool) - n),
                        "bal_acc": float(balanced_accuracy_score(y[te], pred)),
                        "mcc": float(matthews_corrcoef(y[te], pred)),
                    })
                    del m
            gc.collect()
            print(f"  seed={seed} tau={tau:.2f} n_train={n:4d} done", flush=True)

    pd.DataFrame(rows).to_csv(ROWS_CSV, index=False)
    print(f"wrote {ROWS_CSV} ({len(rows)} rows)", flush=True)


def merge():
    rows = pd.read_csv(ROWS_CSV)
    out = {"taus": TAUS, "n_splits": len(SPLITS), "by_model": {}}

    for mname in ("logreg", "hgb", "knn"):
        per_tau = []
        for tau in sorted(rows["tau"].unique()):
            sub = rows[(rows.model == mname) & (np.isclose(rows.tau, tau))]
            e = sub[sub.condition == "excluded"]["bal_acc"].to_numpy()
            m = sub[sub.condition == "matched"]["bal_acc"].to_numpy()
            if len(e) == 0:
                continue
            per_tau.append({
                "tau": float(tau),
                "n_train": int(sub[sub.condition == "excluded"]["n_train"].iloc[0]),
                "bal_acc_excluded": float(e.mean()),
                "bal_acc_excluded_sd": float(e.std(ddof=1)) if len(e) > 1 else 0.0,
                "bal_acc_matched": float(m.mean()),
                "bal_acc_matched_sd": float(m.std(ddof=1)) if len(m) > 1 else 0.0,
                "delta": float(m.mean() - e.mean()),
                "delta_sd": float((m - e).std(ddof=1)) if len(e) > 1 else 0.0,
            })
        out["by_model"][mname] = per_tau

    # headline: pure redundancy effect, integrated over the sweep
    for mname, recs in out["by_model"].items():
        recs = [r for r in recs if r["tau"] >= 0.02]
        out.setdefault("summary", {})[mname] = {
            "mean_delta_ge_0.02": float(np.mean([r["delta"] for r in recs])),
            "max_delta": float(max(r["delta"] for r in recs)),
            "bal_acc_tau0": out["by_model"][mname][0]["bal_acc_excluded"],
            "bal_acc_tau035": out["by_model"][mname][-1]["bal_acc_excluded"],
        }
    save_json(out, RES_JSON)
    for mname, recs in out["by_model"].items():
        print(f"  {mname:7s} " + " ".join(f"{r['tau']:.2f}:{r['bal_acc_excluded']:.3f}"
                                          f"(m{r['bal_acc_matched']:.3f},d{r['delta']:+.3f})"
                                          for r in recs), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run()
    elif mode == "merge":
        merge()
    else:
        raise SystemExit(f"unknown mode {mode!r}")
