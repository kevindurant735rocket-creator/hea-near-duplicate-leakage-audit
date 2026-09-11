"""
03_leakage_experiment.py — quantify how much random-split evaluation inflates
reported accuracy for HEA phase prediction.

Staged so each run is short (the sandbox reaps long-lived child processes):
    python 03_leakage_experiment.py stage1   # EXP-2 protocol CV, EXP-3 leave-one-element-out
    python 03_leakage_experiment.py stage2   # EXP-1 controlled paired split, EXP-4 neighbours
    python 03_leakage_experiment.py merge    # combine into out/03_leakage.json

EXP-1 (primary evidence) — fixed 20 % stratified test set T:
    TrainA         : every non-test alloy
    TrainA_matched : TrainA subsampled to |TrainB|
    TrainB         : non-test alloys sharing no element set with any alloy in T
  TrainA_matched vs TrainB is test-set-paired and size-matched, so the only
  remaining difference is the presence of near-duplicate alloys.
"""
import gc
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hea_common import OUT, build, feature_blocks, save_json

warnings.filterwarnings("ignore")
SEEDS = list(range(10))
N_REPEAT_CV = 3
N_SIZE_MATCH = 3


def make_models():
    return {
        "logreg": Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=3000, C=1.0,
                                                       class_weight="balanced"))]),
        "hgb": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
            min_samples_leaf=5, l2_regularization=1.0, random_state=0),
    }


def score_all(y_true, y_pred, y_prob):
    rec = {"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
           "mcc": float(matthews_corrcoef(y_true, y_pred)),
           "accuracy": float((y_true == y_pred).mean())}
    rec["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    return rec


def load():
    df = build()
    fb = feature_blocks(df)
    y = df["y_ss"].to_numpy()
    blocks = {"composition": df[fb["comp"]].to_numpy(float),
              "descriptors": df[fb["desc"]].to_numpy(float),
              "composition+descriptors": df[fb["both"]].to_numpy(float)}
    return df, fb, y, blocks


# ============================== STAGE 1 ==================================
def stage1():
    df, fb, y, blocks = load()
    groups = df["elem_set_key"].to_numpy()
    out = {"exp2_cv_protocols": [], "exp3_loeo": []}

    for block, X in blocks.items():
        for mname in ("logreg", "hgb"):
            for proto in ("random_stratified", "element_set_grouped"):
                scores = {"balanced_accuracy": [], "mcc": [], "roc_auc": []}
                for rep in range(N_REPEAT_CV):
                    if proto == "random_stratified":
                        it = StratifiedKFold(5, shuffle=True, random_state=100 + rep).split(X, y)
                    else:
                        it = StratifiedGroupKFold(5, shuffle=True, random_state=100 + rep).split(X, y, groups)
                    for tr, te in it:
                        m = make_models()[mname]
                        m.fit(X[tr], y[tr])
                        prob = m.predict_proba(X[te])[:, 1]
                        rec = score_all(y[te], (prob >= 0.5).astype(int), prob)
                        for k in scores:
                            scores[k].append(rec[k])
                rec = {"block": block, "model": mname, "protocol": proto,
                       "bal_acc_mean": float(np.nanmean(scores["balanced_accuracy"])),
                       "bal_acc_std": float(np.nanstd(scores["balanced_accuracy"])),
                       "mcc_mean": float(np.nanmean(scores["mcc"])),
                       "auc_mean": float(np.nanmean(scores["roc_auc"])),
                       "n_fits": len(scores["balanced_accuracy"])}
                out["exp2_cv_protocols"].append(rec)
                print(f"  EXP2 {block:24s} {mname:7s} {proto:22s} balAcc={rec['bal_acc_mean']:.3f}", flush=True)

    used = (df[fb["comp"]] > 0).sum()
    used.index = [c.replace("x_", "") for c in used.index]
    elems = [e for e in used.sort_values(ascending=False).index if used[e] >= 45]
    X_all = blocks["composition+descriptors"]
    for e in elems:
        in_e = df[f"x_{e}"].to_numpy() > 0
        tr, te = np.where(~in_e)[0], np.where(in_e)[0]
        if y[te].sum() == 0 or y[tr].sum() == 0:
            continue
        for mname in ("logreg", "hgb"):
            m = make_models()[mname]
            m.fit(X_all[tr], y[tr])
            prob = m.predict_proba(X_all[te])[:, 1]
            rec = score_all(y[te], (prob >= 0.5).astype(int), prob)
            rec.update({"element": e, "model": mname, "n_train": int(len(tr)),
                        "n_test": int(len(te)), "test_ss_rate": float(y[te].mean())})
            out["exp3_loeo"].append(rec)
        print(f"  EXP3 {e:4s} done", flush=True)

    pd.DataFrame(out["exp2_cv_protocols"]).to_csv(OUT / "03_exp2_protocols.csv", index=False)
    pd.DataFrame(out["exp3_loeo"]).to_csv(OUT / "03_exp3_loeo.csv", index=False)
    (OUT / "03_stage1.json").write_text(json.dumps(out, default=str))
    print("stage1 complete", flush=True)


# ============================== STAGE 2 ==================================
def stage2():
    df, fb, y, blocks = load()
    groups = df["elem_set_key"].to_numpy()
    X_comp = blocks["composition"]
    rows, neigh, nb = [], {"A": [], "B": []}, []

    for seed in SEEDS:
        print(f"  EXP1 seed {seed} ...", flush=True)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        tr_pool, te_idx = next(iter(skf.split(np.zeros(len(y)), y)))
        test_keys = set(groups[te_idx])
        keep_B = np.array([i for i in tr_pool if groups[i] not in test_keys])
        n_B = len(keep_B)
        rng = np.random.default_rng(seed)

        for block, X in blocks.items():
            for mname in ("logreg", "hgb"):
                for vname, tr in (("TrainA_all", tr_pool), ("TrainB_disjoint", keep_B)):
                    m = make_models()[mname]
                    m.fit(X[tr], y[tr])
                    prob = m.predict_proba(X[te_idx])[:, 1]
                    rec = score_all(y[te_idx], (prob >= 0.5).astype(int), prob)
                    rec.update({"seed": seed, "block": block, "model": mname,
                                "variant": vname, "n_train": int(len(tr))})
                    rows.append(rec)
                for rep in range(N_SIZE_MATCH):
                    sub = rng.choice(tr_pool, size=n_B, replace=False)
                    m = make_models()[mname]
                    m.fit(X[sub], y[sub])
                    prob = m.predict_proba(X[te_idx])[:, 1]
                    rec = score_all(y[te_idx], (prob >= 0.5).astype(int), prob)
                    rec.update({"seed": seed, "block": block, "model": mname,
                                "variant": "TrainA_matched", "n_train": int(n_B), "rep": rep})
                    rows.append(rec)
            gc.collect()

        dA = cdist(X_comp[te_idx], X_comp[tr_pool], metric="cityblock").min(axis=1)
        dB = cdist(X_comp[te_idx], X_comp[keep_B], metric="cityblock").min(axis=1)
        neigh["A"].append(dA)
        neigh["B"].append(dB)
        nb.append({"seed": seed, "A_median": float(np.median(dA)), "B_median": float(np.median(dB)),
                   "A_p05": float(np.percentile(dA, 5)), "B_p05": float(np.percentile(dB, 5)),
                   "A_frac_lt_0.05": float((dA < 0.05).mean()),
                   "B_frac_lt_0.05": float((dB < 0.05).mean()),
                   "A_frac_lt_0.10": float((dA < 0.10).mean()),
                   "B_frac_lt_0.10": float((dB < 0.10).mean())})

    e1 = pd.DataFrame(rows)
    e1.to_csv(OUT / "03_exp1_controlled_raw.csv", index=False)
    pairs = []
    for (block, mname, seed), g in e1.groupby(["block", "model", "seed"]):
        pairs.append({"block": block, "model": mname, "seed": seed,
                      "matched": g[g["variant"] == "TrainA_matched"]["balanced_accuracy"].mean(),
                      "disjoint": g[g["variant"] == "TrainB_disjoint"]["balanced_accuracy"].iloc[0]})
    pp = pd.DataFrame(pairs)
    pp["delta"] = pp["matched"] - pp["disjoint"]
    pp.to_csv(OUT / "03_exp1_paired_deltas.csv", index=False)
    np.save(OUT / "03_neighbour_dist_A.npy", np.concatenate(neigh["A"]))
    np.save(OUT / "03_neighbour_dist_B.npy", np.concatenate(neigh["B"]))
    (OUT / "03_stage2.json").write_text(json.dumps(
        {"exp1_controlled": rows, "exp4_neighbour": nb}, default=str))
    print("stage2 complete", flush=True)


# ============================== MERGE ====================================
def merge():
    df, fb, y, blocks = load()
    s1 = json.loads((OUT / "03_stage1.json").read_text())
    s2 = json.loads((OUT / "03_stage2.json").read_text())
    e1 = pd.read_csv(OUT / "03_exp1_controlled_raw.csv")
    e2 = pd.read_csv(OUT / "03_exp2_protocols.csv")
    e3 = pd.read_csv(OUT / "03_exp3_loeo.csv")
    pp = pd.read_csv(OUT / "03_exp1_paired_deltas.csv")
    summary = {"exp2_cv_protocols": s1["exp2_cv_protocols"], "exp3_loeo_raw": s1["exp3_loeo"]}

    exp1_summary = {}
    for (block, mname), g in pp.groupby(["block", "model"]):
        d = g["delta"].to_numpy()
        try:
            _, p = wilcoxon(g["matched"], g["disjoint"])
        except ValueError:
            p = float("nan")
        exp1_summary[f"{block} | {mname}"] = {
            "bal_acc_matched_mean": float(g["matched"].mean()),
            "bal_acc_disjoint_mean": float(g["disjoint"].mean()),
            "delta_mean": float(d.mean()), "delta_std": float(d.std(ddof=1)),
            "delta_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "wilcoxon_p": float(p), "n_pairs": int(len(g))}
    summary["exp1_size_matched_contrast"] = exp1_summary

    nA = e1[e1["variant"] == "TrainA_all"].groupby("seed")["n_train"].first()
    nB = e1[e1["variant"] == "TrainB_disjoint"].groupby("seed")["n_train"].first()
    summary["exp1_train_size"] = {
        "TrainA_mean": float(nA.mean()), "TrainB_mean": float(nB.mean()),
        "retained_fraction_mean": float((nB / nA).mean()),
        "retained_fraction_min": float((nB / nA).min()),
        "retained_fraction_max": float((nB / nA).max())}

    gaps = []
    for block in blocks:
        for mname in ("logreg", "hgb"):
            r = e2[(e2["block"] == block) & (e2["model"] == mname)].set_index("protocol")
            gaps.append({"block": block, "model": mname,
                         "random": float(r.loc["random_stratified", "bal_acc_mean"]),
                         "grouped": float(r.loc["element_set_grouped", "bal_acc_mean"]),
                         "gap": float(r.loc["random_stratified", "bal_acc_mean"]
                                      - r.loc["element_set_grouped", "bal_acc_mean"]),
                         "random_std": float(r.loc["random_stratified", "bal_acc_std"]),
                         "grouped_std": float(r.loc["element_set_grouped", "bal_acc_std"])})
    summary["exp2_protocol_gap"] = gaps

    used = (df[fb["comp"]] > 0).sum()
    used.index = [c.replace("x_", "") for c in used.index]
    elems = [e for e in used.sort_values(ascending=False).index if used[e] >= 45]
    summary["exp3_loeo"] = {
        "elements_tested": elems,
        "mean_bal_acc": float(e3["balanced_accuracy"].mean()),
        "std_bal_acc": float(e3["balanced_accuracy"].std()),
        "min_bal_acc": float(e3["balanced_accuracy"].min()),
        "max_bal_acc": float(e3["balanced_accuracy"].max()),
        "mean_auc": float(e3["roc_auc"].mean()),
        "per_element_hgb": {r["element"]: round(r["balanced_accuracy"], 3)
                            for _, r in e3[e3["model"] == "hgb"].iterrows()},
        "per_element_logreg": {r["element"]: round(r["balanced_accuracy"], 3)
                               for _, r in e3[e3["model"] == "logreg"].iterrows()}}

    dA = np.load(OUT / "03_neighbour_dist_A.npy")
    dB = np.load(OUT / "03_neighbour_dist_B.npy")
    summary["exp4_neighbour_distance"] = {
        "n_test_instances": int(len(dA)),
        "A_median": float(np.median(dA)), "B_median": float(np.median(dB)),
        "A_mean": float(dA.mean()), "B_mean": float(dB.mean()),
        "A_frac_within_0.05": float((dA < 0.05).mean()),
        "B_frac_within_0.05": float((dB < 0.05).mean()),
        "A_frac_within_0.10": float((dA < 0.10).mean()),
        "B_frac_within_0.10": float((dB < 0.10).mean())}
    summary["config"] = {"seeds": SEEDS, "n_repeat_cv": N_REPEAT_CV,
                         "n_size_match": N_SIZE_MATCH, "n_alloys": int(len(df)),
                         "blocks": list(blocks), "models": list(make_models())}
    save_json(summary, OUT / "03_leakage.json")

    print("=== EXP-1 size-matched contrast ===")
    for k, v in exp1_summary.items():
        print(f"  {k:36s} matched={v['bal_acc_matched_mean']:.3f} "
              f"disjoint={v['bal_acc_disjoint_mean']:.3f} d={v['delta_mean']:+.3f} p={v['wilcoxon_p']:.2e}")
    print("=== EXP-2 gaps ===")
    print(pd.DataFrame(gaps).round(3).to_string(index=False))
    print("=== EXP-3 ===")
    print(json.dumps({k: v for k, v in summary["exp3_loeo"].items() if k != "elements_tested"}, indent=2))
    print("=== EXP-4 ===")
    print(json.dumps(summary["exp4_neighbour_distance"], indent=2))
    print("merge complete", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("stage1", "all"):
        stage1()
    if stage in ("stage2", "all"):
        stage2()
    if stage in ("merge", "all"):
        merge()
