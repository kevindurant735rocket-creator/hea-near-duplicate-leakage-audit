"""
02_criteria_audit.py — how well do the textbook HEA phase-formation criteria actually
work on 1,103 experimentally reported alloys?

Criteria evaluated (as binary predictors of solid-solution formation):
  delta  <= 6.6 %                       (Zhang et al. 2008)
  -15 <= dH_mix <= 5 kJ/mol             (Zhang/Guo)
  omega  >= 1.1                         (Yang & Zhang 2012)
  3 <= VEC <= 8                         (Guo & Liu 2011)
  delta_chi <= 0.3                      (common electronegativity-difference rule)
plus the classical conjunctions of these rules.

Also: honest "best single descriptor" ceiling via 5-fold CV threshold selection.
Also: the VEC >= 8 -> FCC / VEC <= 6.87 -> BCC lattice rule.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             matthews_corrcoef, roc_auc_score)

from hea_common import OUT, ROOT, build, save_json

RNG = np.random.default_rng(20260911)
N_BOOT = 2000

df = build()
y = df["y_ss"].to_numpy()


def metrics(y_true, y_pred, score=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    out = {
        "n": int(len(y_true)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float((tp + tn) / len(y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sens), "specificity": float(spec),
        "precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "predicted_positive_rate": float(y_pred.mean()),
    }
    if score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
    return out


def boot_ci(y_true, y_pred, fn, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        try:
            vals.append(fn(y_true[idx], y_pred[idx]))
        except ValueError:
            continue
    vals = np.asarray(vals, dtype=float)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def add_ci(rec, y_true, y_pred):
    rec["balanced_accuracy_ci95"] = boot_ci(
        y_true, y_pred, balanced_accuracy_score, seed=11)
    rec["mcc_ci95"] = boot_ci(y_true, y_pred, matthews_corrcoef, seed=12)
    return rec


d = df["delta"].to_numpy()
dh = df["dH_mix"].to_numpy()
om = df["omega"].to_numpy()
vec = df["VEC"].to_numpy()
dchi = df["delta_chi"].to_numpy()

preds = {}
preds["majority_class_baseline"] = np.zeros_like(y)
preds["delta <= 6.6%"] = (d <= 0.066).astype(int)
preds["-15 <= dH_mix <= 5"] = ((dh >= -15) & (dh <= 5)).astype(int)
preds["omega >= 1.1"] = (np.nan_to_num(om, nan=-1) >= 1.1).astype(int)
preds["3 <= VEC <= 8"] = ((vec >= 3) & (vec <= 8)).astype(int)
preds["delta_chi <= 0.3"] = (dchi <= 0.3).astype(int)
preds["delta <= 6.6 AND -15<=dH<=5"] = (
    (d <= 0.066) & (dh >= -15) & (dh <= 5)).astype(int)
preds["classic 3-rule conjunction"] = (
    (d <= 0.066) & (dh >= -15) & (dh <= 5) & (np.nan_to_num(om, nan=-1) >= 1.1)).astype(int)
preds["classic 4-rule (+VEC 3-8)"] = (
    (d <= 0.066) & (dh >= -15) & (dh <= 5) & (np.nan_to_num(om, nan=-1) >= 1.1)
    & (vec >= 3) & (vec <= 8)).astype(int)

rows = []
for name, p in preds.items():
    rec = metrics(y, p)
    rec = add_ci(rec, y, p)
    rec["rule"] = name
    rec["n_predicted_ss"] = int(p.sum())
    rows.append(rec)

# ---- univariate ROC-AUC of each descriptor (best direction) --------------
uni = {}
for col, sign in [("delta", -1), ("dH_mix", -1), ("dH_mix", 1), ("omega", 1),
                  ("VEC", -1), ("VEC", 1), ("sigma_VEC", -1), ("delta_chi", -1),
                  ("chi", -1), ("Tm", 1), ("S_id", 1), ("n_elem", 1)]:
    v = sign * df[col].to_numpy(dtype=float)
    m = ~np.isnan(v)
    if m.sum() < 50:
        continue
    auc = roc_auc_score(y[m], v[m])
    key = f"{col} ({'higher->SS' if sign > 0 else 'lower->SS'})"
    uni[key] = {"auc": float(auc), "auc_dist_from_chance": float(abs(auc - 0.5))}

# ---- honest ceiling: threshold tuned by 5-fold CV ------------------------
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cv_rows = []
cand_specs = [("delta", -1), ("dH_mix", 1), ("omega", 1), ("VEC", -1),
              ("delta_chi", -1), ("sigma_VEC", -1), ("S_id", 1), ("Tm", 1)]
for col, sign in cand_specs:
    v = df[col].to_numpy(dtype=float) * sign
    v = np.nan_to_num(v, nan=float(np.nanmedian(v)))
    fold_scores = []
    for tr, te in skf.split(v.reshape(-1, 1), y):
        grid = np.quantile(v[tr], np.linspace(0.02, 0.98, 200))  # candidate cut from TRAIN only
        best, best_t = -1.0, float(grid[0])
        for t in grid:
            s = balanced_accuracy_score(y[tr], (v[tr] >= t).astype(int))
            if s > best:
                best, best_t = s, float(t)
        fold_scores.append(balanced_accuracy_score(y[te], (v[te] >= best_t).astype(int)))
    cv_rows.append({
        "descriptor": col,
        "direction": "higher->SS" if sign > 0 else "lower->SS",
        "cv_balanced_accuracy_mean": float(np.mean(fold_scores)),
        "cv_balanced_accuracy_std": float(np.std(fold_scores)),
    })

# ---- VEC lattice rule ----------------------------------------------------
lat = df[df["y_lattice"].notna()].copy()
lv = lat["VEC"].to_numpy()
ly = (lat["y_lattice"] == "FCC").to_numpy().astype(int)
vec_pred = np.where(lv >= 8, 1, np.where(lv <= 6.87, 0, -1))
covered = vec_pred >= 0
vec_rule = {
    "n_total": int(len(ly)),
    "n_fcc": int(ly.sum()), "n_bcc": int((1 - ly).sum()),
    "coverage": float(covered.mean()),
    "n_ambiguous": int((~covered).sum()),
    "accuracy_on_covered": float((vec_pred[covered] == ly[covered]).mean()) if covered.any() else None,
    "balanced_accuracy_on_covered": float(balanced_accuracy_score(ly[covered], vec_pred[covered])) if covered.any() else None,
    "accuracy_all_treating_ambiguous_as_wrong": float((vec_pred == ly).mean()),
    "VEC_stats_fcc": {"min": float(lv[ly == 1].min()), "median": float(np.median(lv[ly == 1])),
                      "max": float(lv[ly == 1].max()),
                      "n_below_8": int((lv[ly == 1] < 8).sum())},
    "VEC_stats_bcc": {"min": float(lv[ly == 0].min()), "median": float(np.median(lv[ly == 0])),
                      "max": float(lv[ly == 0].max()),
                      "n_above_6_87": int((lv[ly == 0] > 6.87).sum())},
}

# ---- where does the classical rule fail? ---------------------------------
fail = df[(preds["classic 4-rule (+VEC 3-8)"] == 1) & (y == 0)]
miss = df[(preds["classic 4-rule (+VEC 3-8)"] == 0) & (y == 1)]
failure_notes = {
    "n_predicted_SS_but_not_SS": int(len(fail)),
    "n_SS_missed": int(len(miss)),
    "top_element_sets_false_positive": {
        k: int(v) for k, v in fail["elem_set_key"].value_counts().head(10).items()},
    "top_element_sets_false_negative": {
        k: int(v) for k, v in miss["elem_set_key"].value_counts().head(10).items()},
    "false_positive_S_Phase_mix": fail["s_phase"].value_counts().to_dict(),
    "false_negative_n_elem_median": float(miss["n_elem"].median()) if len(miss) else None,
}

res = pd.DataFrame(rows).set_index("rule")
res.to_csv(OUT / "02_criteria_table.csv")

summary = {
    "n_alloys": int(len(df)), "n_ss": int(y.sum()), "base_rate_ss": float(y.mean()),
    "criteria": rows,
    "univariate_auc": dict(sorted(uni.items(), key=lambda kv: -kv[1]["auc"])),
    "cv_tuned_single_descriptor": cv_rows,
    "vec_lattice_rule": vec_rule,
    "failure_analysis": failure_notes,
}
save_json(summary, OUT / "02_criteria.json")

pd.set_option("display.width", 200)
print(res[["n", "tp", "tn", "fp", "fn", "accuracy", "balanced_accuracy",
           "sensitivity", "specificity", "mcc", "predicted_positive_rate"]].round(3).to_string())
print("\n--- univariate AUC (top 8) ---")
for k, v in list(summary["univariate_auc"].items())[:8]:
    print(f"  {k:34s} AUC={v['auc']:.3f}")
print("\n--- CV-tuned single descriptor (honest ceiling) ---")
print(pd.DataFrame(cv_rows).round(3).to_string(index=False))
print("\n--- VEC lattice rule ---")
print(json.dumps(vec_rule, indent=2))
print("\n--- failure analysis ---")
print(json.dumps(failure_notes, indent=2, ensure_ascii=False))
