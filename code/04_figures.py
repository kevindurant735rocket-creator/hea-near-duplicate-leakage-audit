"""
04_figures.py — publication-style figures from the saved result files.
Run after 01/02/03 have produced their outputs.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from hea_common import FIG, OUT, ROOT, build, feature_blocks

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "font.family": "DejaVu Sans",
})

CLS_COLORS = {"SS": "#1f77b4", "SS+IM": "#ff7f0e", "IM": "#d62728", "AM": "#2ca02c"}
ACCENT, ACCENT2, GREY = "#0b6e99", "#c1440e", "#7a7a7a"

df = build()
fb = feature_blocks(df)
prof = json.loads((OUT / "01_integrity.json").read_text())
crit = json.loads((OUT / "02_criteria.json").read_text())
leak = json.loads((OUT / "03_leakage.json").read_text())
crit_tab = pd.read_csv(OUT / "02_criteria_table.csv")
e2 = pd.read_csv(OUT / "03_exp2_protocols.csv")
e3 = pd.read_csv(OUT / "03_exp3_loeo.csv")
paired = pd.read_csv(OUT / "03_exp1_paired_deltas.csv")


def save(fig, name):
    p = FIG / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.name)


# ---------------------------------------------------------------- FIG 1 ---
fig, ax = plt.subplots(2, 2, figsize=(9.2, 6.2))
order = ["SS", "SS+IM", "IM", "AM"]
cnt = df["s_phase"].value_counts().reindex(order)
ax[0, 0].bar(cnt.index, cnt.values, color=[CLS_COLORS[c] for c in cnt.index])
for i, v in enumerate(cnt.values):
    ax[0, 0].text(i, v + 6, str(v), ha="center", fontsize=8)
ax[0, 0].set_title("(a) Phase-class composition of the dataset")
ax[0, 0].set_ylabel("number of alloys")
ax[0, 0].set_ylim(0, cnt.max() * 1.18)

gs = df["elem_set_key"].value_counts()
bins = np.arange(0.5, gs.max() + 1.5)
ax[0, 1].hist(gs.values, bins=bins, color=ACCENT)
ax[0, 1].set_yscale("log")
ax[0, 1].set_xlabel("alloys sharing the same element set")
ax[0, 1].set_ylabel("number of element sets (log)")
ax[0, 1].set_title("(b) How many alloys share a chemical system?")
ax[0, 1].axvline(1, color=GREY, ls=":", lw=1)
ax[0, 1].text(1.6, ax[0, 1].get_ylim()[1] * 0.35,
              f"{prof['n_unique_elem_sets']} sets\nfor {prof['n_rows']} alloys", fontsize=8)

srt = np.sort(gs.values)[::-1]
share = np.cumsum(srt) / prof["n_rows"]
ax[1, 0].plot(np.arange(1, len(srt) + 1), share * 100, color=ACCENT2, lw=1.6)
ax[1, 0].axhline(prof["pct_alloys_in_shared_elem_sets"], color=GREY, ls="--", lw=1)
ax[1, 0].annotate(f"{prof['pct_alloys_in_shared_elem_sets']:.1f}% of alloys sit in a\n"
                  f"non-singleton element set",
                  xy=(len(srt) * 0.35, prof["pct_alloys_in_shared_elem_sets"]),
                  xytext=(len(srt) * 0.18, 30), fontsize=8,
                  arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
ax[1, 0].set_xlabel("element sets ranked by size")
ax[1, 0].set_ylabel("cumulative share of alloys (%)")
ax[1, 0].set_title("(c) Concentration of the dataset in a few chemical systems")

ne = df["n_elem"].value_counts().sort_index()
ax[1, 1].bar(ne.index, ne.values, color="#4c72b0")
ax[1, 1].set_xlabel("number of distinct elements in the alloy")
ax[1, 1].set_ylabel("number of alloys")
ax[1, 1].set_title("(d) Principal-element count")
save(fig, "fig1_dataset_structure.png")

# ---------------------------------------------------------------- FIG 2 ---
fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.1))
for cls in order:
    s = df[df["s_phase"] == cls]
    ax[0].scatter(s["VEC"], s["delta"] * 100, s=9, alpha=0.55,
                  color=CLS_COLORS[cls], label=cls, linewidths=0)
ax[0].axhline(6.6, color="k", ls="--", lw=1)
ax[0].axvspan(6.87, 8.0, color="grey", alpha=0.13)
ax[0].text(7.05, 30, "VEC\nambiguity\nwindow", fontsize=7, color=GREY)
ax[0].text(9.6, 6.9, r"$\delta \leq 6.6\%$", fontsize=8)
ax[0].set_xlabel("VEC")
ax[0].set_ylabel(r"$\delta$ (%)")
ax[0].set_title("(a) Classical SS-forming window in (VEC, δ)")
ax[0].legend(markerscale=2, fontsize=8, loc="upper right")
ax[0].set_ylim(0, 35)

for cls in order:
    s = df[df["s_phase"] == cls]
    ax[1].scatter(s["dH_mix"], np.log10(s["omega"].clip(lower=1e-2)), s=9, alpha=0.55,
                  color=CLS_COLORS[cls], linewidths=0)
ax[1].axvspan(-15, 5, color="grey", alpha=0.13)
ax[1].axhline(np.log10(1.1), color="k", ls="--", lw=1)
ax[1].set_xlabel(r"$\Delta H_{mix}$ (kJ/mol)")
ax[1].set_ylabel(r"$\log_{10}\Omega$")
ax[1].set_title(r"(b) $\Omega \geq 1.1$ and $-15 \leq \Delta H_{mix} \leq 5$ window")
ax[1].set_xlim(-60, 15)
save(fig, "fig2_phase_map.png")

# ---------------------------------------------------------------- FIG 3 ---
fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.3))
for a, col, lab in zip(axes, ["VEC", "delta", "dH_mix", "omega"],
                       ["VEC", r"$\delta$ (fraction)", r"$\Delta H_{mix}$ (kJ/mol)", r"$\Omega$"]):
    data, labels, colors = [], [], []
    for cls in order:
        v = df.loc[df["s_phase"] == cls, col].dropna().to_numpy()
        data.append(v)
        labels.append(cls)
        colors.append(CLS_COLORS[cls])
    bp = a.boxplot(data, patch_artist=True, showfliers=False, widths=0.62,
                   medianprops=dict(color="k", lw=1))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    a.set_xticklabels(labels, fontsize=8)
    a.set_title(lab)
    if col == "omega":
        a.set_yscale("log")
fig.suptitle("Descriptor distributions by phase class (box = IQR, whiskers = 1.5·IQR)", y=1.04, fontsize=9)
save(fig, "fig3_descriptors_by_class.png")

# ---------------------------------------------------------------- FIG 4 ---
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ct = crit_tab[crit_tab["rule"] != "majority_class_baseline"].copy()
ct = ct.sort_values("balanced_accuracy")
yy = np.arange(len(ct))
ax.barh(yy, ct["balanced_accuracy"], color=ACCENT, height=0.62)
ax.errorbar(ct["balanced_accuracy"], yy,
            xerr=[ct["balanced_accuracy"] - ct["balanced_accuracy_ci95"].str.strip("[]").str.split(",").str[0].astype(float),
                  ct["balanced_accuracy_ci95"].str.strip("[]").str.split(",").str[1].astype(float) - ct["balanced_accuracy"]],
            fmt="none", ecolor="k", elinewidth=0.8, capsize=2.5)
ax.set_yticks(yy)
ax.set_yticklabels(ct["rule"], fontsize=8)
ax.axvline(0.5, color=GREY, ls="--", lw=1)
ax.text(0.497, len(ct) - 0.35, "majority-class baseline (0.500)", fontsize=7.5,
        color=GREY, ha="right", va="top")
best_cv = max(crit["cv_tuned_single_descriptor"], key=lambda r: r["cv_balanced_accuracy_mean"])
ax.axvline(best_cv["cv_balanced_accuracy_mean"], color=ACCENT2, ls=":", lw=1.4)
ax.annotate(f"best CV-tuned single descriptor\n({best_cv['descriptor']}) = {best_cv['cv_balanced_accuracy_mean']:.3f}",
            xy=(best_cv["cv_balanced_accuracy_mean"], 1.0),
            xytext=(0.585, 1.9), fontsize=7.5, color=ACCENT2,
            arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=0.8))
ax.set_xlim(0.4, 0.85)
ax.set_xlabel("balanced accuracy for SS vs non-SS (95 % bootstrap CI)")
ax.set_title("Textbook HEA phase-formation rules under-perform even a tuned single descriptor")
save(fig, "fig4_criteria_performance.png")

# ---------------------------------------------------------------- FIG 5 ---
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))
blocks = ["composition", "descriptors", "composition+descriptors"]
models = ["logreg", "hgb"]
w = 0.36
xs = np.arange(len(blocks))
for k, (m, off, c) in enumerate(zip(models, [-w / 2, w / 2], [ACCENT, ACCENT2])):
    r = [e2[(e2.block == b) & (e2.model == m) & (e2.protocol == "random_stratified")]["bal_acc_mean"].iloc[0] for b in blocks]
    g = [e2[(e2.block == b) & (e2.model == m) & (e2.protocol == "element_set_grouped")]["bal_acc_mean"].iloc[0] for b in blocks]
    ax[0].bar(xs + off, r, w, color=c, alpha=0.95, label=f"{m} – random stratified")
    ax[0].bar(xs + off, g, w, color=c, alpha=0.42, label=f"{m} – element-set grouped")
    for x, rv, gv in zip(xs + off, r, g):
        ax[0].annotate(f"{rv - gv:+.3f}", xy=(x, max(rv, gv) + 0.012), ha="center", fontsize=7.5, color=c)
ax[0].set_xticks(xs)
ax[0].set_xticklabels([b.replace("+", "+\n") for b in blocks], fontsize=8)
ax[0].set_ylim(0.6, 0.95)
ax[0].set_ylabel("balanced accuracy (5-fold, 5 repeats)")
ax[0].set_title("(a) Random vs chemical-system-grouped CV")
ax[0].legend(fontsize=7.2, ncol=1, loc="lower left")

groups = list(paired.groupby(["block", "model"]))
short = {"composition": "comp", "descriptors": "desc", "composition+descriptors": "comp+desc"}
labels = [f"{short[b]}\n{m}" for (b, m), _ in groups]
data = [g["delta"].to_numpy() for _, g in groups]
bp = ax[1].boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                   medianprops=dict(color="k", lw=1.2))
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(ACCENT if i % 2 == 0 else ACCENT2)
    patch.set_alpha(0.55)
for i, d in enumerate(data):
    ax[1].scatter(np.full(len(d), i + 1) + np.linspace(-0.12, 0.12, len(d)), d,
                  s=10, color="k", alpha=0.55, zorder=3)
ax[1].axhline(0, color=GREY, ls="--", lw=1)
ax[1].set_xticklabels(labels, fontsize=7.2)
ax[1].set_ylabel(r"$\Delta$ balanced accuracy" + "\n(size-matched − disjoint)")
ax[1].set_title("(b) Paired, size-matched leakage effect (10 test splits)")
save(fig, "fig5_leakage_gap.png")

# ---------------------------------------------------------------- FIG 6 ---
dA = np.load(OUT / "03_neighbour_dist_A.npy")
dB = np.load(OUT / "03_neighbour_dist_B.npy")
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.7))
bins = np.linspace(0, 1.2, 61)
ax[0].hist(dA, bins=bins, color=ACCENT, alpha=0.75, label="TrainA (all non-test alloys)")
ax[0].hist(dB, bins=bins, color=ACCENT2, alpha=0.6, label="TrainB (chemical-system disjoint)")
ax[0].set_xlabel("L1 distance in composition space to nearest training alloy")
ax[0].set_ylabel("test alloys")
ax[0].set_title("(a) Nearest-neighbour distance")
ax[0].legend(fontsize=7.5)
for d, c, lab in [(dA, ACCENT, "TrainA"), (dB, ACCENT2, "TrainB")]:
    x = np.sort(d)
    ax[1].plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=1.6, label=lab)
ax[1].set_xlim(0, 0.6)
ax[1].set_xlabel("L1 distance to nearest training alloy")
ax[1].set_ylabel("cumulative fraction of test alloys")
ax[1].set_title("(b) Empirical CDF (zoomed)")
ax[1].legend(fontsize=7.5)
save(fig, "fig6_neighbour_distance.png")

# ---------------------------------------------------------------- FIG 7 ---
fig, ax = plt.subplots(figsize=(8.2, 3.9))
piv = e3.pivot(index="element", columns="model", values="balanced_accuracy")
piv = piv.sort_values("hgb", ascending=False)
yy = np.arange(len(piv))
ax.barh(yy - 0.19, piv["logreg"], 0.36, color=ACCENT, label="logistic regression")
ax.barh(yy + 0.19, piv["hgb"], 0.36, color=ACCENT2, label="gradient boosting")
ax.set_yticks(yy)
ax.set_yticklabels(piv.index, fontsize=8)
ax.axvline(0.5, color=GREY, ls="--", lw=1)
ax.axvline(leak["exp3_loeo"]["mean_bal_acc"], color="k", ls=":", lw=1.2)
ax.text(leak["exp3_loeo"]["mean_bal_acc"] + 0.008, len(piv) - 0.6,
        f"mean = {leak['exp3_loeo']['mean_bal_acc']:.3f}", fontsize=7.5)
ax.set_xlabel("balanced accuracy (train without element X → test alloys containing X)")
ax.set_title("Leave-one-element-out: performance on chemically unseen alloys")
ax.set_xlim(0, 1.0)
ax.legend(fontsize=8, loc="upper right")
save(fig, "fig7_loeo.png")

print("\nall figures written to", FIG)
