"""
08_paper_figures.py — figures for the manuscript, built from out/07_exclusion_sweep.json.

fig8 : exclusion-radius sweep (accuracy vs tau, excluded vs size-matched)
fig9 : variance decomposition — how much of the apparent drop is data volume
       and how much is near-duplicate redundancy.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hea_common import FIG, OUT

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "font.family": "DejaVu Sans",
})

MODEL_COLOR = {"logreg": "#4c72b0", "hgb": "#c1440e", "knn": "#2a7f62"}
MODEL_LABEL = {"logreg": "Logistic regression", "hgb": "Gradient-boosted trees",
               "knn": "k-NN (k=5)"}

d = json.loads((OUT / "07_exclusion_sweep.json").read_text())
bm = d["by_model"]


def series(mname, key):
    return np.array([r[key] for r in bm[mname]])


def taus():
    return np.array([r["tau"] for r in bm["hgb"]])


def save(fig, name):
    p = FIG / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.name)


# ---------------------------------------------------------------- fig 8
def fig8():
    t = taus()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

    ax = axes[0]
    for m in ("logreg", "hgb", "knn"):
        c = MODEL_COLOR[m]
        ax.plot(t, series(m, "bal_acc_matched"), "--", color=c, lw=1.3, alpha=0.85,
                marker="s", ms=3.2, label=f"{MODEL_LABEL[m]} — size-matched")
        ax.plot(t, series(m, "bal_acc_excluded"), "-", color=c, lw=1.9,
                marker="o", ms=4.0, label=f"{MODEL_LABEL[m]} — near-duplicates removed")
        ax.fill_between(t, series(m, "bal_acc_matched"), series(m, "bal_acc_excluded"),
                        color=c, alpha=0.10, lw=0)
    ax.set_xlabel(r"exclusion radius $\tau$  (L1 distance in mole-fraction space)")
    ax.set_ylabel("balanced accuracy on the fixed test set")
    ax.set_title("(a) Training volume held constant")
    ax.set_ylim(0.62, 0.92)
    ax.legend(fontsize=6.4, loc="lower left", ncol=1, handlelength=1.9)

    ax = axes[1]
    for m in ("logreg", "hgb", "knn"):
        y = series(m, "delta")
        ax.plot(t, y, "-", color=MODEL_COLOR[m], lw=1.9, marker="o", ms=4.0,
                label=MODEL_LABEL[m])
        ax.fill_between(t, 0, y, color=MODEL_COLOR[m], alpha=0.10, lw=0)
    ax.axhline(0, color="#444444", lw=1.0)
    ax.set_xlabel(r"exclusion radius $\tau$")
    ax.set_ylabel(r"$\Delta$ balanced accuracy  (size-matched $-$ cleaned)")
    ax.set_title("(b) The part of the score that redundancy was buying")
    ax.set_ylim(-0.03, 0.175)
    ax.annotate("linear model barely\nuses the redundancy",
                xy=(0.16, 0.012), xytext=(0.055, 0.052), fontsize=6.8, color="#4c72b0",
                arrowprops=dict(arrowstyle="->", color="#4c72b0", lw=0.8))
    ax.legend(fontsize=7.0, loc="upper left")

    save(fig, "fig8_exclusion_sweep.png")


# ---------------------------------------------------------------- fig 9
def fig9():
    """Volume vs redundancy split of the total apparent loss, at tau = 0.35 and 0.20."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharey=True)
    models = ["logreg", "hgb", "knn"]

    for ax, tau in zip(axes, (0.20, 0.35)):
        base, vol, red, noise = [], [], [], []
        for m in models:
            recs = bm[m]
            r0 = recs[0]
            r = next(x for x in recs if np.isclose(x["tau"], tau))
            base.append(r0["bal_acc_excluded"])
            vol.append(r0["bal_acc_excluded"] - r["bal_acc_matched"])
            red.append(r["delta"])
            # a bar is only interpretable if it clears the between-split spread
            noise.append(r["delta_sd"] > r["delta"])
        x = np.arange(len(models))
        w = 0.36
        ax.bar(x - w / 2, vol, w, color="#9aa7b4", edgecolor="none",
               label="lost to smaller training set")
        ax.bar(x + w / 2, red, w, color="#c1440e", edgecolor="none",
               label="lost to near-duplicate redundancy")
        for xi, (v, rr, ns) in enumerate(zip(vol, red, noise)):
            ax.text(xi - w / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.8)
            ax.text(xi + w / 2, rr + 0.004, f"{rr:.3f}", ha="center", fontsize=6.8,
                    color="#c1440e", fontweight="bold")
            tot = v + rr
            if tot > 0.02:
                lab = f"{rr / tot:.0%} redundancy" + ("  (n.s.)" if ns else "")
                ax.text(xi, max(v, rr) + 0.030, lab,
                        ha="center", fontsize=6.8, color="#c1440e")
            if ns:
                ax.plot([xi - w / 2, xi + w / 2], [0, 0], color="#c1440e", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m].replace(" (k=5)", "\n(k=5)")
                            .replace("Gradient-boosted trees", "Gradient-boosted\ntrees")
                            .replace("Logistic regression", "Logistic\nregression")
                            for m in models], fontsize=7.6)
        ax.set_title(rf"total training data removed: {100 * (1 - next(x['n_train'] for x in bm['hgb'] if np.isclose(x['tau'], tau)) / 882):.0f}%"
                     f"   ($\\tau$ = {tau})", fontsize=9)
        ax.set_ylim(0, 0.215)

    axes[0].set_ylabel("balanced accuracy lost vs. baseline")
    axes[0].legend(fontsize=7.2, loc="upper left")
    save(fig, "fig9_variance_split.png")


if __name__ == "__main__":
    fig8()
    fig9()
