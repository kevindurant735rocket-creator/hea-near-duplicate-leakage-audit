# HEA Near-Duplicate Leakage Audit

> **One-line significance:** A *random* train/test split quietly inflates machine-learning
> accuracy on high-entropy-alloy (HEA) phase prediction — and most of that inflation is
> **near-duplicate redundancy**, not data volume. This repo lets anyone reproduce the number.

This is an **honest, undergraduate-level methodological audit**. It is **not** a breakthrough,
and it does **not** allege misconduct in any specific paper. It is a small, reproducible
correction to a widespread habit in empirical ML-for-materials work, packaged so the community
can verify it and (hopefully) adopt a better protocol.

---

## The problem in one paragraph

Many papers use machine learning to predict whether a high-entropy alloy forms a solid
solution. They report accuracy after a *random* split of the samples into train/test. But the
dataset is dense with compositionally near-identical alloys (e.g. several `AlCoCrFeNi`-family
recipes that differ only by small mole-fraction tweaks). After a random split, the test set's
"hard" cases have near-twins in the training set. A model that merely *memorises* those twins
looks skilful — this is **near-duplicate leakage**, and it is invisible under a random split.

## What this repo does

We isolate the leakage with an **exclusion-radius sweep (EXP-5)**:

1. Fix one stratified 20 % test set `T`.
2. For an exclusion radius `τ` in mole-fraction L1 space:
   - **excluded(τ)** — drop every training alloy that is within `τ` of *any* test alloy
     (near-twins removed);
   - **matched(τ)** — a random subsample of the *full* training pool, forced to the **same
     size** as `excluded(τ)` (identical volume, twins kept).
3. `Δ(τ) = balAcc(matched) − balAcc(excluded)` isolates the **pure redundancy** contribution.
   `τ = 0` is a built-in self-check: the two sets are identical, so `Δ = 0` by construction.

As `τ` grows, `matched` loses only volume while `excluded` loses volume **and** redundancy, so
the two curves separate by exactly the amount the near-duplicates were "buying".

## Headline result (gradient-boosted trees, hgb)

| quantity                                | value |
|-----------------------------------------|-------|
| excluded(τ=0.35) bal. acc.              | 0.685 |
| size-matched control bal. acc.          | 0.836 |
| **pure redundancy gap Δ**              | **+0.151** |
| total drop τ=0→0.35 (0.883→0.685)       | 0.198 |
| ├─ data-volume share                    | ~24 % |
| └─ near-duplicate redundancy share      | **~76 %** |

By model: KNN shows `Δ = +0.134`; logistic regression shows only `Δ = +0.032` (not
significant). The leakage mainly hurts **memory-prone / non-linear / neighbour-based** models,
not a simple linear baseline — exactly what "leakage through near-twins" predicts.

All numbers regenerate deterministically with `python reproduce.py`.

## Why this matters (the significance)

Random splitting on compositionally dense datasets **systematically overstates model skill**.
That is a concrete, reproducible instance of the broader ML-science reproducibility problem —
and, unlike vague warnings, it ships with a *ready-to-use protocol* (`excluded` vs `matched`)
the community can drop into any similar study. Significance here = **an honest, checkable,
reusable correction**, not a headline claim.

---

## Landing page (share-ready)

A polished, self-contained landing page lives at `public-release/index.html`. Open it
locally, or deploy the `public-release/` folder as a static site to get a shareable URL.
It carries the same honest framing and the one-command reproduction CTA.

## Quick start

```bash
# 1. (optional) create an environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. get the data (CC-BY-4.0, ~0.5 MB)
#    Download 2022-03-31-HEAs_dataset_v2.xlsx from
#    https://doi.org/10.5281/zenodo.6403257  and place it at  data/heas_v2.xlsx
#    (a copy is already included if you cloned the full research/ tree)

# 3. reproduce
python public-release/reproduce.py
# -> runs the sweep (~60 s), writes out/repro_rows.csv + out/repro_sweep.json,
#    compares to the committed reference, prints PASS.
```

`reproduce.py` exits `0` only when the recomputed `Δ` matches the committed reference
(`out/07_exclusion_sweep.json`) within tolerance. Determinism is guaranteed by fixed random
seeds throughout (HGB `random_state=0`, matched subsample `numpy.default_rng(1000+seed)`,
split `random_state=seed`).

## Repository layout (this is the `research/` tree)

```
research/
  data/heas_v2.xlsx                 # raw dataset (CC-BY-4.0, Zenodo 6403257)
  code/hea_common.py                # loading + featurisation
  code/07_exclusion_sweep.py        # EXP-5 source (also imported by reproduce.py)
  out/07_exclusion_sweep.json       # committed reference numbers
  figures/fig8_exclusion_sweep.png  # the headline figure
  public-release/                   # this folder: article + reproducible packaging
```

## Honest limitations

- **Scope:** a single public dataset (Zenodo 6403257, n ≈ 1100 alloys), one task
  (solid-solution vs non-solid-solution classification).
- **Level:** an undergraduate-methodology audit, not a new model or theory.
- **Claim:** it demonstrates a *systematic, reproducible over-estimation* under random splits;
  it does **not** prove any specific paper is wrong or fraudulent.
- **Statistics:** the `matched`/`excluded` gap is the estimand; small-sample noise at large `τ`
  (few excluded alloys remain) is visible in the `delta_sd` fields and is reported, not hidden.

## AI-use disclosure (transparency)

This project was **AI-assisted**: drafting, code scaffolding and analysis support were
machine-generated. The research design, the claims, the honesty framing, and responsibility for
correctness are **human**. No result was fabricated; every number traces to `out/*.json`, which
`reproduce.py` regenerates from raw data.

## Data source & attribution

> M. Af screw et al., *High Entropy Alloys*, Materials for Design Open Repository,
> Zenodo record 6403257, DOI [10.5281/zenodo.6403257](https://doi.org/10.5281/zenodo.6403257),
> CC-BY-4.0.

## Cite this work

```bibtex
@misc{hea-leakage-audit,
  title         = {Near-Duplicate Leakage in ML Prediction of High-Entropy-Alloy Phase Formation},
  author        = {[YOUR NAME]},
  year          = {2026},
  howpublished  = {\url{[REPO URL]}},
  note          = {CC-BY-4.0 data; MIT code}
}
```

A machine-readable `CITATION.cff` is included; an `executive_summary.md` one-pager is
provided for research leads and reviewers.

## Author & preprint (fill in before publishing)

- **Author:** <YOUR NAME>
- **Affiliation:** <YOUR SCHOOL / LAB>
- **Contact:** <YOUR EMAIL>
- **Preprint:** <arXiv / repository link — add after submission>
- **Repo:** <GitHub link — add after push>

---

*Significance is earned by being reproducible and honest. The repo is the durable asset; the
attention is a bonus.*
