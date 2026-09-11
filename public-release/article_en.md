# Near-Duplicate Leakage in ML Prediction of High-Entropy-Alloy Phase Formation

**TL;DR** — A *random* train/test split quietly inflates balanced accuracy in
machine-learning prediction of HEA solid-solution formation. Using a public dataset
(n ≈ 1100 alloys, Zenodo 6403257, CC-BY-4.0), we show that, for gradient-boosted trees,
most of the accuracy drop under an honest split is **near-duplicate redundancy (≈76%)**,
not lost training volume (≈24%). Every number regenerates with `python reproduce.py`.

---

## What we did

We revisited a common evaluation habit in empirical ML-for-materials work: train/test
splitting *at random*. HEA datasets are dense with compositionally near-identical alloys
(several `AlCoCrFeNi`-family recipes differing only by small mole-fraction tweaks). After a
random split, "hard" test alloys have near-twins in training — a model that memorises those
twins looks skilful. This is **near-duplicate leakage**, and it is invisible under a random
split.

To *isolate* it, we ran an **exclusion-radius sweep (EXP-5)**:

1. Fix one stratified 20 % test set `T`.
2. For radius `τ` in mole-fraction L1 space:
   - **excluded(τ)** — drop every training alloy within `τ` of any test alloy (near-twins removed);
   - **matched(τ)** — a random subsample of the *full* pool, forced to the **same size** (identical
     volume, twins kept).
3. `Δ(τ) = balAcc(matched) − balAcc(excluded)` isolates the pure redundancy contribution.
   `τ = 0` is a self-check: the two sets coincide, so `Δ = 0` by construction.

As `τ` grows, `matched` loses only volume while `excluded` loses volume **and** redundancy, so
the curve gap equals exactly what the near-duplicates were buying.

## Headline result (gradient-boosted trees, hgb)

| quantity                          | value |
|-----------------------------------|-------|
| excluded(τ=0.35) bal. acc.        | 0.685 |
| size-matched control bal. acc.    | 0.836 |
| **pure redundancy gap Δ**        | **+0.151** |
| total drop τ=0→0.35 (0.883→0.685) | 0.198 |
| ├─ data-volume share              | ~24 % |
| └─ near-duplicate redundancy      | **~76 %** |

By model: KNN `Δ = +0.134`; logistic regression only `Δ = +0.032` (not significant). Leakage
hits **memory-prone / non-linear / neighbour-based** models hardest — exactly what
"leakage through near-twins" predicts.

## Why it matters

Random splitting on compositionally dense datasets **systematically overstates model skill**.
This is a concrete, reproducible instance of the broader ML-science reproducibility problem —
and, unlike vague warnings, it ships with a *ready-to-use protocol* (`excluded` vs `matched`)
the community can adopt in any similar study.

## Limitations (stated up front)

- Single public dataset, one task (solid-solution classification).
- Undergraduate-level *methodological* audit — not a new model or theory.
- It demonstrates a systematic, reproducible over-estimation under random splits; it does
  **not** allege misconduct in any specific paper.
- Numbers are empirical, not a theoretical bound; small-sample noise at large `τ` is reported
  in the `delta_sd` fields, not hidden.

## Reproducibility & disclosure

All figures regenerate deterministically via `python reproduce.py` (fixed seeds throughout:
HGB `random_state=0`, matched subsample `numpy.default_rng(1000+seed)`, split
`random_state=seed`). This work was **AI-assisted** (drafting, code scaffolding, analysis
support); the design, the claims, and responsibility for correctness are human. No result was
fabricated — every number traces to `out/*.json`. Data: Zenodo 6403257, CC-BY-4.0.

**Links:** repository (to be added) · preprint (to be added after submission).
