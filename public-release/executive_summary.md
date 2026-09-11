# Executive Summary — HEA Near-Duplicate Leakage Audit

**One-liner:** Random train/test splits quietly inflate ML accuracy on
high-entropy-alloy (HEA) phase prediction — and most of that inflation is
**near-duplicate redundancy (≈76%)**, not data volume (≈24%).

**Audience:** research leads, reviewers, and anyone building ML models on
compositionally dense datasets (materials, chemistry, biology).

---

## Problem
HEA datasets are dense with near-identical recipes (several `AlCoCrFeNi`-family
alloys differing only by small mole-fraction tweaks). After a *random* split,
"hard" test alloys have near-twins in training; a model that memorises those
twins looks skilful. This **near-duplicate leakage** is invisible under a random
split and systematically overstates model skill — a concrete instance of the
broader ML-science reproducibility problem.

## Approach (reproducible protocol)
Fix one stratified test set `T`. For exclusion radius `τ` in mole-fraction L1 space:
- **excluded(τ)** — drop every training alloy within `τ` of any test alloy.
- **matched(τ)** — a random subsample of the full pool, forced to the *same size*.

`Δ(τ) = balAcc(matched) − balAcc(excluded)` isolates the pure redundancy
contribution. `τ = 0` is a built-in self-check (`Δ = 0`).

## Headline result (gradient-boosted trees, τ=0.35)
| quantity | value |
|---|---|
| excluded(τ) bal. acc. | 0.685 |
| size-matched control | 0.836 |
| **pure redundancy gap Δ** | **+0.151** |
| total drop decomposition | 24% volume · **76% redundancy** |

KNN `Δ = +0.134`; logistic regression `Δ = +0.032` (not significant). Leakage
hits memory-prone / neighbour-based models hardest.

## Why it matters
A ready-to-use evaluation protocol (`excluded` vs `matched`) the community can
adopt in any similar study — no new model required, just a more honest split.

## Reproducibility
Deterministic. `python reproduce.py` re-runs the sweep and asserts the headline
against the committed reference; verified deviation = 0.00e+00.

## Honest limitations
Single public dataset, one task (solid-solution classification); undergraduate-level
methodological audit; demonstrates systematic over-estimation, **does not** allege
misconduct in any specific paper.

## Disclosure
AI-assisted (drafting, scaffolding, analysis support); design, claims, and
responsibility are human. Data: Zenodo 6403257, CC-BY-4.0.

## Suggested next steps
1. Adopt `excluded/matched` as a routine robustness check in ML-for-materials papers.
2. Extend the sweep to other dense-composition domains.
3. Pair with group/LOSO splits for a full leakage report.
