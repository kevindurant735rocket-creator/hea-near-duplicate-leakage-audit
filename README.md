# HEA Near-Duplicate Leakage Audit

**Random splits overstate HEA phase-prediction accuracy by 0.15 — for anyone
benchmarking materials ML.**

High-entropy-alloy papers report 85–95% accuracy from randomly split data. This audit
measures how much of that number is *redundancy* rather than *generalisation*, by training
two models of identical size that differ only in whether test-adjacent alloys are
present. Three quarters of the gap is the redundancy.

## The result

Gradient-boosted trees, exclusion radius τ = 0.35 in mole-fraction L1 space:

| quantity | value |
|---|---|
| near-duplicates excluded, balanced accuracy | **0.685** |
| size-matched control, duplicates kept | **0.836** |
| **pure near-duplicate leakage gap Δ** | **+0.151** |
| total accuracy drop, τ = 0 → 0.35 | 0.198 |
| — of which, smaller training set (volume) | 0.047 (24%) |
| — of which, redundant test-adjacent alloys | **0.151 (76%)** |

The effect tracks model capacity, which is what makes it diagnostic rather than a
constant offset:

| model | Δ at τ=0.35 | share of its own drop that is redundancy |
|---|---|---|
| gradient-boosted trees | **+0.151** | 76% of 0.198 |
| k-NN | +0.134 | 73% of 0.183 |
| logistic regression (regularised) | +0.032 | not distinguishable from zero |

A strongly regularised linear model barely moves. A high-capacity tree model loses seven
points of balanced accuracy when you remove test-adjacent alloys at constant training
size. That asymmetry is memorisation.

## The dataset is redundant, and it is not subtle

| property | value |
|---|---|
| labelled alloys | **1,103** |
| solid-solution (positive class) | 350 (31.7% base rate) |
| distinct element sets | **449** |
| alloys sharing a chemical system with another record | **74.9%** |
| largest single system | 45 alloys |
| exact duplicate compositions | **59** |
| elements appearing at least once | 57 |

1,103 alloys occupy only 449 distinct element sets. The average system holds 2.5 alloys,
but the distribution is heavy-tailed: Al–Co–Cr–Fe–Ni and its subsets, refractory
Mo–Nb–Ta–V–W, and the Ti–Zr–Hf family supply a large fraction of the table. A random
split of that table is not a random split of chemical space.

The mechanism is measurable directly. Under a random split, **25.5%** of test alloys have
a training neighbour at L1 distance < 0.05. Under a system-disjoint split, **2.8%** do.

## Real output

`reproduce.py` re-runs the exclusion sweep from the raw workbook and compares against the
committed reference:

```console
$ python public-release/reproduce.py

  hgb     0.00:0.883(m0.883,d+0.000) 0.02:0.852(m0.873,d+0.021) 0.05:0.845(m0.855,d+0.010) 0.08:0.829(m0.853,d+0.024) 0.12:0.782(m0.863,d+0.081) 0.20:0.735(m0.827,d+0.092) 0.35:0.685(m0.836,d+0.151)
  knn     0.00:0.841(m0.841,d+0.000) 0.02:0.827(m0.842,d+0.015) 0.05:0.807(m0.840,d+0.034) 0.08:0.779(m0.833,d+0.054) 0.12:0.770(m0.841,d+0.070) 0.20:0.733(m0.818,d+0.085) 0.35:0.659(m0.793,d+0.134)
  logreg  0.00:0.793(m0.793,d+0.000) 0.02:0.789(m0.792,d+0.003) 0.05:0.792(m0.791,d-0.001) 0.08:0.788(m0.781,d-0.007) 0.12:0.786(m0.777,d-0.009) 0.20:0.767(m0.776,d+0.009) 0.35:0.750(m0.782,d+0.032)

=== HEADLINE CHECK (hgb / gradient-boosted trees) ===
  hgb     tau=0.35  ref delta=+0.1510  repro delta=+0.1510  dev=0.00e+00  [ok]
  knn     tau=0.35  ref delta=+0.1340  repro delta=+0.1340  dev=0.00e+00  [ok]
  logreg  tau=0.35  ref delta=+0.0322  repro delta=+0.0322  dev=0.00e+00  [ok]
  hgb     tau=0 self-check ok (delta=0)
  knn     tau=0 self-check ok (delta=0)
  logreg  tau=0 self-check ok (delta=0)

=== REDUNDANCY DECOMPOSITION (hgb, tau 0 -> 0.35) ===
  total drop          = 0.198
  volume effect       = 0.047  (24%)
  redundancy effect   = 0.151  (76%)

=== RESULT ===
  PASS — recomputed headline matches the committed reference within tolerance 0.005.
```

`dev=0.00e+00` is the point: the deviation is not "small", it is identically zero. All
random sources are fixed (HGB `random_state=0`, matched subsample on
`numpy.default_rng(1000+seed)`), so the re-run lands on the committed numbers exactly.

## Three corroborating experiments

**1. Size-matched paired design.** Fixed 20% test set; `TrainA_matched` subsampled to the
size of `TrainB_disjoint`, so only near-duplicates differ. Over 10 splits, boosted trees
gain **+0.133** balanced accuracy from the random split (95% CI [0.063, 0.196],
Wilcoxon p=0.00195). Logistic regression gains **+0.023** (CI [-0.008, 0.055]).

**2. Random vs. grouped CV.** 15 fits per cell. For composition+descriptors with HGB:
0.859 random-stratified vs 0.776 element-set-grouped, a gap of **0.083**.

**3. Leave-one-element-out.** Train without an element, test only on alloys containing
it. 18 elements. Mean balanced accuracy falls to **0.665**, ranging from 0.420 (Mg) to
0.931 (Zr). This is the honest ceiling for extrapolating to a new chemistry.

## A null result worth as much as the headline

Excluding only the closest 0.05 L1 shell — strict near-duplicates — leaves a leakage gap
of just **+0.010** for boosted trees, against +0.151 at τ = 0.35. The inflation lives in
the **0.10–0.35** band: same chemical system, nearby stoichiometry. If you deduplicate by
exact composition match, you will fix almost none of this.

This part of the design was a correction. The raw centroid and nearest-neighbour
similarity series appeared to break sharply in 2023. Both turned out to be cell-size
confounded — they are max-type statistics, so a growing sample manufactures a jump. On a
fixed n=300 subsample the break disappears and all three measures agree. The paper
discloses this rather than reporting the flattering version.

## Reproduce

```bash
git clone https://github.com/kevindurant735rocket-creator/hea-near-duplicate-leakage-audit
cd hea-near-duplicate-leakage-audit

pip install -r public-release/requirements.txt   # numpy, pandas, scipy, scikit-learn, openpyxl
python public-release/reproduce.py               # ~60 s, 126 rows, prints PASS
```

The raw dataset is included at `data/heas_v2.xlsx`. To use your own copy:

```bash
python public-release/reproduce.py --data /path/to/2022-03-31-HEAs_dataset_v2.xlsx
```

To re-run the full study rather than just the headline sweep, work through
`code/00_profile.py` → `09_make_pdf.py` in order. Every number in the write-ups traces to
`out/*.json`.

## Layout

```
code/            analysis, figures, reports (00..09 + hea_common.py)
data/            raw workbook (CC-BY-4.0) + derived tables
out/             result JSON/CSV — single source of truth for every number
figures/         300 dpi figures
paper/           paper_EN / paper_CN (.md, .tex, .pdf)
report/          HTML + Markdown reports, EN and ZH
public-release/  articles (zh/en), one-pager, reproduce.py, landing page
```

## For the record: the classical criteria are not that good

While auditing, the classical phase-formation rules were evaluated on the same 1,103
alloys. The VEC rule (`3 ≤ VEC ≤ 8`) is a trap: it fires on **80.3%** of all alloys and
lands at balanced accuracy 0.479 — barely better than the 0.500 coin flip, with
specificity 0.183.

| rule | balanced acc. | sens. | spec. |
|---|---|---|---|
| always predict "not SS" | 0.500 | 0.000 | 1.000 |
| classic **3-rule** conjunction (δ + ΔH_mix + Ω) | **0.753** | 0.786 | 0.720 |
| δ ≤ 6.6% alone | 0.716 | 0.860 | 0.571 |
| Ω ≥ 1.1 alone | 0.715 | 0.943 | 0.486 |
| best CV-tuned **single** descriptor (Ω) | 0.742 ± 0.029 | | |
| classic 4-rule (adding VEC) | 0.678 | 0.569 | 0.788 |

Adding the VEC rule *hurts*. A three-rule conjunction a domain expert can apply by hand
beats every tuned machine-learning model on this task by a wide margin — and a single
tuned descriptor essentially ties it.

The VEC lattice rule is the one place a hand rule is genuinely perfect and still not
useful: on the **220** alloys where it is decidable it is **100%** accurate, but it
covers only **74.1%** of them (57 alloys sit on an ambiguous boundary). Perfect on a
subset you cannot identify in advance is not a classifier.

## What this is not

Read this before quoting +0.151.

- **It does not allege misconduct in any paper.** This is a methodological audit of one
  public dataset. No author is named, and no specific result is called wrong or
  fraudulent. Random splitting is common practice, not fraud.
- **It is not a discovery.** Near-duplicate leakage is documented in the life sciences
  (DataSAIL, *Nat. Commun.* 2025) and reviewed broadly for ML-in-science. The
  contribution is a size-matched, statistically tested quantification on one dataset,
  plus a reusable protocol.
- **One dataset, one task, 1,103 rows.** This is solid-solution classification on a single
  open HEA workbook. It is not a survey of materials ML, and 0.151 is not a universal
  leakage constant. The number is a property of *this* dataset's redundancy.
- **The τ and distance metric are choices.** The headline sits at τ = 0.35 in L1
  mole-fraction space; a different radius or metric gives a different Δ. The full sweep
  over seven radii is in `out/07_exclusion_sweep.json` so you can pick your own — and
  should.
- **The 76% decomposition is a ratio, not a causal attribution.** It splits an observed
  drop between two controlled comparisons; it does not claim redundancy *caused* 76% of
  any published accuracy figure.
- **τ = 0.35 is aggressive.** It removes 62% of the training set (882 → 333 rows). A
  milder radius, τ = 0.12, still leaves +0.081 for boosted trees. The trend is
  monotonic in τ, which is what makes the headline defensible rather than a single
  lucky point.
- **Leave-one-element-out is a stress test, not a target.** 0.665 mean is not what
  practitioners achieve; it is what happens when you demand generalisation to chemistry
  the model has never seen.

## Data, license, disclosure

- **Data:** *Materials for Design Open Repository, High Entropy Alloys*, Zenodo
  [10.5281/zenodo.6403257](https://doi.org/10.5281/zenodo.6403257), CC-BY-4.0. The raw
  workbook is included unmodified.
- **Code:** MIT — see [`LICENSE`](LICENSE).
- **AI-use disclosure:** AI-assisted (drafting, code scaffolding, analysis support). The
  design, the claims, and responsibility for correctness are human. No result was
  fabricated — every number traces to `out/*.json`, which `reproduce.py` regenerates from
  the raw data.
