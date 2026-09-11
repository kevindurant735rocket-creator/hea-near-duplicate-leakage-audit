# How near-duplicate leakage inflates reported accuracy in high-entropy-alloy phase prediction

**A reproducible audit of 1103 experimentally reported alloys, and the true ceiling of the classical phase-formation criteria**

Fengrui (AI-assisted computational analysis) · 2026-09-11 11:46 Beijing time
Data: Zenodo DOI 10.5281/zenodo.6403257 (CC-BY-4.0)

## Abstract

**Background.** HEA phase-prediction papers routinely report 85–95% accuracy, almost always from randomly split
train/test sets, while experimental alloy data are highly redundant in composition space.

**Methods.** On 1103 experimental records (350 solid-solution / 753 non-solid-solution) I ran four experiments:
a redundancy audit, an evaluation of the classical criteria, a random-versus-system-grouped CV comparison, and a
controlled size-matched paired experiment with nearest-neighbour composition-distance analysis.

**Results.** The 1103 alloys map to only 449 element sets; 74.9% of alloys share their chemical system;
the largest system holds 45 alloys; 59 compositions are exact duplicates. The classical three-rule
conjunction reaches balanced accuracy 0.753, comparable to the best
CV-tuned single descriptor (Ω, 0.742). In the controlled paired experiment the
gradient-boosted model gains **+0.133** balanced accuracy from a random split over a
system-disjoint split (95% CI [0.063, 0.196], p = 2.0e-03), while
logistic regression gains only +0.023. Under a random split 25.5% of
test alloys have a training neighbour at L1 distance < 0.05 (versus 2.8% when disjoint).
Leave-one-element-out extrapolation falls to 0.665.

**Conclusions.** Random-split inflation is about 0.12–0.15 balanced accuracy for high-capacity models and 0.02–0.05 for
strongly regularised linear models. It is driven by near-duplicate samples, not by composition features. HEA ML studies
should report at least one chemical-system-disjoint split and use leave-one-element extrapolation as a stress test.

## Key numbers

| Quantity | Value |
|---|---|
| Labelled alloys | 1103 (SS 350 / non-SS 753) |
| Distinct element sets | 449 |
| Alloys in a shared chemical system | 74.9% |
| Exact duplicate compositions | 59 |
| Classical 3-rule conjunction (balanced accuracy) | 0.753 |
| Best CV-tuned single descriptor (Ω) | 0.742 |
| VEC lattice rule (accuracy on decidable / coverage) | 100% / 74.1% |
| Controlled paired inflation, hgb (composition + descriptors) | +0.133 |
| Controlled paired inflation, logreg (composition + descriptors) | +0.023 |
| Test alloys with training neighbour at distance < 0.05 (random split) | 25.5% |
| Leave-one-element-out mean balanced accuracy | 0.665 |

## 1. Research question

The standard pipeline aggregates literature data, computes empirical descriptors, trains a classifier and reports
accuracy on a random split (typically 0.85–0.95). But experimental HEA data are not i.i.d.: within one chemical system
there are many near-duplicates differing only in stoichiometry, so a random split lets them fall on both sides of the
boundary and a model can score by memorising system labels.

Three questions: **Q1** how severe is the redundancy; **Q2** how much accuracy disappears once it is removed, and does
this depend on model capacity; **Q3** what is the true discriminative power of the classical criteria.

This work does not claim leakage itself is new (see DataSAIL, *Nat. Commun.* 2025); it contributes a reproducible,
size-matched, statistically tested quantification on one specific dataset plus a reusable protocol.

## 2. Data

Source: *Materials for Design Open Repository. High Entropy Alloys*, Zenodo 6403257, CC-BY-4.0. Raw xlsx never modified.

Primary label `S_Phase` (SS / SS+IM / IM / AM); 14 unlabelled records dropped, leaving 1103. Main task binarised to
SS versus non-SS (350 positives, base rate 0.317). Features: 78 elemental mole fractions plus 9 descriptors
(δ, ΔH_mix, S_id, Ω, VEC, σ_VEC, Δχ, χ, T_m), with Ω recomputed from the Yang & Zhang definition.

### 2.1 Redundancy structure

![Figure 1. Redundancy structure of the dataset](figures/fig1_dataset_structure.png)

*Figure 1. Redundancy structure of the dataset*


## 3. Methods

Models: logistic regression (strongly regularised, low capacity) and histogram gradient-boosted trees (150 iterations,
high capacity).

Four protocols:
1. Random stratified CV (StratifiedKFold, 3 repeats).
2. Chemical-system-grouped CV (StratifiedGroupKFold, group = element set, 3 repeats).
3. **Controlled paired experiment**: fixed 20% test set; `TrainA_matched` (subsampled to matched size) versus
   `TrainB_disjoint` (system-disjoint), same test set and same training size, so the only difference is near-duplicates.
   10 splits, Wilcoxon signed-rank test. (TrainB retains only 50.6% of TrainA, hence the
   size matching.)
4. Leave-one-element-out: 18 frequent elements, train without X, test only on alloys containing X.

Metrics: balanced accuracy, MCC, ROC-AUC; 2,000-resample bootstrap CIs; fixed seeds.

## 4. Results

### 4.1 What the classical criteria are really worth

![Figure 2. Class overlap inside the classical windows](figures/fig2_phase_map.png)

*Figure 2. Class overlap inside the classical windows*

![Figure 3. Descriptor distributions by phase class](figures/fig3_descriptors_by_class.png)

*Figure 3. Descriptor distributions by phase class*


| Criterion / rule | Balanced accuracy | 95% CI | Sensitivity | Specificity | MCC |
|---|---|---|---|---|---|
| Majority-class baseline (all predicted non-SS) | 0.500 | [0.500, 0.500] | 0.000 | 1.000 | 0.000 |
| δ ≤ 6.6% | 0.716 | [0.690, 0.740] | 0.860 | 0.571 | 0.405 |
| −15 ≤ ΔHmix ≤ 5 kJ/mol | 0.689 | [0.663, 0.714] | 0.874 | 0.505 | 0.363 |
| Ω ≥ 1.1 | 0.714 | [0.692, 0.736] | 0.943 | 0.486 | 0.419 |
| 3 ≤ VEC ≤ 8 | 0.479 | [0.453, 0.504] | 0.774 | 0.183 | -0.050 |
| Δχ ≤ 0.3 | 0.533 | [0.518, 0.548] | 0.966 | 0.101 | 0.114 |
| δ + ΔHmix combined | 0.731 | [0.704, 0.757] | 0.797 | 0.665 | 0.431 |
| Classical 3-rule conjunction (δ + ΔH + Ω) | 0.753 | [0.725, 0.779] | 0.786 | 0.720 | 0.474 |
| Classical 4-rule conjunction (+ VEC) | 0.678 | [0.649, 0.708] | 0.569 | 0.788 | 0.354 |

![Figure 4. Classical rule performance versus the CV-tuned ceiling](figures/fig4_criteria_performance.png)

*Figure 4. Classical rule performance versus the CV-tuned ceiling*


Single-descriptor ceiling after CV threshold tuning:

| Descriptor (direction) | CV balanced accuracy |
|---|---|
| omega (higher->SS) | 0.742 ± 0.029 |
| S_id (higher->SS) | 0.730 ± 0.020 |
| delta (lower->SS) | 0.728 ± 0.021 |
| dH_mix (higher->SS) | 0.684 ± 0.025 |
| delta_chi (lower->SS) | 0.677 ± 0.023 |
| Tm (higher->SS) | 0.628 ± 0.024 |
| sigma_VEC (lower->SS) | 0.609 ± 0.022 |
| VEC (lower->SS) | 0.513 ± 0.024 |

**Conclusion: even with optimal thresholds the best single descriptor reaches only 0.742 (Ω).**

VEC lattice rule (220 pure FCC/BCC alloys): 163 decidable cases
(coverage 74.1%), accuracy **100%**, 57 ambiguous.
→ "Can a solid solution form" and "which lattice forms" are problems of very different difficulty.

### 4.2 Where the classical criteria fail

- False positives concentrate in SS+IM (104 alloys): the criterion tests
  possibility, not uniqueness.
- False negatives concentrate in refractory HEAs: Mo+Nb+Ti+V+Zr accounts for
  17 missed cases, the most of any system.

### 4.3 Accuracy inflation from the splitting protocol (core result)

![Figure 5. Accuracy inflation caused by the splitting protocol](figures/fig5_leakage_gap.png)

*Figure 5. Accuracy inflation caused by the splitting protocol*


Two protocols compared:

| Feature block | Model | Random stratified | System-grouped | Difference |
|---|---|---|---|---|
| Composition (mole fractions) | logreg | 0.787 | 0.730 | **+0.057** |
| Composition (mole fractions) | hgb | 0.837 | 0.760 | **+0.077** |
| 9 design descriptors | logreg | 0.798 | 0.796 | **+0.002** |
| 9 design descriptors | hgb | 0.850 | 0.763 | **+0.087** |
| Composition + descriptors | logreg | 0.807 | 0.769 | **+0.038** |
| Composition + descriptors | hgb | 0.859 | 0.776 | **+0.083** |

Controlled paired experiment (fixed test set, matched training size):

| Feature block | Model | TrainA_matched | TrainB_disjoint | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
| Composition (mole fractions) | hgb | 0.787 | 0.638 | **+0.150** | [+0.106, +0.230] | 2.0e-03 |
| Composition + descriptors | hgb | 0.829 | 0.696 | **+0.133** | [+0.063, +0.196] | 2.0e-03 |
| 9 design descriptors | hgb | 0.806 | 0.690 | **+0.116** | [+0.067, +0.155] | 2.0e-03 |
| Composition (mole fractions) | logreg | 0.774 | 0.721 | **+0.053** | [+0.025, +0.090] | 2.0e-03 |
| Composition + descriptors | logreg | 0.807 | 0.784 | **+0.023** | [-0.008, +0.055] | 2.7e-02 |
| 9 design descriptors | logreg | 0.803 | 0.783 | **+0.021** | [-0.010, +0.049] | 9.8e-03 |

**Headline number.** With a fixed test set and matched training size, near-duplicate training alloys alone give the
gradient-boosted model **+0.133** balanced accuracy
(95% CI [0.063, 0.196]); logistic regression differs by only +0.023
(CI includes zero). → **Inflation is governed by model capacity.**

With only the 9 descriptors and no composition, boosted trees are still inflated by
**+0.116**. → The cause is near-duplicate samples, not composition features.

### 4.4 Mechanism: nearest-neighbour composition distance

![Figure 6. Nearest-neighbour composition-distance distributions](figures/fig6_neighbour_distance.png)

*Figure 6. Nearest-neighbour composition-distance distributions*


| Quantity | TrainA | TrainB |
|---|---|---|
| Median nearest-neighbour distance | 0.120 | 0.333 |
| Share with distance < 0.05 | **25.5%** | 2.8% |
| Share with distance < 0.10 | **43.5%** | 6.5% |

Under a random split a quarter of test alloys have an almost identical training counterpart; high scores there reflect
memory rather than generalisation.

### 4.5 Leave-one-element-out extrapolation

![Figure 7. Leave-one-element-out performance](figures/fig7_loeo.png)

*Figure 7. Leave-one-element-out performance*


Mean balanced accuracy **0.665** (range 0.420–0.931),
mean AUC 0.776; against the random-split figure of
0.859,
a gap of nearly 0.2.

| Held-out element | Logistic regression | Gradient boosting |
|---|---|---|
| Zr | 0.783 | 0.931 |
| Hf | 0.863 | 0.904 |
| Sn | 0.837 | 0.867 |
| Ta | 0.726 | 0.817 |
| Mn | 0.663 | 0.773 |
| Cu | 0.788 | 0.769 |
| V | 0.551 | 0.747 |
| Ti | 0.634 | 0.734 |
| Mo | 0.575 | 0.731 |
| Ni | 0.682 | 0.681 |
| Fe | 0.598 | 0.662 |
| Co | 0.610 | 0.637 |
| Nb | 0.477 | 0.624 |
| Al | 0.589 | 0.600 |
| Si | 0.458 | 0.567 |
| Cr | 0.595 | 0.555 |
| Mg | 0.420 | 0.500 |
| B | 0.500 | 0.500 |

## 5. Discussion

**Practical implications:** (1) random-split numbers are not generalisation ability — inflation here is roughly
0.12–0.15 balanced accuracy for high-capacity models; (2) report at least one chemical-system-disjoint split (one line
of `GroupKFold`); (3) report balanced accuracy and MCC, not accuracy (SS base rate is only 31.7%, so predicting
"not SS" already yields 0.683); (4) use leave-one-element extrapolation as a
stress test; (5) separate "can a solid solution form" from "FCC versus BCC".

**Relation to prior work:** leakage is mature territory (DataSAIL 2025; Kapoor & Narayanan 2023). This work does not
claim to discover the phenomenon; it quantifies the magnitude on a specific dataset, explains the mechanism, and shows
the magnitude tracks model capacity.

**Transferability:** "fixed test set + group-wise exclusion + size matching + paired test" applies to any data with a
"one system, many stoichiometries" structure.

## 6. Limitations

- Single data source; magnitude not directly transferable to all HEA literature.
- Label noise (literature aggregation, inconsistent processing routes, SS+IM is inherently intermediate).
- The element set is an approximate grouping; stricter grouping would give a larger inflation, so these numbers are a
  **lower bound**.
- Only two model families; capacity spectrum not swept.
- Descriptor conventions inherited from the dataset (Ω recomputed here).
- No hyper-parameter tuning, to keep protocols comparable.

## 7. Reproducibility

```
research/
├── data/     raw xlsx (read-only) + cleaned tables
├── code/     hea_common.py, 00_profile.py, 01_build_dataset.py, 02_criteria_audit.py,
│             03_leakage_experiment.py, 04_figures.py, 05_report.py, 06_report_en.py
├── out/      result JSON/CSV and raw numbers
├── figures/  7 figures at 300 dpi
└── report/   HTML + Markdown (Chinese and English)
```

```
cd research/code
python 00_profile.py && python 01_build_dataset.py && python 02_criteria_audit.py
python 03_leakage_experiment.py stage1
python 03_leakage_experiment.py stage2
python 03_leakage_experiment.py merge
python 04_figures.py && python 05_report.py && python 06_report_en.py
```

Environment: Python 3.13; numpy / pandas / scipy / scikit-learn / matplotlib / openpyxl / pyarrow. Seeds fixed in-script.

## References

1. Precker, C. E., Gregores Coto, A., Muíños Landín, S. *Materials for Design Open Repository. High Entropy Alloys*. Zenodo (2021). DOI 10.5281/zenodo.6403257, CC-BY-4.0. (data source)
2. Zhang, Y. et al. *Adv. Eng. Mater.* 10, 534–538 (2008). (δ criterion)
3. Yang, X. & Zhang, Y. *Mater. Chem. Phys.* 132, 233–238 (2012). (Ω criterion)
4. Guo, S. & Liu, C. T. *Prog. Nat. Sci.* 21, 433–446 (2011). (VEC criterion)
5. Guo, S. et al. *J. Appl. Phys.* 109, 103505 (2011). (VEC ≥ 8 → FCC)
6. Joachimiak, M. P. et al. Data splitting to avoid information leakage with DataSAIL. *Nature Communications* 16 (2025).
7. Kapoor, S. & Narayanan, A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4, 100804 (2023).

---

*Produced by a reproducible pipeline: every number is inserted at build time by `06_report_en.py` from the result files
under `out/`. The data come from a public third-party dataset; conclusions apply to that dataset only and are not an
accusation against any specific published work.*
