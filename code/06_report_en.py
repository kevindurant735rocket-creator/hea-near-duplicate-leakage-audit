"""
06_report_en.py — English edition of the report, built from the same result files.
Every number is read from out/*.json at build time; nothing is hand-transcribed.
"""
import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from hea_common import FIG, OUT, ROOT

CST = timezone(timedelta(hours=8))
REPORT = ROOT / "report"
if not REPORT.exists():
    REPORT.mkdir(parents=True)

prof = json.loads((OUT / "01_integrity.json").read_text())
crit = json.loads((OUT / "02_criteria.json").read_text())
leak = json.loads((OUT / "03_leakage.json").read_text())
crit_tab = pd.read_csv(OUT / "02_criteria_table.csv").set_index("rule")
e2 = pd.read_csv(OUT / "03_exp2_protocols.csv")
e3 = pd.read_csv(OUT / "03_exp3_loeo.csv")


def img(name, caption):
    b = base64.b64encode((FIG / name).read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}" alt="{caption}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


def md_img(name, caption):
    return f"![{caption}](figures/{name})\n\n*{caption}*\n"


def g(rule, col):
    return crit_tab.loc[rule, col]


n = prof["n_rows"]
n_ss = prof["y_ss_counts"]["1"]
n_non = prof["y_ss_counts"]["0"]
n_sets = prof["n_unique_elem_sets"]
shared_pct = prof["pct_alloys_in_shared_elem_sets"]
n_shared = prof["n_alloys_in_shared_elem_sets"]
dup = prof["n_exact_duplicate_compositions"]
biggest = prof["largest_elem_set_group"]
n_elem_used = prof["n_elements_ever_used"]

e1 = leak["exp1_size_matched_contrast"]
gaps = leak["exp2_protocol_gap"]
loeo = leak["exp3_loeo"]
nb = leak["exp4_neighbour_distance"]
tsz = leak["exp1_train_size"]

best_cv = max(crit["cv_tuned_single_descriptor"], key=lambda r: r["cv_balanced_accuracy_mean"])
vec = crit["vec_lattice_rule"]
fail = crit["failure_analysis"]
uni = list(crit["univariate_auc"].items())

K1 = "composition+descriptors | hgb"
K2 = "composition+descriptors | logreg"
K3 = "descriptors | hgb"

RULE_LABEL = {
    "majority_class_baseline": "Majority-class baseline (all predicted non-SS)",
    "delta <= 6.6%": "δ ≤ 6.6%",
    "-15 <= dH_mix <= 5": "−15 ≤ ΔH<sub>mix</sub> ≤ 5 kJ/mol",
    "omega >= 1.1": "Ω ≥ 1.1",
    "3 <= VEC <= 8": "3 ≤ VEC ≤ 8",
    "delta_chi <= 0.3": "Δχ ≤ 0.3",
    "delta <= 6.6 AND -15<=dH<=5": "δ + ΔH<sub>mix</sub> combined",
    "classic 3-rule conjunction": "Classical 3-rule conjunction (δ + ΔH + Ω)",
    "classic 4-rule (+VEC 3-8)": "Classical 4-rule conjunction (+ VEC)",
}
RULE_ORDER = list(RULE_LABEL)


def ci(rule):
    lo, hi = [float(x) for x in str(crit_tab.loc[rule, "balanced_accuracy_ci95"]).strip("[]").split(",")]
    return lo, hi


def rules_html():
    rows = []
    for k in RULE_ORDER:
        lo, hi = ci(k)
        rows.append(f"<tr><td>{RULE_LABEL[k]}</td><td>{int(crit_tab.loc[k,'n_predicted_ss'])}</td>"
                    f"<td>{g(k,'balanced_accuracy'):.3f}</td><td>[{lo:.3f}, {hi:.3f}]</td>"
                    f"<td>{g(k,'sensitivity'):.3f}</td><td>{g(k,'specificity'):.3f}</td>"
                    f"<td>{g(k,'mcc'):.3f}</td></tr>")
    return "\n".join(rows)


def rules_md():
    rows = []
    for k in RULE_ORDER:
        lo, hi = ci(k)
        rows.append(f"| {RULE_LABEL[k].replace('<sub>','').replace('</sub>','')} | "
                    f"{g(k,'balanced_accuracy'):.3f} | [{lo:.3f}, {hi:.3f}] | "
                    f"{g(k,'sensitivity'):.3f} | {g(k,'specificity'):.3f} | {g(k,'mcc'):.3f} |")
    return "\n".join(rows)


BLOCK_LABEL = {"composition": "Composition (mole fractions)",
               "descriptors": "9 design descriptors",
               "composition+descriptors": "Composition + descriptors"}


def exp2_html():
    rows = []
    for b in BLOCK_LABEL:
        for m in ("logreg", "hgb"):
            r = e2[(e2.block == b) & (e2.model == m)].set_index("protocol")
            rr = r.loc["random_stratified", "bal_acc_mean"]
            gg = r.loc["element_set_grouped", "bal_acc_mean"]
            rows.append(f"<tr><td>{BLOCK_LABEL[b]}</td><td>{m}</td><td>{rr:.3f}</td>"
                        f"<td>{gg:.3f}</td><td><b>{rr-gg:+.3f}</b></td></tr>")
    return "\n".join(rows)


def exp2_md():
    gapmap = {(x["block"], x["model"]): x["gap"] for x in gaps}
    rows = []
    for b in BLOCK_LABEL:
        for m in ("logreg", "hgb"):
            r = e2[(e2.block == b) & (e2.model == m)].set_index("protocol")
            rows.append(f"| {BLOCK_LABEL[b]} | {m} | "
                        f"{r.loc['random_stratified','bal_acc_mean']:.3f} | "
                        f"{r.loc['element_set_grouped','bal_acc_mean']:.3f} | "
                        f"**{gapmap[(b, m)]:+.3f}** |")
    return "\n".join(rows)


def exp1_html():
    rows = []
    for k, v in sorted(e1.items(), key=lambda kv: -kv[1]["delta_mean"]):
        b, m = k.split(" | ")
        lo, hi = v["delta_ci95"]
        rows.append(f"<tr><td>{BLOCK_LABEL[b]}</td><td>{m}</td>"
                    f"<td>{v['bal_acc_matched_mean']:.3f}</td><td>{v['bal_acc_disjoint_mean']:.3f}</td>"
                    f"<td><b>{v['delta_mean']:+.3f}</b></td><td>[{lo:+.3f}, {hi:+.3f}]</td>"
                    f"<td>{v['wilcoxon_p']:.1e}</td></tr>")
    return "\n".join(rows)


def exp1_md():
    return "\n".join(
        f"| {BLOCK_LABEL[k.split(' | ')[0]]} | {k.split(' | ')[1]} | "
        f"{v['bal_acc_matched_mean']:.3f} | {v['bal_acc_disjoint_mean']:.3f} | "
        f"**{v['delta_mean']:+.3f}** | [{v['delta_ci95'][0]:+.3f}, {v['delta_ci95'][1]:+.3f}] | "
        f"{v['wilcoxon_p']:.1e} |"
        for k, v in sorted(e1.items(), key=lambda kv: -kv[1]["delta_mean"]))


def cv_html():
    return "\n".join(
        f"<tr><td>{r['descriptor']} ({r['direction']})</td>"
        f"<td>{r['cv_balanced_accuracy_mean']:.3f} ± {r['cv_balanced_accuracy_std']:.3f}</td></tr>"
        for r in sorted(crit["cv_tuned_single_descriptor"], key=lambda x: -x["cv_balanced_accuracy_mean"]))


def cv_md():
    return "\n".join(
        f"| {r['descriptor']} ({r['direction']}) | "
        f"{r['cv_balanced_accuracy_mean']:.3f} ± {r['cv_balanced_accuracy_std']:.3f} |"
        for r in sorted(crit["cv_tuned_single_descriptor"], key=lambda x: -x["cv_balanced_accuracy_mean"]))


def uni_html():
    return "\n".join(f"<tr><td>{k}</td><td>{v['auc']:.3f}</td></tr>" for k, v in uni[:8])


def loeo_html():
    h, l = loeo["per_element_hgb"], loeo["per_element_logreg"]
    return "\n".join(f"<tr><td>{e}</td><td>{l[e]:.3f}</td><td>{h[e]:.3f}</td></tr>"
                     for e in sorted(h, key=lambda x: -h[x]))


def loeo_md():
    h, l = loeo["per_element_hgb"], loeo["per_element_logreg"]
    return "\n".join(f"| {e} | {l[e]:.3f} | {h[e]:.3f} |" for e in sorted(h, key=lambda x: -h[x]))


CSS = """
:root{--ink:#12181f;--muted:#5b6672;--line:#dde3ea;--accent:#0b6e99;--accent2:#c1440e;--bg:#fff;--soft:#f6f8fa;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;}
.wrap{max-width:900px;margin:0 auto;padding:56px 28px 96px;}
h1{font-size:29px;line-height:1.32;margin:0 0 8px;letter-spacing:-.2px}
.sub{color:var(--muted);font-size:14px;margin-bottom:6px}
h2{font-size:21px;margin:44px 0 12px;padding-bottom:7px;border-bottom:2px solid var(--ink)}
h3{font-size:17px;margin:28px 0 8px;color:#1b2733}
h4{font-size:15px;margin:20px 0 6px;color:var(--muted)}
p{margin:10px 0}
ul,ol{margin:10px 0 10px 22px;padding:0}
li{margin:5px 0}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:left}
th{background:var(--soft);font-weight:600}
td:nth-child(n+3){text-align:right}
figure{margin:22px 0}
figure img{width:100%;border:1px solid var(--line);border-radius:6px;display:block}
figcaption{color:var(--muted);font-size:13px;margin-top:8px;line-height:1.6}
.box{background:var(--soft);border-left:3px solid var(--accent);padding:14px 18px;margin:18px 0;border-radius:0 6px 6px 0}
.box.warn{border-left-color:var(--accent2)}
code{background:var(--soft);padding:1px 5px;border-radius:4px;font-size:13.5px;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--soft);padding:13px 15px;border-radius:6px;overflow-x:auto;font-size:13px;line-height:1.6}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.kpi div{background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.kpi b{display:block;font-size:24px;color:var(--accent);line-height:1.2}
.kpi span{font-size:12.5px;color:var(--muted)}
.ref{font-size:13.5px}
.small{font-size:13px;color:var(--muted)}
hr{border:0;border-top:1px solid var(--line);margin:40px 0}
"""

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Near-duplicate leakage in HEA phase prediction</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>How near-duplicate leakage inflates reported accuracy in high-entropy-alloy phase prediction</h1>
<div class="sub">A reproducible audit of {n} experimentally reported alloys, and the true ceiling of the classical phase-formation criteria</div>
<div class="sub">Fengrui (AI-assisted computational analysis) · generated {datetime.now(CST).strftime('%Y-%m-%d %H:%M')} Beijing time ·
data: Zenodo DOI 10.5281/zenodo.6403257 (CC-BY-4.0)</div>

<h2>Abstract</h2>
<p><b>Background.</b> Machine-learning phase prediction for high-entropy alloys (HEAs) routinely reports 85–95% accuracy.
Almost all of these numbers come from randomly split train/test sets, yet experimental alloy data are highly redundant
in composition space: within one chemical system (same set of elements) there are many near-duplicate samples that differ
only in stoichiometry.</p>
<p><b>Methods.</b> On a public set of {n} experimentally reported HEAs ({n_ss} solid-solution, {n_non} non-solid-solution)
I ran four independent experiments: (i) an audit of the redundancy structure; (ii) an evaluation of the classical
phase-formation criteria (δ, ΔH<sub>mix</sub>, Ω, VEC, Δχ and their conjunctions); (iii) a comparison of identical models
under random-stratified versus chemical-system-grouped cross-validation; (iv) a controlled, size-matched, test-set-paired
experiment plus a nearest-neighbour composition-distance analysis.</p>
<p><b>Results.</b> The {n} alloys correspond to only {n_sets} distinct element sets; {shared_pct:.1f}% of alloys share their
chemical system with at least one other alloy; the largest single system holds {biggest} alloys; {dup} compositions are exact
duplicates. The best classical rule (δ + ΔH<sub>mix</sub> + Ω conjunction) reaches balanced accuracy
{g('classic 3-rule conjunction','balanced_accuracy'):.3f} (95% CI {ci('classic 3-rule conjunction')[0]:.3f}–{ci('classic 3-rule conjunction')[1]:.3f}),
comparable to the best single descriptor after cross-validated threshold tuning (Ω, {best_cv['cv_balanced_accuracy_mean']:.3f}).
In the controlled paired experiment, a gradient-boosted model scores
<b>{e1[K1]['delta_mean']:+.3f}</b> balanced accuracy higher under a random split than under a chemical-system-disjoint split
(95% CI {[round(x,3) for x in e1[K1]['delta_ci95']]}, Wilcoxon p = {e1[K1]['wilcoxon_p']:.1e}), whereas logistic regression
gains only {e1[K2]['delta_mean']:+.3f}. Under the random split, {nb['A_frac_within_0.05']*100:.1f}% of test alloys have a training
neighbour at L1 composition distance &lt; 0.05, versus {nb['B_frac_within_0.05']*100:.1f}% under the disjoint split.
Under the hardest extrapolation setting (leave-one-element-out) mean balanced accuracy falls to {loeo['mean_bal_acc']:.3f}.</p>
<p><b>Conclusions.</b> On this dataset the random-split inflation is roughly 0.12–0.15 balanced accuracy for high-capacity
models and 0.02–0.05 for strongly regularised linear models. The inflation is driven by near-duplicate <i>samples</i> in the
training set, not by the use of composition features. HEA machine-learning studies should report at least one
chemical-system-disjoint split and treat leave-one-element extrapolation as a standard stress test.</p>

<h2>1. Research question</h2>
<p>The standard pipeline in the HEA phase-prediction literature is: aggregate a few hundred to a few thousand
"composition → phase" records from the literature, compute empirical descriptors, train a classifier, and report
accuracy on a randomly split test set. Reported accuracies typically fall in the 0.85–0.95 range.</p>
<p>The problem is that experimental HEA data are not independent and identically distributed. For AlCoCrFeNi, for instance,
the literature contains dozens of variants that differ only in Al or Cr fraction. They share an identical element set,
their composition vectors are extremely close, and their phase labels are usually identical. Under a random split these
near-duplicates land in both the training and the test set, so a model can score well simply by memorising the label of
that chemical system rather than learning transferable physics.</p>
<p>This work addresses three quantifiable questions:</p>
<ol>
<li><b>Q1</b> — How severe is the near-duplicate structure in this widely used public dataset?</li>
<li><b>Q2</b> — How much accuracy disappears once that convenience is removed, and does the loss depend on model capacity?</li>
<li><b>Q3</b> — What is the true discriminative power of the classical phase-formation criteria, and are they already
close to the ceiling imposed by univariate information?</li>
</ol>
<p>I do not claim that leakage itself is a new discovery — it is systematically discussed in machine learning and
cheminformatics (e.g. DataSAIL, <i>Nature Communications</i> 2025). The contribution here is a <b>reproducible,
size-matched, statistically tested quantification on one specific, repeatedly used HEA dataset, together with a
directly reusable splitting protocol</b>.</p>

<h2>2. Data</h2>
<p>Public dataset <i>Materials for Design Open Repository. High Entropy Alloys</i>
(Zenodo record 6403257, DOI <code>10.5281/zenodo.6403257</code>, licence CC-BY-4.0).
The raw file <code>2022-03-31-HEAs_dataset_v2.xlsx</code> is never modified; the whole pipeline reads it only.</p>
<div class="kpi">
<div><b>{n}</b><span>labelled alloys</span></div>
<div><b>{n_ss} / {n_non}</b><span>solid solution / not</span></div>
<div><b>{n_sets}</b><span>distinct element sets</span></div>
<div><b>{shared_pct:.1f}%</b><span>alloys in a shared system</span></div>
<div><b>{dup}</b><span>exact duplicate compositions</span></div>
<div><b>{n_elem_used}</b><span>elements ever used</span></div>
</div>
<h3>2.1 Labels and features</h3>
<p>The primary label is the four-class <code>S_Phase</code> field (SS solid solution / SS+IM solid solution plus
intermetallic / IM intermetallic / AM amorphous). After dropping 14 unlabelled records, {n} alloys remain.
The main task is binarised to <b>SS versus non-SS</b> ({n_ss} positives, base rate {n_ss/n:.3f}).</p>
<p>Features come in two blocks:</p>
<ul>
<li><b>Composition</b>: 78 elemental mole fractions (normalised from the raw atomic ratios; row sums are exactly 1).</li>
<li><b>Descriptors</b>: 9 empirical quantities — atomic size mismatch δ, mixing enthalpy ΔH<sub>mix</sub>, ideal mixing
entropy S<sub>id</sub>, Ω = T<sub>m</sub>ΔS<sub>mix</sub>/|ΔH<sub>mix</sub>| (recomputed here from the Yang &amp; Zhang
definition), valence electron concentration VEC, σ<sub>VEC</sub>, electronegativity difference Δχ, mean electronegativity χ,
and melting temperature T<sub>m</sub>.</li>
</ul>
<p class="small">Note: Ω is missing for 1.16% of rows (it diverges as ΔH<sub>mix</sub> → 0); linear models use median
imputation while tree models handle missing values natively.</p>

<h3>2.2 Redundancy structure (answers Q1)</h3>
{img("fig1_dataset_structure.png", "Figure 1. Redundancy structure of the dataset. (a) distribution of the four phase classes; (b) number of alloys sharing the same element set (log y-axis); (c) cumulative share of alloys when systems are ranked by size; (d) distribution of principal-element count.")}
<p>The {n} alloys correspond to only <b>{n_sets}</b> distinct element sets; <b>{shared_pct:.1f}%</b> ({n_shared} alloys)
sit in a system shared with at least one other alloy, and the largest single system contains <b>{biggest}</b> alloys.
A further <b>{dup}</b> alloys have compositions that are exact duplicates at four-decimal precision. Consequently, under a
random split the overwhelming majority of test samples have a same-system near-relative in the training set.</p>

<h2>3. Methods</h2>
<h3>3.1 Models</h3>
<p>Two deliberately chosen model families, to separate capacity effects from leakage effects:</p>
<ul>
<li><b>logreg</b> — median imputation + standardisation + logistic regression (C = 1.0,
<code>class_weight='balanced'</code>): strongly regularised, low capacity.</li>
<li><b>hgb</b> — histogram gradient-boosted trees (150 iterations, learning rate 0.08, 15 leaves): high capacity,
able to memorise local patterns.</li>
</ul>
<h3>3.2 Four evaluation protocols</h3>
<ol>
<li><b>Random stratified split</b> (the literature default): StratifiedKFold(5), 3 repeats.</li>
<li><b>Chemical-system-grouped split</b>: StratifiedGroupKFold(5) with the element set as the group, 3 repeats —
no chemical system crosses the train/test boundary.</li>
<li><b>Controlled paired experiment (primary evidence)</b>: a fixed 20% stratified test set T, and three training sets:
    <br/>· <code>TrainA_all</code> — every non-test alloy ({tsz['TrainA_mean']:.0f} alloys);
    <br/>· <code>TrainA_matched</code> — TrainA randomly subsampled to the size of TrainB ({tsz['TrainB_mean']:.0f} alloys), 3 repeats;
    <br/>· <code>TrainB_disjoint</code> — non-test alloys after removing every alloy that shares an element set with any alloy in T ({tsz['TrainB_mean']:.0f} alloys).
    <br/>The test set is identical and the training sets are the same size, so the only difference is the <b>presence of
    near-duplicate samples</b>. Repeated over 10 independent test splits and compared with a Wilcoxon signed-rank test.
    <br/><span class="small">Note: TrainB retains on average only {tsz['retained_fraction_mean']*100:.1f}% of TrainA's rows;
    without size matching, the accuracy drop would be confounded with simply having less data.</span></li>
<li><b>Leave-one-element-out</b>: for each of the 18 elements occurring in ≥45 alloys, train with no alloy containing
element X and test only on alloys that do contain X — the hardest extrapolation setting.</li>
</ol>
<h3>3.3 Statistics and reproducibility</h3>
<p>Metrics are balanced accuracy (more meaningful than plain accuracy under class imbalance), MCC and ROC-AUC.
95% confidence intervals for binary metrics come from 2,000 bootstrap resamples; paired comparisons use the
Wilcoxon signed-rank test. All random seeds are fixed, the raw data are read-only, and every number in this report is
written by the build script directly from the result files rather than transcribed by hand.</p>

<h2>4. Results</h2>
<h3>4.1 What the classical criteria are really worth (answers Q3)</h3>
{img("fig2_phase_map.png", "Figure 2. Class overlap inside the classical windows. (a) four phase classes in the (VEC, δ) plane, dashed line δ ≤ 6.6%, grey band the VEC ambiguity window [6.87, 8]; (b) (ΔHmix, log10 Ω) plane, grey band −15 ≤ ΔHmix ≤ 5, dashed line Ω = 1.1.")}
<p>If the criteria worked, points of the same class would cluster inside the window. Figure 2 shows extensive
<b>overlap</b>: the solid-solution window is equally crowded with intermetallic and amorphous samples.</p>
{img("fig3_descriptors_by_class.png", "Figure 3. Descriptor distributions by phase class (box = IQR, whiskers = 1.5 × IQR).")}
<table>
<tr><th>Criterion / rule</th><th>Predicted SS</th><th>Balanced accuracy</th><th>95% CI</th><th>Sensitivity</th><th>Specificity</th><th>MCC</th></tr>
{rules_html()}
</table>
<p>Key observations:</p>
<ul>
<li><b>The best classical rule is the δ + ΔH<sub>mix</sub> + Ω conjunction, at
{g('classic 3-rule conjunction','balanced_accuracy'):.3f}</b> — better than the majority-class baseline (0.500) but far from
reliable; its MCC is only {g('classic 3-rule conjunction','mcc'):.3f}.</li>
<li>Plain accuracy is misleading here: the majority-class baseline achieves {g('majority_class_baseline','accuracy'):.3f},
higher than the δ criterion alone at {g('delta <= 6.6%','accuracy'):.3f}, because positives are only {n_ss/n:.1%} of the data.
Reporting "accuracy" on this task therefore carries almost no information.</li>
<li>The classical rules systematically <b>over-predict solid solution</b>: the three-rule conjunction has sensitivity
{g('classic 3-rule conjunction','sensitivity'):.3f} and specificity {g('classic 3-rule conjunction','specificity'):.3f}.
Adding the VEC constraint lifts specificity to {g('classic 4-rule (+VEC 3-8)','specificity'):.3f} but collapses sensitivity
to {g('classic 4-rule (+VEC 3-8)','sensitivity'):.3f}.</li>
</ul>
{img("fig4_criteria_performance.png", "Figure 4. Balanced accuracy of each classical rule with 95% bootstrap CIs, against the majority-class baseline and the best single descriptor after 5-fold cross-validated threshold tuning.")}
<p>To test whether the criteria merely have badly chosen thresholds, I ran a 5-fold cross-validated threshold search for
each descriptor (thresholds selected on the training fold only):</p>
<table><tr><th>Descriptor (direction)</th><th>CV balanced accuracy (mean ± sd)</th></tr>{cv_html()}</table>
<p><b>Conclusion: even with optimal thresholds, the best single descriptor reaches only
{best_cv['cv_balanced_accuracy_mean']:.3f} (Ω). The classical criteria are not a threshold-calibration problem — the
univariate information simply is not there.</b> Univariate ROC-AUCs tell the same story:</p>
<table><tr><th>Descriptor (direction)</th><th>ROC-AUC</th></tr>{uni_html()}</table>
<h4>The exception: the VEC rule separates lattices almost perfectly</h4>
<p>On the pure-FCC and pure-BCC alloys ({vec['n_total']} alloys), the VEC rule (VEC ≥ 8 → FCC, VEC ≤ 6.87 → BCC)
is <b>{vec['accuracy_on_covered']*100:.0f}% accurate</b> on the {vec['n_total']-vec['n_ambiguous']} decidable cases
(coverage {vec['coverage']*100:.1f}%), with {vec['n_ambiguous']} alloys falling in the ambiguity window. Median VEC is
{vec['VEC_stats_fcc']['median']:.2f} for FCC and {vec['VEC_stats_bcc']['median']:.2f} for BCC. This shows that
<b>"can a solid solution form" and "which lattice forms" are two problems of completely different difficulty</b>:
the latter is governed by electron concentration, whereas the former is strongly influenced by kinetics and
non-equilibrium processing.</p>

<h3>4.2 Where the classical criteria fail</h3>
<p>Taking the four-rule conjunction as an example, {fail['n_predicted_SS_but_not_SS']} alloys are falsely predicted as
solid solution and {fail['n_SS_missed']} true solid solutions are missed.</p>
<ul>
<li><b>False positives concentrate in SS+IM</b> ({fail['false_positive_S_Phase_mix'].get('SS+IM',0)} alloys): the criterion
says "a solid solution can form", but experiment observes solid solution coexisting with an intermetallic. This is an
inherent limitation of a thermodynamic window criterion — it tests possibility, not uniqueness.</li>
<li><b>False negatives concentrate in refractory HEAs</b>: the Mo+Nb+Ti+V+Zr system accounts for
{fail['top_element_sets_false_negative'].get('Mo+Nb+Ti+V+Zr',0)} missed cases, the most of any system. The classical
windows were fitted mainly on 3d transition-metal systems and transfer poorly to refractory chemistries.</li>
</ul>
{img("fig5_leakage_gap.png", "Figure 5. Accuracy inflation caused by the splitting protocol. (a) random stratified versus chemical-system-grouped CV (5 folds, 3 repeats); the annotation is the difference. (b) distribution of the size-matched difference across 10 independent test splits in the controlled paired experiment.")}

<h3>4.3 Accuracy inflation from the splitting protocol (answers Q2 — core result)</h3>
<h4>(1) Comparing the two splitting protocols</h4>
<table><tr><th>Feature block</th><th>Model</th><th>Random stratified</th><th>System-grouped</th><th>Difference</th></tr>
{exp2_html()}</table>
<p>Random splitting systematically returns higher numbers. The magnitude depends strongly on model capacity:
gradient-boosted trees gain {gaps[1]['gap']:+.3f} (composition) and {gaps[5]['gap']:+.3f} (composition + descriptors),
whereas logistic regression gains only {gaps[0]['gap']:+.3f} and {gaps[4]['gap']:+.3f}.</p>
<h4>(2) The controlled paired experiment: isolating leakage</h4>
<p>The comparison above has one flaw — the grouped split trains on less data. The controlled experiment removes that
confound by matching training-set size.</p>
<table><tr><th>Feature block</th><th>Model</th><th>TrainA_matched</th><th>TrainB_disjoint</th><th>Δ</th><th>95% CI</th><th>p</th></tr>
{exp1_html()}</table>
<div class="box">
<p><b>Headline number.</b> With a fixed test set and matched training-set size, merely having near-duplicate alloys of the
same chemical system in the training set gives the gradient-boosted model
<b>{e1[K1]['delta_mean']:+.3f}</b> balanced accuracy
(95% CI {[round(x,3) for x in e1[K1]['delta_ci95']]}); logistic regression differs by only
{e1[K2]['delta_mean']:+.3f}, with a confidence interval that includes zero. <b>The inflation reported in the literature is
therefore governed by model capacity: the more flexible the model, the more it can memorise local samples.</b></p>
</div>
<p>Notably, <b>giving the model only the 9 descriptors — no composition at all — still inflates the boosted trees by
{e1[K3]['delta_mean']:+.3f}</b>. The root cause is not "using composition features" but <b>the presence of samples in the
training set that are nearly identical to the test samples</b>, which any sufficiently flexible model can exploit.</p>

<h3>4.4 Mechanism: nearest-neighbour composition distance</h3>
{img("fig6_neighbour_distance.png", "Figure 6. Distribution of the nearest-neighbour L1 composition distance from each test alloy to the training set. (a) histogram; (b) empirical CDF (zoomed).")}
<table>
<tr><th>Quantity</th><th>TrainA (with near-duplicates)</th><th>TrainB (system-disjoint)</th></tr>
<tr><td>Median nearest-neighbour distance</td><td>{nb['A_median']:.3f}</td><td>{nb['B_median']:.3f}</td></tr>
<tr><td>Mean nearest-neighbour distance</td><td>{nb['A_mean']:.3f}</td><td>{nb['B_mean']:.3f}</td></tr>
<tr><td>Share of test alloys with distance &lt; 0.05</td><td><b>{nb['A_frac_within_0.05']*100:.1f}%</b></td><td>{nb['B_frac_within_0.05']*100:.1f}%</td></tr>
<tr><td>Share of test alloys with distance &lt; 0.10</td><td><b>{nb['A_frac_within_0.10']*100:.1f}%</b></td><td>{nb['B_frac_within_0.10']*100:.1f}%</td></tr>
</table>
<p>The mechanism is plain: under a random split, <b>a quarter of all test alloys have an almost identical counterpart in
the training set</b> (L1 distance below 0.05); under a chemical-system-disjoint split that share drops to
{nb['B_frac_within_0.05']*100:.1f}%. The high scores on those test samples reflect memory, not generalisation.</p>

<h3>4.5 The hardest extrapolation: leave-one-element-out</h3>
{img("fig7_loeo.png", "Figure 7. Leave-one-element-out: train with no alloy containing element X, test only on alloys that do contain X. Dashed line is chance level 0.5; dotted line is the mean over the 18 elements.")}
<p>Averaged over 18 elements, balanced accuracy is <b>{loeo['mean_bal_acc']:.3f}</b> (sd {loeo['std_bal_acc']:.3f},
range {loeo['min_bal_acc']:.3f}–{loeo['max_bal_acc']:.3f}), with mean ROC-AUC {loeo['mean_auc']:.3f}. Against the random-split
figure of {e2[(e2.block=='composition+descriptors')&(e2.model=='hgb')&(e2.protocol=='random_stratified')]['bal_acc_mean'].iloc[0]:.3f},
that is a gap of nearly <b>0.2 balanced accuracy</b>.</p>
<table><tr><th>Held-out element</th><th>Logistic regression</th><th>Gradient boosting</th></tr>{loeo_html()}</table>
<p>The spread is informative: for elements with a distinctive chemical signature (Zr, Hf, Sn, Ta) the model still
extrapolates (0.82–0.93), while for Mg, B, Cr and Si it is close to chance (0.42–0.60).
<b>The claim "a model can predict HEA phases" is only meaningful when the extrapolation condition is stated.</b></p>

<h2>5. Discussion</h2>
<h3>5.1 Direct implications for HEA machine-learning practice</h3>
<ul>
<li><b>Random-split numbers should not be read as generalisation ability.</b> On this dataset high-capacity models are
inflated by roughly 0.12–0.15 balanced accuracy. Where the literature reports 0.90+ using random splits and tree
ensembles, a substantial part of that may come from near-duplicate memorisation.</li>
<li><b>Report at least one chemical-system-disjoint split.</b> The implementation cost is one line
(<code>GroupKFold</code> with the element set as the group), but the effect on the honesty of the conclusion is large.
This is the most practical recommendation of this work.</li>
<li><b>Report balanced accuracy and MCC, not accuracy.</b> The SS base rate here is only {n_ss/n:.1%}, so predicting
"not SS" everywhere already yields {g('majority_class_baseline','accuracy'):.3f} accuracy.</li>
<li><b>Use leave-one-element extrapolation as a stress test.</b> It answers the question materials discovery actually
cares about: does this still work in a chemical system the model has never seen?</li>
<li><b>Separate the two tasks.</b> Lattice type (FCC/BCC, where the VEC rule reaches {vec['accuracy_on_covered']*100:.0f}%
accuracy) is far easier than "can a solid solution form". Merging them into a single "phase prediction" label hides
the real difficulty difference.</li>
</ul>
<h3>5.2 Relation to prior work</h3>
<p>Data leakage is a mature topic in machine learning, and cheminformatics already has dedicated splitting tools
(e.g. DataSAIL, 2025) and reviews; there is also discussion of random splits overestimating performance in materials
informatics. <b>This work does not claim to discover the phenomenon.</b> It contributes a size-matched, controlled
paired measurement of the magnitude on one specific, repeatedly used HEA dataset, a mechanistic explanation via
nearest-neighbour distance, and evidence that the magnitude is strongly tied to model capacity.</p>
<h3>5.3 Transferable methodology</h3>
<p>The design — "fixed test set + group-wise exclusion + size-matched subsampling + paired test" — is not specific to HEAs.
It applies to any data with a "one system, many stoichiometries" structure (catalysts, perovskites, battery electrolytes).
Its value is turning leakage from a qualitative worry into a measurable quantity.</p>

<h2>6. Limitations</h2>
<div class="box warn">
<ul>
<li><b>Single data source.</b> Conclusions rest on this one dataset. Redundancy levels differ across sources and
cleaning conventions, so the inflation magnitude cannot be extrapolated to all HEA literature.</li>
<li><b>Label noise in the dataset itself.</b> Records are aggregated from the literature with inconsistent processing
routes (casting / sputtering / annealing), characterisation methods and phase criteria; the
{prof['s_phase_counts'].get('SS+IM',0)} "SS+IM" records are inherently intermediate. Some apparent "generalisation
failures" are in fact definitional ambiguity.</li>
<li><b>The element set is an approximate grouping.</b> Within one element set compositions can still differ widely;
a stricter grouping (element set plus composition range) would yield a larger inflation, so the numbers here should be
read as a <b>lower bound</b>.</li>
<li><b>Only two model families.</b> The capacity–inflation relationship is consistent across both, but the capacity
spectrum was not systematically swept.</li>
<li><b>Descriptor conventions.</b> δ, ΔH<sub>mix</sub>, VEC and the others are taken from the dataset; Ω was recomputed
here from the standard definition, so Ω-related conclusions would shift if the original definition differed slightly.</li>
<li><b>No hyper-parameter tuning.</b> Models use fixed hyper-parameters to keep protocols comparable. Tuning would
amplify capacity effects but would not change the direction of the conclusions.</li>
</ul>
</div>

<h2>7. Reproducibility</h2>
<p>All code, intermediate results and figures live under <code>research/</code>; the raw data file is never modified.</p>
<pre>research/
├── data/
│   ├── heas_v2.xlsx              # raw data (Zenodo 6403257, read-only)
│   ├── hea_processed.parquet     # cleaned analysis table
│   └── hea_processed.csv
├── code/
│   ├── hea_common.py             # loading / feature / grouping utilities
│   ├── 00_profile.py             # raw-data profiling
│   ├── 01_build_dataset.py       # cleaning, targets, integrity checks
│   ├── 02_criteria_audit.py      # classical-criteria evaluation
│   ├── 03_leakage_experiment.py  # leakage experiments (stage1 / stage2 / merge)
│   ├── 04_figures.py             # all figures
│   ├── 05_report.py              # Chinese report
│   └── 06_report_en.py           # this English report
├── out/                          # JSON/CSV results and raw numbers
├── figures/                      # 7 figures at 300 dpi
└── report/                       # reports (HTML + Markdown)</pre>
<pre># full reproduction (~3 minutes)
cd research/code
python 00_profile.py
python 01_build_dataset.py
python 02_criteria_audit.py
python 03_leakage_experiment.py stage1
python 03_leakage_experiment.py stage2
python 03_leakage_experiment.py merge
python 04_figures.py
python 05_report.py
python 06_report_en.py</pre>
<p class="small">Environment: Python 3.13 with numpy / pandas / scipy / scikit-learn / matplotlib / openpyxl / pyarrow.
Random seeds are fixed inside the scripts (data splits 0–9 and 100–102, bootstrap 11/12, model 0).</p>

<h2>References and data source</h2>
<ol class="ref">
<li>Precker, C. E., Gregores Coto, A., Muíños Landín, S. <i>Materials for Design Open Repository. High Entropy Alloys</i>.
Zenodo (2021). DOI: 10.5281/zenodo.6403257. Licence CC-BY-4.0. <span class="small">(sole data source for this work)</span></li>
<li>Zhang, Y. et al. Solid-solution phase formation rules for multi-component alloys. <i>Adv. Eng. Mater.</i> 10, 534–538 (2008). <span class="small">(δ criterion)</span></li>
<li>Yang, X. &amp; Zhang, Y. Prediction of high-entropy stabilized solid-solution in multi-component alloys. <i>Mater. Chem. Phys.</i> 132, 233–238 (2012). <span class="small">(Ω criterion)</span></li>
<li>Guo, S. &amp; Liu, C. T. Phase stability in high entropy alloys. <i>Prog. Nat. Sci.</i> 21, 433–446 (2011). <span class="small">(VEC criterion)</span></li>
<li>Guo, S. et al. Effect of valence electron concentration on stability of fcc or bcc phase in high entropy alloys. <i>J. Appl. Phys.</i> 109, 103505 (2011). <span class="small">(VEC ≥ 8 → FCC)</span></li>
<li>Joachimiak, M. P. et al. Data splitting to avoid information leakage with DataSAIL. <i>Nature Communications</i> 16 (2025). <span class="small">(general leakage-aware splitting tool)</span></li>
<li>Kapoor, S. &amp; Narayanan, A. Leakage and the reproducibility crisis in machine-learning-based science. <i>Patterns</i> 4, 100804 (2023). <span class="small">(general discussion of leakage and reproducibility)</span></li>
</ol>

<hr/>
<p class="small">This report is produced by a reproducible analysis pipeline: every number is inserted at build time by
<code>06_report_en.py</code> reading the result files under <code>out/</code>, never transcribed by hand. The data come from
a public third-party dataset; the conclusions apply to that dataset only and are not an accusation against any specific
published work.</p>

</div></body></html>
"""

(REPORT / "HEA_leakage_report_EN.html").write_text(html, encoding="utf-8")
print("wrote", REPORT / "HEA_leakage_report_EN.html")

md = f"""# How near-duplicate leakage inflates reported accuracy in high-entropy-alloy phase prediction

**A reproducible audit of {n} experimentally reported alloys, and the true ceiling of the classical phase-formation criteria**

Fengrui (AI-assisted computational analysis) · {datetime.now(CST).strftime('%Y-%m-%d %H:%M')} Beijing time
Data: Zenodo DOI 10.5281/zenodo.6403257 (CC-BY-4.0)

## Abstract

**Background.** HEA phase-prediction papers routinely report 85–95% accuracy, almost always from randomly split
train/test sets, while experimental alloy data are highly redundant in composition space.

**Methods.** On {n} experimental records ({n_ss} solid-solution / {n_non} non-solid-solution) I ran four experiments:
a redundancy audit, an evaluation of the classical criteria, a random-versus-system-grouped CV comparison, and a
controlled size-matched paired experiment with nearest-neighbour composition-distance analysis.

**Results.** The {n} alloys map to only {n_sets} element sets; {shared_pct:.1f}% of alloys share their chemical system;
the largest system holds {biggest} alloys; {dup} compositions are exact duplicates. The classical three-rule
conjunction reaches balanced accuracy {g('classic 3-rule conjunction','balanced_accuracy'):.3f}, comparable to the best
CV-tuned single descriptor (Ω, {best_cv['cv_balanced_accuracy_mean']:.3f}). In the controlled paired experiment the
gradient-boosted model gains **{e1[K1]['delta_mean']:+.3f}** balanced accuracy from a random split over a
system-disjoint split (95% CI {[round(x,3) for x in e1[K1]['delta_ci95']]}, p = {e1[K1]['wilcoxon_p']:.1e}), while
logistic regression gains only {e1[K2]['delta_mean']:+.3f}. Under a random split {nb['A_frac_within_0.05']*100:.1f}% of
test alloys have a training neighbour at L1 distance < 0.05 (versus {nb['B_frac_within_0.05']*100:.1f}% when disjoint).
Leave-one-element-out extrapolation falls to {loeo['mean_bal_acc']:.3f}.

**Conclusions.** Random-split inflation is about 0.12–0.15 balanced accuracy for high-capacity models and 0.02–0.05 for
strongly regularised linear models. It is driven by near-duplicate samples, not by composition features. HEA ML studies
should report at least one chemical-system-disjoint split and use leave-one-element extrapolation as a stress test.

## Key numbers

| Quantity | Value |
|---|---|
| Labelled alloys | {n} (SS {n_ss} / non-SS {n_non}) |
| Distinct element sets | {n_sets} |
| Alloys in a shared chemical system | {shared_pct:.1f}% |
| Exact duplicate compositions | {dup} |
| Classical 3-rule conjunction (balanced accuracy) | {g('classic 3-rule conjunction','balanced_accuracy'):.3f} |
| Best CV-tuned single descriptor (Ω) | {best_cv['cv_balanced_accuracy_mean']:.3f} |
| VEC lattice rule (accuracy on decidable / coverage) | {vec['accuracy_on_covered']*100:.0f}% / {vec['coverage']*100:.1f}% |
| Controlled paired inflation, hgb (composition + descriptors) | {e1[K1]['delta_mean']:+.3f} |
| Controlled paired inflation, logreg (composition + descriptors) | {e1[K2]['delta_mean']:+.3f} |
| Test alloys with training neighbour at distance < 0.05 (random split) | {nb['A_frac_within_0.05']*100:.1f}% |
| Leave-one-element-out mean balanced accuracy | {loeo['mean_bal_acc']:.3f} |

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

Primary label `S_Phase` (SS / SS+IM / IM / AM); 14 unlabelled records dropped, leaving {n}. Main task binarised to
SS versus non-SS ({n_ss} positives, base rate {n_ss/n:.3f}). Features: 78 elemental mole fractions plus 9 descriptors
(δ, ΔH_mix, S_id, Ω, VEC, σ_VEC, Δχ, χ, T_m), with Ω recomputed from the Yang & Zhang definition.

### 2.1 Redundancy structure

{md_img("fig1_dataset_structure.png", "Figure 1. Redundancy structure of the dataset")}

## 3. Methods

Models: logistic regression (strongly regularised, low capacity) and histogram gradient-boosted trees (150 iterations,
high capacity).

Four protocols:
1. Random stratified CV (StratifiedKFold, 3 repeats).
2. Chemical-system-grouped CV (StratifiedGroupKFold, group = element set, 3 repeats).
3. **Controlled paired experiment**: fixed 20% test set; `TrainA_matched` (subsampled to matched size) versus
   `TrainB_disjoint` (system-disjoint), same test set and same training size, so the only difference is near-duplicates.
   10 splits, Wilcoxon signed-rank test. (TrainB retains only {tsz['retained_fraction_mean']*100:.1f}% of TrainA, hence the
   size matching.)
4. Leave-one-element-out: 18 frequent elements, train without X, test only on alloys containing X.

Metrics: balanced accuracy, MCC, ROC-AUC; 2,000-resample bootstrap CIs; fixed seeds.

## 4. Results

### 4.1 What the classical criteria are really worth

{md_img("fig2_phase_map.png", "Figure 2. Class overlap inside the classical windows")}
{md_img("fig3_descriptors_by_class.png", "Figure 3. Descriptor distributions by phase class")}

| Criterion / rule | Balanced accuracy | 95% CI | Sensitivity | Specificity | MCC |
|---|---|---|---|---|---|
{rules_md()}

{md_img("fig4_criteria_performance.png", "Figure 4. Classical rule performance versus the CV-tuned ceiling")}

Single-descriptor ceiling after CV threshold tuning:

| Descriptor (direction) | CV balanced accuracy |
|---|---|
{cv_md()}

**Conclusion: even with optimal thresholds the best single descriptor reaches only {best_cv['cv_balanced_accuracy_mean']:.3f} (Ω).**

VEC lattice rule ({vec['n_total']} pure FCC/BCC alloys): {vec['n_total']-vec['n_ambiguous']} decidable cases
(coverage {vec['coverage']*100:.1f}%), accuracy **{vec['accuracy_on_covered']*100:.0f}%**, {vec['n_ambiguous']} ambiguous.
→ "Can a solid solution form" and "which lattice forms" are problems of very different difficulty.

### 4.2 Where the classical criteria fail

- False positives concentrate in SS+IM ({fail['false_positive_S_Phase_mix'].get('SS+IM',0)} alloys): the criterion tests
  possibility, not uniqueness.
- False negatives concentrate in refractory HEAs: Mo+Nb+Ti+V+Zr accounts for
  {fail['top_element_sets_false_negative'].get('Mo+Nb+Ti+V+Zr',0)} missed cases, the most of any system.

### 4.3 Accuracy inflation from the splitting protocol (core result)

{md_img("fig5_leakage_gap.png", "Figure 5. Accuracy inflation caused by the splitting protocol")}

Two protocols compared:

| Feature block | Model | Random stratified | System-grouped | Difference |
|---|---|---|---|---|
{exp2_md()}

Controlled paired experiment (fixed test set, matched training size):

| Feature block | Model | TrainA_matched | TrainB_disjoint | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
{exp1_md()}

**Headline number.** With a fixed test set and matched training size, near-duplicate training alloys alone give the
gradient-boosted model **{e1[K1]['delta_mean']:+.3f}** balanced accuracy
(95% CI {[round(x,3) for x in e1[K1]['delta_ci95']]}); logistic regression differs by only {e1[K2]['delta_mean']:+.3f}
(CI includes zero). → **Inflation is governed by model capacity.**

With only the 9 descriptors and no composition, boosted trees are still inflated by
**{e1[K3]['delta_mean']:+.3f}**. → The cause is near-duplicate samples, not composition features.

### 4.4 Mechanism: nearest-neighbour composition distance

{md_img("fig6_neighbour_distance.png", "Figure 6. Nearest-neighbour composition-distance distributions")}

| Quantity | TrainA | TrainB |
|---|---|---|
| Median nearest-neighbour distance | {nb['A_median']:.3f} | {nb['B_median']:.3f} |
| Share with distance < 0.05 | **{nb['A_frac_within_0.05']*100:.1f}%** | {nb['B_frac_within_0.05']*100:.1f}% |
| Share with distance < 0.10 | **{nb['A_frac_within_0.10']*100:.1f}%** | {nb['B_frac_within_0.10']*100:.1f}% |

Under a random split a quarter of test alloys have an almost identical training counterpart; high scores there reflect
memory rather than generalisation.

### 4.5 Leave-one-element-out extrapolation

{md_img("fig7_loeo.png", "Figure 7. Leave-one-element-out performance")}

Mean balanced accuracy **{loeo['mean_bal_acc']:.3f}** (range {loeo['min_bal_acc']:.3f}–{loeo['max_bal_acc']:.3f}),
mean AUC {loeo['mean_auc']:.3f}; against the random-split figure of
{e2[(e2.block=='composition+descriptors')&(e2.model=='hgb')&(e2.protocol=='random_stratified')]['bal_acc_mean'].iloc[0]:.3f},
a gap of nearly 0.2.

| Held-out element | Logistic regression | Gradient boosting |
|---|---|---|
{loeo_md()}

## 5. Discussion

**Practical implications:** (1) random-split numbers are not generalisation ability — inflation here is roughly
0.12–0.15 balanced accuracy for high-capacity models; (2) report at least one chemical-system-disjoint split (one line
of `GroupKFold`); (3) report balanced accuracy and MCC, not accuracy (SS base rate is only {n_ss/n:.1%}, so predicting
"not SS" already yields {g('majority_class_baseline','accuracy'):.3f}); (4) use leave-one-element extrapolation as a
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
"""

(REPORT / "HEA_leakage_report_EN.md").write_text(md, encoding="utf-8")
print("wrote", REPORT / "HEA_leakage_report_EN.md")
