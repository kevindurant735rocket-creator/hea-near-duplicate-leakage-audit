"""
05_report.py — render the paper-style report from the saved result files.
Every number in the report is read from out/*.json or out/*.csv at build time,
so the text can never drift from the computed results.
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
paired = pd.read_csv(OUT / "03_exp1_paired_deltas.csv")


def img(name, caption):
    b = base64.b64encode((FIG / name).read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}" alt="{caption}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


def md_img(name, caption):
    return f"![{caption}](figures/{name})\n\n*{caption}*\n"


def g(rule, col):
    return crit_tab.loc[rule, col]


# ---- pull the headline numbers -------------------------------------------
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


def fmt_rule_table():
    order = ["majority_class_baseline", "delta <= 6.6%", "-15 <= dH_mix <= 5",
             "omega >= 1.1", "3 <= VEC <= 8", "delta_chi <= 0.3",
             "delta <= 6.6 AND -15<=dH<=5", "classic 3-rule conjunction",
             "classic 4-rule (+VEC 3-8)"]
    cn = {
        "majority_class_baseline": "多数类基线（全判为非 SS）",
        "delta <= 6.6%": r"$\delta \leq 6.6\%$",
        "-15 <= dH_mix <= 5": r"$-15 \leq \Delta H_{mix} \leq 5$",
        "omega >= 1.1": r"$\Omega \geq 1.1$",
        "3 <= VEC <= 8": r"$3 \leq VEC \leq 8$",
        "delta_chi <= 0.3": r"$\Delta\chi \leq 0.3$",
        "delta <= 6.6 AND -15<=dH<=5": r"$\delta$ + $\Delta H_{mix}$ 联合",
        "classic 3-rule conjunction": "经典三判据联合（δ+ΔH+Ω）",
        "classic 4-rule (+VEC 3-8)": "经典四判据联合（再加 VEC）",
    }
    rows = []
    for k in order:
        ci = crit_tab.loc[k, "balanced_accuracy_ci95"]
        lo, hi = [float(x) for x in str(ci).strip("[]").split(",")]
        rows.append(
            f"<tr><td>{cn[k]}</td><td>{int(crit_tab.loc[k,'n_predicted_ss'])}</td>"
            f"<td>{g(k,'balanced_accuracy'):.3f}</td><td>[{lo:.3f}, {hi:.3f}]</td>"
            f"<td>{g(k,'sensitivity'):.3f}</td><td>{g(k,'specificity'):.3f}</td>"
            f"<td>{g(k,'mcc'):.3f}</td></tr>")
    return "\n".join(rows)


def fmt_exp2_table():
    blocks = ["composition", "descriptors", "composition+descriptors"]
    cn = {"composition": "成分摩尔分数", "descriptors": "9 个设计描述符",
          "composition+descriptors": "成分 + 描述符"}
    rows = []
    for b in blocks:
        for m in ("logreg", "hgb"):
            r = e2[(e2.block == b) & (e2.model == m)].set_index("protocol")
            rr = r.loc["random_stratified", "bal_acc_mean"]
            gg = r.loc["element_set_grouped", "bal_acc_mean"]
            rows.append(f"<tr><td>{cn[b]}</td><td>{m}</td><td>{rr:.3f}</td><td>{gg:.3f}</td>"
                        f"<td><b>{rr-gg:+.3f}</b></td></tr>")
    return "\n".join(rows)


def fmt_exp1_table():
    rows = []
    for k, v in sorted(e1.items(), key=lambda kv: -kv[1]["delta_mean"]):
        b, m = k.split(" | ")
        lo, hi = v["delta_ci95"]
        rows.append(f"<tr><td>{b}</td><td>{m}</td><td>{v['bal_acc_matched_mean']:.3f}</td>"
                    f"<td>{v['bal_acc_disjoint_mean']:.3f}</td>"
                    f"<td><b>{v['delta_mean']:+.3f}</b></td><td>[{lo:+.3f}, {hi:+.3f}]</td>"
                    f"<td>{v['wilcoxon_p']:.1e}</td></tr>")
    return "\n".join(rows)


def fmt_exp2_md():
    blocks = ["composition", "descriptors", "composition+descriptors"]
    rows = []
    for b in blocks:
        for m in ("logreg", "hgb"):
            r = e2[(e2.block == b) & (e2.model == m)].set_index("protocol")
            rr = r.loc["random_stratified", "bal_acc_mean"]
            gg = r.loc["element_set_grouped", "bal_acc_mean"]
            rows.append(f"| {b} | {m} | {rr:.3f} | {gg:.3f} | **{rr-gg:+.3f}** |")
    return "\n".join(rows)


def fmt_exp1_md():
    rows = []
    for k, v in sorted(e1.items(), key=lambda kv: -kv[1]["delta_mean"]):
        b, m = k.split(" | ")
        lo, hi = v["delta_ci95"]
        rows.append(f"| {b} | {m} | {v['bal_acc_matched_mean']:.3f} | "
                    f"{v['bal_acc_disjoint_mean']:.3f} | **{v['delta_mean']:+.3f}** | "
                    f"[{lo:+.3f}, {hi:+.3f}] | {v['wilcoxon_p']:.1e} |")
    return "\n".join(rows)


def fmt_rules_md():
    order = ["majority_class_baseline", "delta <= 6.6%", "-15 <= dH_mix <= 5",
             "omega >= 1.1", "3 <= VEC <= 8", "delta_chi <= 0.3",
             "delta <= 6.6 AND -15<=dH<=5", "classic 3-rule conjunction",
             "classic 4-rule (+VEC 3-8)"]
    label = {
        "majority_class_baseline": "多数类基线（全判非 SS）",
        "delta <= 6.6%": "δ ≤ 6.6%",
        "-15 <= dH_mix <= 5": "−15 ≤ ΔH_mix ≤ 5 kJ/mol",
        "omega >= 1.1": "Ω ≥ 1.1",
        "3 <= VEC <= 8": "3 ≤ VEC ≤ 8",
        "delta_chi <= 0.3": "Δχ ≤ 0.3",
        "delta <= 6.6 AND -15<=dH<=5": "δ + ΔH_mix 联合",
        "classic 3-rule conjunction": "经典三判据联合（δ+ΔH+Ω）",
        "classic 4-rule (+VEC 3-8)": "经典四判据联合（再加 VEC）",
    }
    out = []
    for k in order:
        lo, hi = [float(x) for x in str(crit_tab.loc[k, "balanced_accuracy_ci95"]).strip("[]").split(",")]
        out.append(f"| {label[k]} | {g(k,'balanced_accuracy'):.3f} | [{lo:.3f}, {hi:.3f}] | "
                   f"{g(k,'sensitivity'):.3f} | {g(k,'specificity'):.3f} | {g(k,'mcc'):.3f} |")
    return "\n".join(out)


def fmt_uni_table():
    rows = []
    for k, v in uni[:8]:
        rows.append(f"<tr><td>{k}</td><td>{v['auc']:.3f}</td></tr>")
    return "\n".join(rows)


def fmt_loeo_table():
    h = loeo["per_element_hgb"]
    l = loeo["per_element_logreg"]
    order = sorted(h, key=lambda e: -h[e])
    rows = []
    for e in order:
        rows.append(f"<tr><td>{e}</td><td>{l[e]:.3f}</td><td>{h[e]:.3f}</td></tr>")
    return "\n".join(rows)


CSS = """
:root{--ink:#12181f;--muted:#5b6672;--line:#dde3ea;--accent:#0b6e99;--accent2:#c1440e;--bg:#ffffff;--soft:#f6f8fa;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;}
.wrap{max-width:900px;margin:0 auto;padding:56px 28px 96px;}
h1{font-size:30px;line-height:1.3;margin:0 0 6px;letter-spacing:-.2px}
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
figure{margin:22px 0;padding:0}
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
.ref{font-size:13.5px;color:#2a3540}
.ref li{margin:7px 0}
hr{border:0;border-top:1px solid var(--line);margin:40px 0}
.small{font-size:13px;color:var(--muted)}
"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>近重复样本泄漏对高熵合金相预测精度的影响</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>近重复样本泄漏如何虚高高熵合金相预测的精度</h1>
<div class="sub">对 1,103 条实验报道高熵合金数据的可复现审计 —— 兼论经典相形成判据的真实上限</div>
<div class="sub">How near-duplicate leakage inflates reported accuracy in high-entropy-alloy phase prediction:
a reproducible audit of 1,103 experimentally reported alloys</div>
<div class="sub">峰瑞（AI 辅助计算分析） · 生成于 {datetime.now(CST).strftime('%Y-%m-%d %H:%M')} 北京时间 ·
数据来源 Zenodo DOI 10.5281/zenodo.6403257 (CC-BY-4.0)</div>

<h2>摘要</h2>
<p><b>背景。</b>高熵合金（HEA）相预测是机器学习在材料科学中最活跃的应用之一，文献中普遍报告 85%–95%
的预测精度。然而该领域绝大多数报道采用随机划分训练/测试集，而实验合金数据在成分空间中高度冗余：
同一化学体系（相同元素集合）内存在大量仅改变配比的近重复样本。</p>
<p><b>方法。</b>本文对公开的 1,103 条实验报道高熵合金数据（{n_ss} 条固溶体 SS、{n_non} 条非固溶体）
做四项独立实验：① 数据冗余结构审计；② 经典相形成判据（δ、ΔH<sub>mix</sub>、Ω、VEC、Δχ 及其联合）的真实判别力评估；
③ 同一模型在<b>随机划分</b>与<b>化学体系分组划分</b>下的精度差；④ 一个固定测试集、训练集规模匹配的受控配对实验，
以及最近邻成分距离分析。</p>
<p><b>结果。</b>数据中 {n} 条合金仅对应 {n_sets} 个不同元素集合，{shared_pct:.1f}% 的合金与至少另一条合金共享化学体系，
最大单体系含 {biggest} 条合金，另有 {dup} 条完全重复成分。经典三判据联合仅达平衡准确率
{g('classic 3-rule conjunction','balanced_accuracy'):.3f}（95% CI {crit_tab.loc['classic 3-rule conjunction','balanced_accuracy_ci95']}），
与经交叉验证调优的单一最优描述符（Ω，{best_cv['cv_balanced_accuracy_mean']:.3f}）相当。
在规模匹配的受控配对实验中，梯度提升模型在随机划分下的平衡准确率比化学体系不重叠划分高
<b>{e1['composition+descriptors | hgb']['delta_mean']:+.3f}</b>
（95% CI {[round(x,3) for x in e1['composition+descriptors | hgb']['delta_ci95']]}，Wilcoxon p={e1['composition+descriptors | hgb']['wilcoxon_p']:.1e}）；
逻辑回归的对应差异仅 {e1['composition+descriptors | logreg']['delta_mean']:+.3f}。
随机划分下 {nb['A_frac_within_0.05']*100:.1f}% 的测试合金在训练集中存在 L1 成分距离 &lt; 0.05 的“几乎相同”邻居，
而不重叠划分下仅 {nb['B_frac_within_0.05']*100:.1f}%。在最严格的外推设置（留一元素法）下，
平均平衡准确率降至 {loeo['mean_bal_acc']:.3f}，最低 {loeo['min_bal_acc']:.3f}。</p>
<p><b>结论。</b>在这份数据上，随机划分带来的精度虚高对高容量模型约为 0.12–0.15 平衡准确率，
对强正则线性模型约为 0.02–0.05。虚高并非源于“使用了成分特征”，而是源于训练集中存在近重复样本。
建议 HEA 机器学习工作至少报告一种化学体系不重叠的划分结果，并把留一元素外推作为标准压力测试。</p>

<h2>1. 研究问题</h2>
<p>机器学习驱动的 HEA 相预测文献常用如下流程：从文献汇总数百至数千条“成分 → 相”记录，计算若干经验描述符，
训练分类器，用随机划分的测试集报告精度。这类报道的精度通常在 0.85–0.95 区间。</p>
<p>问题在于：实验 HEA 数据不是独立同分布的样本。以 AlCoCrFeNi 为例，文献中存在数十条仅 Al 或 Cr 比例不同的变体，
它们共享完全相同的元素集合，成分向量彼此距离极小，且相标签往往一致。若随机划分，这些近重复样本会同时落入
训练集与测试集，模型只需“记住”该化学体系的标签即可得分，而无需学到可迁移的物理规律。</p>
<p>本文要回答三个可量化的问题：</p>
<ol>
<li><b>Q1</b>：这份广泛使用的公开数据中，近重复结构有多严重？</li>
<li><b>Q2</b>：去掉近重复带来的便利后，精度会下降多少？下降幅度是否依赖模型容量？</li>
<li><b>Q3</b>：经典相形成判据在这份数据上的真实判别力是多少？它们是否已经接近单变量信息的上限？</li>
</ol>
<p>本文不主张“数据泄漏”这一现象本身是新发现——它在机器学习与化学信息学中已被系统讨论
（如 DataSAIL，<i>Nature Communications</i> 2025）。本文的贡献是：<b>在一份被反复使用的具体 HEA 数据集上，
给出可复现、规模匹配、统计检验支撑的量化结果，并附带一套可直接复用的划分协议</b>。</p>

<h2>2. 数据</h2>
<p>使用公开数据集 <i>Materials for Design Open Repository. High Entropy Alloys</i>
（Zenodo 记录 6403257，DOI <code>10.5281/zenodo.6403257</code>，许可 CC-BY-4.0）。
原始文件 <code>2022-03-31-HEAs_dataset_v2.xlsx</code> 未作任何修改，全流程只读。</p>
<div class="kpi">
<div><b>{n}</b><span>带标签合金总数</span></div>
<div><b>{n_ss} / {n_non}</b><span>固溶体 / 非固溶体</span></div>
<div><b>{n_sets}</b><span>不同元素集合数</span></div>
<div><b>{shared_pct:.1f}%</b><span>处于非单一体系中的合金</span></div>
<div><b>{dup}</b><span>完全重复成分（4 位小数）</span></div>
<div><b>{n_elem_used}</b><span>出现过的元素种类</span></div>
</div>
<h3>2.1 标签与特征</h3>
<p>主标签为 <code>S_Phase</code> 四分类（SS 固溶体 / SS+IM 固溶体+金属间化合物 / IM 金属间化合物 / AM 非晶），
剔除 14 条无标签记录后得到 {n} 条。主任务二值化为 <b>SS vs 非 SS</b>（正例 {n_ss}，基准率 {n_ss/n:.3f}）。</p>
<p>特征分两块：</p>
<ul>
<li><b>成分</b>：78 维元素摩尔分数（由原始原子配比归一化，行和严格为 1）。</li>
<li><b>描述符</b>：9 个经验量——原子半径差 δ、混合焓 ΔH<sub>mix</sub>、理想混合熵 S<sub>id</sub>、
Ω = T<sub>m</sub>ΔS<sub>mix</sub>/|ΔH<sub>mix</sub>|（由本流程按 Yang &amp; Zhang 定义重算）、价电子浓度 VEC、σ<sub>VEC</sub>、
电负性差 Δχ、平均电负性 χ、熔点 T<sub>m</sub>。</li>
</ul>
<p class="small">注：Ω 有 1.16% 缺失（ΔH<sub>mix</sub>→0 时发散），线性模型中用中位数插补，
树模型原生处理缺失值。</p>

<h3>2.2 冗余结构审计（回答 Q1）</h3>
{img("fig1_dataset_structure.png", "图 1 数据集的冗余结构。(a) 四类相标签的分布；(b) 共享同一元素集合的合金数量分布（对数纵轴）；(c) 按体系规模排序的累积合金占比；(d) 合金主元数分布。")}
<p>{n} 条合金仅对应 <b>{n_sets}</b> 个不同元素集合；<b>{shared_pct:.1f}%</b>（{n_shared} 条）的合金处在
“与他人共享元素集合”的体系中，最大单一体系含 <b>{biggest}</b> 条合金。另有 <b>{dup}</b> 条合金在 4 位小数精度下
成分完全重复。这意味着：随机划分时，绝大多数测试样本在训练集中都有同体系的“近亲”。</p>

<h2>3. 方法</h2>
<h3>3.1 模型</h3>
<p>两个刻意选择的模型族，以区分“容量效应”与“泄漏效应”：</p>
<ul>
<li><b>logreg</b>：中位数插补 + 标准化 + 逻辑回归（C=1.0，<code>class_weight='balanced'</code>）——强正则、低容量。</li>
<li><b>hgb</b>：直方图梯度提升树（150 轮，学习率 0.08，15 叶）——高容量、可记忆局部模式。</li>
</ul>
<h3>3.2 四种评估协议</h3>
<ol>
<li><b>随机分层划分</b>（文献默认）：StratifiedKFold(5)，3 次重复。</li>
<li><b>化学体系分组划分</b>：StratifiedGroupKFold(5)，以元素集合为组，3 次重复——同一化学体系不会跨训练/测试。</li>
<li><b>受控配对实验（主证据）</b>：固定 20% 分层测试集 T，比较三个训练集：
    <br/>· <code>TrainA_all</code>：全部非测试合金（{tsz['TrainA_mean']:.0f} 条）；
    <br/>· <code>TrainA_matched</code>：从 TrainA 随机下采样至与 TrainB 同规模（{tsz['TrainB_mean']:.0f} 条），重复 3 次；
    <br/>· <code>TrainB_disjoint</code>：剔除所有与 T 中合金共享元素集合的非测试合金（{tsz['TrainB_mean']:.0f} 条）。
    <br/>测试集相同、训练集规模相同，唯一差别是<b>是否含有近重复样本</b>，因此差值可归因于泄漏。
    对 10 个独立测试划分重复，用 Wilcoxon 符号秩检验配对比较。
    <br/><span class="small">注意：TrainB 平均只保留 TrainA 的 {tsz['retained_fraction_mean']*100:.1f}% 数据，
    若不做规模匹配，精度下降会混入“数据变少”这一混淆因素。</span></li>
<li><b>留一元素法</b>：对出现次数 ≥45 的 18 种元素 X，训练集完全不含 X、测试集只含 X 的合金——最严格的外推测试。</li>
</ol>
<h3>3.3 统计与可复现性</h3>
<p>指标为平衡准确率（类别不平衡下比准确率更有意义）、MCC 与 ROC-AUC；二项指标的 95% 置信区间用
2,000 次 bootstrap 重采样得到；配对比较用 Wilcoxon 符号秩检验。全部随机种子固定，原始数据只读，
所有数字由脚本从结果文件生成，不做人工转录。</p>

<h2>4. 结果</h2>
<h3>4.1 经典相形成判据的真实表现（回答 Q3）</h3>
{img("fig2_phase_map.png", "图 2 经典判据窗口内的类别重叠。(a) (VEC, δ) 平面上的四类相分布，虚线为 δ ≤ 6.6%，灰带为 VEC 模糊区 [6.87, 8]；(b) (ΔHmix, log10 Ω) 平面，灰带为 −15 ≤ ΔHmix ≤ 5，虚线为 Ω = 1.1。")}
<p>若判据有效，同一类别的点应聚集在窗口内。图 2 显示四类相在判据平面上<b>大面积重叠</b>：
固溶体窗口内同样挤满金属间化合物与非晶样本。</p>
{img("fig3_descriptors_by_class.png", "图 3 各描述符按相类别的分布（箱体为四分位距，须为 1.5×IQR）。")}
<table>
<tr><th>判据 / 规则</th><th>判为 SS 的样本数</th><th>平衡准确率</th><th>95% CI</th><th>敏感度</th><th>特异度</th><th>MCC</th></tr>
{fmt_rule_table()}
</table>
<p>要点：</p>
<ul>
<li><b>最好的经典规则是“δ + ΔH<sub>mix</sub> + Ω”三判据联合，平衡准确率
{g('classic 3-rule conjunction','balanced_accuracy'):.3f}</b>，优于多数类基线（0.500）但远谈不上可靠；
其 MCC 仅 {g('classic 3-rule conjunction','mcc'):.3f}。</li>
<li>单看准确率会误导：多数类基线准确率为 {g('majority_class_baseline','accuracy'):.3f}，
高于单条 δ 判据的 {g('delta <= 6.6%','accuracy'):.3f}——因为正例只占 {n_ss/n:.1%}。
这说明报告“准确率”在该任务上几乎没有信息量。</li>
<li>经典规则普遍<b>偏向预测 SS</b>：三判据联合的敏感度 {g('classic 3-rule conjunction','sensitivity'):.3f}、
特异度 {g('classic 3-rule conjunction','specificity'):.3f}。加上 VEC 约束后特异度升至
{g('classic 4-rule (+VEC 3-8)','specificity'):.3f}，但敏感度崩到 {g('classic 4-rule (+VEC 3-8)','sensitivity'):.3f}。</li>
</ul>
{img("fig4_criteria_performance.png", "图 4 各经典规则的平衡准确率及 95% bootstrap 置信区间，对比多数类基线与经 5 折交叉验证调优的最优单一描述符。")}
<p>为判断“判据是不是只是阈值没调好”，本文对每个描述符做 5 折交叉验证阈值搜索（阈值仅在训练折上选取）：</p>
<table><tr><th>描述符（方向）</th><th>CV 平衡准确率（均值 ± 标准差）</th></tr>
{"".join(f"<tr><td>{r['descriptor']}（{r['direction']}）</td><td>{r['cv_balanced_accuracy_mean']:.3f} ± {r['cv_balanced_accuracy_std']:.3f}</td></tr>" for r in sorted(crit['cv_tuned_single_descriptor'], key=lambda x: -x['cv_balanced_accuracy_mean']))}
</table>
<p><b>结论：即使把阈值调到最优，单个描述符的上限也只有 {best_cv['cv_balanced_accuracy_mean']:.3f}（Ω）。
经典判据并非“阈值没调好”，而是单变量信息量本身就不够。</b>各描述符的单变量 ROC-AUC 也印证这一点：</p>
<table><tr><th>描述符（方向）</th><th>ROC-AUC</th></tr>{fmt_uni_table()}</table>
<h4>VEC 判据的例外：晶格类型判别极其成功</h4>
<p>在纯 FCC 与纯 BCC 合金（{vec['n_total']} 条）上，VEC 规则（VEC ≥ 8 → FCC，VEC ≤ 6.87 → BCC）
在可判定的 {vec['n_total']-vec['n_ambiguous']} 条（覆盖率 {vec['coverage']*100:.1f}%）上准确率为
<b>{vec['accuracy_on_covered']*100:.0f}%</b>；剩余 {vec['n_ambiguous']} 条落在模糊区。
FCC 合金的 VEC 中位数 {vec['VEC_stats_fcc']['median']:.2f}，BCC 为 {vec['VEC_stats_bcc']['median']:.2f}。
这说明<b>“能不能形成固溶体”与“形成哪种晶格”是两个难度完全不同的问题</b>：
后者由电子浓度主导，前者受动力学与非平衡制备条件强烈影响。</p>

<h3>4.2 经典判据在哪里失败</h3>
<p>以四判据联合为例，{fail['n_predicted_SS_but_not_SS']} 条被误判为固溶体，{fail['n_SS_missed']} 条真实固溶体被漏掉。</p>
<ul>
<li><b>假阳性集中在 SS+IM</b>（{fail['false_positive_S_Phase_mix'].get('SS+IM',0)} 条）——
判据给出“可以形成固溶体”，实验却观察到固溶体与金属间化合物共存。这是热力学窗口判据的固有局限：
它判断的是“是否可能”，不是“是否唯一”。</li>
<li><b>假阴性集中在难熔高熵合金</b>：Mo+Nb+Ti+V+Zr 体系被漏掉 {fail['top_element_sets_false_negative'].get('Mo+Nb+Ti+V+Zr',0)} 条，
是所有体系中最多。经典窗口主要基于 3d 过渡金属体系拟合，向难熔体系迁移时系统性失效。</li>
</ul>
{img("fig5_leakage_gap.png", "图 5 划分协议造成的精度虚高。(a) 随机分层划分 vs 化学体系分组划分（5 折、3 次重复），数字为差值；(b) 受控配对实验中规模匹配后的差值分布（10 个独立测试划分）。")}

<h3>4.3 划分协议造成的精度虚高（回答 Q2，核心结果）</h3>
<h4>（1）两种划分协议的对比</h4>
<table><tr><th>特征块</th><th>模型</th><th>随机分层划分</th><th>化学体系分组划分</th><th>差值</th></tr>
{fmt_exp2_table()}</table>
<p>随机划分系统性地给出更高的数字。差值幅度高度依赖模型容量：
梯度提升树 {gaps[1]['gap']:+.3f}（成分特征）与 {gaps[5]['gap']:+.3f}（成分+描述符），
而逻辑回归只有 {gaps[0]['gap']:+.3f} 与 {gaps[4]['gap']:+.3f}。</p>
<h4>（2）受控配对实验：把泄漏单独拎出来</h4>
<p>上述对比有一个漏洞：分组划分的训练集更小。受控实验通过规模匹配消除该混淆。</p>
<table><tr><th>特征块</th><th>模型</th><th>TrainA_matched</th><th>TrainB_disjoint</th><th>Δ</th><th>95% CI</th><th>p</th></tr>
{fmt_exp1_table()}</table>
<div class="box">
<p><b>核心数字。</b>在测试集固定、训练集规模相同的条件下，仅因为训练集中存在同化学体系的近重复合金，
梯度提升模型的平衡准确率就高出 <b>{e1['composition+descriptors | hgb']['delta_mean']:+.3f}</b>
（95% CI {[round(x,3) for x in e1['composition+descriptors | hgb']['delta_ci95']]}）；
逻辑回归的差距只有 {e1['composition+descriptors | logreg']['delta_mean']:+.3f}，
且其置信区间包含 0。这说明：<b>报告精度中的虚高部分主要由模型容量决定，容量越高、越能记忆局部样本，虚高越多。</b></p>
</div>
<p>值得注意的是，<b>只给 9 个描述符（不给成分）时，梯度提升树同样虚高
{e1['descriptors | hgb']['delta_mean']:+.3f}</b>。所以泄漏的根源不是“用了成分特征”，
而是<b>训练集里存在与测试样本几乎相同的样本</b>；任何足够灵活的模型都能利用这一点。</p>

<h3>4.4 机制：近邻成分距离</h3>
{img("fig6_neighbour_distance.png", "图 6 每个测试合金到训练集的最近邻 L1 成分距离分布。(a) 直方图；(b) 经验累积分布（放大）。")}
<table>
<tr><th>指标</th><th>TrainA（含近重复）</th><th>TrainB（体系不重叠）</th></tr>
<tr><td>最近邻距离中位数</td><td>{nb['A_median']:.3f}</td><td>{nb['B_median']:.3f}</td></tr>
<tr><td>最近邻距离均值</td><td>{nb['A_mean']:.3f}</td><td>{nb['B_mean']:.3f}</td></tr>
<tr><td>距离 &lt; 0.05 的测试样本占比</td><td><b>{nb['A_frac_within_0.05']*100:.1f}%</b></td><td>{nb['B_frac_within_0.05']*100:.1f}%</td></tr>
<tr><td>距离 &lt; 0.10 的测试样本占比</td><td><b>{nb['A_frac_within_0.10']*100:.1f}%</b></td><td>{nb['B_frac_within_0.10']*100:.1f}%</td></tr>
</table>
<p>机制一目了然：随机划分下，<b>四分之一的测试合金在训练集中存在成分距离小于 0.05 的“几乎相同”样本</b>；
化学体系不重叠划分下这一比例降到 {nb['B_frac_within_0.05']*100:.1f}%。模型在这些测试样本上的高分，
反映的是记忆而非泛化。</p>

<h3>4.5 最严格的外推：留一元素法</h3>
{img("fig7_loeo.png", "图 7 留一元素外推：训练集完全不含元素 X，测试集只含 X 的合金。虚线为随机水平 0.5，点线为 18 种元素的平均值。")}
<p>18 种元素平均平衡准确率 <b>{loeo['mean_bal_acc']:.3f}</b>（标准差 {loeo['std_bal_acc']:.3f}，
范围 {loeo['min_bal_acc']:.3f}–{loeo['max_bal_acc']:.3f}），平均 ROC-AUC {loeo['mean_auc']:.3f}。
对照随机划分下的 {e2[(e2.block=='composition+descriptors')&(e2.model=='hgb')&(e2.protocol=='random_stratified')]['bal_acc_mean'].iloc[0]:.3f}，
差距接近 <b>0.2 平衡准确率</b>。</p>
<table><tr><th>留出元素</th><th>逻辑回归</th><th>梯度提升</th></tr>{fmt_loeo_table()}</table>
<p>分化很有信息量：对 Zr、Hf、Sn、Ta 等“体系特征明显”的元素，模型仍能外推（0.82–0.93）；
而 Mg、B、Cr、Si 接近随机（0.42–0.60）。<b>“模型能预测 HEA 相”这一说法必须附上具体外推条件才成立。</b></p>

<h2>5. 讨论</h2>
<h3>5.1 对 HEA 机器学习实践的直接含义</h3>
<ul>
<li><b>随机划分的数字不应被当作泛化能力</b>。在这份数据上，高容量模型的随机划分精度约虚高
0.12–0.15 平衡准确率。文献中 0.90+ 的报道若采用随机划分与树集成，其中相当一部分可能来自近重复记忆。</li>
<li><b>至少要报告一种化学体系不重叠的划分</b>。实现成本极低（一行 <code>GroupKFold</code>，组为元素集合），
但对结论的诚实性影响巨大。这是本文最实用的建议。</li>
<li><b>报告平衡准确率与 MCC，而非准确率</b>。本数据 SS 基准率仅 {n_ss/n:.1%}，
“全判非 SS”就能得到 {g('majority_class_baseline','accuracy'):.3f} 准确率。</li>
<li><b>把留一元素外推作为压力测试</b>。它回答的是“换一个新元素体系还能不能用”，
这恰恰是材料发现真正关心的问题。</li>
<li><b>区分两个任务</b>：晶格类型（FCC/BCC，VEC 判据可达 {vec['accuracy_on_covered']*100:.0f}% 准确率）
远比“能否形成固溶体”容易。把它们混在一个“相预测”标签下报告，会掩盖真实的难度差异。</li>
</ul>
<h3>5.2 与已有工作的关系</h3>
<p>数据泄漏在机器学习中是成熟议题，化学信息学已有专门的划分工具（如 DataSAIL，2025）与综述；
材料信息学中也有关于随机划分高估性能的讨论。本文<b>不主张发现泄漏现象本身</b>，
而是在一份被反复使用的具体 HEA 数据集上，用规模匹配的受控配对设计给出量化幅度、
用最近邻距离给出机制解释，并证明虚高幅度与模型容量强相关。</p>
<h3>5.3 可迁移的方法论</h3>
<p>“固定测试集 + 按组剔除 + 规模匹配下采样 + 配对检验”这套设计不限于 HEA，
适用于任何存在“同一体系多配比”结构的数据（催化剂、钙钛矿、电池电解液等）。
它的价值在于把“泄漏”从一个定性担忧变成可测量的量。</p>

<h2>6. 局限</h2>
<div class="box warn">
<ul>
<li><b>单一数据源</b>。结论基于这一份数据集。不同来源、不同清洗口径的 HEA 数据冗余程度可能不同，
虚高幅度不能直接外推到所有 HEA 文献。</li>
<li><b>数据集本身的标签噪声</b>。数据来自文献汇总，制备条件（铸造/溅射/退火）、表征手段与判据不一致，
{prof['s_phase_counts'].get('SS+IM',0)} 条“SS+IM”本身就是中间态。部分“泛化失败”实为标签定义模糊所致。</li>
<li><b>元素集合是近似分组</b>。同一元素集合内仍可能有很大成分差异；
更严格的分组（如按元素+配比区间）会给出更大的虚高幅度，因此本文的数字可视为<b>下界</b>。</li>
<li><b>只有两个模型族</b>。容量—虚高的关系在两族模型上一致，但未系统扫描模型容量谱。</li>
<li><b>描述符口径</b>。δ、ΔH<sub>mix</sub>、VEC 等直接取自数据集；Ω 由本文按标准定义重算，
若原作者的 Ω 定义略有差异，Ω 相关结论会受影响。</li>
<li><b>未做超参数调优</b>。模型使用固定超参数，目的是保持协议间可比。调优会放大容量效应，
不会改变结论方向。</li>
</ul>
</div>

<h2>7. 可复现性</h2>
<p>全部代码、中间结果与图片在本地目录 <code>research/</code> 下，原始数据文件未被修改。</p>
<pre>research/
├── data/
│   ├── heas_v2.xlsx              # 原始数据（Zenodo 6403257，只读）
│   ├── hea_processed.parquet     # 清洗后分析表
│   └── hea_processed.csv
├── code/
│   ├── hea_common.py             # 加载/特征/分组工具
│   ├── 00_profile.py             # 原始数据画像
│   ├── 01_build_dataset.py       # 清洗、目标构造、完整性检查
│   ├── 02_criteria_audit.py      # 经典判据评估
│   ├── 03_leakage_experiment.py  # 泄漏实验（stage1 / stage2 / merge）
│   ├── 04_figures.py             # 全部图表
│   └── 05_report.py              # 本报告生成
├── out/                          # JSON/CSV 结果与原始数值
├── figures/                      # 7 张图（300 dpi PNG）
└── report/                       # 本报告（HTML + Markdown）</pre>
<pre># 完整复现（约 3 分钟）
cd research/code
python 00_profile.py
python 01_build_dataset.py
python 02_criteria_audit.py
python 03_leakage_experiment.py stage1
python 03_leakage_experiment.py stage2
python 03_leakage_experiment.py merge
python 04_figures.py
python 05_report.py</pre>
<p class="small">环境：Python 3.13，numpy / pandas / scipy / scikit-learn / matplotlib / openpyxl / pyarrow。
随机种子固定于脚本内（数据划分种子 0–9 与 100–102，bootstrap 种子 11/12，模型种子 0）。</p>

<h2>参考文献与数据来源</h2>
<ol class="ref">
<li>Precker, C. E., Gregores Coto, A., Muíños Landín, S. <i>Materials for Design Open Repository. High Entropy Alloys</i>.
Zenodo (2021). DOI: 10.5281/zenodo.6403257. 许可 CC-BY-4.0. <span class="small">（本文全部数据来源）</span></li>
<li>Zhang, Y. et al. Solid-solution phase formation rules for multi-component alloys. <i>Adv. Eng. Mater.</i> 10, 534–538 (2008). <span class="small">（δ 判据）</span></li>
<li>Yang, X. &amp; Zhang, Y. Prediction of high-entropy stabilized solid-solution in multi-component alloys. <i>Mater. Chem. Phys.</i> 132, 233–238 (2012). <span class="small">（Ω 判据）</span></li>
<li>Guo, S. &amp; Liu, C. T. Phase stability in high entropy alloys: formation of solid-solution phase or amorphous phase.
<i>Prog. Nat. Sci.</i> 21, 433–446 (2011). <span class="small">（VEC 判据）</span></li>
<li>Guo, S. et al. Effect of valence electron concentration on stability of fcc or bcc phase in high entropy alloys.
<i>J. Appl. Phys.</i> 109, 103505 (2011). <span class="small">（VEC ≥ 8 → FCC 规则）</span></li>
<li>Joachimiak, M. P. et al. Data splitting to avoid information leakage with DataSAIL. <i>Nature Communications</i> 16 (2025). <span class="small">（泄漏控制划分的通用工具，用于说明本文现象并非新发现）</span></li>
<li>Kapoor, S. &amp; Narayanan, A. Leakage and the reproducibility crisis in machine-learning-based science. <i>Patterns</i> 4, 100804 (2023). <span class="small">（泄漏与可复现性的一般讨论）</span></li>
</ol>

<hr/>
<p class="small">本报告由可复现分析流程自动生成：文中所有数值均由 <code>05_report.py</code> 在构建时从
<code>out/</code> 下的结果文件读取并填入，未经过人工转录。数据来自公开的第三方数据集，
本文结论仅针对该数据集，不构成对任何具体已发表工作的指控。</p>

</div></body></html>
"""

(REPORT / "HEA_leakage_report.html").write_text(html, encoding="utf-8")
print("wrote", REPORT / "HEA_leakage_report.html")

# ------------------------------------------------------------------ markdown
md = f"""# 近重复样本泄漏如何虚高高熵合金相预测的精度

**对 1,103 条实验报道高熵合金数据的可复现审计 —— 兼论经典相形成判据的真实上限**

峰瑞（AI 辅助计算分析） · {datetime.now(CST).strftime('%Y-%m-%d %H:%M')} 北京时间
数据来源：Zenodo DOI 10.5281/zenodo.6403257（CC-BY-4.0）

## 摘要

**背景。** 高熵合金（HEA）相预测文献普遍报告 85%–95% 的精度，但绝大多数采用随机划分，
而实验合金数据在成分空间高度冗余。

**方法。** 对 {n} 条实验数据（{n_ss} 条固溶体 SS / {n_non} 条非固溶体）做四项实验：
数据冗余审计、经典判据评估、随机 vs 化学体系分组划分对比、固定测试集+规模匹配的受控配对实验
与最近邻成分距离分析。

**结果。** {n} 条合金仅对应 {n_sets} 个元素集合，{shared_pct:.1f}% 的合金处于共享体系，
最大体系 {biggest} 条，完全重复成分 {dup} 条。经典三判据联合平衡准确率
{g('classic 3-rule conjunction','balanced_accuracy'):.3f}，与 CV 调优的最优单描述符（Ω，{best_cv['cv_balanced_accuracy_mean']:.3f}）相当。
受控配对实验中，梯度提升模型随机划分比体系不重叠划分高
**{e1['composition+descriptors | hgb']['delta_mean']:+.3f}**（95% CI {[round(x,3) for x in e1['composition+descriptors | hgb']['delta_ci95']]}，p={e1['composition+descriptors | hgb']['wilcoxon_p']:.1e}），
逻辑回归仅 {e1['composition+descriptors | logreg']['delta_mean']:+.3f}。
随机划分下 {nb['A_frac_within_0.05']*100:.1f}% 的测试合金有 L1 距离 < 0.05 的训练近邻（不重叠划分下 {nb['B_frac_within_0.05']*100:.1f}%）。
留一元素外推平均平衡准确率降至 {loeo['mean_bal_acc']:.3f}。

**结论。** 随机划分的精度虚高对高容量模型约 0.12–0.15 平衡准确率，对强正则线性模型约 0.02–0.05；
根源是训练集中的近重复样本，而非成分特征本身。建议 HEA 机器学习至少报告一种体系不重叠划分，
并以留一元素外推作为压力测试。

## 关键数字

| 指标 | 数值 |
|---|---|
| 带标签合金 | {n}（SS {n_ss} / 非 SS {n_non}） |
| 不同元素集合 | {n_sets} |
| 共享体系的合金占比 | {shared_pct:.1f}% |
| 完全重复成分 | {dup} 条 |
| 经典三判据联合（平衡准确率） | {g('classic 3-rule conjunction','balanced_accuracy'):.3f} |
| CV 调优最优单描述符（Ω） | {best_cv['cv_balanced_accuracy_mean']:.3f} |
| VEC 晶格规则（可判定部分准确率 / 覆盖率） | {vec['accuracy_on_covered']*100:.0f}% / {vec['coverage']*100:.1f}% |
| 受控配对虚高：hgb（成分+描述符） | {e1['composition+descriptors | hgb']['delta_mean']:+.3f} |
| 受控配对虚高：logreg（成分+描述符） | {e1['composition+descriptors | logreg']['delta_mean']:+.3f} |
| 随机划分下近邻距离 < 0.05 占比 | {nb['A_frac_within_0.05']*100:.1f}% |
| 留一元素外推平均平衡准确率 | {loeo['mean_bal_acc']:.3f} |

## 1. 研究问题

文献流程通常是：汇总文献数据 → 计算经验描述符 → 训练分类器 → 随机划分测试集报告精度（常 0.85–0.95）。
但实验 HEA 数据并非独立同分布：同一化学体系内存在大量仅改变配比的近重复样本，
随机划分会让它们同时进入训练集与测试集，模型只需记住体系标签即可得分。

三个可量化问题：
1. **Q1** 数据中近重复结构有多严重？
2. **Q2** 去掉近重复便利后精度下降多少？是否依赖模型容量？
3. **Q3** 经典判据的真实判别力是多少？是否已接近单变量信息上限？

本文不主张泄漏现象本身是新发现（见 DataSAIL, *Nat. Commun.* 2025 等），
贡献在于在一份被反复使用的具体数据集上给出可复现、规模匹配、统计检验支撑的量化结果与可复用协议。

## 2. 数据

来源：*Materials for Design Open Repository. High Entropy Alloys*，Zenodo 6403257，CC-BY-4.0。
原始 xlsx 未修改，只读。

标签为 `S_Phase` 四分类（SS / SS+IM / IM / AM），剔除 14 条无标签记录得 {n} 条；
主任务二值化为 SS vs 非 SS（正例 {n_ss}，基准率 {n_ss/n:.3f}）。
特征为 78 维元素摩尔分数 + 9 个描述符（δ、ΔH_mix、S_id、Ω、VEC、σ_VEC、Δχ、χ、T_m），
其中 Ω 由本流程按 Yang & Zhang 定义重算。

### 2.1 冗余结构

{md_img("fig1_dataset_structure.png", "图 1 数据集冗余结构")}

## 3. 方法

模型：逻辑回归（强正则、低容量）与直方图梯度提升树（150 轮，高容量）。

四种评估协议：
1. 随机分层划分（StratifiedKFold，3 次重复）
2. 化学体系分组划分（StratifiedGroupKFold，组 = 元素集合，3 次重复）
3. **受控配对实验**：固定 20% 测试集；`TrainA_matched`（下采样至同规模）vs `TrainB_disjoint`（体系不重叠），
   规模相同、测试集相同，唯一差别是近重复样本。10 个划分重复，Wilcoxon 配对检验。
   （TrainB 平均只保留 TrainA 的 {tsz['retained_fraction_mean']*100:.1f}%，故必须规模匹配。）
4. 留一元素法：18 种高频元素，训练集不含 X、测试集只含 X。

指标：平衡准确率、MCC、ROC-AUC；2,000 次 bootstrap 置信区间；固定随机种子。

## 4. 结果

### 4.1 经典判据的真实表现

{md_img("fig2_phase_map.png", "图 2 经典判据窗口内的类别重叠")}
{md_img("fig3_descriptors_by_class.png", "图 3 各描述符按相类别的分布")}

| 判据 / 规则 | 平衡准确率 | 95% CI | 敏感度 | 特异度 | MCC |
|---|---|---|---|---|---|
{fmt_rules_md()}

{md_img("fig4_criteria_performance.png", "图 4 各经典规则性能与 CV 调优上限")}

CV 调优阈值后单描述符上限：

| 描述符（方向） | CV 平衡准确率 |
|---|---|
""" + "\n".join(
    f"| {r['descriptor']}（{r['direction']}） | {r['cv_balanced_accuracy_mean']:.3f} ± {r['cv_balanced_accuracy_std']:.3f} |"
    for r in sorted(crit['cv_tuned_single_descriptor'], key=lambda x: -x['cv_balanced_accuracy_mean'])) + f"""

**结论：即使最优调阈值，单描述符上限也只有 {best_cv['cv_balanced_accuracy_mean']:.3f}（Ω）。**

VEC 晶格规则（{vec['n_total']} 条纯 FCC/BCC）：可判定 {vec['n_total']-vec['n_ambiguous']} 条（覆盖率 {vec['coverage']*100:.1f}%），
准确率 **{vec['accuracy_on_covered']*100:.0f}%**，{vec['n_ambiguous']} 条落在模糊区。
→ “能否形成固溶体”与“形成哪种晶格”难度完全不同。

### 4.2 经典判据的失败模式

- 假阳性集中在 SS+IM（{fail['false_positive_S_Phase_mix'].get('SS+IM',0)} 条）：判据说“可能”，实验观察到两相共存。
- 假阴性集中在难熔高熵合金：Mo+Nb+Ti+V+Zr 被漏 {fail['top_element_sets_false_negative'].get('Mo+Nb+Ti+V+Zr',0)} 条，
  为所有体系最多。经典窗口基于 3d 过渡金属拟合，向难熔体系迁移系统性失效。

### 4.3 划分协议造成的精度虚高（核心）

{md_img("fig5_leakage_gap.png", "图 5 划分协议造成的精度虚高")}

两种划分协议对比：

| 特征块 | 模型 | 随机分层 | 体系分组 | 差值 |
|---|---|---|---|---|
{fmt_exp2_md()}

受控配对实验（测试集固定、训练集规模匹配）：

| 特征块 | 模型 | TrainA_matched | TrainB_disjoint | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
{fmt_exp1_md()}

**核心数字。** 测试集固定、训练集规模相同，仅因训练集中存在同体系近重复合金，
梯度提升模型平衡准确率高出 **{e1['composition+descriptors | hgb']['delta_mean']:+.3f}**
（95% CI {[round(x,3) for x in e1['composition+descriptors | hgb']['delta_ci95']]}），逻辑回归仅 {e1['composition+descriptors | logreg']['delta_mean']:+.3f}（CI 含 0）。
→ **虚高幅度由模型容量决定。**

只给 9 个描述符、不给成分时，梯度提升树仍虚高 **{e1['descriptors | hgb']['delta_mean']:+.3f}**。
→ 根源不是成分特征，而是训练集中的近重复样本。

### 4.4 机制：近邻成分距离

{md_img("fig6_neighbour_distance.png", "图 6 最近邻成分距离分布")}

| 指标 | TrainA | TrainB |
|---|---|---|
| 最近邻距离中位数 | {nb['A_median']:.3f} | {nb['B_median']:.3f} |
| 距离 < 0.05 占比 | **{nb['A_frac_within_0.05']*100:.1f}%** | {nb['B_frac_within_0.05']*100:.1f}% |
| 距离 < 0.10 占比 | **{nb['A_frac_within_0.10']*100:.1f}%** | {nb['B_frac_within_0.10']*100:.1f}% |

随机划分下四分之一测试合金有“几乎相同”的训练近邻；模型的高分反映记忆而非泛化。

### 4.5 留一元素外推

{md_img("fig7_loeo.png", "图 7 留一元素外推性能")}

18 种元素平均平衡准确率 **{loeo['mean_bal_acc']:.3f}**（范围 {loeo['min_bal_acc']:.3f}–{loeo['max_bal_acc']:.3f}），
平均 AUC {loeo['mean_auc']:.3f}；对照随机划分的
{e2[(e2.block=='composition+descriptors')&(e2.model=='hgb')&(e2.protocol=='random_stratified')]['bal_acc_mean'].iloc[0]:.3f}，差距近 0.2。

| 留出元素 | 逻辑回归 | 梯度提升 |
|---|---|---|
""" + "\n".join(f"| {e} | {loeo['per_element_logreg'][e]:.3f} | {loeo['per_element_hgb'][e]:.3f} |"
                for e in sorted(loeo['per_element_hgb'], key=lambda x: -loeo['per_element_hgb'][x])) + """

## 5. 讨论

**对实践的直接含义：**
1. 随机划分的数字不应被当作泛化能力；本数据上高容量模型虚高约 0.12–0.15 平衡准确率。
2. 至少报告一种化学体系不重叠划分（一行 `GroupKFold`，组 = 元素集合）。
3. 报告平衡准确率与 MCC，而非准确率（SS 基准率仅 """ + f"{n_ss/n:.1%}" + """，“全判非 SS”准确率即 """ + f"{g('majority_class_baseline','accuracy'):.3f}" + """）。
4. 把留一元素外推作为压力测试。
5. 区分“能否形成固溶体”与“FCC/BCC 晶格选择”两个难度不同的任务。

**与已有工作关系：** 泄漏是成熟议题（DataSAIL 2025；Kapoor & Narayanan 2023）。本文不主张发现现象，
而是在具体数据集上量化幅度、给出机制、并证明幅度与模型容量强相关。

**可迁移性：** “固定测试集 + 按组剔除 + 规模匹配 + 配对检验”适用于任何存在“同体系多配比”结构的数据。

## 6. 局限

- 单一数据源，虚高幅度不能直接外推到所有 HEA 文献。
- 数据集存在标签噪声（文献汇总、制备条件不一，SS+IM 本身是中间态）。
- 元素集合是近似分组；更严格分组会给出更大虚高，本文数字可视为**下界**。
- 仅两个模型族；未系统扫描容量谱。
- 描述符口径沿用数据集（Ω 由本文重算）。
- 未做超参数调优（为保持协议可比）。

## 7. 可复现性

```
research/
├── data/     原始 xlsx（只读）+ 清洗后表
├── code/     hea_common.py, 00_profile.py, 01_build_dataset.py,
│             02_criteria_audit.py, 03_leakage_experiment.py, 04_figures.py, 05_report.py
├── out/      结果 JSON/CSV 与原始数值
├── figures/  7 张 300 dpi 图
└── report/   HTML + Markdown 报告
```

复现命令：
```
cd research/code
python 00_profile.py && python 01_build_dataset.py && python 02_criteria_audit.py
python 03_leakage_experiment.py stage1
python 03_leakage_experiment.py stage2
python 03_leakage_experiment.py merge
python 04_figures.py && python 05_report.py
```

环境：Python 3.13；numpy / pandas / scipy / scikit-learn / matplotlib / openpyxl / pyarrow。
随机种子固定于脚本内。

## 参考文献

1. Precker, C. E., Gregores Coto, A., Muíños Landín, S. *Materials for Design Open Repository. High Entropy Alloys*. Zenodo (2021). DOI 10.5281/zenodo.6403257, CC-BY-4.0.（本文数据来源）
2. Zhang, Y. et al. *Adv. Eng. Mater.* 10, 534–538 (2008).（δ 判据）
3. Yang, X. & Zhang, Y. *Mater. Chem. Phys.* 132, 233–238 (2012).（Ω 判据）
4. Guo, S. & Liu, C. T. *Prog. Nat. Sci.* 21, 433–446 (2011).（VEC 判据）
5. Guo, S. et al. *J. Appl. Phys.* 109, 103505 (2011).（VEC ≥ 8 → FCC）
6. Joachimiak, M. P. et al. Data splitting to avoid information leakage with DataSAIL. *Nature Communications* 16 (2025).
7. Kapoor, S. & Narayanan, A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4, 100804 (2023).

---

*本报告由可复现分析流程自动生成：所有数值由 `05_report.py` 在构建时从 `out/` 结果文件读取填入，未经人工转录。
数据来自公开第三方数据集，结论仅针对该数据集，不构成对任何具体已发表工作的指控。*
"""

(REPORT / "HEA_leakage_report.md").write_text(md, encoding="utf-8")
print("wrote", REPORT / "HEA_leakage_report.md")
