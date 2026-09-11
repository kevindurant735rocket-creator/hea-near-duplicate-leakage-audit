# HEA Near-Duplicate Leakage Audit

> **A *random* train/test split quietly inflates machine-learning accuracy on
> high-entropy-alloy (HEA) phase prediction — and most of that inflation is
> near-duplicate redundancy, not data volume.**

**🌐 Live overview page:** https://7564e6f7b2a54736889ff6b986ed445d.sg2.agentos-app.run

This repository is a small, **honest and fully reproducible** methodological audit. It is
*not* a breakthrough, and it does **not** allege misconduct in any specific paper.

## Headline result (gradient-boosted trees, τ = 0.35)

| quantity | value |
|---|---|
| excluded(τ) balanced accuracy | 0.685 |
| size-matched control | 0.836 |
| **pure near-duplicate leakage gap Δ** | **+0.151** |
| total drop τ=0→0.35 | 0.198 = ~24% volume + **~76% redundancy** |

KNN Δ = +0.134; logistic regression Δ = +0.032 (not significant). Every number regenerates
deterministically.

## Reproduce in one command

```bash
pip install -r public-release/requirements.txt
python public-release/reproduce.py
# -> HEADLINE CHECK hgb tau=0.35  ref +0.1510  repro +0.1510  dev 0.00e+00  [ok]
# -> RESULT: PASS
```

The raw dataset (Zenodo 6403257, CC-BY-4.0) is included at `data/heas_v2.xlsx`.

## Layout

```
code/            analysis + figure + report scripts (00..09, hea_common, build_public_site)
data/            raw dataset (CC-BY-4.0) + derived tables
out/             result JSON/CSV — the single source of truth for every number
figures/         figures (300 dpi)
paper/           paper_EN / paper_CN (.md / .tex / .pdf)
report/          HTML + Markdown reports
public-release/  public packaging: articles (zh/en), README, reproduce.py, landing page, OG card
```

## Read the write-ups

- 中文长文 — `public-release/article_zh.md`
- English note — `public-release/article_en.md`
- One-pager — `public-release/executive_summary.md`
- Site-facing README — `public-release/README.md`

## Honest limitations

Single public dataset; one task (solid-solution classification); undergraduate-level
*methodological* audit; it demonstrates systematic over-estimation under random splits — it
does **not** claim any specific paper is wrong or fraudulent.

## Data, licence, disclosure

- **Data:** Materials for Design Open Repository, Zenodo
  [10.5281/zenodo.6403257](https://doi.org/10.5281/zenodo.6403257), CC-BY-4.0.
- **Code:** MIT (`LICENSE`).
- **AI-use disclosure:** AI-assisted (drafting, code scaffolding, analysis support); the
  design, the claims, and responsibility for correctness are human. No result was fabricated —
  every number traces to `out/*.json`, which `reproduce.py` regenerates from raw data.
