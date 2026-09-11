# Release Playbook — getting this audit seen (without overclaiming)

**The honest premise:** significance comes from *being honest + reproducible + reusable*.
Attention comes from *packaging + the right community*. **AI cannot guarantee virality, and I
will not post on your behalf (no credentials).** The repo is the durable asset; attention is a
bonus. Below is the distribution plan.

---

## 1. Channels (you already chose: both)

| bucket | where | format | best hook |
|--------|-------|--------|-----------|
| **Chinese mass** | 知乎 (column/answer) | long-form repost of `article_zh.md` | "我用一次重复实验，戳破了高熵合金 ML 论文里一个被忽视的坑" |
| | B站 | 8–12 min screencast: run `reproduce.py` live, show Δ=+0.151 | "一个高中生跑通的可复现科研：随机切分为什么高估了 AI" |
| | 微信公众号 | push to your network + school | same as 知乎, trimmed to ~1500 字 |
| | 小红书 (optional) | 图文卡片, 3 张图讲清"近亲偷看" | "你的 90% 准确率，可能偷看了考卷" |
| **English / academic** | GitHub repo | the durable asset; README + `reproduce.py` | "one-command reproduction of a leakage audit" |
| | arXiv preprint | `article_en.md` as a 2-page note (cs.LG / cond-mat.mtrl-sci) | concrete protocol + reproducible number |
| | X / Twitter | thread: hook + figure + repo link | "Random splits overstate HEA ML accuracy by ~15 pts. Reproduce it yourself 👇" |
| | Reddit | r/MachineLearning, r/MaterialsScience, r/learnedtutorials | "Showed near-duplicate leakage in HEA phase ML — code + data公开" |

---

## 2. Per-platform copy notes

- **知乎/B站:** lead with the *concrete* 0.151 number and the "考试偷看" analogy — it travels.
  Put the repo link at top and bottom. Never say " breakthrough"; say "可复现的方法学审计".
- **arXiv:** do **not** claim novelty of method; frame as a *reproducibility note*. Category
  `cs.LG` primary, `cond-mat.mtrl-sci` cross. Keep it ≤ 4 pages.
- **X thread:** 1 hook + 1 figure (fig8) + 1 "how to reproduce" + 1 "limitations/AI-disclosure".
  Tag the dataset authors and a couple of ML-for-science accounts.
- **Reddit:** post the figure + a one-paragraph plain-English summary, link the repo, and
  *actively answer* "why not just use group/LOSO split?" (answer: this isolates redundancy
  *on top of* a held-out split; see paper §4.4). Engagement, not broadcast.

---

## 3. Pre-publish conditions (check before posting)

- [ ] GitHub repo created, `reproduce.py` runs green on a clean clone (CI or local proof).
- [ ] Author / affiliation / email / repo URL filled in README placeholders.
- [ ] AI-use disclosure present in README + both articles (already written).
- [ ] Data downloaded + Zenodo 6403257 (CC-BY-4.0) attribution included.
- [ ] If submitted to a student journal (e.g. JEI): **mentor co-authorship / sign-off** secured
      before any public preprint that names the institution.
- [ ] Figures (`fig8_exclusion_sweep.png`, `fig9_variance_split.png`) included in posts.

---

## 4. Honest guardrails (do not cross)

- No "我证明了 / breakthrough / 实验已验证用户体验" language. Say *audited / reproduced / showed*.
- Do not claim any specific paper is fraudulent. Claim: *systematic over-estimation under random splits*.
- No fabricated engagement, no bought likes, no astroturfing.
- If a critic finds a bug, fix it in the repo publicly. The audit's credibility IS the product.

---

## 5. Suggested order & timeline

1. **Day 0 — anchor asset:** push GitHub repo (README + reproduce.py + data note). This exists
   independent of any single post's reach.
2. **Day 1 — English technical:** X thread + Reddit (r/MachineLearning). Technical audiences
   validate first; their questions sharpen later Chinese posts.
3. **Day 3 — Chinese long-form:** 知乎 article + 公众号. Link the live repo.
4. **Day 5 — video:** B站 screencast running `reproduce.py` end-to-end (highest trust, lowest
   reach-effort ratio for a student audience).
5. **Week 2 — preprint:** arXiv after the above surfaced no fatal critique.

---

## 6. Metrics to watch (and not over-read)

- Repo: clones / `reproduce.py` issues opened (people *running* it = real signal).
- Posts: saves & genuine questions > likes. A good critique in comments is a win, not a loss.
- **Reality check:** attention is not guaranteed. A clean, citable, reproducible repo has value
  even at zero views — it is a real asset for university applications and future work.

---

*Build the asset first, broadcast second, and let the reproducibility speak.*
