"""
build_public_site.py — render the public-release markdown docs into styled HTML pages.

Why: the deployed site is served by a plain static server, which returns the raw
.md/.py files as text/markdown and text/x-python — so the footer "Resources"
links showed raw source or triggered a download instead of a readable page.

This script produces, inside research/public-release/:
    readme.html, article_zh.html, article_en.html, code.html
so those links open proper, styled pages. Re-run after editing any source doc.

Usage:
    <venv-python> build_public_site.py
"""
from __future__ import annotations

import html
from pathlib import Path

import markdown
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REL = HERE.parent / "public-release"

F_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
F_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
F_BLACK = "/System/Library/Fonts/Supplemental/Arial Black.ttf"


def make_og() -> None:
    """Render a 1200x630 Open Graph share card (social preview image)."""
    W, H = 1200, 630
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    top, bot = (15, 23, 42), (49, 46, 129)
    for y in range(H):
        t = y / H
        d.line([(0, y), (W, y)],
               fill=tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)))
    f_badge = ImageFont.truetype(F_BOLD, 20)
    f_title = ImageFont.truetype(F_BOLD, 52)
    f_num = ImageFont.truetype(F_BLACK, 128)
    f_sub = ImageFont.truetype(F_REG, 27)
    f_small = ImageFont.truetype(F_REG, 24)
    d.rounded_rectangle([72, 60, 72 + 424, 104], radius=22, fill=(79, 70, 229))
    d.text((98, 69), "REPRODUCIBLE RESEARCH AUDIT", font=f_badge, fill=(224, 231, 255))
    for i, line in enumerate(["Random splits quietly inflate",
                              "ML accuracy in high-entropy-alloy",
                              "phase prediction."]):
        d.text((72, 150 + i * 60), line, font=f_title, fill=(255, 255, 255))
    d.text((68, 348), "+0.151", font=f_num, fill=(252, 165, 165))
    d.text((74, 500), "pure near-duplicate leakage gap", font=f_sub, fill=(226, 232, 240))
    d.text((74, 536), "76% of the accuracy drop is redundancy - not data volume",
           font=f_sub, fill=(148, 163, 184))
    d.rounded_rectangle([72, H - 56, 96, H - 32], radius=6, fill=(79, 70, 229))
    d.text((106, H - 55), "HEA Leakage Audit  -  Zenodo 6403257 (CC-BY-4.0)",
           font=f_small, fill=(165, 180, 252))
    out = REL / "assets" / "og.png"
    img.save(out, optimize=True)
    print(f"wrote {out.relative_to(REL)}")

CSS = """
:root{--ink:#0f172a;--muted:#475569;--border:#e2e8f0;--primary:#4f46e5;--surface:#f8fafc;--code:#0f172a}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif;color:var(--ink);line-height:1.7;background:#fff}
a{color:var(--primary);text-decoration:none}a:hover{text-decoration:underline}
.topbar{position:sticky;top:0;background:rgba(255,255,255,.9);backdrop-filter:saturate(180%) blur(10px);border-bottom:1px solid var(--border);z-index:10}
.tb-in{max-width:860px;margin:0 auto;padding:0 24px;height:56px;display:flex;align-items:center;justify-content:space-between}
.brand{display:flex;align-items:center;gap:9px;font-weight:700}
.brand .m{width:22px;height:22px;border-radius:6px;background:var(--primary);display:inline-block}
.back{font-size:14px;color:var(--muted)}
.doc{max-width:860px;margin:0 auto;padding:44px 24px 80px}
.doc h1{font-size:30px;line-height:1.25;margin:0 0 18px;letter-spacing:-.01em}
.doc h2{font-size:22px;margin:38px 0 12px;padding-top:8px;border-top:1px solid var(--border)}
.doc h3{font-size:17px;margin:24px 0 8px}
.doc p,.doc li{color:#1e293b}
.doc code{background:#f1f5f9;border:1px solid var(--border);border-radius:6px;padding:1px 6px;font-size:.9em;font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace}
.doc pre{background:var(--code);color:#e2e8f0;border-radius:12px;padding:18px 20px;overflow:auto}
.doc pre code{background:none;border:none;color:inherit;padding:0}
.doc table{width:100%;border-collapse:collapse;margin:18px 0;font-size:14.5px}
.doc th,.doc td{border:1px solid var(--border);padding:9px 12px;text-align:left}
.doc th{background:var(--surface)}
.doc blockquote{margin:18px 0;padding:12px 18px;border-left:4px solid var(--primary);background:var(--surface);color:var(--muted);border-radius:0 10px 10px 0}
.doc hr{border:none;border-top:1px solid var(--border);margin:34px 0}
.doc img{max-width:100%}
.dl{display:inline-block;margin:0 0 16px;font-weight:600}
footer{border-top:1px solid var(--border);background:var(--surface);color:var(--muted);font-size:13px}
footer .f-in{max-width:860px;margin:0 auto;padding:22px 24px}
"""

BRAND_SVG = ('<span class="m"></span>')

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body>
<div class="topbar"><div class="tb-in">
  <a class="brand" href="index.html">__BRAND__ HEA Leakage Audit</a>
  <a class="back" href="index.html">&larr; 返回首页 / Back to overview</a>
</div></div>
<main class="doc">__BODY__</main>
<footer><div class="f-in">
  Reproducible research release &middot; Data: Zenodo 6403257 (CC-BY-4.0) &middot; Code: MIT &middot;
  AI-assisted, human-verified. &middot; <a href="index.html">Overview</a>
</div></footer>
</body>
</html>
"""

LINK_REWRITE = [
    ("README.md", "readme.html"),
    ("article_zh.md", "article_zh.html"),
    ("article_en.md", "article_en.html"),
    ("reproduce.py", "code.html"),
]


def page(title: str, body: str) -> str:
    return (TEMPLATE.replace("__TITLE__", title)
                    .replace("__CSS__", CSS)
                    .replace("__BRAND__", BRAND_SVG)
                    .replace("__BODY__", body))


def render_md(src: Path, title: str) -> str:
    body = markdown.markdown(
        src.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    for a, b in LINK_REWRITE:
        body = body.replace(f'href="{a}"', f'href="{b}"')
    return page(title, body)


def render_code(src: Path, title: str) -> str:
    code = html.escape(src.read_text(encoding="utf-8"))
    body = (f'<h1>{html.escape(src.name)}</h1>'
            f'<a class="dl" href="{src.name}" download>Download {html.escape(src.name)} &darr;</a>'
            f'<pre><code>{code}</code></pre>')
    return page(title, body)


def main() -> None:
    jobs = [
        ("README.md", "readme.html", "README — HEA Leakage Audit"),
        ("article_zh.md", "article_zh.html", "中文长文 — 高熵合金近重复泄漏审计"),
        ("article_en.md", "article_en.html", "English note — Near-Duplicate Leakage Audit"),
    ]
    for src_name, out_name, title in jobs:
        src = REL / src_name
        if not src.exists():
            print(f"skip (missing): {src_name}")
            continue
        (REL / out_name).write_text(render_md(src, title), encoding="utf-8")
        print(f"wrote {out_name}  <-  {src_name}")

    code_src = REL / "reproduce.py"
    if code_src.exists():
        (REL / "code.html").write_text(render_code(code_src, "reproduce.py — HEA Leakage Audit"),
                                       encoding="utf-8")
        print("wrote code.html  <-  reproduce.py")

    make_og()


if __name__ == "__main__":
    main()
