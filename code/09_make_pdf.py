"""
09_make_pdf.py — render the manuscripts to print-ready HTML (figures inlined as
base64) so that headless Chrome can turn them into a submission PDF.

No LaTeX is installed on this machine, so the PDF path is Markdown -> styled
HTML -> Chrome --print-to-pdf. Run this first, then the chrome command printed
at the end.

    python 09_make_pdf.py
"""
import base64
import mimetypes
import re
from pathlib import Path

import markdown

PAPER = Path(__file__).resolve().parents[1] / "paper"

CSS = """
@page { size: A4; margin: 19mm 17mm 20mm 17mm; }
* { box-sizing: border-box; }
body { font-family: %(SERIF)s; font-size: 10.4pt; line-height: 1.52;
       color: #111; margin: 0; text-align: justify; hyphens: auto; }
h1 { font-size: 16.5pt; line-height: 1.3; text-align: center; margin: 0 0 10px 0; }
h1 + p { text-align: center; font-size: 9.6pt; color: #333; margin: 2px 0; }
h2 { font-size: 12.4pt; margin: 20px 0 7px 0; padding-bottom: 3px;
     border-bottom: 0.6pt solid #bbb; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 14px 0 5px 0; page-break-after: avoid; }
p { margin: 6px 0; }
strong { font-weight: 700; }
hr { border: 0; border-top: 0.6pt solid #ccc; margin: 14px 0; }
table { border-collapse: collapse; width: 100%%; font-size: 8.5pt;
        margin: 10px 0 12px 0; page-break-inside: avoid; }
th { border-top: 1.1pt solid #222; border-bottom: 0.6pt solid #222;
     padding: 4px 5px; text-align: left; font-weight: 700; }
td { border-bottom: 0.4pt solid #ddd; padding: 3.4px 5px; vertical-align: top; }
tr:last-child td { border-bottom: 1.1pt solid #222; }
img { max-width: 100%%; display: block; margin: 10px auto 4px auto; }
p > em:only-child { display: block; font-size: 8.8pt; color: #444;
                    text-align: center; margin: -2px 0 14px 0; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 8.8pt;
       background: #f4f4f4; padding: 0 2px; border-radius: 2px; }
pre { background: #f7f7f7; border: 0.5pt solid #ddd; padding: 7px 9px;
      font-size: 8.4pt; line-height: 1.35; overflow-x: hidden;
      page-break-inside: avoid; white-space: pre-wrap; }
pre code { background: none; padding: 0; }
blockquote { margin: 8px 0; padding-left: 10px; border-left: 2pt solid #ccc;
             color: #333; }
ul, ol { margin: 6px 0 6px 0; padding-left: 20px; }
li { margin: 3px 0; }
a { color: #111; text-decoration: none; }
h2, h3 { break-after: avoid; }
"""

SERIF = {
    "EN": '"Times New Roman", "Liberation Serif", Georgia, serif',
    "CN": '"Songti SC", "STSong", "Times New Roman", serif',
}


def inline_images(html: str, base: Path) -> str:
    def repl(m):
        src = m.group(1)
        p = base / src
        if not p.exists():
            print("  ! missing figure", src)
            return m.group(0)
        mime = mimetypes.guess_type(p.name)[0] or "image/png"
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'src="data:{mime};base64,{b64}"'
    return re.sub(r'src="([^"]+)"', repl, html)


def build(lang: str, src_md: Path, out_html: Path) -> None:
    md = src_md.read_text()
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "attr_list"])
    body = inline_images(body, PAPER)
    css = CSS % {"SERIF": SERIF[lang]}
    html = (f"<!DOCTYPE html>\n<html lang=\"{'en' if lang == 'EN' else 'zh-CN'}\">\n"
            f"<head><meta charset=\"utf-8\"><title>{src_md.stem}</title>"
            f"<style>{css}</style></head>\n<body>\n{body}\n</body>\n</html>\n")
    out_html.write_text(html, encoding="utf-8")
    print(f"wrote {out_html.name}  ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    build("EN", PAPER / "paper_EN.md", PAPER / "paper_EN_print.html")
    build("CN", PAPER / "paper_CN.md", PAPER / "paper_CN_print.html")
    print("\nnow render with headless Chrome:")
    for n in ("paper_EN", "paper_CN"):
        print(f'  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" '
              f'--headless --disable-gpu --no-pdf-header-footer '
              f'--virtual-time-budget=15000 '
              f'--print-to-pdf="{PAPER / (n + ".pdf")}" '
              f'"file://{PAPER / (n + "_print.html")}"')
