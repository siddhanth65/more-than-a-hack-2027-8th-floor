#!/usr/bin/env python
"""Build docs/technical-document.pdf from docs/technical-document.md.

Markdown -> HTML (mistune) -> PDF (headless Chrome or Edge, whatever is installed),
styled by docs/pdf/style.css. Only system fonts are used; no network access is needed.

    python scripts/build_pdf.py            # writes docs/technical-document.pdf
    python scripts/build_pdf.py --out X.pdf --browser "C:\\path\\to\\chrome.exe"

Requires: mistune (pip install mistune) and Chrome 131+ or Edge (for page-number margin boxes).
"""
import argparse, html, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]
NUM = re.compile(r"^[\s*(\[]*[-+\u2212]?\d[\d.,]*\s?(%|s|min)?([\s/,)\]*-]+[-+\u2212]?\d[\d.,]*\s?(%|s)?)*[\s*)\]]*$")


def find_browser(explicit):
    if explicit:
        return explicit
    for c in CANDIDATES:
        p = c if os.path.isabs(c) else shutil.which(c)
        if p and os.path.exists(p):
            return p
    sys.exit("no Chrome/Edge found; pass --browser")


def table_post(m):
    tbl = m.group(0)
    ncols = len(re.findall(r"<th", tbl))
    text = len(re.sub(r"<[^>]+>", "", tbl))
    cls = "wide" if (ncols >= 4 or text > 2500) else ""
    # right-align a column only when every body cell in it is numeric
    rows = re.findall(r"<tr>(.*?)</tr>", tbl, flags=re.S)
    body = [re.findall(r"<td[^>]*>(.*?)</td>", r, flags=re.S) for r in rows if "<td" in r]
    plain = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s)).strip()
    numcols = set()
    for j in range(ncols):
        vals = [plain(r[j]) for r in body if len(r) > j]
        if vals and all(v and NUM.match(v) for v in vals):
            numcols.add(j)
    def fix_row(m):
        r = m.group(0); j = [-1]
        def c(mm):
            j[0] += 1
            if j[0] in numcols:
                return f'<{mm.group(1)} class="num"{mm.group(2)}>'
            return mm.group(0)
        return re.sub(r"<(t[dh])([^>]*)>", c, r)
    if numcols:
        tbl = re.sub(r"<tr>.*?</tr>", fix_row, tbl, flags=re.S)
    tbl = tbl.replace("<table>", f'<table class="{cls}">' if cls else "<table>", 1)
    short = " short" if tbl.count("<tr>") <= 7 and text < 1800 else ""
    return f'<div class="tw{short}">{tbl}</div>'


def render(md_text):
    md = mistune.create_markdown(escape=False, plugins=["table", "url"])
    body = md(md_text)
    body = re.sub(r"<table>.*?</table>", table_post, body, flags=re.S)
    body = re.sub(r"<p><em>(Table \d+\..*?)</em></p>", r'<p class="caption"><em>\1</em></p>', body, flags=re.S)
    body = body.replace("<blockquote>", '<blockquote class="glance">', 1)
    # title block = everything before the first h2
    k = body.index("<h2")
    front = body[:k]
    front = front.replace("<p><strong>", '<p class="subtitle"><strong>', 1)
    front, _, glance = front.partition('<blockquote class="glance">')
    body = f'<div class="title">{front}</div><blockquote class="glance">{glance}{body[k:]}'
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(ROOT / "docs" / "technical-document.md"))
    ap.add_argument("--css", default=str(ROOT / "docs" / "pdf" / "style.css"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "technical-document.pdf"))
    ap.add_argument("--browser")
    ap.add_argument("--keep-html")
    a = ap.parse_args()
    md_text = Path(a.md).read_text(encoding="utf-8")
    css = Path(a.css).read_text(encoding="utf-8")
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           "<title>More Than A Hack 2027 \u2014 team 8th floor: technical document</title>"
           f"<style>{css}</style></head><body>{render(md_text)}</body></html>")
    with tempfile.TemporaryDirectory() as td:
        h = Path(td) / "doc.html"
        h.write_text(doc, encoding="utf-8")
        if a.keep_html:
            Path(a.keep_html).write_text(doc, encoding="utf-8")
        out = Path(a.out).resolve()
        cmd = [find_browser(a.browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
               f"--user-data-dir={Path(td) / 'profile'}", "--no-first-run",
               f"--print-to-pdf={out}", h.as_uri()]
        subprocess.run(cmd, check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
