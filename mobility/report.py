"""Standalone HTML report (open in a browser, use Print → Save as PDF).

Pure string building: figures are passed in as ready-made HTML fragments
(for example ``plotly_fig.to_html(full_html=False, include_plotlyjs="cdn")``).
"""

import html
from datetime import datetime

import pandas as pd

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
     margin:0;background:#f5f7fb;color:#1f2937;line-height:1.5}
.wrap{max-width:1000px;margin:0 auto;padding:32px 24px 64px}
header{background:#111827;color:#fff;padding:28px 24px}
header .wrap{padding:0 24px}
h1{margin:0 0 4px;font-size:28px} h2{margin:36px 0 8px;font-size:21px;
  border-bottom:2px solid #e5e7eb;padding-bottom:6px}
.sub{color:#9ca3af;font-size:14px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px 20px;
      margin:14px 0}
ul{padding-left:22px} li{margin:6px 0}
.metrics{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.metric{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:12px 16px;
        min-width:140px}
.metric b{display:block;font-size:22px} .metric span{color:#6b7280;font-size:13px}
table.tbl{border-collapse:collapse;width:100%;font-size:13px;background:#fff}
table.tbl th,table.tbl td{border:1px solid #e5e7eb;padding:6px 8px;text-align:right}
table.tbl th{background:#f3f4f6;text-align:center}
table.tbl td:first-child,table.tbl th:first-child{text-align:left}
.caption{color:#6b7280;font-size:13px;margin:12px 0 4px}
.note{color:#6b7280;font-size:12px}
@media print{body{background:#fff}.card,.metric{break-inside:avoid}
  header{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""


def _esc(value):
    return html.escape(str(value))


def table_html(df, max_rows=60):
    if df is None or len(df) == 0:
        return "<p class='note'>No data.</p>"
    shown = df.head(max_rows)
    out = shown.to_html(index=False, escape=True, classes="tbl", border=0,
                        float_format=lambda v: f"{v:,.2f}", na_rep="")
    if len(df) > max_rows:
        out += f"<p class='note'>Showing the first {max_rows} of {len(df):,} rows.</p>"
    return out


def metric_html(label, value):
    return f"<div class='metric'><b>{_esc(value)}</b><span>{_esc(label)}</span></div>"


def build_report_html(title, subtitle="", sections=None, footer=""):
    """Build the HTML document.

    ``sections`` is a list of dicts with optional keys:
      title, paragraphs[list[str]], bullets[list[str]], metrics[list[(label, value)]],
      tables[list[(caption, DataFrame)]], figures[list[(caption, html_fragment)]]
    """
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{_esc(title)}</title><style>{CSS}</style></head><body>",
        f"<header><div class='wrap'><h1>{_esc(title)}</h1>",
        f"<div class='sub'>{_esc(subtitle)}</div></div></header><div class='wrap'>",
    ]

    for section in sections or []:
        if section.get("title"):
            parts.append(f"<h2>{_esc(section['title'])}</h2>")
        for text in section.get("paragraphs", []):
            parts.append(f"<p>{_esc(text)}</p>")
        if section.get("metrics"):
            parts.append("<div class='metrics'>" + "".join(
                metric_html(label, value) for label, value in section["metrics"]) + "</div>")
        if section.get("bullets"):
            parts.append("<div class='card'><ul>" + "".join(
                f"<li>{_esc(b)}</li>" for b in section["bullets"]) + "</ul></div>")
        for caption, frame in section.get("tables", []):
            parts.append(f"<div class='caption'>{_esc(caption)}</div>{table_html(frame)}")
        for caption, fragment in section.get("figures", []):
            parts.append(f"<div class='caption'>{_esc(caption)}</div>"
                         f"<div class='card'>{fragment}</div>")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts.append(f"<p class='note'>{_esc(footer)} Generated {stamp}.</p></div></body></html>")
    return "".join(parts)


def sanitize_for_filename(name):
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))
    return keep.strip("_")[:60] or "report"
