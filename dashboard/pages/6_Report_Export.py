"""Report export page — download mermaid charts as Markdown or HTML."""

import html
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from data import clear_all_caches, fetch_mermaid_reports


# ── Helpers ──────────────────────────────────────────────────────────


def _slugify(text: str) -> str:
    """Create an anchor-friendly slug from text."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"[\s_]+", "-", slug)


def _dedupe_anchors(report_list: list[dict]) -> list[tuple[str, str]]:
    """Return (title, unique_anchor) pairs, appending a suffix on duplicates."""
    seen: dict[str, int] = {}
    results: list[tuple[str, str]] = []
    for r in report_list:
        title = f"{r['doc_name']} — {r['institution']}"
        base = _slugify(title)
        count = seen.get(base, 0)
        seen[base] = count + 1
        anchor = base if count == 0 else f"{base}-{count}"
        results.append((title, anchor))
    return results


def _safe_md(text: str) -> str:
    """Strip newlines to prevent structure injection in Markdown output."""
    return text.replace("\n", " ").replace("\r", "")


def _build_markdown(report_list: list[dict]) -> str:
    """Build a Markdown document with TOC and one section per report."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title_anchors = _dedupe_anchors(report_list)

    lines = [
        "# Mermaid Charts Export",
        f"Generated: {now}",
        "",
        "## Table of Contents",
    ]

    for i, (title, anchor) in enumerate(title_anchors, 1):
        lines.append(f"{i}. [{_safe_md(title)}](#{anchor})")

    lines.append("")
    lines.append("---")
    lines.append("")

    for r, (title, _anchor) in zip(report_list, title_anchors):
        lines.append(f"## {_safe_md(title)}")
        lines.append("")
        if r["model_name"]:
            lines.append(f"**Model:** {_safe_md(r['model_name'])}")
            lines.append("")
        if r["steps_summary"]:
            lines.append(f"**Summary:** {_safe_md(r['steps_summary'])}")
            lines.append("")
        lines.append("```mermaid")
        lines.append(r["mermaid_diagram"].strip())
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _build_html(report_list: list[dict]) -> str:
    """Build a self-contained HTML file with mermaid.js rendering."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title_anchors = _dedupe_anchors(report_list)

    toc_items = []
    sections = []

    for i, (r, (title, anchor)) in enumerate(
        zip(report_list, title_anchors), 1
    ):
        escaped_title = html.escape(title)
        toc_items.append(
            f'      <li><a href="#{html.escape(anchor)}">{escaped_title}</a></li>'
        )

        meta_lines = []
        if r["model_name"]:
            meta_lines.append(
                f"<p><strong>Model:</strong> {html.escape(r['model_name'])}</p>"
            )
        if r["steps_summary"]:
            meta_lines.append(
                f"<p><strong>Summary:</strong> {html.escape(r['steps_summary'])}</p>"
            )
        meta_html = "\n        ".join(meta_lines)

        # Mermaid diagrams must NOT be HTML-escaped — mermaid.js parses raw text
        diagram = r["mermaid_diagram"].strip()

        sections.append(f"""    <section id="{html.escape(anchor)}">
      <h2>{i}. {escaped_title}</h2>
      {meta_html}
      <pre class="mermaid">
{diagram}
      </pre>
    </section>""")

    toc_html = "\n".join(toc_items)
    sections_html = "\n\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mermaid Charts Export</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      max-width: 900px;
      margin: 0 auto;
      padding: 2rem;
      color: #1a1a1a;
      line-height: 1.6;
    }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
    h2 {{ color: #2c3e50; margin-top: 2rem; }}
    .meta {{ color: #555; font-size: 0.9rem; }}
    nav ol {{ columns: 2; column-gap: 2rem; }}
    nav a {{ text-decoration: none; color: #2563eb; }}
    nav a:hover {{ text-decoration: underline; }}
    section {{ margin-bottom: 2rem; }}
    pre.mermaid {{ background: #f8f9fa; padding: 1rem; border-radius: 6px; }}
    hr {{ border: none; border-top: 1px solid #ddd; margin: 2rem 0; }}

    @media print {{
      body {{ padding: 0; max-width: 100%; }}
      nav {{ page-break-after: always; }}
      section {{ page-break-before: always; }}
      a {{ color: inherit; text-decoration: none; }}
    }}
  </style>
</head>
<body>
  <h1>Mermaid Charts Export</h1>
  <p class="meta">Generated: {now}</p>

  <nav>
    <h2>Table of Contents</h2>
    <ol>
{toc_html}
    </ol>
  </nav>

  <hr>

{sections_html}

  <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({{ startOnLoad: true, theme: "default" }});</script>
</body>
</html>"""


# ── Page ─────────────────────────────────────────────────────────────

if st.sidebar.button("\U0001f504 Refresh Data", key="refresh_report_export"):
    clear_all_caches()
    st.rerun()

st.title("Report Export")
st.markdown("Download all reports with mermaid diagrams as **Markdown** or **HTML**.")

reports = fetch_mermaid_reports()

if not reports:
    st.warning("No reports with mermaid diagrams found.")
    st.stop()

st.metric("Reports with diagrams", len(reports))

with st.expander("Preview available reports", expanded=False):
    for report in reports:
        st.text(
            f"- {report['doc_name']} — {report['institution']}  "
            f"(model: {report['model_name']})"
        )

col1, col2 = st.columns(2)

with col1:
    md_content = _build_markdown(reports)
    st.download_button(
        "Download Markdown",
        data=md_content.encode("utf-8"),
        file_name="mermaid_charts_export.md",
        mime="text/markdown",
    )

with col2:
    html_content = _build_html(reports)
    st.download_button(
        "Download HTML",
        data=html_content.encode("utf-8"),
        file_name="mermaid_charts_export.html",
        mime="text/html",
    )

st.info(
    "The HTML file renders mermaid diagrams in-browser. "
    "Open it and use **Print → Save as PDF** for a PDF with page breaks between reports."
)
