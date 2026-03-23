"""Report export page — auto-generates PDF from mermaid charts."""

import base64
import html
import re
import sys
import tempfile
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

    .print-btn {{
      position: fixed;
      top: 1rem;
      right: 1rem;
      padding: 0.5rem 1.2rem;
      background: #2563eb;
      color: white;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.9rem;
      z-index: 1000;
    }}
    .print-btn:hover {{ background: #1d4ed8; }}

    @media print {{
      body {{ padding: 0; max-width: 100%; }}
      nav {{ page-break-after: always; }}
      section {{ page-break-before: always; }}
      a {{ color: inherit; text-decoration: none; }}
      .print-btn {{ display: none; }}
    }}
  </style>
</head>
<body>
  <button class="print-btn" onclick="window.print()">Print / Save as PDF</button>
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


def _render_pdf(html_content: str) -> bytes:
    """Render HTML with mermaid diagrams to PDF using a headless browser."""
    from playwright.sync_api import sync_playwright

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        f.write(html_content.encode("utf-8"))
        html_path = f.name

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{html_path}")
            page.wait_for_function("document.querySelectorAll('.mermaid svg').length > 0", timeout=15000)
            pdf_bytes = page.pdf(
                format="A4",
                print_background=True,
                margin={"top": "1cm", "right": "1cm", "bottom": "1cm", "left": "1cm"},
            )
            browser.close()
    finally:
        Path(html_path).unlink(missing_ok=True)

    return pdf_bytes


# ── Page ─────────────────────────────────────────────────────────────

if st.sidebar.button("\U0001f504 Refresh Data", key="refresh_report_export"):
    clear_all_caches()
    st.rerun()

st.title("Report Export")

reports = fetch_mermaid_reports()

if not reports:
    st.warning("No reports with mermaid diagrams found.")
    st.stop()

st.metric("Reports with diagrams", len(reports))

html_content = _build_html(reports)


@st.cache_data(show_spinner=False)
def _cached_pdf(html_src: str) -> bytes:
    """Generate and cache the PDF so it's only rendered once per data change."""
    return _render_pdf(html_src)


with st.spinner("Rendering mermaid diagrams to PDF..."):
    try:
        pdf_bytes = _cached_pdf(html_content)
    except Exception as e:
        pdf_bytes = None
        st.error(f"PDF generation failed: {e}")

if pdf_bytes is not None:
    b64 = base64.b64encode(pdf_bytes).decode()
    pdf_data_uri = f"data:application/pdf;base64,{b64}"
    st.markdown(
        f'<iframe src="{pdf_data_uri}" width="100%" height="800" type="application/pdf"></iframe>',
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="mermaid_charts_export.pdf",
        mime="application/pdf",
    )
