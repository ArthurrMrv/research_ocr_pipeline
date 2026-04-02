"""Report export page — auto-generates PDF from mermaid charts."""

import base64
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# AUTH DISABLED FOR DEV — re-enable for production
# from auth import require_auth
from data import clear_all_caches, fetch_mermaid_reports
from mermaid_utils import build_mermaid_export_html

# require_auth()


def _render_pdf(html_content: str) -> bytes:
    """Render HTML with mermaid diagrams to PDF using a headless browser."""
    from playwright.sync_api import sync_playwright

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        f.write(html_content.encode("utf-8"))
        html_path = f.name

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=["--disable-dev-shm-usage"])
            page = browser.new_page()
            page.goto(f"file://{html_path}", wait_until="load")
            try:
                page.wait_for_function(
                    "window.__mermaidStatus && window.__mermaidStatus.done === true",
                    timeout=30000,
                )
            except Exception:
                page.wait_for_timeout(1500)
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

html_content = build_mermaid_export_html(reports)


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
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="mermaid_charts_export.pdf",
        mime="application/pdf",
    )
    if st.toggle("Show PDF preview", value=False):
        b64 = base64.b64encode(pdf_bytes).decode()
        pdf_data_uri = f"data:application/pdf;base64,{b64}"
        st.markdown(
            f'<iframe src="{pdf_data_uri}" width="100%" height="800" type="application/pdf"></iframe>',
            unsafe_allow_html=True,
        )
