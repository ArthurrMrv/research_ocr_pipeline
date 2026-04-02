"""Pipeline Dashboard — entry point.

Read-only Streamlit app for visualizing the ingestion pipeline state.
Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# AUTH DISABLED FOR DEV — re-enable for production
# from auth import require_auth, logout
# require_auth()

st.sidebar.title("Pipeline Dashboard")
st.sidebar.markdown("Read-only view of the ingestion pipeline.")

if st.sidebar.button("🔄 Refresh Data"):
    from data import clear_all_caches

    clear_all_caches()
    st.rerun()

# AUTH DISABLED FOR DEV — re-enable for production
# if st.sidebar.button("🚪 Logout"):
#     logout()
#     st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Navigate using the pages in the sidebar.")

st.title("📊 Pipeline Dashboard")
st.markdown(
    """
Welcome to the ingestion pipeline dashboard. Use the sidebar to navigate:

- **Overview** — Pipeline status summary & stage funnel
- **Reports** — Per-report drill-down with OCR, scout, and formatting details
- **Step Results** — Cross-document extraction results by step
- **Scout Analysis** — Score distributions, heatmaps, and per-doc profiles
- **Model CSV Export** — Flat `extract_model_name` table with CSV download
"""
)
