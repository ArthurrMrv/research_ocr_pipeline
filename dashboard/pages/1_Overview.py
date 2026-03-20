"""Overview page — pipeline status summary."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from data import clear_all_caches, fetch_overview

if st.sidebar.button("🔄 Refresh Data", key="refresh_overview"):
    clear_all_caches()
    st.rerun()

st.title("Pipeline Overview")

df = fetch_overview()

if df.empty:
    st.warning("No documents found in the database.")
    st.stop()

# ── Metric Cards ────────────────────────────────────────────────────

total_docs = len(df)
ocr_done = int(df["last_ocr"].notna().sum()) if "last_ocr" in df.columns else 0
scout_done = int(df["last_scout"].notna().sum()) if "last_scout" in df.columns else 0
formatting_done = int(
    (df["formatting_nb"].fillna(0) > 0).sum() if "formatting_nb" in df.columns else 0
)

cols = st.columns(4)
cols[0].metric("Total Docs", total_docs)
cols[1].metric("OCR Done", ocr_done, delta=f"{ocr_done}/{total_docs}")
cols[2].metric("Scout Done", scout_done, delta=f"{scout_done}/{total_docs}")
cols[3].metric("Formatting Done", formatting_done, delta=f"{formatting_done}/{total_docs}")

st.markdown("---")

# ── Pipeline Stage Funnel ──────────────────────────────────────────

st.subheader("Pipeline Stage Funnel")

funnel_data = pd.DataFrame(
    {
        "Stage": ["Ingested", "OCR", "Scout", "Formatting"],
        "Documents": [total_docs, ocr_done, scout_done, formatting_done],
    }
)

fig_funnel = px.bar(
    funnel_data,
    x="Stage",
    y="Documents",
    text="Documents",
    color="Stage",
    color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
)
fig_funnel.update_layout(showlegend=False, yaxis_title="Documents")
fig_funnel.update_traces(textposition="outside")
st.plotly_chart(fig_funnel, use_container_width=True)

# ── Institution Breakdown ──────────────────────────────────────────

st.subheader("Institution Breakdown")

if "institution" in df.columns:
    inst_counts = (
        df["institution"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )
    inst_counts.columns = ["Institution", "Documents"]
    fig_inst = px.bar(
        inst_counts,
        x="Institution",
        y="Documents",
        text="Documents",
        color="Institution",
    )
    fig_inst.update_layout(showlegend=False)
    fig_inst.update_traces(textposition="outside")
    st.plotly_chart(fig_inst, use_container_width=True)
else:
    st.info("No institution data available.")

# ── Error Summary ──────────────────────────────────────────────────

st.subheader("Error Summary")

if "error" in df.columns:
    error_rows = df[df["error"].apply(lambda e: bool(e) if isinstance(e, list) else False)]
    if error_rows.empty:
        st.success("No errors recorded.")
    else:
        error_records = []
        for _, row in error_rows.iterrows():
            errors = row["error"]
            latest = errors[-1] if errors else {}
            error_records.append(
                {
                    "doc_name": row.get("doc_name", ""),
                    "institution": row.get("institution", ""),
                    "error_count": len(errors),
                    "latest_error": latest.get("message", ""),
                    "latest_ts": latest.get("ts", ""),
                }
            )
        st.dataframe(
            pd.DataFrame(error_records),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("No pipeline error data available.")
