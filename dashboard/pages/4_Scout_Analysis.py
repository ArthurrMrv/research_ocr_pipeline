"""Scout Analysis page — score distributions, heatmaps, and per-doc profiles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from data import clear_all_caches, fetch_overview, fetch_scout_scores

if st.sidebar.button("🔄 Refresh Data", key="refresh_scout"):
    clear_all_caches()
    st.rerun()

st.title("Scout Analysis")

SCOUT_THRESHOLD = 0.6

scores_df = fetch_scout_scores()

if scores_df.empty:
    st.warning("No scout scores found in the database.")
    st.stop()

# Join doc names for display
overview = fetch_overview()
if not overview.empty:
    name_map = overview.set_index("doc_id")["doc_name"].to_dict()
    scores_df = scores_df.copy()
    scores_df["doc_name"] = scores_df["doc_id"].map(name_map).fillna(scores_df["doc_id"].str[:12])
else:
    scores_df = scores_df.copy()
    scores_df["doc_name"] = scores_df["doc_id"].str[:12]

# ── Score Distribution Histogram ───────────────────────────────────

st.subheader("Score Distribution")

fig_hist = px.histogram(
    scores_df,
    x="score",
    nbins=20,
    color_discrete_sequence=["#636EFA"],
    labels={"score": "Scout Score"},
)
fig_hist.add_vline(
    x=SCOUT_THRESHOLD,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Threshold ({SCOUT_THRESHOLD})",
)
fig_hist.update_layout(yaxis_title="Count")
st.plotly_chart(fig_hist, use_container_width=True)

# ── Step filter for heatmap and profiles ───────────────────────────

steps = sorted(scores_df["step_name"].unique())
selected_step = st.selectbox("Filter by step", ["All"] + steps)

filtered = (
    scores_df if selected_step == "All"
    else scores_df[scores_df["step_name"] == selected_step]
)

# ── Heatmap: Docs x Pages ─────────────────────────────────────────

st.subheader("Score Heatmap (Docs × Pages)")

pivot = filtered.pivot_table(
    index="doc_name",
    columns="page_number",
    values="score",
    aggfunc="max",
)

if not pivot.empty:
    fig_heat = px.imshow(
        pivot.values,
        x=[str(c) for c in pivot.columns],
        y=list(pivot.index),
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=1,
        aspect="auto",
        labels={"x": "Page", "y": "Document", "color": "Score"},
    )
    fig_heat.update_layout(height=max(300, len(pivot) * 30))
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("No data for heatmap.")

# ── Per-Doc Score Profile ──────────────────────────────────────────

st.subheader("Per-Document Score Profile")

doc_names = sorted(filtered["doc_name"].unique())
selected_doc = st.selectbox("Select document", doc_names, key="profile_doc")

doc_data = filtered[filtered["doc_name"] == selected_doc]

if not doc_data.empty:
    if len(doc_data["step_name"].unique()) > 1:
        fig_line = px.line(
            doc_data.sort_values("page_number"),
            x="page_number",
            y="score",
            color="step_name",
            markers=True,
            labels={"page_number": "Page", "score": "Score", "step_name": "Step"},
        )
    else:
        fig_line = px.line(
            doc_data.sort_values("page_number"),
            x="page_number",
            y="score",
            markers=True,
            labels={"page_number": "Page", "score": "Score"},
        )
    fig_line.add_hline(
        y=SCOUT_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
    )
    fig_line.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig_line, use_container_width=True)

# ── Summary Stats Table ───────────────────────────────────────────

st.subheader("Summary Statistics")

stats_records = []
for doc_name in doc_names:
    doc_scores = filtered[filtered["doc_name"] == doc_name]["score"]
    stats_records.append(
        {
            "Document": doc_name,
            "Avg Score": round(float(doc_scores.mean()), 3),
            "Max Score": round(float(doc_scores.max()), 3),
            "Pages Above Threshold": int((doc_scores >= SCOUT_THRESHOLD).sum()),
            "Total Pages": len(doc_scores),
        }
    )

stats_df = pd.DataFrame(stats_records).sort_values("Avg Score", ascending=False)
st.dataframe(stats_df, use_container_width=True, hide_index=True)
