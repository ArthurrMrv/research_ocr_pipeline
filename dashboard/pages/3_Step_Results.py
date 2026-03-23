"""Step Results page — cross-document extraction results by step."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from data import clear_all_caches, fetch_formatting_with_meta, fetch_overview

if st.sidebar.button("🔄 Refresh Data", key="refresh_steps"):
    clear_all_caches()
    st.rerun()

st.title("Step Results")

KNOWN_STEPS = ["extract_model_inputs", "extract_model_methodology", "extract_table"]

selected_step = st.selectbox("Select a step", KNOWN_STEPS)

overview = fetch_overview()
total_docs = len(overview) if not overview.empty else 0

fmt_df = fetch_formatting_with_meta(step=selected_step)

if fmt_df.empty:
    st.warning(f"No results found for step `{selected_step}`.")
    st.stop()

# ── Coverage ────────────────────────────────────────────────────────

docs_with_results = set(fmt_df["doc_id"].unique())
coverage = len(docs_with_results)
st.metric("Coverage", f"{coverage} / {total_docs} docs")

if not overview.empty:
    all_doc_ids = set(overview["doc_id"].unique())
    missing_ids = all_doc_ids - docs_with_results
    if missing_ids:
        missing_docs = overview[overview["doc_id"].isin(missing_ids)][
            ["doc_name", "institution"]
        ]
        with st.expander(f"Missing docs ({len(missing_ids)})"):
            st.dataframe(missing_docs, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Results Table ──────────────────────────────────────────────────

if selected_step == "extract_model_inputs":
    records = []
    for _, row in fmt_df.iterrows():
        content = row.get("content") or {}
        records.append(
            {
                "doc_name": row.get("doc_name", ""),
                "institution": row.get("institution", ""),
                "model_name": content.get("model_name", ""),
                "variables": ", ".join(content.get("variables", [])),
            }
        )
    results_df = pd.DataFrame(records)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    for _, row in fmt_df.iterrows():
        content = row.get("content") or {}
        label = f"{row.get('doc_name', '')} — {row.get('institution', '')}"
        with st.expander(label):
            st.markdown(f"**Model:** {content.get('model_name', 'N/A')}")
            if content.get("variables"):
                st.markdown("**Variables:**")
                st.write(content["variables"])
            if content.get("variables_important"):
                st.markdown("**Key Variables:**")
                st.write(content["variables_important"])
            if content.get("assumptions"):
                st.markdown("**Assumptions (values):**")
                st.write(content["assumptions"])

elif selected_step == "extract_model_methodology":
    records = []
    for _, row in fmt_df.iterrows():
        content = row.get("content") or {}
        records.append(
            {
                "doc_name": row.get("doc_name", ""),
                "institution": row.get("institution", ""),
                "summary": content.get("steps_summary", ""),
            }
        )
    results_df = pd.DataFrame(records)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    for _, row in fmt_df.iterrows():
        content = row.get("content") or {}
        label = f"{row.get('doc_name', '')} — {row.get('institution', '')}"
        with st.expander(label):
            if content.get("steps_summary"):
                st.markdown(f"**Summary:** {content['steps_summary']}")
            if content.get("steps_detailed"):
                st.markdown(content["steps_detailed"])
            if content.get("mermaid_diagram"):
                st.code(content["mermaid_diagram"], language="mermaid")
            if content.get("sub_models"):
                st.markdown("**Sub-models:**")
                st.write(content["sub_models"])
            if content.get("assumptions"):
                st.markdown("**Assumptions (structural):**")
                st.write(content["assumptions"])

elif selected_step == "extract_table":
    records = []
    for _, row in fmt_df.iterrows():
        content = row.get("content") or {}
        table_data = content.get("table")
        if table_data is None:
            records.append(
                {
                    "doc_name": row.get("doc_name", ""),
                    "institution": row.get("institution", ""),
                    "table_title": "No table found",
                    "rows": 0,
                    "cols": 0,
                }
            )
        elif isinstance(table_data, dict):
            records.append(
                {
                    "doc_name": row.get("doc_name", ""),
                    "institution": row.get("institution", ""),
                    "table_title": table_data.get("title", ""),
                    "rows": len(table_data.get("rows", [])),
                    "cols": len(table_data.get("headers", [])),
                }
            )
        else:
            records.append(
                {
                    "doc_name": row.get("doc_name", ""),
                    "institution": row.get("institution", ""),
                    "table_title": "Unknown format",
                    "rows": 0,
                    "cols": 0,
                }
            )
    results_df = pd.DataFrame(records)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # Expandable full tables
    for _, row in fmt_df.iterrows():
        content = row.get("content") or {}
        table_data = content.get("table")
        if table_data and isinstance(table_data, dict):
            headers = table_data.get("headers", [])
            rows = table_data.get("rows", [])
            if headers and rows:
                label = f"{row.get('doc_name', '')} — {table_data.get('title', 'Table')}"
                with st.expander(label):
                    st.dataframe(
                        pd.DataFrame(rows, columns=headers),
                        use_container_width=True,
                        hide_index=True,
                    )

else:
    st.dataframe(fmt_df, use_container_width=True, hide_index=True)
