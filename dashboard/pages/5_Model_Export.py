"""Model export page — merged model inputs + methodology results with CSV download."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from data import clear_all_caches, fetch_model_export

if st.sidebar.button("🔄 Refresh Data", key="refresh_model_export"):
    clear_all_caches()
    st.rerun()

st.title("Model CSV Export")
st.markdown(
    "Merged view of model inputs and methodology results for spreadsheet inspection and CSV download."
)

results_df = fetch_model_export()

if results_df.empty:
    st.warning("No model extraction results found in the database.")
    st.stop()

search_term = st.text_input(
    "Filter by document or model name",
    placeholder="Type part of a document name or model name",
)

filtered_df = results_df
if search_term:
    query = search_term.strip().lower()
    filtered_df = results_df[
        results_df["document_name"].fillna("").str.lower().str.contains(query, regex=False)
        | results_df["model_name"].fillna("").str.lower().str.contains(query, regex=False)
    ].reset_index(drop=True)

st.metric("Rows", len(filtered_df))
st.dataframe(filtered_df, use_container_width=True, hide_index=True)

csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    data=csv_bytes,
    file_name="model_export.csv",
    mime="text/csv",
)
