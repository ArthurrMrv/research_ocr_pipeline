"""Reports page — per-report drill-down."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

# AUTH DISABLED FOR DEV — re-enable for production
# from auth import require_auth
from data import (
    clear_all_caches,
    fetch_formatting,
    fetch_ocr_page,
    fetch_ocr_summary,
    fetch_overview,
    fetch_scout_scores,
)

# require_auth()

if st.sidebar.button("🔄 Refresh Data", key="refresh_reports"):
    clear_all_caches()
    st.rerun()

st.title("Report Drill-Down")

df = fetch_overview()

if df.empty:
    st.warning("No documents found.")
    st.stop()

# ── Report Selector ────────────────────────────────────────────────

df["label"] = df.apply(
    lambda r: f"{r.get('doc_name', '')} — {r.get('institution', 'N/A')}", axis=1
)
selected_label = st.selectbox("Select a report", df["label"].tolist())
row = df[df["label"] == selected_label].iloc[0]
doc_id = row["doc_id"]

# ── Header with Status Badges ──────────────────────────────────────

st.subheader(row.get("doc_name", ""))

badge_cols = st.columns(4)


def _badge(col, label: str, done: bool) -> None:
    icon = "✅" if done else "⬜"
    col.markdown(f"**{icon} {label}**")


_badge(badge_cols[0], "Ingested", True)
_badge(badge_cols[1], "OCR", pd.notna(row.get("last_ocr")))
_badge(badge_cols[2], "Scout", pd.notna(row.get("last_scout")))
_badge(
    badge_cols[3],
    "Formatting",
    (row.get("formatting_nb") or 0) > 0,
)

meta_cols = st.columns(3)
meta_cols[0].markdown(f"**Institution:** {row.get('institution', 'N/A')}")
meta_cols[1].markdown(f"**Report Date:** {row.get('report_date', 'N/A')}")
meta_cols[2].markdown(f"**Doc ID:** `{doc_id[:12]}…`")

st.markdown("---")

# ── OCR Pages ──────────────────────────────────────────────────────

with st.expander("📄 OCR Pages", expanded=False):
    ocr_df = fetch_ocr_summary(doc_id)
    if ocr_df.empty:
        st.info("No OCR results for this document.")
    else:
        st.dataframe(ocr_df, use_container_width=True, hide_index=True)
        page_num = st.selectbox(
            "View full OCR text for page:",
            ocr_df["page_number"].tolist(),
            key="ocr_page_select",
        )
        if page_num is not None:
            text = fetch_ocr_page(doc_id, int(page_num))
            if text:
                st.text_area("OCR Content", text, height=400, disabled=True)
            else:
                st.info("No content for this page.")

# ── Scout Scores ───────────────────────────────────────────────────

SCOUT_THRESHOLD = 0.6

with st.expander("🔍 Scout Scores", expanded=False):
    scout_df = fetch_scout_scores(doc_id)
    if scout_df.empty:
        st.info("No scout scores for this document.")
    else:
        display_df = scout_df[["step_name", "page_number", "score", "scout_model"]].copy()
        display_df["above_threshold"] = display_df["score"] >= SCOUT_THRESHOLD
        st.dataframe(
            display_df.style.map(
                lambda v: "background-color: #d4edda" if v else "",
                subset=["above_threshold"],
            ),
            use_container_width=True,
            hide_index=True,
        )

# ── Formatting Results ─────────────────────────────────────────────

with st.expander("📊 Formatting Results", expanded=True):
    fmt_df = fetch_formatting(doc_id=doc_id)
    if fmt_df.empty:
        st.info("No formatting results for this document.")
    else:
        for _, fmt_row in fmt_df.iterrows():
            step = fmt_row["step_name"]
            model = fmt_row.get("formatting_model", "")
            content = fmt_row.get("content", {})

            st.markdown(f"#### Step: `{step}` (model: `{model}`)")

            if not content:
                st.warning("Empty content.")
                continue

            if step == "extract_model_inputs":
                st.markdown(f"**Model Name:** {content.get('model_name', 'N/A')}")
                if content.get("notes_model"):
                    st.markdown(f"**Notes:** {content['notes_model']}")

                if content.get("variables"):
                    st.markdown("**Variables:**")
                    st.write(content["variables"])

                if content.get("variables_important"):
                    st.markdown("**Key Variables:**")
                    st.write(content["variables_important"])

                if content.get("assumptions"):
                    st.markdown("**Assumptions (values):**")
                    st.write(content["assumptions"])

            elif step == "extract_model_methodology":
                st.markdown(f"**Summary:** {content.get('steps_summary', '')}")

                if content.get("steps_detailed"):
                    with st.expander("Detailed Steps"):
                        st.markdown(content["steps_detailed"])

                if content.get("mermaid_diagram"):
                    with st.expander("Mermaid Diagram (source)"):
                        st.code(content["mermaid_diagram"], language="mermaid")

                if content.get("sub_models"):
                    st.markdown("**Sub-models:**")
                    st.write(content["sub_models"])

                if content.get("assumptions"):
                    st.markdown("**Assumptions (structural):**")
                    st.write(content["assumptions"])

                technique_flags = ["uses_regressions", "uses_simulations", "uses_averages", "uses_mean_reversion"]
                active_techniques = [f.replace("uses_", "") for f in technique_flags if content.get(f) == 1]
                if active_techniques:
                    st.markdown(f"**Techniques used:** {', '.join(active_techniques)}")

            elif step == "extract_model_assumptions":
                st.markdown(f"**Sophistication Index:** {content.get('sophistication_index', 'N/A')}")
                st.markdown(f"**Explanation:** {content.get('sophistication_explanation', '')}")
                techniques = content.get("techniques_used", [])
                if techniques:
                    st.markdown("**Techniques Used:**")
                    for t in techniques:
                        st.markdown(f"- {t.get('technique_name', '')} (complexity: {t.get('complexity', '?')})")

                assumptions = content.get("assumptions", [])
                if assumptions:
                    st.markdown("**Classified Assumptions:**")
                    for a in assumptions:
                        classification = a.get("classification", "?")
                        block = a.get("building_block", "")
                        text = a.get("assumption", "")
                        st.markdown(f"- **[{classification}]** {text} _(re: {block})_")

            elif step == "extract_table":
                table_data = content.get("table")
                if table_data is None:
                    st.info("No table found in this document.")
                elif isinstance(table_data, dict):
                    title = table_data.get("title", "Extracted Table")
                    if title:
                        st.markdown(f"**{title}**")
                    headers = table_data.get("headers", [])
                    rows = table_data.get("rows", [])
                    if headers and rows:
                        table_df = pd.DataFrame(rows, columns=headers)
                        st.dataframe(table_df, use_container_width=True, hide_index=True)
                    else:
                        st.json(table_data)
                else:
                    st.json(table_data)
            else:
                st.json(content)

            st.markdown("---")

# ── Errors ─────────────────────────────────────────────────────────

with st.expander("⚠️ Errors", expanded=False):
    errors = row.get("error")
    if not errors or not isinstance(errors, list) or len(errors) == 0:
        st.success("No errors recorded.")
    else:
        error_df = pd.DataFrame(errors)
        st.dataframe(error_df, use_container_width=True, hide_index=True)
