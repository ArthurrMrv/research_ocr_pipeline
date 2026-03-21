"""Read-only Supabase data layer for the pipeline dashboard.

All functions return new DataFrames — no mutation.
All queries use .select() only — zero write operations.
"""

import os

import pandas as pd
import streamlit as st
from supabase import Client, create_client

MODEL_EXPORT_COLUMNS = [
    "document_name",
    "model_name",
    "notes_model",
    "variables",
    "variables_nb",
    "steps_summary",
    "steps_detailed",
    "variables_importants",
]


@st.cache_resource
def get_client() -> Client:
    """Cached Supabase client (one per app lifetime)."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)


def _query(table: str, select: str = "*", **eq_filters) -> list[dict]:
    """Run a select query with optional equality filters."""
    q = get_client().table(table).select(select)
    for col, val in eq_filters.items():
        q = q.eq(col, val)
    return q.execute().data or []


# ── Overview ────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def fetch_overview() -> pd.DataFrame:
    """Join bronze_mapping + pipeline for the overview page."""
    bronze = _query("bronze_mapping")
    pipeline = _query("pipeline")
    df_bronze = pd.DataFrame(bronze)
    df_pipeline = pd.DataFrame(pipeline)
    if df_bronze.empty:
        return pd.DataFrame()
    if df_pipeline.empty:
        return df_bronze
    return pd.merge(df_bronze, df_pipeline, on="doc_id", how="left")


# ── OCR ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def fetch_ocr_summary(doc_id: str) -> pd.DataFrame:
    """Page numbers + content lengths for a doc (no full text)."""
    rows = _query("ocr_results", select="doc_id,page_number,ocr_model,added", doc_id=doc_id)
    if not rows:
        return pd.DataFrame(columns=["page_number", "ocr_model", "added"])
    # We need content length but don't want to cache full text in summary
    full = _query("ocr_results", doc_id=doc_id)
    records = [
        {
            "page_number": r["page_number"],
            "ocr_model": r.get("ocr_model", ""),
            "content_length": len(r.get("content") or ""),
            "added": r.get("added"),
        }
        for r in full
    ]
    return pd.DataFrame(records).sort_values("page_number").reset_index(drop=True)


@st.cache_data(ttl=300)
def fetch_ocr_page(doc_id: str, page: int) -> str:
    """Full OCR text for one page."""
    rows = _query("ocr_results", doc_id=doc_id, page_number=page)
    if rows:
        return rows[0].get("content") or ""
    return ""


# ── Scout ───────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def fetch_scout_scores(doc_id: str | None = None) -> pd.DataFrame:
    """All or per-doc scout scores."""
    if doc_id:
        rows = _query("scout_page_scores", doc_id=doc_id)
    else:
        rows = _query("scout_page_scores")
    if not rows:
        return pd.DataFrame(columns=["doc_id", "step_name", "page_number", "score", "scout_model"])
    return pd.DataFrame(rows)


# ── Formatting ──────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def fetch_formatting(doc_id: str | None = None, step: str | None = None) -> pd.DataFrame:
    """Formatting results with optional filters."""
    filters: dict = {}
    if doc_id:
        filters["doc_id"] = doc_id
    if step:
        filters["step_name"] = step
    rows = _query("formatting", **filters)
    if not rows:
        return pd.DataFrame(columns=["doc_id", "step_name", "formatting_model", "content"])
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def fetch_formatting_with_meta(step: str | None = None) -> pd.DataFrame:
    """Formatting results joined with bronze_mapping for doc metadata."""
    fmt = fetch_formatting(step=step)
    if fmt.empty:
        return fmt
    bronze = pd.DataFrame(_query("bronze_mapping", select="doc_id,doc_name,institution"))
    if bronze.empty:
        return fmt
    return pd.merge(fmt, bronze, on="doc_id", how="left")


def _serialize_list(values: object) -> str:
    """Render list values as a flat comma-separated string for CSV export."""
    if not isinstance(values, list):
        return ""
    return ", ".join(str(value) for value in values if value is not None)


def _build_extract_model_name_export_df(
    formatting_rows: list[dict],
    bronze_rows: list[dict],
) -> pd.DataFrame:
    """Flatten extract_model_name formatting rows into export-ready columns."""
    if not formatting_rows:
        return pd.DataFrame(columns=MODEL_EXPORT_COLUMNS)

    name_map = {
        row.get("doc_id"): row.get("doc_name", "")
        for row in bronze_rows
    }

    records = []
    for row in formatting_rows:
        content = row.get("content") or {}
        variables = content.get("variables")
        important_variables = content.get("variables_important")
        records.append(
            {
                "document_name": name_map.get(row.get("doc_id"), ""),
                "model_name": content.get("model_name", ""),
                "notes_model": content.get("notes_model", ""),
                "variables": _serialize_list(variables),
                "variables_nb": len(variables) if isinstance(variables, list) else 0,
                "steps_summary": content.get("steps_summary", ""),
                "steps_detailed": content.get("steps_detailed", ""),
                "variables_importants": _serialize_list(important_variables),
            }
        )

    return pd.DataFrame(records, columns=MODEL_EXPORT_COLUMNS)


@st.cache_data(ttl=300)
def fetch_mermaid_reports() -> list[dict]:
    """Return reports that have mermaid diagrams, with doc metadata.

    Each dict has: doc_name, institution, model_name, mermaid_diagram, steps_summary.
    """
    formatting_rows = _query(
        "formatting",
        select="doc_id,content",
        step_name="extract_model_name",
    )
    bronze_rows = _query("bronze_mapping", select="doc_id,doc_name,institution")
    meta_map = {
        row.get("doc_id"): row
        for row in bronze_rows
    }

    reports = []
    for row in formatting_rows:
        content = row.get("content") or {}
        diagram = content.get("mermaid_diagram")
        if not diagram:
            continue
        meta = meta_map.get(row.get("doc_id"), {})
        reports.append({
            "doc_name": meta.get("doc_name", "Unknown"),
            "institution": meta.get("institution", "Unknown"),
            "model_name": content.get("model_name", ""),
            "mermaid_diagram": diagram,
            "steps_summary": content.get("steps_summary", ""),
        })
    return sorted(reports, key=lambda r: (r["institution"], r["doc_name"]))


@st.cache_data(ttl=300)
def fetch_extract_model_name_export() -> pd.DataFrame:
    """Flattened extract_model_name results for dashboard display and CSV export."""
    formatting_rows = _query(
        "formatting",
        select="doc_id,content",
        step_name="extract_model_name",
    )
    bronze_rows = _query("bronze_mapping", select="doc_id,doc_name")
    return _build_extract_model_name_export_df(formatting_rows, bronze_rows)


# ── Helpers ─────────────────────────────────────────────────────────


def clear_all_caches() -> None:
    """Clear all st.cache_data caches (called by sidebar refresh button)."""
    st.cache_data.clear()
