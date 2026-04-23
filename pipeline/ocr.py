from __future__ import annotations

import math
import re
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone

import fitz  # pymupdf
from PIL import Image
from supabase import Client

from config import MAX_PAGES_PER_BATCH, OCR_MODEL_ID, OCR_PROVIDER, ZAI_MAX_PAGES
from pipeline import debug_logger
from pipeline.ocr_providers.registry import get_ocr_provider
from pipeline.page_utils import is_empty_page_content
from pipeline.tracker import (
    append_error,
    delete_ocr_rows,
    get_ocr_chunks,
    pipeline_get,
    pipeline_update,
    silver_bulk_upsert,
    silver_upsert,
)


def pdf_to_images(pdf_path: str, start: int = 0, end: int | None = None) -> list[Image.Image]:
    """Render PDF pages [start, end) to PIL Images at 150 DPI.

    If end is None, renders from start to the last page.
    """
    dpi = 150
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    images = []
    with fitz.open(pdf_path) as doc:
        stop = end if end is not None else len(doc)
        for page in doc[start:stop]:
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images


_LEADING_IMAGE_RE = re.compile(r"^(!\[.*?\]\(.*?\))\s*", re.DOTALL)


def _add_page_marker(content: str, page_num: int) -> str:
    """Prepend '--- Page N ---' respecting any leading markdown image."""
    marker = f"--- Page {page_num} ---"
    m = _LEADING_IMAGE_RE.match(content)
    if m:
        return f"{m.group(1)}\n{marker}\n{content[m.end():]}"
    return f"{marker}\n{content}"


def _split_chunk_text_into_pages(chunk_text: str) -> list[tuple[int, str]]:
    """Split combined OCR text into (page_number, page_content) sections."""
    sections = re.split(r"(?=---\s*Page\s+\d+\s*---)", chunk_text)
    pages: list[tuple[int, str]] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        match = re.match(r"---\s*Page\s+(\d+)\s*---", section)
        if not match:
            continue
        pages.append((int(match.group(1)), section))
    return pages


def _extract_sub_pdf_from_doc(src: fitz.Document, start: int, end: int) -> bytes:
    """Extract 0-indexed pages [start, end) from an open PDF document and return as bytes."""
    with fitz.open() as dst:
        dst.insert_pdf(src, from_page=start, to_page=end - 1)
        return dst.tobytes()


def extract_sub_pdf_bytes(pdf_path: str, start: int, end: int) -> bytes:
    """Extract 0-indexed pages [start, end) from PDF and return as bytes."""
    with fitz.open(pdf_path) as src:
        return _extract_sub_pdf_from_doc(src, start, end)


def _after_date(row: dict, since: str | None) -> bool:
    """Return True if row['added'] >= since (ISO date string). Returns False when since is None."""
    if since is None:
        return False
    added = row.get("added")
    if added is None:
        return False
    added_dt = datetime.fromisoformat(str(added).replace("Z", "+00:00"))
    since_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
    return added_dt >= since_dt


def _reocr_pages(
    doc_id: str,
    client: Client,
    page_nums: list[int],
    progress_callback: Callable[[int, int], None] | None,
) -> dict:
    """Re-OCR specific pages of an already-processed document and upsert results.

    Batches consecutive pages together to minimise PDF opens and API calls.
    """
    bronze_row = (
        client.table("bronze_mapping").select("file_path").eq("doc_id", doc_id).execute()
    )
    file_path = bronze_row.data[0]["file_path"]
    provider = get_ocr_provider(OCR_PROVIDER)

    newly_empty: list[int] = []
    pending_rows: list[dict] = []

    # Group consecutive pages into batches to reduce PDF opens and API calls
    batches = _group_consecutive(sorted(page_nums))

    pages_done = 0
    for batch in batches:
        batch_start = batch[0] - 1  # 0-indexed
        batch_end = batch[-1]       # exclusive end for fitz
        chunk = extract_sub_pdf_bytes(file_path, batch_start, batch_end)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(chunk)
            tmp.flush()
            try:
                pages = provider.ocr_document(tmp.name)
            except NotImplementedError:
                continue
            except Exception as exc:
                append_error(client, doc_id, f"OCR re-OCR batch error: {exc}")
                continue

        for i, page_num in enumerate(batch):
            raw_content = pages[i] if i < len(pages) else ""
            is_empty = not raw_content.strip()
            if is_empty:
                newly_empty.append(page_num)
            debug_logger.print_ocr_page(page_num, empty=is_empty)
            content = _add_page_marker(raw_content, page_num)
            pending_rows.append({
                "doc_id": doc_id,
                "page_number": page_num,
                "ocr_model": OCR_MODEL_ID,
                "content": content,
            })
            pages_done += 1
            if progress_callback:
                progress_callback(pages_done, len(page_nums))

    if pending_rows:
        silver_bulk_upsert(client, "ocr_results", pending_rows, on_conflict="doc_id,page_number")

    return {"status": "done", "empty_pages": len(newly_empty)}


def _group_consecutive(nums: list[int]) -> list[list[int]]:
    """Group sorted integers into lists of consecutive runs.

    Example: [1,2,3,7,8,12] -> [[1,2,3],[7,8],[12]]
    """
    if not nums:
        return []
    groups: list[list[int]] = [[nums[0]]]
    for n in nums[1:]:
        if n == groups[-1][-1] + 1:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def run_ocr(
    doc_id: str,
    client: Client,
    *,
    force: bool = False,
    since: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """
    Run OCR for doc_id and store result in ocr_results.
    Chunks pages into batches of MAX_PAGES_PER_BATCH.
    Each chunk is stored as a separate row with a 'pages' column.
    Skips if already OCR'd unless force=True or since date matches.
    If already done but has empty pages, re-OCRs only those pages.
    Returns dict with 'status' ("skipped"|"done") and 'empty_pages' count.
    """
    pipeline_row = pipeline_get(client, doc_id)
    if pipeline_row is None:
        raise ValueError(f"No pipeline row for doc_id={doc_id}")

    already_done = pipeline_row.get("last_ocr") is not None
    bronze_date_row = (
        client.table("bronze_mapping").select("added").eq("doc_id", doc_id).execute()
    )
    bronze_added = bronze_date_row.data[0] if bronze_date_row.data else {}
    if already_done and not force and not _after_date(bronze_added, since):
        existing_rows = get_ocr_chunks(client, doc_id)
        empty_page_nums = [
            r["page_number"] for r in existing_rows if is_empty_page_content(r["content"])
        ]
        if not empty_page_nums:
            return {"status": "skipped", "empty_pages": 0}
        return _reocr_pages(doc_id, client, empty_page_nums, progress_callback)

    # Fetch file path from bronze_mapping
    bronze_row = (
        client.table("bronze_mapping")
        .select("file_path")
        .eq("doc_id", doc_id)
        .execute()
    )
    if not bronze_row.data:
        raise ValueError(f"No bronze_mapping row for doc_id={doc_id}")
    file_path = bronze_row.data[0]["file_path"]

    try:
        provider = get_ocr_provider(OCR_PROVIDER)
        delete_ocr_rows(client, doc_id)

        empty_pages: list[int] = []

        try:
            # Fast path: provider can OCR the PDF natively
            with fitz.open(file_path) as doc:
                total_pages = len(doc)

            if total_pages <= ZAI_MAX_PAGES:
                pages = provider.ocr_document(file_path)
                pending_rows: list[dict] = []
                for i, raw_content in enumerate(pages):
                    page_num = i + 1
                    if not raw_content.strip():
                        empty_pages.append(page_num)
                    debug_logger.print_ocr_page(page_num, empty=not raw_content.strip())
                    content = _add_page_marker(raw_content, page_num)
                    pending_rows.append({
                        "doc_id": doc_id,
                        "page_number": page_num,
                        "ocr_model": OCR_MODEL_ID,
                        "content": content,
                    })
                silver_bulk_upsert(client, "ocr_results", pending_rows, on_conflict="doc_id,page_number")
                if progress_callback is not None:
                    progress_callback(1, 1)
            else:
                total_batches = math.ceil(total_pages / ZAI_MAX_PAGES)
                with fitz.open(file_path) as src_doc:
                    for batch_idx in range(total_batches):
                        batch_start = batch_idx * ZAI_MAX_PAGES
                        batch_end = min(batch_start + ZAI_MAX_PAGES, total_pages)
                        chunk_bytes = _extract_sub_pdf_from_doc(src_doc, batch_start, batch_end)
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                            tmp.write(chunk_bytes)
                            tmp.flush()
                            pages = provider.ocr_document(tmp.name)
                        pending_rows = []
                        for i, raw_content in enumerate(pages):
                            page_num = batch_start + i + 1
                            if not raw_content.strip():
                                empty_pages.append(page_num)
                            debug_logger.print_ocr_page(page_num, empty=not raw_content.strip())
                            content = _add_page_marker(raw_content, page_num)
                            pending_rows.append({
                                "doc_id": doc_id,
                                "page_number": page_num,
                                "ocr_model": OCR_MODEL_ID,
                                "content": content,
                            })
                        silver_bulk_upsert(client, "ocr_results", pending_rows, on_conflict="doc_id,page_number")
                        if progress_callback is not None:
                            progress_callback(batch_idx + 1, total_batches)

            if empty_pages:
                append_error(
                    client,
                    doc_id,
                    f"OCR produced {len(empty_pages)} empty page(s): {empty_pages}",
                )
        except NotImplementedError:
            # Slow path: rasterise and send page images per batch
            with fitz.open(file_path) as doc:
                total_pages = len(doc)
            total_batches = math.ceil(total_pages / MAX_PAGES_PER_BATCH)
            for batch_start in range(0, total_pages, MAX_PAGES_PER_BATCH):
                batch_end = min(batch_start + MAX_PAGES_PER_BATCH, total_pages)
                # Render only this batch's pages (lazy per-batch)
                batch_images = pdf_to_images(file_path, start=batch_start, end=batch_end)
                chunk_text = provider.ocr_pages(batch_images, page_offset=batch_start)
                pending_rows = []
                for page_num, page_content in _split_chunk_text_into_pages(chunk_text):
                    is_empty = is_empty_page_content(page_content)
                    if is_empty:
                        empty_pages.append(page_num)
                    debug_logger.print_ocr_page(page_num, empty=is_empty)
                    pending_rows.append({
                        "doc_id": doc_id,
                        "page_number": page_num,
                        "ocr_model": OCR_MODEL_ID,
                        "content": page_content,
                    })
                silver_bulk_upsert(client, "ocr_results", pending_rows, on_conflict="doc_id,page_number")
                if progress_callback is not None:
                    batch_index = batch_start // MAX_PAGES_PER_BATCH
                    progress_callback(batch_index + 1, total_batches)

        pipeline_update(
            client,
            doc_id,
            {"last_ocr": datetime.now(timezone.utc).isoformat()},
        )
        return {"status": "done", "empty_pages": len(empty_pages)}
    except Exception as exc:
        append_error(client, doc_id, f"OCR error: {exc}")
        raise
