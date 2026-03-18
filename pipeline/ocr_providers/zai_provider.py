from __future__ import annotations

import base64
import os
import re

from pipeline.ocr_providers.base import OCRProvider

_PAGE_MARKER_RE = re.compile(r"!\[\]\(page=(\d+),bbox=")


def _split_by_pages(md: str) -> list[str]:
    """Split a single markdown string into per-page strings using page markers.

    The ZAI API embeds markers like ![](page=N,bbox=...) throughout the output.
    Page numbers from the API are 0-indexed.
    Returns a list where index i contains all content belonging to API page i.
    """
    markers = list(_PAGE_MARKER_RE.finditer(md))
    if not markers:
        return [md]

    max_page = max(int(m.group(1)) for m in markers)
    pages: list[list[str]] = [[] for _ in range(max_page + 1)]

    # Split content at each marker boundary
    for i, marker in enumerate(markers):
        page_num = int(marker.group(1))
        start = marker.start()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(md)
        pages[page_num].append(md[start:end])

    # Any content before the first marker belongs to page 0
    preamble = md[: markers[0].start()].strip()
    if preamble:
        pages[0].insert(0, preamble)

    return ["\n".join(parts).strip() for parts in pages]


class ZAIProvider(OCRProvider):
    """OCR provider using ZAI SDK's layout_parsing API (PDF-native via base64)."""

    def __init__(self) -> None:
        api_key = os.environ.get("ZAI_API_KEY")
        if not api_key:
            raise ValueError("ZAI_API_KEY environment variable is required for ZAI OCR provider")
        from zai import ZaiClient  # type: ignore[import-untyped]

        self._client = ZaiClient(api_key=api_key)

    def ocr_pages(self, images, *, page_offset: int = 0) -> str:
        raise NotImplementedError("ZAIProvider uses ocr_document; call run_ocr with a PDF path")

    def ocr_document(self, pdf_path: str) -> list[str]:
        with open(pdf_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("utf-8")
        file_data = f"data:application/pdf;base64,{pdf_b64}"
        result = self._client.layout_parsing.create(model="glm-ocr", file=file_data)
        md = result.md_results
        if isinstance(md, list):
            return [str(p) for p in md]
        return _split_by_pages(str(md))
