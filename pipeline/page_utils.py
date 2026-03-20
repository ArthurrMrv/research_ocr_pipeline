from __future__ import annotations

import re

PAGE_MARKER_RE = re.compile(r"^---\s*Page\s+\d+\s*---\s*$", re.MULTILINE)


def is_empty_page_content(content: str) -> bool:
    """Return True if a stored OCR row contains no real text beyond the page marker."""
    return not PAGE_MARKER_RE.sub("", content).strip()


def get_total_pages(ocr_text: str) -> int:
    """Return the highest page number found in OCR text."""
    matches = re.findall(r"--- Page (\d+) ---", ocr_text)
    if not matches:
        return 0
    return max(int(m) for m in matches)


def extract_pages(ocr_text: str, start_page: int, end_page: int) -> str:
    """
    Extract text for pages start_page..end_page (inclusive) from OCR text
    that has absolute '--- Page N ---' markers.
    Returns concatenated sections for the requested range.
    """
    sections = re.split(r"(?=--- Page \d+ ---)", ocr_text)
    result_sections = []
    for section in sections:
        m = re.match(r"--- Page (\d+) ---", section)
        if m:
            page_num = int(m.group(1))
            if start_page <= page_num <= end_page:
                result_sections.append(section.strip())
    return "\n\n".join(result_sections)
