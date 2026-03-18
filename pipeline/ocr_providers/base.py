from abc import ABC, abstractmethod

from PIL import Image


class OCRProvider(ABC):
    """Abstract base class for OCR providers."""

    @abstractmethod
    def ocr_pages(self, images: list[Image.Image], *, page_offset: int = 0) -> str:
        """
        Run OCR on a list of page images and return the combined text.
        Each page is separated by a page header using absolute page numbers.
        page_offset is added to enumerate start so page numbers are absolute.
        """

    def ocr_document(self, pdf_path: str) -> list[str]:
        """
        Run OCR on a whole PDF and return a list of page strings (one per page).
        Override in providers that support native PDF ingestion.
        Raises NotImplementedError by default (triggers image-based fallback).
        """
        raise NotImplementedError
