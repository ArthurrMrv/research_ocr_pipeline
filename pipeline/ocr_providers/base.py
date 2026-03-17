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
