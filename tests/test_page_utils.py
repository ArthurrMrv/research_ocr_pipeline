import pytest

from pipeline.page_utils import extract_pages, get_total_pages

SAMPLE_OCR = (
    "--- Page 1 ---\nIntroduction text.\n\n"
    "--- Page 2 ---\nModel specifications.\n\n"
    "--- Page 3 ---\nResults table.\n\n"
    "--- Page 4 ---\nConclusion."
)


class TestGetTotalPages:
    def test_returns_highest_page_number(self):
        assert get_total_pages(SAMPLE_OCR) == 4

    def test_returns_zero_for_empty_string(self):
        assert get_total_pages("") == 0

    def test_returns_zero_for_text_without_markers(self):
        assert get_total_pages("some text without page markers") == 0

    def test_handles_non_contiguous_pages(self):
        text = "--- Page 5 ---\nfoo\n\n--- Page 10 ---\nbar"
        assert get_total_pages(text) == 10

    def test_single_page(self):
        text = "--- Page 1 ---\nonly page"
        assert get_total_pages(text) == 1


class TestExtractPages:
    def test_extracts_single_page(self):
        result = extract_pages(SAMPLE_OCR, 2, 2)
        assert "Model specifications" in result
        assert "Introduction" not in result
        assert "Results table" not in result

    def test_extracts_page_range(self):
        result = extract_pages(SAMPLE_OCR, 2, 3)
        assert "Model specifications" in result
        assert "Results table" in result
        assert "Introduction" not in result
        assert "Conclusion" not in result

    def test_extracts_all_pages(self):
        result = extract_pages(SAMPLE_OCR, 1, 4)
        assert "Introduction" in result
        assert "Model specifications" in result
        assert "Results table" in result
        assert "Conclusion" in result

    def test_returns_empty_string_for_out_of_range(self):
        result = extract_pages(SAMPLE_OCR, 10, 20)
        assert result == ""

    def test_preserves_page_markers(self):
        result = extract_pages(SAMPLE_OCR, 1, 1)
        assert "--- Page 1 ---" in result

    def test_handles_text_without_markers(self):
        result = extract_pages("no markers here", 1, 5)
        assert result == ""

    def test_large_chunk_absolute_page_numbers(self):
        # Simulate chunk 2 of a 150-page doc (pages 76-100)
        chunk2_pages = "\n\n".join(
            f"--- Page {i} ---\nContent of page {i}" for i in range(76, 101)
        )
        result = extract_pages(chunk2_pages, 80, 82)
        assert "--- Page 80 ---" in result
        assert "--- Page 82 ---" in result
        assert "--- Page 79 ---" not in result
        assert "--- Page 83 ---" not in result
