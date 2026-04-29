from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pipeline.ocr import _after_date, pdf_to_images, run_ocr
from pipeline.ocr_providers.zai_provider import _split_by_pages


class TestAfterDate:
    def test_returns_false_when_since_is_none(self):
        row = {"last_ocr": "2024-01-01T00:00:00+00:00"}
        assert _after_date(row, None) is False

    def test_returns_true_when_ocr_after_since(self):
        row = {"last_ocr": "2024-06-01T00:00:00+00:00"}
        assert _after_date(row, "2024-01-01") is True

    def test_returns_false_when_ocr_before_since(self):
        row = {"last_ocr": "2023-01-01T00:00:00+00:00"}
        assert _after_date(row, "2024-01-01") is False

    def test_returns_true_when_ocr_equals_since(self):
        row = {"last_ocr": "2024-01-01T00:00:00+00:00"}
        assert _after_date(row, "2024-01-01") is True

    def test_returns_true_when_last_ocr_is_none(self):
        # No last_ocr means never processed — always eligible when --parse-date is used
        row = {"last_ocr": None}
        assert _after_date(row, "2024-01-01") is True


class TestSplitByPages:
    def test_no_markers_returns_single_page(self):
        md = "Some plain text without markers"
        result = _split_by_pages(md)
        assert result == [md]

    def test_single_page_marker(self):
        md = "![](page=0,bbox=0,0,100,100) Page zero content"
        result = _split_by_pages(md)
        assert len(result) == 1
        assert "Page zero content" in result[0]

    def test_two_pages(self):
        md = (
            "![](page=0,bbox=0,0,100,100) First page stuff\n"
            "![](page=1,bbox=0,0,100,100) Second page stuff"
        )
        result = _split_by_pages(md)
        assert len(result) == 2
        assert "First page" in result[0]
        assert "Second page" in result[1]

    def test_preamble_goes_to_page_zero(self):
        md = (
            "Preamble text\n"
            "![](page=0,bbox=0,0,100,100) Page zero\n"
            "![](page=1,bbox=0,0,100,100) Page one"
        )
        result = _split_by_pages(md)
        assert len(result) == 2
        assert "Preamble text" in result[0]
        assert "Page zero" in result[0]
        assert "Page one" in result[1]

    def test_multiple_markers_same_page(self):
        md = (
            "![](page=0,bbox=0,0,50,50) First block\n"
            "![](page=0,bbox=50,0,100,50) Second block\n"
            "![](page=1,bbox=0,0,100,100) Page one"
        )
        result = _split_by_pages(md)
        assert len(result) == 2
        assert "First block" in result[0]
        assert "Second block" in result[0]
        assert "Page one" in result[1]

    def test_three_pages(self):
        md = (
            "![](page=0,bbox=0,0,100,100) P0\n"
            "![](page=1,bbox=0,0,100,100) P1\n"
            "![](page=2,bbox=0,0,100,100) P2"
        )
        result = _split_by_pages(md)
        assert len(result) == 3
        assert "P0" in result[0]
        assert "P1" in result[1]
        assert "P2" in result[2]


class TestPdfToImages:
    def test_returns_list_of_images(self, fixture_pdf):
        images = pdf_to_images(fixture_pdf)
        assert len(images) >= 1
        assert all(isinstance(img, Image.Image) for img in images)


class TestRunOcr:
    def _make_pipeline_row(self, last_ocr=None):
        return {"doc_id": "doc1", "last_ocr": last_ocr, "added": "2024-01-01T00:00:00+00:00"}

    def _make_bronze_row(self, file_path="/path/test.pdf"):
        return {"file_path": file_path}

    def _make_client(self, pipeline_row, bronze_row=None):
        client = MagicMock()

        def table_side_effect(name):
            chain = MagicMock()
            result = MagicMock()
            if name == "pipeline":
                result.data = [pipeline_row] if pipeline_row else []
            elif name == "bronze_mapping":
                result.data = [bronze_row] if bronze_row else []
            else:
                result.data = []
            chain.execute.return_value = result
            chain.select.return_value = chain
            chain.update.return_value = chain
            chain.upsert.return_value = chain
            chain.delete.return_value = chain
            chain.eq.return_value = chain
            return chain

        client.table.side_effect = table_side_effect
        return client

    def test_skips_if_already_ocrd_no_force(self):
        pipeline_row = self._make_pipeline_row(last_ocr="2024-01-01T00:00:00+00:00")
        client = self._make_client(pipeline_row)

        with patch("pipeline.ocr.get_ocr_provider") as mock_provider:
            result = run_ocr("doc1", client, force=False, since=None)
            mock_provider.assert_not_called()
            assert result == {"status": "skipped", "empty_pages": 0}

    def test_raises_if_no_pipeline_row(self):
        client = self._make_client(pipeline_row=None)
        with pytest.raises(ValueError, match="No pipeline row"):
            run_ocr("doc1", client)

    def test_runs_ocr_for_new_doc(self, fixture_pdf):
        pipeline_row = self._make_pipeline_row(last_ocr=None)
        bronze_row = self._make_bronze_row(file_path=fixture_pdf)
        client = self._make_client(pipeline_row, bronze_row)

        mock_provider = MagicMock()
        mock_provider.ocr_document.side_effect = NotImplementedError
        mock_provider.ocr_pages.return_value = "page text"

        with (
            patch("pipeline.ocr.get_ocr_provider", return_value=mock_provider),
            patch("pipeline.ocr.silver_bulk_upsert") as mock_upsert,
        ):
            run_ocr("doc1", client)

        mock_provider.ocr_pages.assert_called_once()
        mock_upsert.assert_called_once()

    def test_force_reruns_ocr(self, fixture_pdf):
        pipeline_row = self._make_pipeline_row(last_ocr="2024-01-01T00:00:00+00:00")
        bronze_row = self._make_bronze_row(file_path=fixture_pdf)
        client = self._make_client(pipeline_row, bronze_row)

        mock_provider = MagicMock()
        mock_provider.ocr_document.side_effect = NotImplementedError
        mock_provider.ocr_pages.return_value = "page text"

        with patch("pipeline.ocr.get_ocr_provider", return_value=mock_provider):
            run_ocr("doc1", client, force=True)

        mock_provider.ocr_pages.assert_called()

    def test_chunks_pages_in_batches(self, fixture_pdf):
        """Test that pages are chunked when exceeding MAX_PAGES_PER_BATCH."""
        pipeline_row = self._make_pipeline_row(last_ocr=None)
        bronze_row = self._make_bronze_row(file_path=fixture_pdf)
        client = self._make_client(pipeline_row, bronze_row)

        mock_provider = MagicMock()
        mock_provider.ocr_document.side_effect = NotImplementedError
        mock_provider.ocr_pages.return_value = "chunk text"

        _real_fitz_open = __import__("fitz").open
        call_count = [0]

        def _fake_fitz_open(path=None):
            if path is None:
                return _real_fitz_open()
            call_count[0] += 1
            if call_count[0] == 1:
                # Fast path: return real doc so provider.ocr_document raises NotImplementedError
                return _real_fitz_open(path)
            # Slow path: return mock 100-page doc
            mock_doc = MagicMock()
            mock_doc.__len__ = lambda s: 100
            mock_doc.__enter__ = lambda s: s
            mock_doc.__exit__ = MagicMock(return_value=False)
            return mock_doc

        def fake_pdf_to_images(path, start=0, end=None):
            count = (end or 100) - start
            return [Image.new("RGB", (10, 10)) for _ in range(count)]

        with (
            patch("pipeline.ocr.get_ocr_provider", return_value=mock_provider),
            patch("pipeline.ocr.pdf_to_images", side_effect=fake_pdf_to_images),
            patch("pipeline.ocr.fitz.open", side_effect=_fake_fitz_open),
            patch("pipeline.ocr.MAX_PAGES_PER_BATCH", 75),
        ):
            run_ocr("doc1", client)

        # Should be called twice: 75 pages + 25 pages
        assert mock_provider.ocr_pages.call_count == 2
        first_batch = mock_provider.ocr_pages.call_args_list[0][0][0]
        second_batch = mock_provider.ocr_pages.call_args_list[1][0][0]
        assert len(first_batch) == 75
        assert len(second_batch) == 25

    def test_page_offset_passed_to_provider(self, fixture_pdf):
        """Second chunk should have page_offset=75 so pages are numbered 76+."""
        pipeline_row = self._make_pipeline_row(last_ocr=None)
        bronze_row = self._make_bronze_row(file_path=fixture_pdf)
        client = self._make_client(pipeline_row, bronze_row)

        mock_provider = MagicMock()
        mock_provider.ocr_document.side_effect = NotImplementedError
        mock_provider.ocr_pages.return_value = "chunk text"

        _real_fitz_open = __import__("fitz").open
        call_count = [0]

        def _fake_fitz_open(path=None):
            if path is None:
                return _real_fitz_open()
            call_count[0] += 1
            if call_count[0] == 1:
                return _real_fitz_open(path)
            mock_doc = MagicMock()
            mock_doc.__len__ = lambda s: 100
            mock_doc.__enter__ = lambda s: s
            mock_doc.__exit__ = MagicMock(return_value=False)
            return mock_doc

        def fake_pdf_to_images(path, start=0, end=None):
            count = (end or 100) - start
            return [Image.new("RGB", (10, 10)) for _ in range(count)]

        with (
            patch("pipeline.ocr.get_ocr_provider", return_value=mock_provider),
            patch("pipeline.ocr.pdf_to_images", side_effect=fake_pdf_to_images),
            patch("pipeline.ocr.fitz.open", side_effect=_fake_fitz_open),
            patch("pipeline.ocr.MAX_PAGES_PER_BATCH", 75),
        ):
            run_ocr("doc1", client)

        first_call_kwargs = mock_provider.ocr_pages.call_args_list[0][1]
        second_call_kwargs = mock_provider.ocr_pages.call_args_list[1][1]
        assert first_call_kwargs["page_offset"] == 0
        assert second_call_kwargs["page_offset"] == 75

    def test_ocr_document_fast_path_stores_per_page(self, fixture_pdf):
        """When ocr_document returns multiple pages, each gets its own row."""
        pipeline_row = self._make_pipeline_row(last_ocr=None)
        bronze_row = self._make_bronze_row(file_path=fixture_pdf)
        client = self._make_client(pipeline_row, bronze_row)

        mock_provider = MagicMock()
        mock_provider.ocr_document.return_value = ["Page 0 content", "Page 1 content", "Page 2 content"]

        with (
            patch("pipeline.ocr.get_ocr_provider", return_value=mock_provider),
            patch("pipeline.ocr.ZAI_MAX_PAGES", 100),
            patch("pipeline.ocr.silver_bulk_upsert") as mock_bulk_upsert,
        ):
            result = run_ocr("doc1", client)

        # Should store 3 rows in a single bulk upsert
        mock_bulk_upsert.assert_called_once()
        rows = mock_bulk_upsert.call_args[0][2]
        assert len(rows) == 3
        page_numbers = [r["page_number"] for r in rows]
        assert page_numbers == [1, 2, 3]
        assert result["status"] == "done"
        assert result["empty_pages"] == 0

    def test_empty_pages_flagged_in_result(self, fixture_pdf):
        """Empty OCR pages are counted and logged as errors."""
        pipeline_row = self._make_pipeline_row(last_ocr=None)
        bronze_row = self._make_bronze_row(file_path=fixture_pdf)
        client = self._make_client(pipeline_row, bronze_row)

        mock_provider = MagicMock()
        mock_provider.ocr_document.return_value = ["Content", "", "  ", "More content"]

        with (
            patch("pipeline.ocr.get_ocr_provider", return_value=mock_provider),
            patch("pipeline.ocr.ZAI_MAX_PAGES", 100),
            patch("pipeline.ocr.silver_bulk_upsert"),
            patch("pipeline.ocr.append_error") as mock_err,
        ):
            result = run_ocr("doc1", client)

        assert result["status"] == "done"
        assert result["empty_pages"] == 2
        mock_err.assert_called_once()
        assert "2 empty page(s)" in mock_err.call_args[0][2]

    def test_does_not_delete_existing_rows_before_reocr(self, fixture_pdf):
        """OCR upserts in place; previous results are preserved if the run fails partway."""
        pipeline_row = self._make_pipeline_row(last_ocr=None)
        bronze_row = self._make_bronze_row(file_path=fixture_pdf)
        client = self._make_client(pipeline_row, bronze_row)

        mock_provider = MagicMock()
        mock_provider.ocr_document.side_effect = NotImplementedError
        mock_provider.ocr_pages.return_value = "page text"

        with (
            patch("pipeline.ocr.get_ocr_provider", return_value=mock_provider),
            patch("pipeline.ocr.delete_ocr_rows") as mock_delete,
        ):
            run_ocr("doc1", client)

        mock_delete.assert_not_called()
