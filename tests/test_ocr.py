from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pipeline.ocr import _after_date, make_page_range, pdf_to_images, run_ocr


class TestAfterDate:
    def test_returns_false_when_since_is_none(self):
        row = {"added": "2024-01-01T00:00:00+00:00"}
        assert _after_date(row, None) is False

    def test_returns_true_when_added_after_since(self):
        row = {"added": "2024-06-01T00:00:00+00:00"}
        assert _after_date(row, "2024-01-01") is True

    def test_returns_false_when_added_before_since(self):
        row = {"added": "2023-01-01T00:00:00+00:00"}
        assert _after_date(row, "2024-01-01") is False

    def test_returns_true_when_added_equals_since(self):
        row = {"added": "2024-01-01T00:00:00+00:00"}
        assert _after_date(row, "2024-01-01") is True

    def test_returns_false_when_added_is_none(self):
        row = {"added": None}
        assert _after_date(row, "2024-01-01") is False


class TestMakePageRange:
    def test_single_batch(self):
        assert make_page_range(1, 75) == "1-75"

    def test_second_batch(self):
        assert make_page_range(76, 150) == "76-150"

    def test_partial_batch(self):
        assert make_page_range(151, 200) == "151-200"


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
            run_ocr("doc1", client, force=False, since=None)
            mock_provider.assert_not_called()

    def test_raises_if_no_pipeline_row(self):
        client = self._make_client(pipeline_row=None)
        with pytest.raises(ValueError, match="No pipeline row"):
            run_ocr("doc1", client)

    def test_runs_ocr_for_new_doc(self, fixture_pdf):
        pipeline_row = self._make_pipeline_row(last_ocr=None)
        bronze_row = self._make_bronze_row(file_path=fixture_pdf)
        client = self._make_client(pipeline_row, bronze_row)

        mock_provider = MagicMock()
        mock_provider.ocr_pages.return_value = "page text"

        with patch("pipeline.ocr.get_ocr_provider", return_value=mock_provider):
            run_ocr("doc1", client)

        mock_provider.ocr_pages.assert_called_once()
        # Should have called upsert on ocr_results
        upsert_calls = [
            c for c in client.table.call_args_list if c[0][0] == "ocr_results"
        ]
        assert len(upsert_calls) >= 1

    def test_force_reruns_ocr(self, fixture_pdf):
        pipeline_row = self._make_pipeline_row(last_ocr="2024-01-01T00:00:00+00:00")
        bronze_row = self._make_bronze_row(file_path=fixture_pdf)
        client = self._make_client(pipeline_row, bronze_row)

        mock_provider = MagicMock()
        mock_provider.ocr_pages.return_value = "page text"

        with patch("pipeline.ocr.get_ocr_provider", return_value=mock_provider):
            run_ocr("doc1", client, force=True)

        mock_provider.ocr_pages.assert_called()

    def test_chunks_pages_in_batches(self, fixture_pdf):
        """Test that pages are chunked when exceeding MAX_PAGES_PER_BATCH."""
        pipeline_row = self._make_pipeline_row(last_ocr=None)
        bronze_row = self._make_bronze_row(file_path=fixture_pdf)
        client = self._make_client(pipeline_row, bronze_row)

        # Create 100 fake images to simulate a long document
        fake_images = [Image.new("RGB", (10, 10)) for _ in range(100)]
        mock_provider = MagicMock()
        mock_provider.ocr_pages.return_value = "chunk text"

        with (
            patch("pipeline.ocr.get_ocr_provider", return_value=mock_provider),
            patch("pipeline.ocr.pdf_to_images", return_value=fake_images),
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

        fake_images = [Image.new("RGB", (10, 10)) for _ in range(100)]
        mock_provider = MagicMock()
        mock_provider.ocr_pages.return_value = "chunk text"

        with (
            patch("pipeline.ocr.get_ocr_provider", return_value=mock_provider),
            patch("pipeline.ocr.pdf_to_images", return_value=fake_images),
            patch("pipeline.ocr.MAX_PAGES_PER_BATCH", 75),
        ):
            run_ocr("doc1", client)

        first_call_kwargs = mock_provider.ocr_pages.call_args_list[0][1]
        second_call_kwargs = mock_provider.ocr_pages.call_args_list[1][1]
        assert first_call_kwargs["page_offset"] == 0
        assert second_call_kwargs["page_offset"] == 75

    def test_deletes_existing_rows_before_reocr(self, fixture_pdf):
        pipeline_row = self._make_pipeline_row(last_ocr=None)
        bronze_row = self._make_bronze_row(file_path=fixture_pdf)
        client = self._make_client(pipeline_row, bronze_row)

        mock_provider = MagicMock()
        mock_provider.ocr_pages.return_value = "page text"

        with (
            patch("pipeline.ocr.get_ocr_provider", return_value=mock_provider),
            patch("pipeline.ocr.delete_ocr_rows") as mock_delete,
        ):
            run_ocr("doc1", client)

        mock_delete.assert_called_once_with(client, "doc1")
