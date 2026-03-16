from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pipeline.ocr import _after_date, ocr_image, pdf_to_images, run_ocr


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


class TestOcrImage:
    def test_calls_model_and_returns_text(self):
        image = Image.new("RGB", (10, 10))
        mock_processor = MagicMock()
        mock_model = MagicMock()

        mock_processor.return_value = {"input_ids": MagicMock()}
        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = ["  extracted text  "]

        result = ocr_image(image, mock_model, mock_processor)

        assert result == "extracted text"
        mock_model.generate.assert_called_once()
        mock_processor.batch_decode.assert_called_once()


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
            chain.eq.return_value = chain
            return chain

        client.table.side_effect = table_side_effect
        return client

    def test_skips_if_already_ocrd_no_force(self):
        pipeline_row = self._make_pipeline_row(last_ocr="2024-01-01T00:00:00+00:00")
        client = self._make_client(pipeline_row)

        with patch("pipeline.ocr.load_ocr_model") as mock_load:
            run_ocr("doc1", client, force=False, since=None)
            mock_load.assert_not_called()

    def test_raises_if_no_pipeline_row(self):
        client = self._make_client(pipeline_row=None)
        with pytest.raises(ValueError, match="No pipeline row"):
            run_ocr("doc1", client)

    def test_runs_ocr_for_new_doc(self, fixture_pdf):
        pipeline_row = self._make_pipeline_row(last_ocr=None)
        bronze_row = self._make_bronze_row(file_path=fixture_pdf)
        client = self._make_client(pipeline_row, bronze_row)

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.return_value = {}
        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = ["page text"]

        with patch("pipeline.ocr.load_ocr_model", return_value=(mock_model, mock_processor)):
            run_ocr("doc1", client)

        # Should have called upsert on ocr_results
        upsert_calls = [
            c for c in client.table.call_args_list if c[0][0] == "ocr_results"
        ]
        assert len(upsert_calls) >= 1

    def test_force_reruns_ocr(self, fixture_pdf):
        pipeline_row = self._make_pipeline_row(last_ocr="2024-01-01T00:00:00+00:00")
        bronze_row = self._make_bronze_row(file_path=fixture_pdf)
        client = self._make_client(pipeline_row, bronze_row)

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.return_value = {}
        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = ["page text"]

        with patch("pipeline.ocr.load_ocr_model", return_value=(mock_model, mock_processor)):
            run_ocr("doc1", client, force=True)

        mock_model.generate.assert_called()
