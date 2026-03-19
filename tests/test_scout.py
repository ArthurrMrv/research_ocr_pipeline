from unittest.mock import MagicMock, patch

import pytest

from pipeline.scout import run_scout


def _make_client(pipeline_row, ocr_chunks=None, scout_scores=None):
    client = MagicMock()

    def table_side_effect(name):
        chain = MagicMock()
        result = MagicMock()
        if name == "pipeline":
            result.data = [pipeline_row] if pipeline_row else []
        elif name == "ocr_results":
            result.data = ocr_chunks if ocr_chunks is not None else []
        elif name == "scout_page_scores":
            result.data = scout_scores if scout_scores is not None else []
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


class TestRunScout:
    def _make_pipeline_row(self, last_scout=None):
        return {"doc_id": "doc1", "last_scout": last_scout}

    def test_skips_if_already_scouted_no_force(self):
        pipeline_row = self._make_pipeline_row(last_scout="2024-01-01T00:00:00+00:00")
        ocr_chunks = [{"content": "--- Page 1 ---\ntext", "page_number": 1}]
        scout_scores = [
            {"step_name": "extract_model_name", "page_number": 1, "score": 0.9},
            {"step_name": "extract_table", "page_number": 1, "score": 0.6},
        ]
        client = _make_client(pipeline_row, ocr_chunks, scout_scores)

        with patch("pipeline.scout._score_page") as mock_score:
            result = run_scout("doc1", client, force=False)
            assert result == "skipped"
            mock_score.assert_not_called()

    def test_raises_if_no_pipeline_row(self):
        client = _make_client(pipeline_row=None)
        with pytest.raises(ValueError, match="No pipeline row"):
            run_scout("doc1", client)

    def test_raises_if_no_ocr_results(self):
        pipeline_row = self._make_pipeline_row()
        client = _make_client(pipeline_row, ocr_chunks=[])
        with pytest.raises(ValueError, match="No OCR results"):
            run_scout("doc1", client)

    def test_runs_scout_and_upserts_results(self):
        pipeline_row = self._make_pipeline_row()
        ocr_chunks = [{"content": "--- Page 1 ---\nsome text", "page_number": 1}]
        client = _make_client(pipeline_row, ocr_chunks)

        scout_output = {"extract_model_name": 0.9, "extract_table": 0.4}
        step_config = ("prompt", {}, {"provider": "gemini", "model": "gemini-2.0-flash-lite"})

        with (
            patch("pipeline.scout._score_page", return_value=scout_output),
            patch("pipeline.scout.load_step", return_value=step_config),
            patch(
                "pipeline.scout.load_step_config",
                side_effect=[
                    {"definition": "model evidence"},
                    {"definition": "table evidence"},
                ],
            ),
            patch("pipeline.scout.get_provider"),
            patch("pipeline.scout.scout_page_score_upsert") as mock_upsert,
            patch("pipeline.scout.delete_scout_page_scores"),
            patch("pipeline.scout.pipeline_update") as mock_update,
            patch("pipeline.scout.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
        ):
            run_scout("doc1", client)

        assert mock_upsert.call_count == 2
        upserted_steps = {call[0][1]["step_name"] for call in mock_upsert.call_args_list}
        assert upserted_steps == {"extract_model_name", "extract_table"}
        mock_update.assert_called_once()

    def test_skips_empty_ocr_pages(self):
        pipeline_row = self._make_pipeline_row()
        ocr_chunks = [
            {"content": "--- Page 1 ---\ntext", "page_number": 1},
            {"content": "--- Page 2 ---\n", "page_number": 2},
        ]
        client = _make_client(pipeline_row, ocr_chunks)
        step_config = ("prompt", {}, {"provider": "gemini", "model": "gemini-2.0-flash-lite"})

        with (
            patch("pipeline.scout._score_page", return_value={"extract_model_name": 0.8, "extract_table": 0.2}) as mock_score,
            patch("pipeline.scout.load_step", return_value=step_config),
            patch(
                "pipeline.scout.load_step_config",
                side_effect=[
                    {"definition": "model evidence"},
                    {"definition": "table evidence"},
                ],
            ),
            patch("pipeline.scout.get_provider"),
            patch("pipeline.scout.scout_page_score_upsert"),
            patch("pipeline.scout.delete_scout_page_scores"),
            patch("pipeline.scout.pipeline_update"),
        ):
            run_scout("doc1", client)

        mock_score.assert_called_once()

    def test_appends_error_when_run_step_returns_none(self):
        pipeline_row = self._make_pipeline_row()
        ocr_chunks = [{"content": "--- Page 1 ---\ntext", "page_number": 1}]
        client = _make_client(pipeline_row, ocr_chunks)

        with (
            patch("pipeline.scout._score_page", return_value=None),
            patch("pipeline.scout.load_step", return_value=("prompt", {}, {"provider": "gemini", "model": "gemini-2.0-flash-lite"})),
            patch(
                "pipeline.scout.load_step_config",
                side_effect=[
                    {"definition": "model evidence"},
                    {"definition": "table evidence"},
                ],
            ),
            patch("pipeline.scout.get_provider"),
            patch("pipeline.scout.delete_scout_page_scores"),
            patch("pipeline.scout.append_error") as mock_err,
        ):
            run_scout("doc1", client)

        mock_err.assert_called_once()
        assert "Scout" in mock_err.call_args[0][2]

    def test_force_reruns_even_if_scouted(self):
        pipeline_row = self._make_pipeline_row(last_scout="2024-01-01T00:00:00+00:00")
        ocr_chunks = [{"content": "--- Page 1 ---\ntext", "page_number": 1}]
        client = _make_client(pipeline_row, ocr_chunks)

        scout_output = {"extract_model_name": 0.7}
        step_config = ("prompt", {}, {"provider": "gemini", "model": "gemini-2.0-flash-lite"})

        with (
            patch("pipeline.scout._score_page", return_value=scout_output) as mock_score,
            patch("pipeline.scout.load_step", return_value=step_config),
            patch("pipeline.scout.load_step_config", return_value={"definition": "model evidence"}),
            patch("pipeline.scout.get_provider"),
            patch("pipeline.scout.scout_page_score_upsert"),
            patch("pipeline.scout.delete_scout_page_scores"),
            patch("pipeline.scout.pipeline_update"),
            patch("pipeline.scout.ACTIVE_STEPS", ["extract_model_name"]),
        ):
            run_scout("doc1", client, force=True)

        mock_score.assert_called_once()

    def test_raises_if_active_step_definition_is_missing(self):
        pipeline_row = self._make_pipeline_row()
        ocr_chunks = [{"content": "--- Page 1 ---\ntext", "page_number": 1}]
        client = _make_client(pipeline_row, ocr_chunks)

        with (
            patch("pipeline.scout.load_step", return_value=("prompt", {}, {"provider": "gemini", "model": "gemini-2.0-flash-lite"})),
            patch("pipeline.scout.load_step_config", return_value={}),
            patch("pipeline.scout.ACTIVE_STEPS", ["extract_model_name"]),
        ):
            with pytest.raises(ValueError, match="Missing 'definition'"):
                run_scout("doc1", client)

    def test_passes_rendered_us_equity_definition_to_scout_prompt(self):
        pipeline_row = self._make_pipeline_row()
        ocr_chunks = [{"content": "--- Page 1 ---\ntext", "page_number": 1}]
        client = _make_client(pipeline_row, ocr_chunks)
        step_config = (
            "Return scores for {step_names}\n{step_definitions}",
            {},
            {"provider": "gemini", "model": "gemini-2.0-flash-lite"},
        )

        with (
            patch("pipeline.scout._score_page", return_value={"extract_model_name": 0.8}) as mock_score,
            patch("pipeline.scout.load_step", return_value=step_config),
            patch(
                "pipeline.scout.load_step_config",
                return_value={
                    "definition": "Pages about the U.S. equity market return forecast methodology and assumptions."
                },
            ),
            patch("pipeline.scout.get_provider", return_value=MagicMock()),
            patch("pipeline.scout.scout_page_score_upsert"),
            patch("pipeline.scout.delete_scout_page_scores"),
            patch("pipeline.scout.pipeline_update"),
            patch("pipeline.scout.ACTIVE_STEPS", ["extract_model_name"]),
        ):
            run_scout("doc1", client)

        rendered_prompt = mock_score.call_args[0][1]
        assert "U.S. equity market return forecast methodology and assumptions" in rendered_prompt
