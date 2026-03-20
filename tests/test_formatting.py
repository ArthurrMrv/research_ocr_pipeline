import json
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from pipeline.formatting import load_step, run_formatting, run_step, validate_output
from pipeline.providers.base import NonJSONResponseError
from pipeline.step_errors import MissingPromptError


def _make_step_dir(tmp_path, step_name, config=None, schema=None, prompt=None):
    step_dir = tmp_path / step_name
    step_dir.mkdir(exist_ok=True)
    (step_dir / "prompt.txt").write_text(prompt or "Extract: {ocr_text}")
    (step_dir / "schema.json").write_text(
        json.dumps(
            schema
            or {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["result"],
                "properties": {"result": {"type": ["string", "null"]}},
            }
        )
    )
    (step_dir / "config.json").write_text(
        json.dumps(config or {"provider": "moonshot", "model": "kimi-k2.5"})
    )
    return step_dir


class TestLoadStep:
    def test_loads_all_three_files(self, tmp_path):
        _make_step_dir(tmp_path, "my_step", prompt="Hello {ocr_text}")
        with patch("pipeline.formatting.STEPS_DIR", tmp_path):
            prompt, schema, config = load_step("my_step")
        assert "{ocr_text}" in prompt
        assert "type" in schema
        assert "provider" in config

    def test_raises_on_missing_step(self, tmp_path):
        with patch("pipeline.formatting.STEPS_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                load_step("nonexistent_step")

    def test_loads_institution_specific_prompt(self, tmp_path):
        step_dir = _make_step_dir(
            tmp_path, "my_step",
            config={"provider": "moonshot", "model": "kimi-k2.5", "per_company": True},
        )
        prompts_dir = step_dir / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "Apple.txt").write_text("Apple-specific prompt")

        with patch("pipeline.formatting.STEPS_DIR", tmp_path):
            prompt, schema, config = load_step("my_step", institution="Apple")
        assert prompt == "Apple-specific prompt"

    def test_institution_prompt_case_insensitive(self, tmp_path):
        step_dir = _make_step_dir(
            tmp_path, "my_step",
            config={"provider": "moonshot", "model": "kimi-k2.5", "per_company": True},
        )
        prompts_dir = step_dir / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "Apple.txt").write_text("Apple prompt")

        with patch("pipeline.formatting.STEPS_DIR", tmp_path):
            prompt, _, _ = load_step("my_step", institution="apple")
        assert prompt == "Apple prompt"

    def test_per_company_missing_prompt_uses_fallback(self, tmp_path):
        _make_step_dir(
            tmp_path, "my_step",
            config={"provider": "moonshot", "model": "kimi-k2.5", "per_company": True},
            prompt="fallback prompt",
        )
        with patch("pipeline.formatting.STEPS_DIR", tmp_path):
            prompt, _, _ = load_step("my_step", institution="Unknown")
        assert prompt == "fallback prompt"

    def test_per_company_no_institution_uses_fallback(self, tmp_path):
        _make_step_dir(
            tmp_path, "my_step",
            config={"provider": "moonshot", "model": "kimi-k2.5", "per_company": True},
            prompt="fallback prompt",
        )
        with patch("pipeline.formatting.STEPS_DIR", tmp_path):
            prompt, _, _ = load_step("my_step", institution=None)
        assert prompt == "fallback prompt"


class TestValidateOutput:
    _schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["value"],
        "properties": {"value": {"type": "string"}},
    }

    def test_passes_valid_output(self):
        validate_output({"value": "hello"}, self._schema, "test_step")

    def test_raises_on_invalid_output(self):
        with pytest.raises(jsonschema.ValidationError):
            validate_output({"value": 123}, self._schema, "test_step")

    def test_raises_on_missing_required_key(self):
        with pytest.raises(jsonschema.ValidationError):
            validate_output({}, self._schema, "test_step")


class TestRunStep:
    def test_returns_valid_result(self, tmp_path):
        _make_step_dir(tmp_path, "step1")
        mock_provider = MagicMock()
        mock_provider.call.return_value = {"result": "found"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.get_provider", return_value=mock_provider),
        ):
            result, model = run_step("step1", "some ocr text")

        assert result == {"result": "found"}
        assert model == "kimi-k2.5"
        mock_provider.call.assert_called_once()

    def test_retries_on_schema_failure_then_returns_none(self, tmp_path):
        _make_step_dir(tmp_path, "step1")
        mock_provider = MagicMock()
        mock_provider.call.return_value = {"wrong_key": "value"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.get_provider", return_value=mock_provider),
        ):
            result, model = run_step("step1", "ocr text")

        assert result is None
        assert model == "kimi-k2.5"
        assert mock_provider.call.call_count == 2

    def test_returns_result_on_second_attempt(self, tmp_path):
        _make_step_dir(tmp_path, "step1")
        mock_provider = MagicMock()
        mock_provider.call.side_effect = [
            {"wrong_key": "bad"},
            {"result": "ok"},
        ]

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.get_provider", return_value=mock_provider),
        ):
            result, model = run_step("step1", "ocr text")

        assert result == {"result": "ok"}

    def test_raises_missing_prompt_error_when_prompt_is_none(self, tmp_path):
        """When per_company step has no matching prompt, raises MissingPromptError."""
        _make_step_dir(
            tmp_path, "step1",
            config={"provider": "moonshot", "model": "kimi-k2.5", "per_company": True},
        )
        # Remove the fallback prompt so it returns None
        (tmp_path / "step1" / "prompt.txt").unlink()

        with patch("pipeline.formatting.STEPS_DIR", tmp_path):
            with pytest.raises(MissingPromptError):
                run_step("step1", "ocr text", institution="Unknown")

    def test_passes_institution_to_load_step(self, tmp_path):
        _make_step_dir(tmp_path, "step1")
        mock_provider = MagicMock()
        mock_provider.call.return_value = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.get_provider", return_value=mock_provider),
        ):
            result, model = run_step("step1", "ocr text", institution="Apple")

        assert result == {"result": "ok"}


class TestRunFormatting:
    def _make_client(self, pipeline_row, ocr_chunks=None, bronze_row=None, scout_scores=None, formatting_rows=None):
        client = MagicMock()

        def table_side_effect(name):
            chain = MagicMock()
            result = MagicMock()
            if name == "pipeline":
                result.data = [pipeline_row] if pipeline_row else []
            elif name == "ocr_results":
                result.data = ocr_chunks if ocr_chunks else []
            elif name == "bronze_mapping":
                result.data = [bronze_row] if bronze_row else []
            elif name == "scout_page_scores":
                result.data = scout_scores if scout_scores is not None else []
            elif name == "formatting":
                result.data = formatting_rows if formatting_rows is not None else []
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

    def test_skips_if_valid_formatting_rows_exist(self, tmp_path):
        pipeline_row = {
            "doc_id": "doc1",
            "last_formatting": None,
            "formatting_nb": 0,
        }
        formatting_rows = [
            {"step_name": "extract_model_name", "content": {"result": "ok"}},
            {"step_name": "extract_table", "content": {"result": "ok"}},
        ]
        client = self._make_client(
            pipeline_row,
            bronze_row={"doc_id": "doc1", "institution": None},
            formatting_rows=formatting_rows,
        )

        with (
            patch("pipeline.formatting.run_step") as mock_step,
            patch("pipeline.formatting.load_step", return_value=("prompt", {"type": "object", "required": ["result"], "properties": {"result": {"type": "string"}}}, {"provider": "moonshot", "model": "kimi-k2.5"})),
        ):
            result = run_formatting("doc1", client)
            mock_step.assert_not_called()
            assert result["status"] == "skipped"

    def test_force_reruns_even_if_valid_formatting_rows_exist(self, tmp_path):
        pipeline_row = {
            "doc_id": "doc1",
            "last_formatting": "2024-01-01T00:00:00+00:00",
            "formatting_nb": 2,
        }
        ocr_chunks = [{"page_number": 1, "content": "sample ocr text"}]
        bronze_row = {"doc_id": "doc1", "institution": None}
        formatting_rows = [
            {"step_name": "extract_model_name", "content": {"result": "ok"}},
            {"step_name": "extract_table", "content": {"result": "ok"}},
        ]
        client = self._make_client(
            pipeline_row,
            ocr_chunks=ocr_chunks,
            bronze_row=bronze_row,
            scout_scores=[],
            formatting_rows=formatting_rows,
        )

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=({"result": "rerun"}, "kimi-k2.5")) as mock_step,
            patch("pipeline.formatting.load_step", return_value=("prompt", {"type": "object", "required": ["result"], "properties": {"result": {"type": "string"}}}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.delete_formatting_results") as mock_delete,
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
        ):
            result = run_formatting("doc1", client, force=True)

        assert result["status"] == "done"
        assert mock_step.call_count == 2
        mock_delete.assert_called_once_with(client, "doc1")

    def test_reruns_when_legacy_formatting_rows_are_invalid(self, tmp_path):
        pipeline_row = {
            "doc_id": "doc1",
            "last_formatting": "2024-01-01T00:00:00+00:00",
            "formatting_nb": 2,
        }
        ocr_chunks = [{"page_number": 1, "content": "sample ocr text"}]
        bronze_row = {"doc_id": "doc1", "institution": None}
        formatting_rows = [
            {"step_name": "extract_model_name", "content": {"legacy": "shape"}},
            {"step_name": "extract_table", "content": {"legacy": "shape"}},
        ]
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_scores=[], formatting_rows=formatting_rows)
        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=(step_result, "kimi-k2.5")) as mock_step,
            patch("pipeline.formatting.load_step", return_value=("prompt", {"type": "object", "required": ["result"], "properties": {"result": {"type": "string"}}}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
        ):
            result = run_formatting("doc1", client)

        assert result["status"] == "done"
        assert mock_step.call_count == 2

    def test_upserts_successful_step_content(self, tmp_path):
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [{"page_number": 1, "content": "sample ocr text"}]
        bronze_row = {"doc_id": "doc1", "institution": None}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_scores=[])
        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=(step_result, "kimi-k2.5")),
            patch("pipeline.formatting.load_step", return_value=("prompt", {"type": "object", "required": ["result"], "properties": {"result": {"type": "string"}}}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.formatting_upsert") as mock_upsert,
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
        ):
            run_formatting("doc1", client)

        assert mock_upsert.call_count == 2
        for call in mock_upsert.call_args_list:
            row = call[0][1]
            assert row["content"] == step_result
            assert row["pages_given"] == [1]

    def test_raises_if_no_pipeline_row(self):
        client = self._make_client(pipeline_row=None)
        with pytest.raises(ValueError, match="No pipeline row"):
            run_formatting("doc1", client)

    def test_raises_if_no_ocr_results(self):
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        client = self._make_client(pipeline_row, ocr_chunks=[])
        with pytest.raises(ValueError, match="No OCR results"):
            run_formatting("doc1", client)

    def test_runs_all_steps_no_scout(self, tmp_path):
        """When no scout results, all steps run with full OCR text."""
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [{"page_number": 1, "content": "chunk1 text"}]
        bronze_row = {"doc_id": "doc1", "institution": None}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_scores=[])

        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=(step_result, "kimi-k2.5")) as mock_step,
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
        ):
            result = run_formatting("doc1", client)

        assert mock_step.call_count == 2
        assert result["status"] == "done"
        assert result["completed_steps"] == 2
        assert result["failed_steps"] == 0
        assert result["failed_details"] == []

    def test_soft_fails_on_schema_error(self, tmp_path):
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [{"page_number": 1, "content": "sample ocr text"}]
        bronze_row = {"doc_id": "doc1", "institution": None}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_scores=[])

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=(None, "kimi-k2.5")),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.append_error") as mock_err,
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
        ):
            result = run_formatting("doc1", client)

        assert mock_err.call_count == 2
        assert result["status"] == "done"
        assert result["completed_steps"] == 0
        assert result["failed_steps"] == 2
        assert len(result["failed_details"]) == 2
        assert all(d["reason"] == "schema validation failed after retry" for d in result["failed_details"])

    def test_concatenates_multi_chunk_ocr_no_scout(self, tmp_path):
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [
            {"page_number": 1, "content": "--- Page 1 ---\nchunk1"},
            {"page_number": 76, "content": "--- Page 76 ---\nchunk2"},
        ]
        bronze_row = {"doc_id": "doc1", "institution": None}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_scores=[])

        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=(step_result, "kimi-k2.5")) as mock_step,
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
        ):
            run_formatting("doc1", client)

        # Verify the OCR text passed to run_step is the concatenation
        ocr_text_arg = mock_step.call_args_list[0][0][1]
        assert "chunk1" in ocr_text_arg
        assert "chunk2" in ocr_text_arg

    def test_passes_institution_to_run_step(self, tmp_path):
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [{"page_number": 1, "content": "--- Page 1 ---\ntext"}]
        bronze_row = {"doc_id": "doc1", "institution": "Apple"}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_scores=[])

        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=(step_result, "kimi-k2.5")) as mock_step,
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
        ):
            run_formatting("doc1", client)

        # Check institution kwarg was passed
        assert mock_step.call_args_list[0][1]["institution"] == "Apple"

    def test_uses_scout_filtered_pages(self, tmp_path):
        """When scout scores exist, each step only sees pages above threshold."""
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0, "last_scout": "2024-01-01T00:00:00+00:00"}
        page_rows = [
            {"page_number": 1, "content": "--- Page 1 ---\nIntro"},
            {"page_number": 2, "content": "--- Page 2 ---\nModel specs"},
            {"page_number": 3, "content": "--- Page 3 ---\nResults table"},
            {"page_number": 4, "content": "--- Page 4 ---\nConclusion"},
        ]
        bronze_row = {"doc_id": "doc1", "institution": None}
        scout_scores = [
            {"step_name": "extract_model_name", "page_number": 2, "score": 0.9, "scout_model": "gemini-2.0-flash-lite"},
            {"step_name": "extract_model_name", "page_number": 3, "score": 0.3, "scout_model": "gemini-2.0-flash-lite"},
            {"step_name": "extract_table", "page_number": 2, "score": 0.4, "scout_model": "gemini-2.0-flash-lite"},
            {"step_name": "extract_table", "page_number": 3, "score": 0.95, "scout_model": "gemini-2.0-flash-lite"},
        ]
        client = self._make_client(pipeline_row, page_rows, bronze_row, scout_scores)

        step_result = {"result": "ok"}
        received_texts = []

        def capture_run_step(step_name, ocr_text, **kwargs):
            received_texts.append((step_name, ocr_text))
            return step_result, "kimi-k2.5"

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", side_effect=capture_run_step),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
            patch("pipeline.formatting.SCOUT_SCORE_THRESHOLD", 0.5),
            patch("pipeline.formatting.formatting_upsert") as mock_upsert,
        ):
            run_formatting("doc1", client)

        # extract_model_name should only see page 2
        model_name_text = dict(received_texts)["extract_model_name"]
        assert "Model specs" in model_name_text
        assert "Intro" not in model_name_text
        assert "Results table" not in model_name_text

        # extract_table should only see page 3
        table_text = dict(received_texts)["extract_table"]
        assert "Results table" in table_text
        assert "Model specs" not in table_text
        pages_by_step = {call[0][1]["step_name"]: call[0][1]["pages_given"] for call in mock_upsert.call_args_list}
        assert pages_by_step["extract_model_name"] == [2]
        assert pages_by_step["extract_table"] == [3]

    def test_full_ocr_fallback_persists_all_pages_given(self, tmp_path):
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [
            {"page_number": 2, "content": "--- Page 2 ---\nchunk2"},
            {"page_number": 5, "content": "--- Page 5 ---\nchunk5"},
        ]
        bronze_row = {"doc_id": "doc1", "institution": None}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_scores=[])
        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=step_result),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.formatting_upsert") as mock_upsert,
        ):
            run_formatting("doc1", client)

        for call in mock_upsert.call_args_list:
            assert call[0][1]["pages_given"] == [2, 5]

    def test_skips_step_when_scout_has_no_range(self, tmp_path):
        """When scout ran but shortlisted no pages, append error and skip that step."""
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0, "last_scout": "2024-01-01T00:00:00+00:00"}
        ocr_chunks = [{"page_number": 1, "content": "--- Page 1 ---\nsome text"}]
        bronze_row = {"doc_id": "doc1", "institution": None}
        scout_scores = [
            {"step_name": "extract_model_name", "page_number": 1, "score": 0.8, "scout_model": "gemini-2.0-flash-lite"},
            {"step_name": "extract_table", "page_number": 1, "score": 0.2, "scout_model": "gemini-2.0-flash-lite"},
        ]
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_scores)

        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=(step_result, "kimi-k2.5")) as mock_step,
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
            patch("pipeline.formatting.SCOUT_SCORE_THRESHOLD", 0.5),
            patch("pipeline.formatting.append_error") as mock_err,
            patch("pipeline.formatting.formatting_upsert") as mock_upsert,
        ):
            run_formatting("doc1", client)

        # Only extract_model_name should run; extract_table is skipped
        assert mock_step.call_count == 1
        mock_err.assert_called_once()
        assert "extract_table" in mock_err.call_args[0][2]
        assert mock_upsert.call_count == 1
        assert mock_upsert.call_args[0][1]["step_name"] == "extract_model_name"

    def test_failed_details_contains_provider_error(self, tmp_path):
        """Provider exceptions produce failed_details with 'provider error: ...' reason."""
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [{"page_number": 1, "content": "text"}]
        bronze_row = {"doc_id": "doc1", "institution": None}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_scores=[])

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", side_effect=RuntimeError("429 Too Many Requests")),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.append_error"),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
        ):
            result = run_formatting("doc1", client)

        assert result["failed_steps"] == 2
        assert len(result["failed_details"]) == 2
        assert all("provider error:" in d["reason"] for d in result["failed_details"])
        assert "429 Too Many Requests" in result["failed_details"][0]["reason"]

    def test_failed_details_include_raw_output_for_non_json_provider_error_in_debug(self, tmp_path):
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [{"page_number": 1, "content": "text"}]
        bronze_row = {"doc_id": "doc1", "institution": None}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_scores=[])

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", side_effect=NonJSONResponseError("Gemini", "{\n  \"a\": 1\n}\ntrailing")),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.debug_logger.is_enabled", return_value=True),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
        ):
            result = run_formatting("doc1", client)

        assert result["failed_steps"] == 2
        assert result["failed_details"][0]["reason"] == "Gemini returned non-JSON"
        assert result["failed_details"][0]["raw_output"] == "{\n  \"a\": 1\n}\ntrailing"

    def test_failed_details_contains_missing_prompt(self, tmp_path):
        """MissingPromptError produces failed_details with 'no prompt for institution' reason."""
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [{"page_number": 1, "content": "text"}]
        bronze_row = {"doc_id": "doc1", "institution": "Unknown"}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_scores=[])

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", side_effect=MissingPromptError("step1", "Unknown")),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.append_error"),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
        ):
            result = run_formatting("doc1", client)

        assert result["failed_steps"] == 2
        assert all("no prompt for institution" in d["reason"] for d in result["failed_details"])

    def test_failed_details_scout_no_pages_above_threshold(self, tmp_path):
        """Low scout scores produce failed_details with 'no scout pages above threshold' reason."""
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0, "last_scout": "2024-01-01T00:00:00+00:00"}
        ocr_chunks = [{"page_number": 1, "content": "text"}]
        bronze_row = {"doc_id": "doc1", "institution": None}
        scout_scores = [
            {"step_name": "extract_model_name", "page_number": 1, "score": 0.8, "scout_model": "gemini-2.0-flash-lite"},
            {"step_name": "extract_table", "page_number": 1, "score": 0.1, "scout_model": "gemini-2.0-flash-lite"},
        ]
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_scores)

        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=(step_result, "kimi-k2.5")),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
            patch("pipeline.formatting.SCOUT_SCORE_THRESHOLD", 0.5),
            patch("pipeline.formatting.append_error"),
        ):
            result = run_formatting("doc1", client)

        assert result["failed_steps"] == 1
        assert result["failed_details"] == [{"step": "extract_table", "reason": "no scout pages above threshold"}]

    def test_reruns_when_attempts_exhausted_but_formatting_rows_missing(self, tmp_path):
        pipeline_row = {
            "doc_id": "doc1",
            "last_formatting": None,
            "formatting_nb": 0,
            "formatting_attempts": 3,
            "error": [
                {"stage": "formatting", "attempt": 1, "step": "extract_model_name", "reason": "schema validation failed after retry"},
            ],
        }
        ocr_chunks = [{"page_number": 1, "content": "sample ocr text"}]
        bronze_row = {"doc_id": "doc1", "institution": None}
        client = self._make_client(pipeline_row, ocr_chunks=ocr_chunks, bronze_row=bronze_row, scout_scores=[])
        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=(step_result, "kimi-k2.5")) as mock_step,
            patch("pipeline.formatting.load_step", return_value=("prompt", {"type": "object", "required": ["result"], "properties": {"result": {"type": "string"}}}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.increment_formatting_attempts", return_value=4) as mock_increment,
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
        ):
            result = run_formatting("doc1", client)

        assert result["status"] == "done"
        assert mock_step.call_count == 2
        mock_increment.assert_called_once_with(client, "doc1")
