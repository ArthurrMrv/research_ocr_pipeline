import json
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from pipeline.formatting import (
    _build_assumptions_context,
    _build_context,
    _build_methodology_context,
    _merge_assumption_drafts,
    _merge_drafts,
    _run_step_multipass,
    load_step,
    run_formatting,
    run_step,
    validate_output,
)
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
            {"step_name": "extract_model_inputs", "content": {"result": "ok"}},
            {"step_name": "extract_model_methodology", "content": {"result": "ok"}},
            {"step_name": "extract_model_assumptions", "content": {"result": "ok"}},
        ]
        client = self._make_client(
            pipeline_row,
            bronze_row={"doc_id": "doc1", "institution": None},
            formatting_rows=formatting_rows,
        )

        with (
            patch("pipeline.formatting.run_step") as mock_step,
            patch("pipeline.formatting.load_step_config", return_value={}),
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
            patch("pipeline.formatting.load_step_config", return_value={}),
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
            patch("pipeline.formatting.load_step_config", return_value={}),
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
            patch("pipeline.formatting.load_step_config", return_value={}),
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
            patch("pipeline.formatting.load_step_config", return_value={}),
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
            patch("pipeline.formatting.load_step_config", return_value={}),
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
            patch("pipeline.formatting.load_step_config", return_value={}),
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
            patch("pipeline.formatting.load_step_config", return_value={}),
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
            patch("pipeline.formatting.load_step_config", return_value={}),
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
            patch("pipeline.formatting.load_step_config", return_value={}),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.formatting_upsert") as mock_upsert,
        ):
            run_formatting("doc1", client)

        for call in mock_upsert.call_args_list:
            assert call[0][1]["pages_given"] == [2, 5]

    def test_fallback_when_scout_below_threshold(self, tmp_path):
        """When scout scores are below threshold, fallback to top-N + cross-step pages."""
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
            patch("pipeline.formatting.load_step_config", return_value={}),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
            patch("pipeline.formatting.SCOUT_SCORE_THRESHOLD", 0.5),
            patch("pipeline.formatting.formatting_upsert") as mock_upsert,
        ):
            result = run_formatting("doc1", client)

        # Both steps should run: extract_model_name above threshold, extract_table via fallback
        assert mock_step.call_count == 2
        assert mock_upsert.call_count == 2
        assert result["failed_steps"] == 0

    def test_failed_details_contains_provider_error(self, tmp_path):
        """Provider exceptions produce failed_details with 'provider error: ...' reason."""
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [{"page_number": 1, "content": "text"}]
        bronze_row = {"doc_id": "doc1", "institution": None}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_scores=[])

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", side_effect=RuntimeError("429 Too Many Requests")),
            patch("pipeline.formatting.load_step_config", return_value={}),
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
            patch("pipeline.formatting.load_step_config", return_value={}),
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
            patch("pipeline.formatting.load_step_config", return_value={}),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.append_error"),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
        ):
            result = run_formatting("doc1", client)

        assert result["failed_steps"] == 2
        assert all("no prompt for institution" in d["reason"] for d in result["failed_details"])

    def test_scout_fallback_uses_top_n_and_cross_step_pages(self, tmp_path):
        """Low scout scores trigger fallback: top-N pages + pages above threshold in other steps."""
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
            patch("pipeline.formatting.run_step", return_value=(step_result, "kimi-k2.5")) as mock_step,
            patch("pipeline.formatting.load_step_config", return_value={}),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
            patch("pipeline.formatting.SCOUT_SCORE_THRESHOLD", 0.5),
        ):
            result = run_formatting("doc1", client)

        # extract_table runs via fallback (top-N page 1 + cross-step page 1 from extract_model_name)
        assert mock_step.call_count == 2
        assert result["failed_steps"] == 0
        assert result["failed_details"] == []

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
            patch("pipeline.formatting.load_step_config", return_value={}),
            patch("pipeline.formatting.load_step", return_value=("prompt", {"type": "object", "required": ["result"], "properties": {"result": {"type": "string"}}}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.increment_formatting_attempts", return_value=4) as mock_increment,
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
        ):
            result = run_formatting("doc1", client)

        assert result["status"] == "done"
        assert mock_step.call_count == 2
        mock_increment.assert_called_once_with(client, "doc1")


class TestMergeDrafts:
    def test_unions_variables_deduplicates(self):
        drafts = [
            {"model_name": "CMAs", "variables": ["inflation", "GDP growth"]},
            {"model_name": "CMAs", "variables": ["GDP growth", "dividend yield"]},
            {"model_name": "CMAs", "variables": ["inflation", "risk premium"]},
        ]
        merged = _merge_drafts(drafts)
        assert merged["variables"] == ["inflation", "GDP growth", "dividend yield", "risk premium"]

    def test_unions_variables_important(self):
        drafts = [
            {"variables_important": ["inflation"]},
            {"variables_important": ["inflation", "GDP"]},
        ]
        merged = _merge_drafts(drafts)
        assert merged["variables_important"] == ["inflation", "GDP"]

    def test_unions_assumptions(self):
        drafts = [
            {"assumptions": ["inflation = 2%"]},
            {"assumptions": ["inflation = 2%", "GDP > 0 (implied)"]},
        ]
        merged = _merge_drafts(drafts)
        assert merged["assumptions"] == ["inflation = 2%", "GDP > 0 (implied)"]

    def test_picks_most_common_model_name(self):
        drafts = [
            {"model_name": "Capital Market Assumptions"},
            {"model_name": "Capital Market Model"},
            {"model_name": "Capital Market Assumptions"},
        ]
        merged = _merge_drafts(drafts)
        assert merged["model_name"] == "Capital Market Assumptions"

    def test_handles_missing_fields(self):
        drafts = [{"model_name": "X"}, {}]
        merged = _merge_drafts(drafts)
        assert merged["model_name"] == "X"
        assert merged["variables"] == []
        assert merged["notes_model"] == ""

    def test_case_insensitive_dedup(self):
        drafts = [
            {"variables": ["Inflation", "gdp"]},
            {"variables": ["inflation", "GDP"]},
        ]
        merged = _merge_drafts(drafts)
        assert len(merged["variables"]) == 2
        # First-seen wins
        assert merged["variables"][0] == "Inflation"
        assert merged["variables"][1] == "gdp"


class TestBuildMethodologyContext:
    def test_full_result(self):
        result = {
            "steps_summary": "Build blocks combined into expected return.",
            "steps_detailed": "1. Project macro.\n2. Estimate earnings.",
            "sub_models": ["CAPM", "DDM"],
        }
        ctx = _build_methodology_context(result)
        assert "Summary: Build blocks combined" in ctx
        assert "Sub-models used: CAPM, DDM" in ctx
        # steps_detailed should NOT be included (user chose summary + sub_models only)
        assert "Project macro" not in ctx

    def test_none_result(self):
        assert _build_methodology_context(None) == ""

    def test_empty_result(self):
        assert _build_methodology_context({}) == ""

    def test_no_sub_models(self):
        result = {"steps_summary": "Simple approach."}
        ctx = _build_methodology_context(result)
        assert "Summary: Simple approach." in ctx
        assert "Sub-models" not in ctx


class TestRunStepMultipass:
    def test_drafts_merged_and_verified(self, tmp_path):
        step_dir = _make_step_dir(
            tmp_path,
            "mp_step",
            config={
                "provider": "moonshot",
                "model": "pro-model",
                "draft_model": "flash-model",
                "draft_runs": 2,
                "multi_pass": True,
            },
            schema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["model_name", "variables"],
                "properties": {
                    "model_name": {"type": "string"},
                    "variables": {"type": "array", "items": {"type": "string"}},
                },
            },
        )
        (step_dir / "verify_prompt.txt").write_text(
            "Verify: {draft_result}\n\nOCR TEXT:\n{ocr_text}"
        )

        flash_provider = MagicMock()
        flash_provider.call.side_effect = [
            {"model_name": "CMA", "variables": ["inflation"]},
            {"model_name": "CMA", "variables": ["inflation", "GDP"]},
        ]
        pro_provider = MagicMock()
        pro_provider.call.return_value = {
            "model_name": "CMA",
            "variables": ["inflation", "GDP"],
        }

        providers_created = []

        def fake_get_provider(name, config):
            providers_created.append(config["model"])
            if config["model"] == "flash-model":
                return flash_provider
            return pro_provider

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.get_provider", side_effect=fake_get_provider),
        ):
            result, model = _run_step_multipass("mp_step", "some ocr text")

        assert result == {"model_name": "CMA", "variables": ["inflation", "GDP"]}
        assert model == "pro-model"
        assert flash_provider.call.call_count == 2
        assert pro_provider.call.call_count == 1
        # Verify the verify prompt was filled with merged draft
        verify_call_prompt = pro_provider.call.call_args[0][0]
        assert "inflation" in verify_call_prompt
        assert "GDP" in verify_call_prompt

    def test_falls_back_to_single_pass_when_all_drafts_fail(self, tmp_path):
        _make_step_dir(
            tmp_path,
            "mp_step",
            config={
                "provider": "moonshot",
                "model": "pro-model",
                "draft_model": "flash-model",
                "draft_runs": 2,
                "multi_pass": True,
            },
        )

        flash_provider = MagicMock()
        flash_provider.call.side_effect = RuntimeError("API error")

        def fake_get_provider(name, config):
            if config["model"] == "flash-model":
                return flash_provider
            return MagicMock()

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.get_provider", side_effect=fake_get_provider),
            patch(
                "pipeline.formatting.run_step",
                return_value=({"result": "fallback"}, "pro-model"),
            ) as mock_run_step,
        ):
            result, model = _run_step_multipass("mp_step", "ocr text")

        assert result == {"result": "fallback"}
        mock_run_step.assert_called_once()

    def test_returns_none_on_verify_validation_failure(self, tmp_path):
        step_dir = _make_step_dir(
            tmp_path,
            "mp_step",
            config={
                "provider": "moonshot",
                "model": "pro-model",
                "draft_model": "flash-model",
                "draft_runs": 1,
                "multi_pass": True,
            },
            schema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["model_name", "variables"],
                "properties": {
                    "model_name": {"type": "string"},
                    "variables": {"type": "array", "items": {"type": "string"}},
                },
            },
        )
        (step_dir / "verify_prompt.txt").write_text("Verify: {draft_result}\n{ocr_text}")

        flash_provider = MagicMock()
        flash_provider.call.return_value = {"model_name": "X", "variables": ["a"]}
        pro_provider = MagicMock()
        pro_provider.call.return_value = {"wrong_key": "bad"}

        def fake_get_provider(name, config):
            if config["model"] == "flash-model":
                return flash_provider
            return pro_provider

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.get_provider", side_effect=fake_get_provider),
        ):
            result, model = _run_step_multipass("mp_step", "ocr text")

        assert result is None
        assert model == "pro-model"
        assert pro_provider.call.call_count == 2

    def test_run_formatting_dispatches_multipass(self, tmp_path):
        """run_formatting uses _run_step_multipass when config has multi_pass=true."""
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [{"page_number": 1, "content": "sample text"}]
        bronze_row = {"doc_id": "doc1", "institution": None}

        client = MagicMock()

        def table_side_effect(name):
            chain = MagicMock()
            result_mock = MagicMock()
            if name == "pipeline":
                result_mock.data = [pipeline_row]
            elif name == "ocr_results":
                result_mock.data = ocr_chunks
            elif name == "bronze_mapping":
                result_mock.data = [bronze_row]
            elif name == "formatting":
                result_mock.data = []
            else:
                result_mock.data = []
            chain.execute.return_value = result_mock
            chain.select.return_value = chain
            chain.update.return_value = chain
            chain.upsert.return_value = chain
            chain.delete.return_value = chain
            chain.eq.return_value = chain
            return chain

        client.table.side_effect = table_side_effect

        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch(
                "pipeline.formatting.load_step_config",
                return_value={"provider": "moonshot", "model": "m", "multi_pass": True},
            ),
            patch(
                "pipeline.formatting._run_step_multipass",
                return_value=(step_result, "pro-model"),
            ) as mock_mp,
            patch(
                "pipeline.formatting.run_step",
                return_value=(step_result, "kimi-k2.5"),
            ) as mock_single,
            patch(
                "pipeline.formatting.load_step",
                return_value=(
                    "prompt",
                    {},
                    {"provider": "moonshot", "model": "kimi-k2.5"},
                ),
            ),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_inputs"]),
            patch("pipeline.formatting.formatting_upsert"),
        ):
            result = run_formatting("doc1", client)

        mock_mp.assert_called_once()
        mock_single.assert_not_called()
        assert result["status"] == "done"
        assert result["completed_steps"] == 1


class TestRunStepExtraContext:
    def test_run_step_injects_methodology_context(self, tmp_path):
        _make_step_dir(tmp_path, "step1", prompt="Context: {methodology_context}\nExtract: {ocr_text}")
        mock_provider = MagicMock()
        mock_provider.call.return_value = {"result": "found"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.get_provider", return_value=mock_provider),
        ):
            result, model = run_step(
                "step1", "ocr text", extra_context="METHODOLOGY CONTEXT:\nSummary: test\n"
            )

        prompt_sent = mock_provider.call.call_args[0][0]
        assert "METHODOLOGY CONTEXT:" in prompt_sent
        assert "Summary: test" in prompt_sent

    def test_run_step_replaces_placeholder_with_empty_when_no_context(self, tmp_path):
        _make_step_dir(tmp_path, "step1", prompt="Context: {methodology_context}\nExtract: {ocr_text}")
        mock_provider = MagicMock()
        mock_provider.call.return_value = {"result": "found"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.get_provider", return_value=mock_provider),
        ):
            run_step("step1", "ocr text")

        prompt_sent = mock_provider.call.call_args[0][0]
        assert "{methodology_context}" not in prompt_sent

    def test_run_formatting_passes_methodology_to_dependent_step(self, tmp_path):
        """When methodology runs first and inputs depends_on it, context is passed."""
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [{"page_number": 1, "content": "sample text"}]
        bronze_row = {"doc_id": "doc1", "institution": None}

        client = MagicMock()

        def table_side_effect(name):
            chain = MagicMock()
            result_mock = MagicMock()
            if name == "pipeline":
                result_mock.data = [pipeline_row]
            elif name == "ocr_results":
                result_mock.data = ocr_chunks
            elif name == "bronze_mapping":
                result_mock.data = [bronze_row]
            elif name == "formatting":
                result_mock.data = []
            else:
                result_mock.data = []
            chain.execute.return_value = result_mock
            chain.select.return_value = chain
            chain.update.return_value = chain
            chain.upsert.return_value = chain
            chain.delete.return_value = chain
            chain.eq.return_value = chain
            return chain

        client.table.side_effect = table_side_effect

        methodology_result = {
            "steps_summary": "Combine sub-models.",
            "steps_detailed": "1. Run CAPM.\n2. Run DDM.",
            "mermaid_diagram": "flowchart TD\n    A --> B",
            "sub_models": ["CAPM", "DDM"],
        }
        inputs_result = {
            "model_name": "CMA",
            "variables": ["risk premium"],
        }

        call_log: list[tuple[str, str | None]] = []

        def fake_load_step_config(step_name):
            if step_name == "extract_model_inputs":
                return {"provider": "moonshot", "model": "m", "depends_on": "extract_model_methodology"}
            return {"provider": "moonshot", "model": "m"}

        def fake_run_step(step_name, ocr_text, *, institution=None, extra_context=None):
            call_log.append((step_name, extra_context))
            if step_name == "extract_model_methodology":
                return methodology_result, "m"
            return inputs_result, "m"

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.load_step_config", side_effect=fake_load_step_config),
            patch("pipeline.formatting.run_step", side_effect=fake_run_step),
            patch(
                "pipeline.formatting.load_step",
                return_value=("prompt", {}, {"provider": "moonshot", "model": "m"}),
            ),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_methodology", "extract_model_inputs"]),
            patch("pipeline.formatting.formatting_upsert"),
        ):
            result = run_formatting("doc1", client)

        assert result["completed_steps"] == 2
        # Methodology step should have no extra_context
        assert call_log[0] == ("extract_model_methodology", None)
        # Inputs step should have methodology context
        assert call_log[1][0] == "extract_model_inputs"
        assert "Combine sub-models" in call_log[1][1]
        assert "CAPM, DDM" in call_log[1][1]


class TestBuildAssumptionsContext:
    def test_full_results_from_both_steps(self):
        dep_results = {
            "extract_model_methodology": {
                "steps_summary": "Combined building blocks.",
                "sub_models": ["CAPM", "DDM"],
                "assumptions": ["r_equity = r_f + ERP", "mean reversion (implied)"],
            },
            "extract_model_inputs": {
                "variables": ["inflation", "GDP growth"],
                "assumptions": ["inflation = 2%", "GDP > 0 (implied)"],
            },
        }
        ctx = _build_assumptions_context(dep_results)
        assert "Methodology Summary: Combined building blocks." in ctx
        assert "Sub-models: CAPM, DDM" in ctx
        assert "r_equity = r_f + ERP" in ctx
        assert "mean reversion (implied)" in ctx
        assert "inflation, GDP growth" in ctx
        assert "inflation = 2%" in ctx
        assert "GDP > 0 (implied)" in ctx

    def test_methodology_only(self):
        dep_results = {
            "extract_model_methodology": {
                "steps_summary": "Simple approach.",
                "assumptions": ["growth = 3%"],
            },
        }
        ctx = _build_assumptions_context(dep_results)
        assert "Methodology Summary: Simple approach." in ctx
        assert "growth = 3%" in ctx
        assert "Value-type" not in ctx

    def test_inputs_only(self):
        dep_results = {
            "extract_model_inputs": {
                "variables": ["inflation"],
                "assumptions": ["inflation = 2%"],
            },
        }
        ctx = _build_assumptions_context(dep_results)
        assert "inflation" in ctx
        assert "inflation = 2%" in ctx
        assert "Methodology Summary" not in ctx

    def test_empty_deps(self):
        assert _build_assumptions_context({}) == ""


class TestBuildContext:
    def test_dispatches_to_assumptions_context(self):
        dep_results = {
            "extract_model_methodology": {"steps_summary": "test"},
            "extract_model_inputs": {"variables": ["x"]},
        }
        ctx = _build_context("extract_model_assumptions", dep_results)
        assert "PRIOR EXTRACTION CONTEXT:" in ctx

    def test_dispatches_to_methodology_context_by_default(self):
        dep_results = {
            "extract_model_methodology": {
                "steps_summary": "Build blocks.",
                "sub_models": ["CAPM"],
            },
        }
        ctx = _build_context("extract_model_inputs", dep_results)
        assert "METHODOLOGY CONTEXT" in ctx
        assert "Build blocks." in ctx


class TestMergeAssumptionDrafts:
    def test_deduplicates_assumptions(self):
        drafts = [
            {
                "assumptions": [
                    {"assumption": "CAPE reverts to mean", "building_block": "valuation", "classification": "mean-reversion"},
                    {"assumption": "GDP = 3%", "building_block": "growth", "classification": "historical"},
                ],
                "techniques_used": [
                    {"technique_name": "mean-reversion model", "complexity": 5},
                    {"technique_name": "historical averaging", "complexity": 2},
                ],
                "sophistication_index": 4.0,
                "sophistication_explanation": "Relies on history.",
            },
            {
                "assumptions": [
                    {"assumption": "cape reverts to mean", "building_block": "valuation", "classification": "mean-reversion"},
                    {"assumption": "AI boosts growth", "building_block": "growth", "classification": "forward-looking"},
                ],
                "techniques_used": [
                    {"technique_name": "Mean-Reversion Model", "complexity": 5},
                    {"technique_name": "Gordon growth model", "complexity": 6},
                ],
                "sophistication_index": 6.0,
                "sophistication_explanation": "Uses structured models.",
            },
        ]
        merged = _merge_assumption_drafts(drafts)
        assert len(merged["assumptions"]) == 3
        technique_names = [t["technique_name"] for t in merged["techniques_used"]]
        assert len(merged["techniques_used"]) == 3
        assert "mean-reversion model" in [n.lower() for n in technique_names]
        assert "historical averaging" in [n.lower() for n in technique_names]
        assert "gordon growth model" in [n.lower() for n in technique_names]
        assert merged["sophistication_index"] == 5.0

    def test_averages_index(self):
        drafts = [
            {"assumptions": [], "techniques_used": [], "sophistication_index": 8.0, "sophistication_explanation": "a"},
            {"assumptions": [], "techniques_used": [], "sophistication_index": 4.0, "sophistication_explanation": "a"},
            {"assumptions": [], "techniques_used": [], "sophistication_index": 3.0, "sophistication_explanation": "b"},
        ]
        merged = _merge_assumption_drafts(drafts)
        assert merged["sophistication_index"] == 5.0

    def test_handles_empty_drafts(self):
        drafts = [{"assumptions": [], "techniques_used": [], "sophistication_index": 1, "sophistication_explanation": ""}]
        merged = _merge_assumption_drafts(drafts)
        assert merged["assumptions"] == []
        assert merged["techniques_used"] == []
        assert merged["sophistication_index"] == 1
