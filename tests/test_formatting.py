import json
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from pipeline.formatting import load_step, run_formatting, run_step, validate_output


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

    def test_loads_company_specific_prompt(self, tmp_path):
        step_dir = _make_step_dir(
            tmp_path, "my_step",
            config={"provider": "moonshot", "model": "kimi-k2.5", "per_company": True},
        )
        prompts_dir = step_dir / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "Apple.txt").write_text("Apple-specific prompt")

        with patch("pipeline.formatting.STEPS_DIR", tmp_path):
            prompt, schema, config = load_step("my_step", company_name="Apple")
        assert prompt == "Apple-specific prompt"

    def test_company_prompt_case_insensitive(self, tmp_path):
        step_dir = _make_step_dir(
            tmp_path, "my_step",
            config={"provider": "moonshot", "model": "kimi-k2.5", "per_company": True},
        )
        prompts_dir = step_dir / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "Apple.txt").write_text("Apple prompt")

        with patch("pipeline.formatting.STEPS_DIR", tmp_path):
            prompt, _, _ = load_step("my_step", company_name="apple")
        assert prompt == "Apple prompt"

    def test_per_company_missing_prompt_uses_fallback(self, tmp_path):
        _make_step_dir(
            tmp_path, "my_step",
            config={"provider": "moonshot", "model": "kimi-k2.5", "per_company": True},
            prompt="fallback prompt",
        )
        with patch("pipeline.formatting.STEPS_DIR", tmp_path):
            prompt, _, _ = load_step("my_step", company_name="Unknown")
        assert prompt == "fallback prompt"

    def test_per_company_no_company_name_uses_fallback(self, tmp_path):
        _make_step_dir(
            tmp_path, "my_step",
            config={"provider": "moonshot", "model": "kimi-k2.5", "per_company": True},
            prompt="fallback prompt",
        )
        with patch("pipeline.formatting.STEPS_DIR", tmp_path):
            prompt, _, _ = load_step("my_step", company_name=None)
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
            result = run_step("step1", "some ocr text")

        assert result == {"result": "found"}
        mock_provider.call.assert_called_once()

    def test_retries_on_schema_failure_then_returns_none(self, tmp_path):
        _make_step_dir(tmp_path, "step1")
        mock_provider = MagicMock()
        mock_provider.call.return_value = {"wrong_key": "value"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.get_provider", return_value=mock_provider),
        ):
            result = run_step("step1", "ocr text")

        assert result is None
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
            result = run_step("step1", "ocr text")

        assert result == {"result": "ok"}

    def test_returns_none_when_prompt_is_none(self, tmp_path):
        """When per_company step has no matching prompt, returns None."""
        _make_step_dir(
            tmp_path, "step1",
            config={"provider": "moonshot", "model": "kimi-k2.5", "per_company": True},
        )
        # Remove the fallback prompt so it returns None
        (tmp_path / "step1" / "prompt.txt").unlink()

        with patch("pipeline.formatting.STEPS_DIR", tmp_path):
            result = run_step("step1", "ocr text", company_name="Unknown")

        assert result is None

    def test_passes_company_name_to_load_step(self, tmp_path):
        _make_step_dir(tmp_path, "step1")
        mock_provider = MagicMock()
        mock_provider.call.return_value = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.get_provider", return_value=mock_provider),
        ):
            result = run_step("step1", "ocr text", company_name="Apple")

        assert result == {"result": "ok"}


class TestRunFormatting:
    def _make_client(self, pipeline_row, ocr_chunks=None, bronze_row=None, scout_rows=None):
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
            elif name == "scout_results":
                result.data = scout_rows if scout_rows is not None else []
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

    def test_skips_if_all_steps_done(self, tmp_path):
        pipeline_row = {
            "doc_id": "doc1",
            "last_formatting": "2024-01-01T00:00:00+00:00",
            "formatting_nb": 2,
        }
        client = self._make_client(pipeline_row)

        with patch("pipeline.formatting.run_step") as mock_step:
            run_formatting("doc1", client)
            mock_step.assert_not_called()

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
        ocr_chunks = [{"content": "chunk1 text", "pages": "1-75"}]
        bronze_row = {"doc_id": "doc1", "company_name": None}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_rows=[])

        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=step_result) as mock_step,
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
        ):
            run_formatting("doc1", client)

        assert mock_step.call_count == 2

    def test_soft_fails_on_schema_error(self, tmp_path):
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [{"content": "sample ocr text", "pages": "1-75"}]
        bronze_row = {"doc_id": "doc1", "company_name": None}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_rows=[])

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=None),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.append_error") as mock_err,
        ):
            run_formatting("doc1", client)

        assert mock_err.call_count == 2

    def test_concatenates_multi_chunk_ocr_no_scout(self, tmp_path):
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [
            {"content": "--- Page 1 ---\nchunk1", "pages": "1-75"},
            {"content": "--- Page 76 ---\nchunk2", "pages": "76-100"},
        ]
        bronze_row = {"doc_id": "doc1", "company_name": None}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_rows=[])

        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=step_result) as mock_step,
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
        ):
            run_formatting("doc1", client)

        # Verify the OCR text passed to run_step is the concatenation
        ocr_text_arg = mock_step.call_args_list[0][0][1]
        assert "chunk1" in ocr_text_arg
        assert "chunk2" in ocr_text_arg

    def test_passes_company_name_to_run_step(self, tmp_path):
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [{"content": "--- Page 1 ---\ntext", "pages": "1-10"}]
        bronze_row = {"doc_id": "doc1", "company_name": "Apple"}
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_rows=[])

        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=step_result) as mock_step,
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
        ):
            run_formatting("doc1", client)

        # Check company_name kwarg was passed
        assert mock_step.call_args_list[0][1]["company_name"] == "Apple"

    def test_uses_scout_filtered_pages(self, tmp_path):
        """When scout results exist, each step receives only its targeted pages."""
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [
            {
                "content": (
                    "--- Page 1 ---\nIntro\n\n"
                    "--- Page 2 ---\nModel specs\n\n"
                    "--- Page 3 ---\nResults table\n\n"
                    "--- Page 4 ---\nConclusion"
                ),
                "pages": "1-4",
            }
        ]
        bronze_row = {"doc_id": "doc1", "company_name": None}
        scout_rows = [
            {"step_name": "extract_model_name", "start_page": 2, "end_page": 2, "scout_model": "gemini-2.0-flash-lite"},
            {"step_name": "extract_table", "start_page": 3, "end_page": 3, "scout_model": "gemini-2.0-flash-lite"},
        ]
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_rows)

        step_result = {"result": "ok"}
        received_texts = []

        def capture_run_step(step_name, ocr_text, **kwargs):
            received_texts.append((step_name, ocr_text))
            return step_result

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", side_effect=capture_run_step),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
            patch("pipeline.formatting.SCOUT_PAGE_PADDING", 0),
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

    def test_skips_step_when_scout_has_no_range(self, tmp_path):
        """When scout ran but a step has no range, append error and skip that step."""
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_chunks = [{"content": "--- Page 1 ---\nsome text", "pages": "1-5"}]
        bronze_row = {"doc_id": "doc1", "company_name": None}
        # Scout has result for extract_model_name but NOT for extract_table
        scout_rows = [
            {"step_name": "extract_model_name", "start_page": 1, "end_page": 2, "scout_model": "gemini-2.0-flash-lite"},
        ]
        client = self._make_client(pipeline_row, ocr_chunks, bronze_row, scout_rows)

        step_result = {"result": "ok"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=step_result) as mock_step,
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.ACTIVE_STEPS", ["extract_model_name", "extract_table"]),
            patch("pipeline.formatting.SCOUT_PAGE_PADDING", 0),
            patch("pipeline.formatting.append_error") as mock_err,
        ):
            run_formatting("doc1", client)

        # Only extract_model_name should run; extract_table is skipped
        assert mock_step.call_count == 1
        mock_err.assert_called_once()
        assert "extract_table" in mock_err.call_args[0][2]
