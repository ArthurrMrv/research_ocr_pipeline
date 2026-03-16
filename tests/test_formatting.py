import json
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from pipeline.formatting import load_step, run_formatting, run_step, validate_output


def _make_step_dir(tmp_path, step_name, config=None, schema=None, prompt=None):
    step_dir = tmp_path / step_name
    step_dir.mkdir()
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
        # Always returns wrong shape
        mock_provider.call.return_value = {"wrong_key": "value"}

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.get_provider", return_value=mock_provider),
        ):
            result = run_step("step1", "ocr text")

        assert result is None
        assert mock_provider.call.call_count == 2  # tried twice

    def test_returns_result_on_second_attempt(self, tmp_path):
        _make_step_dir(tmp_path, "step1")
        mock_provider = MagicMock()
        mock_provider.call.side_effect = [
            {"wrong_key": "bad"},    # first attempt fails validation
            {"result": "ok"},        # second attempt passes
        ]

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.get_provider", return_value=mock_provider),
        ):
            result = run_step("step1", "ocr text")

        assert result == {"result": "ok"}


class TestRunFormatting:
    def _make_client(self, pipeline_row, ocr_row=None):
        client = MagicMock()

        def table_side_effect(name):
            chain = MagicMock()
            result = MagicMock()
            if name == "pipeline":
                result.data = [pipeline_row] if pipeline_row else []
            elif name == "ocr_results":
                result.data = [ocr_row] if ocr_row else []
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
        client = self._make_client(pipeline_row, ocr_row=None)
        with pytest.raises(ValueError, match="No OCR results"):
            run_formatting("doc1", client)

    def test_runs_all_steps(self, tmp_path):
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_row = {"content": "sample ocr text"}
        client = self._make_client(pipeline_row, ocr_row)

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
        ocr_row = {"content": "sample ocr text"}
        client = self._make_client(pipeline_row, ocr_row)

        with (
            patch("pipeline.formatting.STEPS_DIR", tmp_path),
            patch("pipeline.formatting.run_step", return_value=None),
            patch("pipeline.formatting.load_step", return_value=("prompt", {}, {"provider": "moonshot", "model": "kimi-k2.5"})),
            patch("pipeline.formatting.append_error") as mock_err,
        ):
            run_formatting("doc1", client)

        # Should have logged errors for both steps that returned None
        assert mock_err.call_count == 2
