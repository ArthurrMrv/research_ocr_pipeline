import json
from unittest.mock import MagicMock, patch

import pytest

from pipeline.formatting import call_llm, get_kimi_client, load_prompt, run_formatting


class TestGetKimiClient:
    def test_creates_client_with_env_vars(self):
        with patch("pipeline.formatting.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            client = get_kimi_client()
            mock_openai.assert_called_once_with(
                api_key="test-moonshot-key",
                base_url="https://api.moonshot.ai/v1",
            )

    def test_raises_on_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("MOONSHOT_API_KEY")
        with pytest.raises(KeyError):
            get_kimi_client()


class TestLoadPrompt:
    def test_loads_existing_prompt(self, tmp_path):
        with patch("pipeline.formatting.PROMPTS_DIR", tmp_path):
            prompt_file = tmp_path / "test_step.txt"
            prompt_file.write_text("Test prompt {ocr_text}")
            result = load_prompt("test_step")
            assert result == "Test prompt {ocr_text}"

    def test_raises_on_missing_prompt(self, tmp_path):
        with patch("pipeline.formatting.PROMPTS_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                load_prompt("nonexistent_step")


class TestCallLlm:
    def test_parses_json_response(self):
        mock_client = MagicMock()
        response_content = json.dumps({"model_name": "TestModel"})
        mock_client.chat.completions.create.return_value.choices[
            0
        ].message.content = response_content

        result = call_llm(mock_client, "Extract model: {ocr_text}", "some ocr text")

        assert result == {"model_name": "TestModel"}

    def test_fills_ocr_text_in_prompt(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[
            0
        ].message.content = "{}"
        call_llm(mock_client, "Text: {ocr_text}", "my ocr content")
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert "my ocr content" in messages[0]["content"]

    def test_raises_on_invalid_json(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[
            0
        ].message.content = "not valid json"
        with pytest.raises(ValueError, match="non-JSON"):
            call_llm(mock_client, "prompt", "ocr text")


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

    def test_skips_if_all_steps_done(self):
        pipeline_row = {
            "doc_id": "doc1",
            "last_formatting": "2024-01-01T00:00:00+00:00",
            "formatting_nb": 2,  # matches len(ACTIVE_STEPS) = 2
        }
        client = self._make_client(pipeline_row)
        mock_kimi = MagicMock()

        run_formatting("doc1", client, mock_kimi)

        mock_kimi.chat.completions.create.assert_not_called()

    def test_raises_if_no_pipeline_row(self):
        client = self._make_client(pipeline_row=None)
        mock_kimi = MagicMock()
        with pytest.raises(ValueError, match="No pipeline row"):
            run_formatting("doc1", client, mock_kimi)

    def test_raises_if_no_ocr_results(self):
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        client = self._make_client(pipeline_row, ocr_row=None)
        mock_kimi = MagicMock()
        with pytest.raises(ValueError, match="No OCR results"):
            run_formatting("doc1", client, mock_kimi)

    def test_runs_all_steps(self, tmp_path):
        pipeline_row = {"doc_id": "doc1", "last_formatting": None, "formatting_nb": 0}
        ocr_row = {"content": "sample ocr text"}
        client = self._make_client(pipeline_row, ocr_row)

        mock_kimi = MagicMock()
        mock_kimi.chat.completions.create.return_value.choices[
            0
        ].message.content = '{"result": "ok"}'

        prompt_content = "Process: {ocr_text}"
        with patch("pipeline.formatting.PROMPTS_DIR", tmp_path):
            for step in ["extract_model_name", "extract_table"]:
                (tmp_path / f"{step}.txt").write_text(prompt_content)
            run_formatting("doc1", client, mock_kimi)

        assert mock_kimi.chat.completions.create.call_count == 2
