from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from main import run


class TestCli:
    def test_run_with_empty_dir(self, tmp_path):
        runner = CliRunner()
        with (
            patch("main.get_supabase_client") as mock_supa,
            patch("main.get_kimi_client") as mock_kimi,
            patch("main.ingest", return_value=[]) as mock_ingest,
            patch("main.get_all_doc_ids", return_value=[]) as mock_ids,
        ):
            result = runner.invoke(run, [str(tmp_path)])
            assert result.exit_code == 0
            assert "Done." in result.output
            mock_ingest.assert_called_once()

    def test_run_processes_new_doc(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        runner = CliRunner()

        with (
            patch("main.get_supabase_client") as mock_supa,
            patch("main.get_kimi_client") as mock_kimi,
            patch("main.ingest", return_value=["doc1"]) as mock_ingest,
            patch("main.get_all_doc_ids", return_value=["doc1"]) as mock_ids,
            patch("main.run_ocr") as mock_ocr,
            patch("main.run_formatting") as mock_fmt,
        ):
            result = runner.invoke(run, [str(tmp_path)])
            assert result.exit_code == 0
            mock_ocr.assert_called_once_with("doc1", mock_supa.return_value, force=False, since=None)
            mock_fmt.assert_called_once_with("doc1", mock_supa.return_value, mock_kimi.return_value)

    def test_parse_all_flag_passes_force(self, tmp_path):
        runner = CliRunner()
        with (
            patch("main.get_supabase_client"),
            patch("main.get_kimi_client"),
            patch("main.ingest", return_value=[]),
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr") as mock_ocr,
            patch("main.run_formatting"),
        ):
            result = runner.invoke(run, [str(tmp_path), "--parse-all"])
            assert result.exit_code == 0
            call_kwargs = mock_ocr.call_args[1]
            assert call_kwargs["force"] is True

    def test_parse_date_flag_passes_since(self, tmp_path):
        runner = CliRunner()
        with (
            patch("main.get_supabase_client"),
            patch("main.get_kimi_client"),
            patch("main.ingest", return_value=[]),
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr") as mock_ocr,
            patch("main.run_formatting"),
        ):
            result = runner.invoke(run, [str(tmp_path), "--parse-date", "2024-06-01"])
            assert result.exit_code == 0
            call_kwargs = mock_ocr.call_args[1]
            assert call_kwargs["since"] == "2024-06-01"

    def test_missing_pdf_dir_fails(self):
        runner = CliRunner()
        result = runner.invoke(run, ["/nonexistent/path"])
        assert result.exit_code != 0
