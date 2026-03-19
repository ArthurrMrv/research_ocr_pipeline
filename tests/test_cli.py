from unittest.mock import ANY, patch

import pytest
from click.testing import CliRunner

from main import run

OCR_DONE = {"status": "done", "empty_pages": 0}
FMT_DONE = {"status": "done", "completed_steps": 2, "failed_steps": 0, "failed_details": []}


class TestCli:
    def test_run_with_empty_dir(self, tmp_path):
        runner = CliRunner()
        with (
            patch("main.get_supabase_client") as mock_supa,
            patch("main.ingest", return_value=[]),
            patch("main.get_all_doc_ids", return_value=[]),
            patch("main.run_scout"),
        ):
            result = runner.invoke(run, [str(tmp_path)])
            assert result.exit_code == 0
            assert "Pipeline complete" in result.output

    def test_run_processes_new_doc(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        runner = CliRunner()

        with (
            patch("main.get_supabase_client") as mock_supa,
            patch("main.ingest", return_value=["doc1"]),
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr", return_value=OCR_DONE) as mock_ocr,
            patch("main.run_scout") as mock_scout,
            patch("main.run_formatting", return_value=FMT_DONE) as mock_fmt,
        ):
            result = runner.invoke(run, [str(tmp_path)])
            assert result.exit_code == 0
            mock_ocr.assert_called_once()
            mock_scout.assert_called_once_with("doc1", mock_supa.return_value, force=False)
            mock_fmt.assert_called_once_with("doc1", mock_supa.return_value, force=False)

    def test_single_file_run_targets_exact_doc_id_not_matching_basename(self, tmp_path):
        pdf = tmp_path / "JPMorgan_20170930.pdf"
        pdf.touch()
        runner = CliRunner()

        with (
            patch("main.get_supabase_client") as mock_supa,
            patch("main.ingest", return_value=[]),
            patch("main.get_all_doc_ids", return_value=["target-doc", "stale-doc"]),
            patch("main.make_doc_id", return_value="target-doc"),
            patch("main._doc_label", side_effect=["JPMorgan_20170930.pdf", "JPMorgan_20170930.pdf"]),
            patch("main.run_ocr", return_value=OCR_DONE) as mock_ocr,
            patch("main.run_scout") as mock_scout,
            patch("main.run_formatting", return_value=FMT_DONE) as mock_fmt,
        ):
            result = runner.invoke(run, [str(pdf)])
            assert result.exit_code == 0
            mock_ocr.assert_called_once_with("target-doc", mock_supa.return_value, force=False, since=None, progress_callback=ANY)
            mock_scout.assert_called_once_with("target-doc", mock_supa.return_value, force=False)
            mock_fmt.assert_called_once_with("target-doc", mock_supa.return_value, force=False)

    def test_parse_all_flag_passes_force(self, tmp_path):
        runner = CliRunner()
        with (
            patch("main.get_supabase_client"),
            patch("main.ingest", return_value=[]),
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr", return_value=OCR_DONE) as mock_ocr,
            patch("main.run_scout") as mock_scout,
            patch("main.run_formatting", return_value=FMT_DONE) as mock_fmt,
        ):
            result = runner.invoke(run, [str(tmp_path), "--parse-all"])
            assert result.exit_code == 0
            assert mock_ocr.call_args[1]["force"] is True
            assert mock_scout.call_args[1]["force"] is True
            assert mock_fmt.call_args[1]["force"] is True

    def test_force_flag_passes_per_step_force(self, tmp_path):
        runner = CliRunner()
        with (
            patch("main.get_supabase_client"),
            patch("main.ingest", return_value=[]),
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr", return_value=OCR_DONE) as mock_ocr,
            patch("main.run_scout") as mock_scout,
            patch("main.run_formatting", return_value=FMT_DONE) as mock_fmt,
        ):
            result = runner.invoke(
                run,
                [str(tmp_path), "--step", "ocr", "--step", "formatting", "--force", "formatting"],
            )
            assert result.exit_code == 0
            assert mock_ocr.call_args[1]["force"] is False
            mock_scout.assert_not_called()
            assert mock_fmt.call_args[1]["force"] is True

    def test_parse_date_flag_passes_since(self, tmp_path):
        runner = CliRunner()
        with (
            patch("main.get_supabase_client"),
            patch("main.ingest", return_value=[]),
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr", return_value=OCR_DONE) as mock_ocr,
            patch("main.run_scout"),
            patch("main.run_formatting"),
        ):
            result = runner.invoke(run, [str(tmp_path), "--parse-date", "2024-06-01"])
            assert result.exit_code == 0
            assert mock_ocr.call_args[1]["since"] == "2024-06-01"

    def test_missing_pdf_dir_fails(self):
        runner = CliRunner()
        result = runner.invoke(run, ["/nonexistent/path"])
        assert result.exit_code != 0

    def test_step_ingest_only(self, tmp_path):
        runner = CliRunner()
        with (
            patch("main.get_supabase_client"),
            patch("main.ingest", return_value=["doc1"]) as mock_ingest,
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr", return_value=OCR_DONE) as mock_ocr,
            patch("main.run_scout") as mock_scout,
            patch("main.run_formatting", return_value=FMT_DONE) as mock_fmt,
        ):
            result = runner.invoke(run, [str(tmp_path), "--step", "ingest"])
            assert result.exit_code == 0
            mock_ingest.assert_called_once()
            mock_ocr.assert_not_called()
            mock_scout.assert_not_called()
            mock_fmt.assert_not_called()

    def test_step_ocr_only(self, tmp_path):
        runner = CliRunner()
        with (
            patch("main.get_supabase_client"),
            patch("main.ingest") as mock_ingest,
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr", return_value=OCR_DONE) as mock_ocr,
            patch("main.run_scout") as mock_scout,
            patch("main.run_formatting", return_value=FMT_DONE) as mock_fmt,
        ):
            result = runner.invoke(run, [str(tmp_path), "--step", "ocr"])
            assert result.exit_code == 0
            mock_ingest.assert_not_called()
            mock_ocr.assert_called_once()
            mock_scout.assert_not_called()
            mock_fmt.assert_not_called()

    def test_step_scout_only(self, tmp_path):
        runner = CliRunner()
        with (
            patch("main.get_supabase_client"),
            patch("main.ingest") as mock_ingest,
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr", return_value=OCR_DONE) as mock_ocr,
            patch("main.run_scout") as mock_scout,
            patch("main.run_formatting", return_value=FMT_DONE) as mock_fmt,
        ):
            result = runner.invoke(run, [str(tmp_path), "--step", "scout"])
            assert result.exit_code == 0
            mock_ingest.assert_not_called()
            mock_ocr.assert_not_called()
            mock_scout.assert_called_once()
            mock_fmt.assert_not_called()

    def test_step_formatting_only(self, tmp_path):
        runner = CliRunner()
        with (
            patch("main.get_supabase_client"),
            patch("main.ingest") as mock_ingest,
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr", return_value=OCR_DONE) as mock_ocr,
            patch("main.run_scout") as mock_scout,
            patch("main.run_formatting", return_value=FMT_DONE) as mock_fmt,
        ):
            result = runner.invoke(run, [str(tmp_path), "--step", "formatting"])
            assert result.exit_code == 0
            mock_ingest.assert_not_called()
            mock_ocr.assert_not_called()
            mock_scout.assert_not_called()
            mock_fmt.assert_called_once_with("doc1", ANY, force=False)

    def test_multiple_steps(self, tmp_path):
        runner = CliRunner()
        with (
            patch("main.get_supabase_client"),
            patch("main.ingest", return_value=[]) as mock_ingest,
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr", return_value=OCR_DONE) as mock_ocr,
            patch("main.run_scout") as mock_scout,
            patch("main.run_formatting", return_value=FMT_DONE) as mock_fmt,
        ):
            result = runner.invoke(run, [str(tmp_path), "--step", "ingest", "--step", "ocr"])
            assert result.exit_code == 0
            mock_ingest.assert_called_once()
            mock_ocr.assert_called_once()
            mock_scout.assert_not_called()
            mock_fmt.assert_not_called()

    def test_formatting_warning_panel_shows_step_details(self, tmp_path):
        """When formatting steps fail, the warning panel shows step names and reasons."""
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        runner = CliRunner()

        fmt_with_failures = {
            "status": "done",
            "completed_steps": 0,
            "failed_steps": 2,
            "failed_details": [
                {"step": "extract_model_name", "reason": "schema validation failed after retry"},
                {"step": "extract_table", "reason": "no scout pages above threshold"},
            ],
        }

        with (
            patch("main.get_supabase_client"),
            patch("main.ingest", return_value=["doc1"]),
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr", return_value=OCR_DONE),
            patch("main.run_scout"),
            patch("main.run_formatting", return_value=fmt_with_failures),
        ):
            result = runner.invoke(run, [str(tmp_path)])
            assert result.exit_code == 0
            assert "extract_model_name" in result.output
            assert "schema validation failed after retry" in result.output
            assert "extract_table" in result.output
            assert "no scout pages above threshold" in result.output

    def test_debug_formatting_warning_panel_shows_attempt_history(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        runner = CliRunner()

        fmt_with_history = {
            "status": "skipped",
            "completed_steps": 0,
            "failed_steps": 0,
            "failed_details": [
                {"step": "all", "reason": "exceeded max attempts (3)"},
            ],
            "attempt_history": [
                {
                    "attempt": 1,
                    "failures": [
                        {"step": "extract_model_name", "reason": "schema validation failed after retry"},
                    ],
                },
                {
                    "attempt": 2,
                    "failures": [
                        {
                            "step": "extract_table",
                            "reason": "Gemini returned non-JSON",
                            "raw_output": "{\n  \"tables\": [\n    1\n  ]\n}\ntrailing text",
                        },
                    ],
                },
            ],
        }

        with (
            patch("main.get_supabase_client"),
            patch("main.ingest", return_value=["doc1"]),
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr", return_value=OCR_DONE),
            patch("main.run_scout"),
            patch("main.run_formatting", return_value=fmt_with_history),
        ):
            result = runner.invoke(run, [str(tmp_path), "--debug"])
            assert result.exit_code == 0
            assert "exceeded max attempts (3)" in result.output
            assert "attempt 1" in result.output
            assert "extract_model_name" in result.output
            assert "attempt 2" in result.output
            assert "Gemini returned non-JSON" in result.output
            assert "raw output:" in result.output
            assert "\"tables\": [" in result.output

    def test_no_step_flag_runs_all(self, tmp_path):
        runner = CliRunner()
        with (
            patch("main.get_supabase_client"),
            patch("main.ingest", return_value=[]) as mock_ingest,
            patch("main.get_all_doc_ids", return_value=["doc1"]),
            patch("main.run_ocr", return_value=OCR_DONE) as mock_ocr,
            patch("main.run_scout") as mock_scout,
            patch("main.run_formatting", return_value=FMT_DONE) as mock_fmt,
        ):
            result = runner.invoke(run, [str(tmp_path)])
            assert result.exit_code == 0
            mock_ingest.assert_called_once()
            mock_ocr.assert_called_once()
            mock_scout.assert_called_once()
            mock_fmt.assert_called_once()
