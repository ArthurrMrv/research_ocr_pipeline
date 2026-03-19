from unittest.mock import MagicMock, patch

import pytest

from pipeline.tracker import (
    append_error,
    bronze_insert,
    bronze_update_path,
    delete_ocr_rows,
    delete_scout_page_scores,
    formatting_upsert,
    get_all_doc_ids,
    get_bronze_row,
    get_formatting_results,
    get_ocr_chunks,
    get_scout_page_scores,
    get_supabase_client,
    pipeline_get,
    pipeline_insert,
    pipeline_update,
    scout_page_score_upsert,
    silver_upsert,
)


def _make_client(data=None):
    """Helper: mock supabase client returning given data."""
    client = MagicMock()
    chain = MagicMock()
    result = MagicMock()
    result.data = data or []
    chain.execute.return_value = result
    chain.select.return_value = chain
    chain.insert.return_value = chain
    chain.update.return_value = chain
    chain.upsert.return_value = chain
    chain.delete.return_value = chain
    chain.eq.return_value = chain
    client.table.return_value = chain
    return client


class TestGetSupabaseClient:
    def test_creates_client_from_env(self):
        with patch("pipeline.tracker.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = get_supabase_client()
            mock_create.assert_called_once_with(
                "https://test.supabase.co", "test-service-key"
            )

    def test_raises_on_missing_env(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL")
        with pytest.raises(KeyError):
            get_supabase_client()


class TestBronzeInsert:
    def test_inserts_new_record(self):
        client = _make_client()
        bronze_insert(client, "doc1", "/path/file.pdf", "file.pdf")
        client.table.assert_called_with("bronze_mapping")
        client.table().insert.assert_called_with(
            {"doc_id": "doc1", "file_path": "/path/file.pdf", "doc_name": "file.pdf"}
        )

    def test_inserts_with_company_and_date(self):
        client = _make_client()
        bronze_insert(
            client, "doc1", "/path/file.pdf", "Apple_2024-01-01.pdf",
            institution="Apple", report_date="2024-01-01",
        )
        client.table().insert.assert_called_with(
            {
                "doc_id": "doc1",
                "file_path": "/path/file.pdf",
                "doc_name": "Apple_2024-01-01.pdf",
                "institution": "Apple",
                "report_date": "2024-01-01",
            }
        )

    def test_omits_none_company_and_date(self):
        client = _make_client()
        bronze_insert(
            client, "doc1", "/path/file.pdf", "file.pdf",
            institution=None, report_date=None,
        )
        client.table().insert.assert_called_with(
            {"doc_id": "doc1", "file_path": "/path/file.pdf", "doc_name": "file.pdf"}
        )


class TestPipelineInsert:
    def test_inserts_empty_row(self):
        client = _make_client()
        pipeline_insert(client, "doc1")
        client.table.assert_called_with("pipeline")
        client.table().insert.assert_called_with({"doc_id": "doc1"})


class TestBronzeUpdatePath:
    def test_updates_file_path(self):
        client = _make_client()
        bronze_update_path(client, "doc1", "/new/path/file.pdf")
        client.table.assert_called_with("bronze_mapping")
        client.table().update.assert_called_with({"file_path": "/new/path/file.pdf"})
        client.table().update().eq.assert_called_with("doc_id", "doc1")


class TestPipelineGet:
    def test_returns_row_when_found(self):
        row = {"doc_id": "doc1", "last_ocr": None}
        client = _make_client(data=[row])
        result = pipeline_get(client, "doc1")
        assert result == row

    def test_returns_none_when_not_found(self):
        client = _make_client(data=[])
        result = pipeline_get(client, "missing")
        assert result is None


class TestPipelineUpdate:
    def test_updates_fields(self):
        client = _make_client()
        pipeline_update(client, "doc1", {"last_ocr": "2024-01-01T00:00:00+00:00"})
        client.table().update.assert_called_with(
            {"last_ocr": "2024-01-01T00:00:00+00:00"}
        )
        client.table().update().eq.assert_called_with("doc_id", "doc1")


class TestSilverUpsert:
    def test_upserts_to_table(self):
        client = _make_client()
        row = {"doc_id": "doc1", "ocr_model": "test-model", "content": "text"}
        silver_upsert(client, "ocr_results", row)
        client.table.assert_called_with("ocr_results")
        client.table().upsert.assert_called_with(row, on_conflict=None)


class TestFormattingUpsert:
    def test_upserts_formatting_row(self):
        client = _make_client()
        row = {
            "doc_id": "doc1",
            "step_name": "extract_model_name",
            "formatting_model": "kimi-k2.5",
            "content": {"model_name": "test"},
        }
        formatting_upsert(client, row)
        client.table.assert_called_with("formatting")
        client.table().upsert.assert_called_with(row, on_conflict="doc_id,step_name")


class TestGetFormattingResults:
    def test_returns_rows_by_step_name(self):
        client = _make_client(data=[
            {"doc_id": "doc1", "step_name": "extract_model_name", "content": {"models": []}},
            {"doc_id": "doc1", "step_name": "extract_table", "content": {"tables": []}},
        ])
        result = get_formatting_results(client, "doc1")
        assert set(result) == {"extract_model_name", "extract_table"}
        assert result["extract_table"]["content"] == {"tables": []}


class TestScoutPageScoreUpsert:
    def test_upserts_scout_page_score_row(self):
        client = _make_client()
        row = {
            "doc_id": "doc1",
            "step_name": "extract_table",
            "page_number": 3,
            "score": 0.82,
            "scout_model": "gemini-test",
        }
        scout_page_score_upsert(client, row)
        client.table.assert_called_with("scout_page_scores")
        client.table().upsert.assert_called_with(
            row,
            on_conflict="doc_id,step_name,page_number",
        )


class TestAppendError:
    def test_appends_error_to_empty_array(self):
        row = {"doc_id": "doc1", "error": []}
        client = _make_client(data=[row])
        append_error(client, "doc1", "Something failed")
        update_call = client.table().update.call_args
        updated_error = update_call[0][0]["error"]
        assert isinstance(updated_error, list)
        assert len(updated_error) == 1
        assert updated_error[0]["message"] == "Something failed"

    def test_appends_to_existing_errors(self):
        existing = [{"message": "old error", "ts": "2024-01-01T00:00:00+00:00"}]
        row = {"doc_id": "doc1", "error": existing}
        client = _make_client(data=[row])
        append_error(client, "doc1", "new error")
        update_call = client.table().update.call_args
        updated_error = update_call[0][0]["error"]
        assert isinstance(updated_error, list)
        assert len(updated_error) == 2

    def test_no_op_when_doc_not_found(self):
        client = _make_client(data=[])
        append_error(client, "missing", "error")
        client.table().update.assert_not_called()

    def test_appends_error_with_context(self):
        row = {"doc_id": "doc1", "error": []}
        client = _make_client(data=[row])
        append_error(client, "doc1", "Formatting failed", context={"stage": "formatting", "attempt": 1})
        update_call = client.table().update.call_args
        updated_error = update_call[0][0]["error"]
        assert updated_error[0]["stage"] == "formatting"
        assert updated_error[0]["attempt"] == 1


class TestGetAllDocIds:
    def test_returns_list_of_ids(self):
        data = [{"doc_id": "doc1"}, {"doc_id": "doc2"}]
        client = _make_client(data=data)
        result = get_all_doc_ids(client)
        assert result == ["doc1", "doc2"]

    def test_returns_empty_list_when_none(self):
        client = _make_client(data=[])
        result = get_all_doc_ids(client)
        assert result == []


class TestGetBronzeRow:
    def test_returns_row_when_found(self):
        row = {"doc_id": "doc1", "file_path": "/path.pdf", "institution": "Apple"}
        client = _make_client(data=[row])
        result = get_bronze_row(client, "doc1")
        assert result == row

    def test_returns_none_when_not_found(self):
        client = _make_client(data=[])
        result = get_bronze_row(client, "missing")
        assert result is None


class TestGetOcrChunks:
    def test_returns_sorted_chunks(self):
        data = [
            {"doc_id": "doc1", "page_number": 76, "content": "chunk2"},
            {"doc_id": "doc1", "page_number": 1, "content": "chunk1"},
        ]
        client = _make_client(data=data)
        result = get_ocr_chunks(client, "doc1")
        assert result[0]["page_number"] == 1
        assert result[1]["page_number"] == 76

    def test_returns_empty_list_when_none(self):
        client = _make_client(data=[])
        result = get_ocr_chunks(client, "doc1")
        assert result == []


class TestDeleteOcrRows:
    def test_calls_delete_with_doc_id(self):
        client = _make_client()
        delete_ocr_rows(client, "doc1")
        client.table.assert_called_with("ocr_results")
        client.table().delete.assert_called_once()
        client.table().delete().eq.assert_called_with("doc_id", "doc1")


class TestDeleteScoutPageScores:
    def test_calls_delete_with_doc_id(self):
        client = _make_client()
        delete_scout_page_scores(client, "doc1")
        client.table.assert_called_with("scout_page_scores")
        client.table().delete.assert_called_once()
        client.table().delete().eq.assert_called_with("doc_id", "doc1")


class TestGetScoutPageScores:
    def test_returns_sorted_scores(self):
        data = [
            {"doc_id": "doc1", "step_name": "extract_table", "page_number": 3, "score": 0.8},
            {"doc_id": "doc1", "step_name": "extract_model_name", "page_number": 2, "score": 0.4},
            {"doc_id": "doc1", "step_name": "extract_model_name", "page_number": 1, "score": 0.9},
        ]
        client = _make_client(data=data)
        result = get_scout_page_scores(client, "doc1")
        assert [row["page_number"] for row in result] == [1, 2, 3]

    def test_returns_empty_list_when_none(self):
        client = _make_client(data=[])
        result = get_scout_page_scores(client, "doc1")
        assert result == []
