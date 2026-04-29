import os
from unittest.mock import MagicMock, patch

import pytest

from pipeline.ingest import ingest, make_doc_id


class TestMakeDocId:
    def test_returns_deterministic_id(self, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.touch()
        id1 = make_doc_id(str(pdf))
        id2 = make_doc_id(str(pdf))
        assert id1 == id2

    def test_different_filenames_produce_different_ids(self, tmp_path):
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.touch()
        b.touch()
        assert make_doc_id(str(a)) != make_doc_id(str(b))

    def test_same_filename_different_dirs_produce_same_id(self, tmp_path):
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "report.pdf").touch()
        (dir_b / "report.pdf").touch()
        assert make_doc_id(str(dir_a / "report.pdf")) == make_doc_id(str(dir_b / "report.pdf"))

    def test_relative_and_absolute_produce_same_id(self, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.touch()
        abs_id = make_doc_id(os.path.abspath(str(pdf)))
        rel_id = make_doc_id(str(pdf))
        assert abs_id == rel_id


class TestIngest:
    def _make_client(self, existing_ids=None, bronze_rows=None):
        client = MagicMock()

        bronze_rows = bronze_rows or {}

        def table_side_effect(name):
            chain = MagicMock()
            result = MagicMock()
            if name == "bronze_mapping":
                result.data = [{"doc_id": d} for d in (existing_ids or [])]

                def eq_side_effect(field, value):
                    if field == "doc_id":
                        row = bronze_rows.get(value)
                        result.data = [row] if row else []
                    elif field == "doc_name":
                        # Match by basename of stored file_path
                        matched = [
                            row for row in bronze_rows.values()
                            if os.path.basename(row.get("file_path", "")) == value
                        ]
                        result.data = matched
                    else:
                        result.data = []
                    return chain

                chain.eq.side_effect = eq_side_effect
            else:
                result.data = [{"doc_id": d} for d in (existing_ids or [])]
                chain.eq.return_value = chain
            chain.execute.return_value = result
            chain.select.return_value = chain
            chain.insert.return_value = chain
            chain.update.return_value = chain
            return chain

        client.table.side_effect = table_side_effect
        return client

    def test_ingests_new_pdfs(self, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.touch()
        client = self._make_client(existing_ids=[])

        new_ids = ingest([str(pdf)], client)

        assert len(new_ids) == 1
        assert new_ids[0] == make_doc_id(str(pdf))

    def test_skips_existing_pdfs(self, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.touch()
        existing_id = make_doc_id(str(pdf))
        client = self._make_client(
            existing_ids=[existing_id],
            bronze_rows={existing_id: {"doc_id": existing_id, "file_path": os.path.abspath(str(pdf))}},
        )

        new_ids = ingest([str(pdf)], client)

        assert new_ids == []

    def test_ingests_only_new_among_mixed(self, tmp_path):
        existing_pdf = tmp_path / "existing.pdf"
        new_pdf = tmp_path / "new.pdf"
        existing_pdf.touch()
        new_pdf.touch()
        existing_id = make_doc_id(str(existing_pdf))
        client = self._make_client(
            existing_ids=[existing_id],
            bronze_rows={existing_id: {"doc_id": existing_id, "file_path": os.path.abspath(str(existing_pdf))}},
        )

        new_ids = ingest([str(existing_pdf), str(new_pdf)], client)

        assert len(new_ids) == 1
        assert new_ids[0] == make_doc_id(str(new_pdf))

    def test_updates_stored_path_for_existing_doc(self, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.touch()
        existing_id = make_doc_id(str(pdf))
        client = self._make_client(
            existing_ids=[existing_id],
            bronze_rows={existing_id: {"doc_id": existing_id, "file_path": "/stale/path/report.pdf"}},
        )

        with patch("pipeline.ingest.bronze_update_path") as mock_update:
            new_ids = ingest([str(pdf)], client)

        assert new_ids == []
        mock_update.assert_called_once_with(client, existing_id, os.path.abspath(str(pdf)))

    def test_empty_pdf_list_returns_empty(self):
        client = self._make_client(existing_ids=[])
        new_ids = ingest([], client)
        assert new_ids == []
