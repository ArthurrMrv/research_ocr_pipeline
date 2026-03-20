import pandas as pd

from dashboard import data


class TestExtractModelNameExport:
    def test_build_export_df_flattens_content_and_preserves_column_order(self):
        formatting_rows = [
            {
                "doc_id": "doc-1",
                "content": {
                    "model_name": "Capital Market Model",
                    "notes_model": "scenario-based, valuation",
                    "variables": ["inflation", "real GDP growth", "dividend yield"],
                    "steps_summary": "Build blocks are combined into expected return.",
                    "steps_detailed": "1. Project macro inputs\n2. Estimate earnings\n3. Combine blocks",
                    "variables_important": ["inflation", "dividend yield"],
                },
            }
        ]
        bronze_rows = [{"doc_id": "doc-1", "doc_name": "JPM_2024-01-01.pdf"}]

        result = data._build_extract_model_name_export_df(formatting_rows, bronze_rows)

        assert list(result.columns) == data.MODEL_EXPORT_COLUMNS
        assert result.to_dict("records") == [
            {
                "document_name": "JPM_2024-01-01.pdf",
                "model_name": "Capital Market Model",
                "notes_model": "scenario-based, valuation",
                "variables": "inflation, real GDP growth, dividend yield",
                "variables_nb": 3,
                "steps_summary": "Build blocks are combined into expected return.",
                "steps_detailed": "1. Project macro inputs\n2. Estimate earnings\n3. Combine blocks",
                "variables_importants": "inflation, dividend yield",
            }
        ]

    def test_build_export_df_handles_missing_optional_fields(self):
        formatting_rows = [
            {
                "doc_id": "doc-2",
                "content": {
                    "model_name": "Narrative approach",
                    "variables": [],
                },
            }
        ]

        result = data._build_extract_model_name_export_df(formatting_rows, bronze_rows=[])

        assert result.loc[0, "document_name"] == ""
        assert result.loc[0, "notes_model"] == ""
        assert result.loc[0, "variables"] == ""
        assert result.loc[0, "variables_nb"] == 0
        assert result.loc[0, "steps_summary"] == ""
        assert result.loc[0, "steps_detailed"] == ""
        assert result.loc[0, "variables_importants"] == ""

    def test_fetch_extract_model_name_export_queries_expected_tables(self, monkeypatch):
        def fake_query(table: str, select: str = "*", **eq_filters):
            if table == "formatting":
                assert select == "doc_id,content"
                assert eq_filters == {"step_name": "extract_model_name"}
                return [
                    {
                        "doc_id": "doc-3",
                        "content": {
                            "model_name": "Factor model",
                            "variables": ["risk premium"],
                            "variables_important": ["risk premium"],
                        },
                    }
                ]
            if table == "bronze_mapping":
                assert select == "doc_id,doc_name"
                assert eq_filters == {}
                return [{"doc_id": "doc-3", "doc_name": "MSCI_2024-06-01.pdf"}]
            raise AssertionError(f"Unexpected table queried: {table}")

        monkeypatch.setattr(data, "_query", fake_query)
        data.fetch_extract_model_name_export.clear()

        result = data.fetch_extract_model_name_export()

        assert isinstance(result, pd.DataFrame)
        assert result.loc[0, "document_name"] == "MSCI_2024-06-01.pdf"
        assert result.loc[0, "variables_nb"] == 1
        assert result.loc[0, "variables_importants"] == "risk premium"
