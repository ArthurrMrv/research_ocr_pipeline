import pandas as pd

from dashboard import data


class TestModelExport:
    def test_build_model_export_df_merges_both_steps(self):
        inputs_rows = [
            {
                "doc_id": "doc-1",
                "content": {
                    "model_name": "Capital Market Model",
                    "notes_model": "scenario-based, valuation",
                    "variables": ["inflation", "real GDP growth", "dividend yield"],
                    "variables_important": ["inflation", "dividend yield"],
                    "assumptions": ["inflation = 2%", "steady GDP growth (implied)"],
                },
            }
        ]
        methodology_rows = [
            {
                "doc_id": "doc-1",
                "content": {
                    "steps_summary": "Build blocks are combined into expected return.",
                    "steps_detailed": "1. Project macro inputs\n2. Estimate earnings\n3. Combine blocks",
                    "mermaid_diagram": "flowchart TD\n    A --> B",
                    "assumptions": ["r_equity = r_f + ERP", "mean reversion (implied)"],
                },
            }
        ]
        bronze_rows = [{"doc_id": "doc-1", "doc_name": "JPM_2024-01-01.pdf"}]

        result = data._build_model_export_df(inputs_rows, methodology_rows, bronze_rows)

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
                "assumptions_values": "* inflation = 2%\n* steady GDP growth (implied)",
                "assumptions_structural": "* r_equity = r_f + ERP\n* mean reversion (implied)",
            }
        ]

    def test_build_model_export_df_handles_missing_steps(self):
        inputs_rows = [
            {
                "doc_id": "doc-2",
                "content": {
                    "model_name": "Narrative approach",
                    "variables": [],
                },
            }
        ]

        result = data._build_model_export_df(inputs_rows, methodology_rows=[], bronze_rows=[])

        assert result.loc[0, "document_name"] == ""
        assert result.loc[0, "model_name"] == "Narrative approach"
        assert result.loc[0, "notes_model"] == ""
        assert result.loc[0, "variables"] == ""
        assert result.loc[0, "variables_nb"] == 0
        assert result.loc[0, "steps_summary"] == ""
        assert result.loc[0, "steps_detailed"] == ""
        assert result.loc[0, "variables_importants"] == ""
        assert result.loc[0, "assumptions_values"] == ""
        assert result.loc[0, "assumptions_structural"] == ""

    def test_fetch_model_export_queries_both_steps(self, monkeypatch):
        calls = []

        def fake_query(table: str, select: str = "*", **eq_filters):
            calls.append((table, eq_filters.get("step_name")))
            if table == "formatting" and eq_filters.get("step_name") == "extract_model_inputs":
                return [
                    {
                        "doc_id": "doc-3",
                        "content": {
                            "model_name": "Factor model",
                            "variables": ["risk premium"],
                        },
                    }
                ]
            if table == "formatting" and eq_filters.get("step_name") == "extract_model_methodology":
                return [
                    {
                        "doc_id": "doc-3",
                        "content": {
                            "steps_summary": "Factor-based approach.",
                            "steps_detailed": "1. Identify factors.",
                            "mermaid_diagram": "flowchart TD\n    A --> B",
                        },
                    }
                ]
            if table == "bronze_mapping":
                return [{"doc_id": "doc-3", "doc_name": "MSCI_2024-06-01.pdf"}]
            raise AssertionError(f"Unexpected query: {table} {eq_filters}")

        monkeypatch.setattr(data, "_query", fake_query)
        data.fetch_model_export.clear()

        result = data.fetch_model_export()

        assert isinstance(result, pd.DataFrame)
        assert result.loc[0, "document_name"] == "MSCI_2024-06-01.pdf"
        assert result.loc[0, "model_name"] == "Factor model"
        assert result.loc[0, "steps_summary"] == "Factor-based approach."
        step_names_queried = [c[1] for c in calls if c[0] == "formatting"]
        assert "extract_model_inputs" in step_names_queried
        assert "extract_model_methodology" in step_names_queried
