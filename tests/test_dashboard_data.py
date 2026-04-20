import pandas as pd

from dashboard import data


class TestModelExport:
    def test_build_model_export_df_merges_all_steps(self):
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
                    "uses_regressions": 1,
                    "uses_simulations": 0,
                    "uses_averages": 1,
                    "uses_mean_reversion": 1,
                },
            }
        ]
        assumptions_rows = [
            {
                "doc_id": "doc-1",
                "content": {
                    "assumptions": [
                        {"assumption": "CAPE reverts to mean", "building_block": "valuation", "classification": "mean-reversion"},
                    ],
                    "techniques_used": [
                        {"technique_name": "mean-reversion model", "complexity": 5},
                    ],
                    "sophistication_index": 4.5,
                    "sophistication_explanation": "Relies on mean-reversion with building-block approach.",
                },
            }
        ]
        bronze_rows = [{"doc_id": "doc-1", "doc_name": "JPM_2024-01-01.pdf"}]

        result = data._build_model_export_df(inputs_rows, methodology_rows, bronze_rows, assumptions_rows)

        assert list(result.columns) == data.MODEL_EXPORT_COLUMNS
        record = result.to_dict("records")[0]
        assert record["document_name"] == "JPM_2024-01-01.pdf"
        assert record["model_name"] == "Capital Market Model"
        assert record["variables"] == "inflation, real GDP growth, dividend yield"
        assert record["variables_nb"] == 3
        assert record["assumptions_values"] == "* inflation = 2%\n* steady GDP growth (implied)"
        assert record["assumptions_structural"] == "* r_equity = r_f + ERP\n* mean reversion (implied)"
        assert record["uses_regressions"] == 1
        assert record["uses_simulations"] == 0
        assert record["uses_averages"] == 1
        assert record["uses_mean_reversion"] == 1
        assert "[mean-reversion] CAPE reverts to mean" in record["assumptions_classified"]
        assert record["techniques_used"] == "mean-reversion model (5)"
        assert record["sophistication_index"] == 4.5

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
        assert result.loc[0, "uses_regressions"] == 0
        assert result.loc[0, "uses_simulations"] == 0
        assert result.loc[0, "uses_averages"] == 0
        assert result.loc[0, "uses_mean_reversion"] == 0
        assert result.loc[0, "assumptions_classified"] == ""
        assert result.loc[0, "techniques_used"] == ""
        assert result.loc[0, "sophistication_index"] == ""
        assert result.loc[0, "sophistication_explanation"] == ""

    def test_fetch_model_export_queries_all_steps(self, monkeypatch):
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
            if table == "formatting" and eq_filters.get("step_name") == "extract_model_assumptions":
                return []
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
        assert "extract_model_assumptions" in step_names_queried


class TestFetchMermaidReports:
    def test_sanitizes_diagrams_before_returning_reports(self, monkeypatch):
        def fake_query(table: str, select: str = "*", **eq_filters):
            if table == "formatting" and eq_filters.get("step_name") == "extract_model_methodology":
                return [
                    {
                        "doc_id": "doc-1",
                        "content": {
                            "steps_summary": "Summary",
                            "mermaid_diagram": "flowCchart TD A --> B",
                        },
                    }
                ]
            if table == "formatting" and eq_filters.get("step_name") == "extract_model_inputs":
                return [
                    {
                        "doc_id": "doc-1",
                        "content": {
                            "model_name": "Model",
                        },
                    }
                ]
            if table == "bronze_mapping":
                return [
                    {
                        "doc_id": "doc-1",
                        "doc_name": "Doc.pdf",
                        "institution": "Firm",
                    }
                ]
            raise AssertionError(f"Unexpected query: {table} {eq_filters}")

        monkeypatch.setattr(data, "_query", fake_query)
        data.fetch_mermaid_reports.clear()

        reports = data.fetch_mermaid_reports()

        assert reports == [
            {
                "doc_name": "Doc.pdf",
                "institution": "Firm",
                "model_name": "Model",
                "mermaid_diagram": "flowchart TD\nA --> B",
                "steps_summary": "Summary",
            }
        ]
