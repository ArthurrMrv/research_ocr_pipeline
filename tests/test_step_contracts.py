import jsonschema
import pytest

from pipeline.formatting import load_step, validate_output


class TestExtractModelInputsContract:
    def test_prompt_contains_variable_focus(self):
        prompt, _, config = load_step("extract_model_inputs")

        assert "variables" in prompt.lower()
        assert "model_name" in prompt
        assert config["temperature"] == 0

    def test_accepts_expected_shape(self):
        _, schema, _ = load_step("extract_model_inputs")

        payload = {
            "model_name": "Capital Market Model",
            "notes_model": "scenario-based, valuation",
            "variables": ["risk factors", "CAPM"],
            "variables_important": ["risk factors"],
            "assumptions": ["inflation = 2%", "steady-state economy (implied)"],
        }

        validate_output(payload, schema, "extract_model_inputs")

    def test_requires_model_name_and_variables(self):
        _, schema, _ = load_step("extract_model_inputs")

        with pytest.raises(jsonschema.ValidationError):
            validate_output(
                {"model_name": "Capital Market Model"},
                schema,
                "extract_model_inputs",
            )

    def test_accepts_minimal_required_fields(self):
        _, schema, _ = load_step("extract_model_inputs")

        payload = {
            "model_name": "Factor model",
            "variables": ["risk premium"],
        }

        validate_output(payload, schema, "extract_model_inputs")


class TestExtractModelMethodologyContract:
    def test_prompt_contains_methodology_focus(self):
        prompt, _, config = load_step("extract_model_methodology")

        assert "methodology" in prompt.lower()
        assert "mermaid_diagram" in prompt
        assert config["temperature"] == 0

    def test_accepts_expected_shape(self):
        _, schema, _ = load_step("extract_model_methodology")

        payload = {
            "steps_summary": "The report derives U.S. equity return from risk factors.",
            "steps_detailed": "1. Start from risk factors.\n2. Apply CAPM check.\n3. Produce forecast.",
            "mermaid_diagram": "flowchart TD\n    A[risk factors] --> B[forecast]\n    B --> C[CAPM check]\n    C --> D[output]",
            "sub_models": ["CAPM"],
            "assumptions": ["r_equity = r_f + beta * ERP", "mean reversion in valuations (implied)"],
        }

        validate_output(payload, schema, "extract_model_methodology")

    def test_requires_summary_detailed_and_mermaid(self):
        _, schema, _ = load_step("extract_model_methodology")

        with pytest.raises(jsonschema.ValidationError):
            validate_output(
                {"steps_summary": "Summary only"},
                schema,
                "extract_model_methodology",
            )

    def test_accepts_minimal_required_fields(self):
        _, schema, _ = load_step("extract_model_methodology")

        payload = {
            "steps_summary": "Simple approach.",
            "steps_detailed": "1. One step.",
            "mermaid_diagram": "flowchart TD\n    A --> B",
        }

        validate_output(payload, schema, "extract_model_methodology")


class TestExtractTableContract:
    def test_allows_null_table(self):
        _, schema, _ = load_step("extract_table")
        validate_output({"table": None}, schema, "extract_table")

    def test_accepts_expected_table_shape(self):
        _, schema, _ = load_step("extract_table")
        validate_output(
            {
                "table": {
                    "title": "U.S. Equity Return Assumptions",
                    "headers": ["Assumption", "Value"],
                    "rows": [["Return", "6.0%"]],
                }
            },
            schema,
            "extract_table",
        )
