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
            "uses_regressions": 1,
            "uses_simulations": 0,
            "uses_averages": 0,
            "uses_mean_reversion": 1,
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
            "uses_regressions": 0,
            "uses_simulations": 0,
            "uses_averages": 0,
            "uses_mean_reversion": 0,
        }

        validate_output(payload, schema, "extract_model_methodology")

    def test_rejects_invalid_technique_flag_value(self):
        _, schema, _ = load_step("extract_model_methodology")

        payload = {
            "steps_summary": "Simple approach.",
            "steps_detailed": "1. One step.",
            "mermaid_diagram": "flowchart TD\n    A --> B",
            "uses_regressions": 2,
            "uses_simulations": 0,
            "uses_averages": 0,
            "uses_mean_reversion": 0,
        }

        with pytest.raises(jsonschema.ValidationError):
            validate_output(payload, schema, "extract_model_methodology")


class TestExtractModelAssumptionsContract:
    def test_prompt_contains_classification_focus(self):
        prompt, _, config = load_step("extract_model_assumptions")

        assert "classification" in prompt.lower()
        assert "mean-reversion" in prompt
        assert "historical" in prompt
        assert "forward-looking" in prompt
        assert config["temperature"] == 0

    def test_accepts_expected_shape(self):
        _, schema, _ = load_step("extract_model_assumptions")

        payload = {
            "assumptions": [
                {
                    "assumption": "CAPE reverts to long-run average",
                    "building_block": "valuation",
                    "classification": "mean-reversion",
                },
                {
                    "assumption": "GDP growth = historical 3%",
                    "building_block": "earnings growth",
                    "classification": "historical",
                },
                {
                    "assumption": "AI drives GDP growth to 5%",
                    "building_block": "earnings growth",
                    "classification": "forward-looking",
                },
            ],
            "techniques_used": [
                {"technique_name": "building-block approach", "complexity": 5},
                {"technique_name": "mean-reversion model", "complexity": 6},
            ],
            "sophistication_index": 6,
            "sophistication_explanation": "The model relies on a building-block approach with mean-reversion assumptions, representing intermediate sophistication.",
        }

        validate_output(payload, schema, "extract_model_assumptions")

    def test_requires_all_top_level_fields(self):
        _, schema, _ = load_step("extract_model_assumptions")

        with pytest.raises(jsonschema.ValidationError):
            validate_output(
                {"assumptions": []},
                schema,
                "extract_model_assumptions",
            )

    def test_rejects_invalid_classification(self):
        _, schema, _ = load_step("extract_model_assumptions")

        payload = {
            "assumptions": [
                {
                    "assumption": "test",
                    "building_block": "test",
                    "classification": "speculative",
                }
            ],
            "techniques_used": [],
            "sophistication_index": 5,
            "sophistication_explanation": "test",
        }

        with pytest.raises(jsonschema.ValidationError):
            validate_output(payload, schema, "extract_model_assumptions")

    def test_rejects_index_out_of_range(self):
        _, schema, _ = load_step("extract_model_assumptions")

        payload = {
            "assumptions": [],
            "techniques_used": [],
            "sophistication_index": 11,
            "sophistication_explanation": "test",
        }

        with pytest.raises(jsonschema.ValidationError):
            validate_output(payload, schema, "extract_model_assumptions")

    def test_rejects_extra_properties_on_assumption(self):
        _, schema, _ = load_step("extract_model_assumptions")

        payload = {
            "assumptions": [
                {
                    "assumption": "test",
                    "building_block": "test",
                    "classification": "historical",
                    "extra_field": "not allowed",
                }
            ],
            "techniques_used": [],
            "sophistication_index": 3,
            "sophistication_explanation": "test",
        }

        with pytest.raises(jsonschema.ValidationError):
            validate_output(payload, schema, "extract_model_assumptions")


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
