import jsonschema
import pytest

from pipeline.formatting import load_step, validate_output


class TestExtractModelNameContract:
    def test_prompt_and_config_capture_fb_requirements(self):
        prompt, _, config = load_step("extract_model_name")

        assert (
            "must explicitly include all stated assumptions, constraints, and conditions relevant to the U.S. equity return forecast"
            in prompt
        )
        assert config["temperature"] == 0

    def test_accepts_expected_shape(self):
        _, schema, _ = load_step("extract_model_name")

        payload = {
            "model_name": "Capital Market Model",
            "notes_model": "Starts from risk factors and uses CAPM as a check.",
            "steps_summary": "The report derives the U.S. equity return forecast from key risk factors and cross-checks the result with CAPM.",
            "steps_detailed": "1. The report starts from the relevant risk factors.\n2. It translates those stated inputs into the U.S. equity return forecast.\n3. It explicitly assumes CAPM is used as a consistency check rather than the sole engine.\n4. It applies that constraint when forming the final forecast.",
            "variables": ["risk factors", "CAPM"],
            "variables_important": ["risk factors", "CAPM"],
            "mermaid_diagram": "flowchart TD\n    A[risk factors] --> B[forecast construction]\n    B --> C[CAPM check]\n    C --> D[U.S. equity return forecast]",
            "sub_models": ["CAPM"],
        }

        validate_output(payload, schema, "extract_model_name")

    def test_requires_summaries_and_mermaid(self):
        _, schema, _ = load_step("extract_model_name")

        with pytest.raises(jsonschema.ValidationError):
            validate_output(
                {
                    "model_name": "Capital Market Model",
                    "variables": ["risk factors"],
                },
                schema,
                "extract_model_name",
            )


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
