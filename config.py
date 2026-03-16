from pathlib import Path

OCR_MODEL_ID = "zai-org/GLM-OCR"
STEPS_DIR = Path(__file__).parent / "steps"
ACTIVE_STEPS = ["extract_model_name", "extract_table"]
