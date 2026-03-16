from pathlib import Path

OCR_MODEL_ID = "zai-org/GLM-OCR"
FORMATTING_MODEL = "kimi-k2.5"
FORMATTING_BASE_URL = "https://api.moonshot.ai/v1"
PROMPTS_DIR = Path(__file__).parent / "prompts"
ACTIVE_STEPS = ["extract_model_name", "extract_table"]
