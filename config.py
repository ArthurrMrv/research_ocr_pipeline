import os
from pathlib import Path

OCR_MODEL_ID = "zai-org/GLM-OCR"
STEPS_DIR = Path(__file__).parent / "steps"
ACTIVE_STEPS = ["extract_model_name"] #["extract_model_name", "extract_table"]
MAX_PAGES_PER_BATCH = 75
ZAI_MAX_PAGES = 75  # safe batch size for ZAI layout_parsing API (hard limit is 100)
OCR_PROVIDER = os.environ.get("OCR_PROVIDER", "zai")
SCOUT_SCORE_THRESHOLD = float(os.environ.get("SCOUT_SCORE_THRESHOLD", "0.6"))
