import os
from pathlib import Path

OCR_MODEL_ID = "zai-org/GLM-OCR"
STEPS_DIR = Path(__file__).parent / "steps"
ACTIVE_STEPS = ["extract_model_name", "extract_table"]
MAX_PAGES_PER_BATCH = 75
OCR_PROVIDER = os.environ.get("OCR_PROVIDER", "local")
SCOUT_PAGE_PADDING = 1
