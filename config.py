import os
from pathlib import Path

OCR_MODEL_ID = "zai-org/GLM-OCR"
STEPS_DIR = Path(__file__).parent / "steps"
ACTIVE_STEPS = ["extract_model_name", "extract_table"]
MAX_PAGES_PER_BATCH = 75
ZAI_MAX_PAGES = 100  # ZAI layout_parsing API hard limit
OCR_PROVIDER = os.environ.get("OCR_PROVIDER", "zai")
SCOUT_PAGE_PADDING = 1
