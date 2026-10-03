import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

IMG_SIZE = 128
SEED = 42

# Sheet layout. 4 x 3 = 12 signatures per generated sheet.
# If the image model struggles with 12, use 3 x 3 (9) or 2 x 3 (6) and update the number words in generate_sheets.py
GRID_ROWS = 3
GRID_COLS = 4
PROMPT_VERSION = "v1"
